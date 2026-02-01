"""
Async parquet writer with memory-limited queue.

This module provides thread-based non-blocking I/O for writing similarity
results to parquet files. The key innovation is the MemoryLimitedQueue that
limits queue size by memory usage rather than item count.

Why memory-limited queue:
- Batch sizes vary wildly (some batches have 1M pairs, others have 1K)
- Fixed item count limit (e.g., maxsize=5) doesn't prevent OOM
- Memory-based limit adapts to actual data sizes

Why separate writer thread:
- GPU computation is fast, parquet writes can be slow (compression)
- Overlapping I/O with computation improves throughput
- Queue acts as buffer to smooth out write latency spikes
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from queue import Empty, Full
from threading import Thread
from typing import Optional

import numpy as np
import polars as pl
import psutil
from numpy.typing import NDArray


# =============================================================================
# Memory-Limited Queue
# =============================================================================


class MemoryLimitedQueue:
    """
    Thread-safe queue that limits total memory of items, not count.

    Why: GPU batches produce variable-sized result arrays. A batch with
    many above-threshold pairs produces megabytes of data, while sparse
    batches produce kilobytes. Limiting by memory prevents OOM regardless
    of batch characteristics.

    Behavior:
    - put() blocks when adding would exceed max_memory_bytes
    - get() signals space freed, unblocking waiting put() calls
    - Blocking is cooperative (uses Condition variables)

    Example:
        queue = MemoryLimitedQueue(max_memory_bytes=1_000_000_000)  # 1GB
        queue.put({"arr": np.zeros(100_000_000, dtype=np.float32)})  # ~400MB
        queue.put({"arr": np.zeros(100_000_000, dtype=np.float32)})  # ~400MB
        # Next put would block until get() frees space
    """

    def __init__(self, max_memory_bytes: int):
        """
        Initialize memory-limited queue.

        Args:
            max_memory_bytes: Maximum total memory of queued items in bytes
        """
        assert max_memory_bytes > 0, (
            f"max_memory_bytes must be positive, got {max_memory_bytes}"
        )
        self.max_memory_bytes = max_memory_bytes
        self.current_memory_bytes = 0
        self._queue: list[tuple[dict, int]] = []  # (data, size_bytes)
        self._lock = threading.Lock()
        self._not_full = threading.Condition(self._lock)
        self._not_empty = threading.Condition(self._lock)
        self._stop = False

    @staticmethod
    def estimate_frame_memory(data: dict) -> int:
        """
        Compute memory usage of numpy arrays in data dict.

        Args:
            data: Dictionary with numpy array values

        Returns:
            Total bytes across all numpy arrays
        """
        total = 0
        for val in data.values():
            if isinstance(val, np.ndarray):
                total += val.nbytes
        return total

    @staticmethod
    def compute_default_max_memory(safety_ratio: float = 0.80) -> int:
        """
        Compute max queue memory based on available system RAM.

        Why 80% default: Leaves headroom for OS, Python overhead, and other
        processes. Prevents swapping which would kill performance.

        Args:
            safety_ratio: Fraction of available memory to use (0.0, 1.0]

        Returns:
            Max memory in bytes
        """
        assert 0.0 < safety_ratio <= 1.0, (
            f"safety_ratio must be in (0, 1], got {safety_ratio}"
        )
        mem = psutil.virtual_memory()
        return int(mem.available * safety_ratio)

    def put(
        self, data: dict, block: bool = True, timeout: Optional[float] = None
    ) -> None:
        """
        Add item to queue, blocking if memory limit would be exceeded.

        Args:
            data: Dictionary with numpy array values
            block: If True, wait for space. If False, raise Full immediately.
            timeout: Max seconds to wait (None = wait forever)

        Raises:
            Full: If non-blocking and no space, or timeout exceeded
        """
        item_size = self.estimate_frame_memory(data)

        with self._not_full:
            # Block until there's room
            while (
                self.current_memory_bytes + item_size > self.max_memory_bytes
                and self._queue  # Allow if queue is empty (single large item)
            ):
                if not block:
                    raise Full("Queue memory limit exceeded")
                if not self._not_full.wait(timeout):
                    raise Full("Timed out waiting for queue space")

            self.current_memory_bytes += item_size
            self._queue.append((data, item_size))
            self._not_empty.notify()

    def get(self, block: bool = True, timeout: Optional[float] = None) -> dict:
        """
        Remove and return item from queue.

        Args:
            block: If True, wait for item. If False, raise Empty immediately.
            timeout: Max seconds to wait (None = wait forever)

        Returns:
            Data dictionary

        Raises:
            Empty: If non-blocking and queue empty, timeout exceeded, or stopped
        """
        with self._not_empty:
            while not self._queue and not self._stop:
                if not block:
                    raise Empty("Queue is empty")
                if not self._not_empty.wait(timeout):
                    raise Empty("Timed out waiting for item")

            if not self._queue:
                raise Empty("Queue is empty and stopped")

            data, item_size = self._queue.pop(0)
            self.current_memory_bytes -= item_size
            self._not_full.notify()
            return data

    def stop(self) -> None:
        """Signal queue to stop, waking any blocked get() calls."""
        with self._not_empty:
            self._stop = True
            self._not_empty.notify_all()

    def empty(self) -> bool:
        """Check if queue is empty."""
        with self._lock:
            return len(self._queue) == 0

    def qsize(self) -> int:
        """Return number of items in queue."""
        with self._lock:
            return len(self._queue)

    def memory_usage(self) -> int:
        """Return current memory usage in bytes."""
        with self._lock:
            return self.current_memory_bytes


# =============================================================================
# Result Buffer (Thread-Safe Accumulator)
# =============================================================================


class ResultBuffer:
    """
    Thread-safe accumulator for pair results before writing.

    Why: Collecting multiple GPU batch results before submitting to the writer
    reduces write frequency and improves throughput.
    """

    def __init__(self) -> None:
        self.left_idxs: list[NDArray[np.int32]] = []
        self.right_idxs: list[NDArray[np.int32]] = []
        self.similarities: list[NDArray[np.float32]] = []
        self.lock = threading.Lock()

    def add(
        self,
        left: NDArray[np.int32],
        right: NDArray[np.int32],
        sims: NDArray[np.float32],
    ) -> None:
        """Add a batch of results."""
        with self.lock:
            self.left_idxs.append(left)
            self.right_idxs.append(right)
            self.similarities.append(sims)

    def flush(self) -> Optional[dict]:
        """Flush accumulated results and return as dict (or None if empty)."""
        with self.lock:
            if not self.left_idxs:
                return None

            data = {
                "idx_left": np.concatenate(self.left_idxs).astype(np.int32, copy=False),
                "idx_right": np.concatenate(self.right_idxs).astype(
                    np.int32, copy=False
                ),
                "similarity": np.concatenate(self.similarities).astype(
                    np.float32, copy=False
                ),
            }

            self.left_idxs.clear()
            self.right_idxs.clear()
            self.similarities.clear()

            return data

    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        with self.lock:
            return len(self.left_idxs) == 0


# =============================================================================
# Async Parquet Writer
# =============================================================================


class AsyncParquetWriter:
    """
    Thread-based writer for non-blocking parquet writes with memory limiting.

    Why: GPU computation is fast, but parquet writes can be slow (especially with
    compression). Using a separate writer thread with a memory-limited queue allows
    the GPU to continue computing while previous results are being written.

    The writer accumulates chunks and appends to the parquet file in batches.

    Memory limiting:
    - If writer_max_queue_memory_bytes is specified, use that limit
    - Otherwise, auto-compute as 80% of available system RAM
    - When queue memory is full, write_batch() blocks until writer catches up
    """

    def __init__(
        self,
        output_path: Path,
        max_queue_memory_bytes: Optional[int] = None,
        memory_safety_ratio: float = 0.80,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize async writer.

        Args:
            output_path: Path to output parquet file
            max_queue_memory_bytes: Max queue memory in bytes (None = auto)
            memory_safety_ratio: Safety ratio for auto memory limit (default: 0.80)
            logger: Optional logger for progress reporting
        """
        self.output_path = output_path
        self.logger = logger

        # Compute queue memory limit
        if max_queue_memory_bytes is not None:
            queue_limit = max_queue_memory_bytes
        else:
            queue_limit = MemoryLimitedQueue.compute_default_max_memory(
                memory_safety_ratio
            )

        if logger:
            logger.info(
                f"  [Writer] Queue memory limit: {queue_limit / 1e9:.2f} GB"
            )

        self.queue = MemoryLimitedQueue(queue_limit)
        self.thread = Thread(target=self._writer_loop, daemon=True)
        self._stop_event = threading.Event()
        self._exception: Optional[Exception] = None
        self.chunks_written = 0
        self.pairs_written = 0

    def start(self) -> None:
        """Start the writer thread."""
        self.thread.start()

    def write_batch(self, data: dict) -> None:
        """
        Submit a batch for writing (blocks if queue memory is full).

        Args:
            data: Dictionary with keys 'idx_left', 'idx_right', 'similarity'
        """
        if self._exception is not None:
            raise RuntimeError(f"Writer thread failed: {self._exception}")
        self.queue.put(data)

    def stop(self) -> None:
        """Signal the writer thread to stop and wait for completion."""
        self._stop_event.set()
        self.queue.stop()
        self.thread.join()
        if self._exception is not None:
            raise RuntimeError(f"Writer thread failed: {self._exception}")

    def _writer_loop(self) -> None:
        """Writer thread main loop."""
        try:
            chunks: list[pl.DataFrame] = []

            while not self._stop_event.is_set() or not self.queue.empty():
                try:
                    data = self.queue.get(timeout=0.1)
                    chunks.append(pl.DataFrame(data))
                    self.pairs_written += len(data["idx_left"])

                    # Write every 10 chunks to balance I/O frequency vs memory
                    if len(chunks) >= 10:
                        self._write_chunks(chunks)
                        chunks.clear()

                except Empty:
                    continue

            # Write any remaining chunks
            if chunks:
                self._write_chunks(chunks)

        except Exception as e:
            self._exception = e
            if self.logger:
                self.logger.error(f"AsyncParquetWriter failed: {e}")

    def _write_chunks(self, chunks: list[pl.DataFrame]) -> None:
        """
        Write accumulated chunks to parquet file (append mode).

        Why append mode: We want to incrementally build the output file as
        batches complete, rather than accumulating everything in memory.
        """
        df = pl.concat(chunks)

        # Ensure int32 dtypes for indices
        df = df.with_columns(
            [
                pl.col("idx_left").cast(pl.Int32),
                pl.col("idx_right").cast(pl.Int32),
                pl.col("similarity").cast(pl.Float32),
            ]
        )

        # Append to file (create if first write)
        if self.chunks_written == 0:
            df.write_parquet(self.output_path)
        else:
            # Append mode using pyarrow
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = df.to_arrow()

            # Read existing and append
            existing = pq.read_table(self.output_path)
            combined = pa.concat_tables([existing, table])
            pq.write_table(combined, self.output_path)

        self.chunks_written += 1

        if self.logger:
            self.logger.info(
                f"  [Writer] Wrote chunk {self.chunks_written} "
                f"({len(df)} pairs, total={self.pairs_written})"
            )
