from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

if TYPE_CHECKING:
    from approximate_similarity import SimilarityConfig

"""
Utilities for batched GPU/CPU similarity pipelines.

This module centralizes a small set of helpers that are shared between
`batched_gpu.py` and `batched_exact_cosine.py`:

- `BatchedGPUConfig`: dataclass holding batch-level configuration
- `_yield_batches_dynamic`: generator that splits a CSR matrix into batches
  based on peak counts (non-zero entries)
- `_log_message_to_file`: logging helper that logs to the module logger and
  optionally writes messages to a file

Why: these helpers were pulled out into a separate module to avoid circular
imports between the GPU batching code and the exact-cosine pipeline while
keeping the batching and logging behavior consistent.
"""


logger = logging.getLogger(__name__)

__all__ = [
    "BatchedGPUConfig",
    "_yield_batches_dynamic",
    "_log_message_to_file",
    "logger",
]


@dataclass
class BatchedGPUConfig:
    """
    Configuration for batched GPU similarity computation (batch-level only).

    This dataclass encapsulates the parameters relevant to batch-level behavior
    (batch size, write interval, etc.). The finer grained binning/expansion
    configuration remains in the `SimilarityConfig` used by the approximate stage.

    Fields:
        approx_config: Configuration for approximate similarity (type hinted as a
            forward reference to avoid tight import coupling).
        batch_size: Minimum number of spectra to include in an approximate batch.
        gpu_batch_write_interval: How many GPU batches to process before flushing
            results to disk.
        target_gpu_mem_ratio: Fraction (0, 1] of free GPU memory we aim to use.
        max_peaks_per_batch: Optional user limit (total peaks) to clamp dynamic estimates.
    """

    approx_config: "SimilarityConfig"
    batch_size: int = 1000
    gpu_batch_write_interval: int = 10
    target_gpu_mem_ratio: float = 0.1
    max_peaks_per_batch: int | None = None

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.gpu_batch_write_interval <= 0:
            raise ValueError("gpu_batch_write_interval must be positive")
        if not (0.0 < self.target_gpu_mem_ratio <= 1.0):
            raise ValueError("target_gpu_mem_ratio must be in (0.0, 1.0]")
        if self.max_peaks_per_batch is not None and self.max_peaks_per_batch <= 0:
            raise ValueError("max_peaks_per_batch must be positive")
        # Note: approx_config validation is the responsibility of its own class.


def _log_message_to_file(
    message: str,
    log_path: Optional[Path] = None,
    level: int = logging.INFO,
    overwrite: bool = False,
) -> None:
    """
    Log a message to the module logger and optionally append it to a file.

    Args:
        message: Message string to log.
        log_path: If provided, the message will also be appended to this file.
                  If None, only the logger is used.
        level: Logging level (e.g., logging.INFO).
        overwrite: If True and log_path is provided, the file is opened in write mode
                   (overwriting any existing contents). Otherwise, append mode is used.
    """
    logger.log(level, message)
    if log_path is None:
        return

    mode = "w" if overwrite else "a"
    with open(log_path, mode) as f:
        f.write(f"{message}\n")


def _yield_batches_dynamic(
    csr_matrix: "sp.csr_matrix",
    global_idxs: NDArray[np.int64],
    max_peaks: int,
    min_batch_size: int = 100,
) -> Iterator[tuple[int, int, "sp.csr_matrix", NDArray[np.int64]]]:
    """
    Yield batches (start_idx, end_idx, csr_batch, idxs_batch) based on non-zero counts.

    This generator greedily accumulates rows until the total non-zero element
    (peak) count in the batch is less than or equal to `max_peaks`. A minimum
    batch size is enforced when possible.

    Args:
        csr_matrix: Input CSR matrix where each row corresponds to a spectrum.
        global_idxs: Global indices corresponding to rows of `csr_matrix`.
        max_peaks: Target maximum number of peaks (non-zero entries) per batch.
        min_batch_size: Enforced minimum number of spectra per batch (unless end of data).

    Yields:
        Tuples of (start_idx, end_idx, csr_matrix[start_idx:end_idx], global_idxs[start_idx:end_idx]).
    """
    shape = getattr(csr_matrix, "shape", None)
    assert shape is not None, "csr_matrix.shape must not be None"
    n_spectra = int(shape[0])
    start_idx = 0

    # `indptr` has length n_spectra + 1 and is monotonic; using it allows
    # O(log n) search for the end index that keeps us within `max_peaks`.
    indptr = csr_matrix.indptr
    # Statically assert availability and shape of `indptr` so static type/check
    # tools do not treat it as a possibly-None value (which causes ``None[...]`` errors).
    assert indptr is not None, "csr_matrix.indptr must not be None"
    indptr = np.asarray(indptr)
    assert indptr.ndim == 1 and indptr.size >= n_spectra + 1, (
        f"csr_matrix.indptr must be 1D and have length >= n_spectra + 1, got shape {indptr.shape}"
    )

    while start_idx < n_spectra:
        # Determine target peaks threshold for this batch
        target_peaks = indptr[start_idx] + max_peaks

        # Find the first indptr entry greater than target_peaks, subtract 1
        candidate_end = np.searchsorted(indptr, target_peaks, side="right") - 1

        # Ensure at least one row per batch
        if candidate_end <= start_idx:
            candidate_end = start_idx + 1

        # Enforce minimum batch size if possible (unless we hit end)
        if candidate_end - start_idx < min_batch_size:
            candidate_end = min(start_idx + min_batch_size, n_spectra)

        # Clamp to bounds
        end_idx = min(candidate_end, n_spectra)

        batch_csr = csr_matrix[start_idx:end_idx]
        batch_idxs = global_idxs[start_idx:end_idx]

        yield start_idx, end_idx, batch_csr, batch_idxs

        start_idx = end_idx
