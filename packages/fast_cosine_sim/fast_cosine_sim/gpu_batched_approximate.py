from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import polars as pl
import scipy.sparse as sp

from .config import ApproximateGpuBatchedSimilarityConfig

# Why: conservative overhead multipliers for the batching memory model.
GPU_CSR_OVERHEAD_FACTOR: float = 1.25
GPU_SIM_TEMP_OVERHEAD_FACTOR: float = 1.20


def _collect_if_lazy(frame: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    return frame.collect() if isinstance(frame, pl.LazyFrame) else frame


def _ensure_row_index_column(
    frame: pl.DataFrame | pl.LazyFrame, *, index_column_name: str
) -> pl.DataFrame | pl.LazyFrame:
    if index_column_name in frame.columns:
        return frame
    return frame.with_row_index(index_column_name)


def _apply_intensity_power_if_needed(intensity: np.ndarray, *, power: float) -> np.ndarray:
    # Contract from you: "enable dictating the power ... with a check for 0 to prevent extra work".
    # Semantics: power==0 is treated as identity (skip work), not x**0 == 1.
    if float(power) == 0.0:
        return intensity
    # Why: avoid implicit float64 upcast.
    return np.power(intensity, float(power), dtype=intensity.dtype)


def _sparse_bin_spectra_df_to_csr(
    df: pl.DataFrame,
    *,
    mz_column_name: str,
    intensity_column_name: str,
    upper_bound: float,
    bin_size: float,
    intensity_power: float,
    intensity_dtype: np.dtype,
    csr_index_dtype: np.dtype,
) -> sp.csr_matrix:
    assert mz_column_name in df.columns, f"Missing column: {mz_column_name}"
    assert intensity_column_name in df.columns, f"Missing column: {intensity_column_name}"
    assert upper_bound > 0.0, "upper_bound must be positive"
    assert bin_size > 0.0, "bin_size must be positive"

    nbins = int(np.floor(float(upper_bound) / float(bin_size))) + 1
    assert nbins > 0, f"Computed nbins must be positive, got {nbins}"

    mz_list: list[np.ndarray] = df[mz_column_name].to_list()
    intensity_list: list[np.ndarray] = df[intensity_column_name].to_list()
    assert len(mz_list) == len(intensity_list) == len(df), "List columns must align with df height"

    data_parts: list[np.ndarray] = []
    indices_parts: list[np.ndarray] = []
    indptr = np.zeros(len(df) + 1, dtype=np.int64)

    nnz_total = 0
    for row_i, (mz, intensity) in enumerate(zip(mz_list, intensity_list, strict=True)):
        mz = np.asarray(mz)
        intensity = np.asarray(intensity)

        assert mz.ndim == 1 and intensity.ndim == 1, "mz/intensity must be 1D arrays per spectrum"
        assert mz.shape[0] == intensity.shape[0], "mz and intensity must have same length"

        if mz.size == 0:
            indptr[row_i + 1] = nnz_total
            continue

        bin_idx = np.floor(mz / float(bin_size)).astype(np.int64, copy=False)
        valid = (bin_idx >= 0) & (bin_idx < nbins)
        bin_idx = bin_idx[valid]
        if bin_idx.size == 0:
            indptr[row_i + 1] = nnz_total
            continue

        values = intensity[valid].astype(intensity_dtype, copy=False)
        values = _apply_intensity_power_if_needed(values, power=intensity_power)

        # Aggregate duplicate bins inside the spectrum.
        order = np.argsort(bin_idx, kind="mergesort")
        bin_idx = bin_idx[order]
        values = values[order]

        uniq_bins, start_positions = np.unique(bin_idx, return_index=True)
        summed = np.add.reduceat(values, start_positions)

        # Drop zeros to limit nnz and
        # GPU transfer.
        keep = summed != 0
        uniq_bins = uniq_bins[keep]
        summed = summed[keep]

        if uniq_bins.size:
            indices_parts.append(uniq_bins.astype(csr_index_dtype, copy=False))
            data_parts.append(summed.astype(intensity_dtype, copy=False))
            nnz_total += int(uniq_bins.size)

        indptr[row_i + 1] = nnz_total

    if nnz_total == 0:
        return sp.csr_matrix((len(df), nbins), dtype=intensity_dtype)

    data = np.concatenate(data_parts).astype(intensity_dtype, copy=False)
    indices = np.concatenate(indices_parts).astype(csr_index_dtype, copy=False)

    # CuPy sparse is most compatible with int32 indices/indptr; fail fast if too large.
    assert nnz_total <= np.iinfo(np.int32).max, (
        f"CSR nnz too large for int32 indptr (nnz_total={nnz_total}); "
        "lower batch sizes / reduce nnz or switch approach."
    )
    indptr_i32 = indptr.astype(csr_index_dtype, copy=False)

    return sp.csr_matrix((data, indices, indptr_i32), shape=(len(df), nbins))


def _normalize_csr_rows_inplace_gpu(x_gpu) -> None:
    import cupy as cp

    indptr = x_gpu.indptr
    data = x_gpu.data

    squared = data * data
    row_sums = cp.add.reduceat(squared, indptr[:-1])
    norms = cp.sqrt(row_sums)

    nonzero_rows = norms > 0
    scales = cp.ones_like(norms)
    scales[nonzero_rows] = 1.0 / norms[nonzero_rows]

    row_ids = cp.repeat(cp.arange(x_gpu.shape[0], dtype=cp.int32), cp.diff(indptr))
    data *= scales[row_ids]


def _pairs_above_threshold_from_sparse_dot_gpu(
    left_gpu,
    right_gpu,
    *,
    approx_threshold: float,
    similarity_dtype: np.dtype,
    upper_triangle_by_position: bool,
    left_global_ids: np.ndarray,
    right_global_ids: np.ndarray,
) -> pl.DataFrame:
    import cupy as cp

    sim = (left_gpu @ right_gpu.T).astype(similarity_dtype, copy=False)
    mask = sim >= cp.asarray(float(approx_threshold), dtype=sim.dtype)

    row_idx, col_idx = cp.nonzero(mask)

    if upper_triangle_by_position:
        # Self mode: only keep i<j where positions are within the same ordering.
        # Why: this is stable and avoids moving full id grids to GPU.
        keep = row_idx < col_idx
        row_idx = row_idx[keep]
        col_idx = col_idx[keep]

    if row_idx.size == 0:
        return pl.DataFrame(
            {
                "idx_left": np.empty((0,), dtype=left_global_ids.dtype),
                "idx_right": np.empty((0,), dtype=right_global_ids.dtype),
                "approx_similarity": np.empty((0,), dtype=np.dtype(similarity_dtype)),
            }
        )

    scores = sim[row_idx, col_idx]

    # Map local to global ids.
    left_ids = left_global_ids[cp.asnumpy(row_idx)]
    right_ids = right_global_ids[cp.asnumpy(col_idx)]

    return pl.DataFrame(
        {
            "idx_left": left_ids,
            "idx_right": right_ids,
            "approx_similarity": cp.asnumpy(scores),
        }
    )


def _estimate_max_peaks_per_batch(
    *,
    free_mem_bytes: int,
    target_gpu_memory_usage_ratio: float,
    min_spectra_per_batch: int,
    csr_index_itemsize: int,
    intensity_itemsize: int,
    similarity_itemsize: int,
) -> int:
    target_mem = int(float(free_mem_bytes) * float(target_gpu_memory_usage_ratio))
    assert target_mem > 0, "Computed target GPU memory target must be positive"

    # Similarity matrix dominates memory: min_batch^2 * sim_bytes with overhead.
    sim_bytes = int(min_spectra_per_batch) * int(min_spectra_per_batch) * int(similarity_itemsize)
    sim_bytes = int(sim_bytes * GPU_SIM_TEMP_OVERHEAD_FACTOR)

    remaining = target_mem - sim_bytes
    if remaining <= 0:
        # Fall back: allow at least one peak per spectrum.
        return int(min_spectra_per_batch)

    bytes_per_nnz = int(csr_index_itemsize + intensity_itemsize)
    bytes_per_nnz = int(bytes_per_nnz * GPU_CSR_OVERHEAD_FACTOR)

    max_peaks = remaining // (2 * bytes_per_nnz)
    return max(int(max_peaks), int(min_spectra_per_batch))


def _yield_batches_dynamic(
    csr_matrix: sp.csr_matrix,
    global_ids: np.ndarray,
    *,
    max_peaks: int,
    min_batch_size: int,
) -> Iterator[tuple[int, int, sp.csr_matrix, np.ndarray]]:
    n_rows = int(csr_matrix.shape[0])
    indptr = np.asarray(csr_matrix.indptr)
    assert indptr.ndim == 1 and indptr.size == n_rows + 1, "CSR indptr shape mismatch"

    start = 0
    while start < n_rows:
        target_peaks = int(indptr[start]) + int(max_peaks)
        end = int(np.searchsorted(indptr, target_peaks, side="right") - 1)

        if end <= start:
            end = start + 1
        if end - start < min_batch_size:
            end = min(start + min_batch_size, n_rows)
        end = min(end, n_rows)

        yield start, end, csr_matrix[start:end], global_ids[start:end]
        start = end


@dataclass(frozen=True, slots=True)
class _ParquetPartitionWriter:
    base_dir: Path

    def __post_init__(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def write_partition(self, df: pl.DataFrame, *, partition_index: int) -> None:
        path = self.base_dir / f"part-{partition_index:06d}.parquet"
        df.write_parquet(path)


def compute_gpu_batched_approximate_similarity_pairs(
    left: pl.DataFrame | pl.LazyFrame,
    config: ApproximateGpuBatchedSimilarityConfig,
    *,
    right: pl.DataFrame | pl.LazyFrame | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    """GPU batched approximate similarity candidate generation.

    What this does:
      - bins per-spectrum peak lists into sparse vectors (CSR)
      - transfers CSR batches to GPU
      - computes approximate cosine (via L2-normalized sparse dot)
      - thresholds to emit candidate pairs
      - (optionally) writes batches to parquet partitions and returns a scan

    Modes:
      - self-comparison (right is None): upper triangle only (i<j in batch positions)
      - cross-comparison (right provided): full left x right (no triangular filter)

    Output:
      Columns: `idx_left`, `idx_right`, `approx_similarity`
      If config.batching.flush_to_parquet_every_n_batches is None:
        returns a `pl.DataFrame`
      Else:
        requires `output_parquet_path` be provided by setting it on `config` via your wrapper
        and returns a `pl.LazyFrame` via `pl.scan_parquet(dir/*.parquet)`
    """
    import cupy as cp

    if right is None:
        assert config.comparison_mode == "self", (
            "If right is None, config.comparison_mode must be 'self'"
        )
    else:
        assert config.comparison_mode == "cross", (
            "If right is provided, config.comparison_mode must be 'cross'"
        )

    left = _ensure_row_index_column(left, index_column_name=config.spectrum_id_column)
    right_frame = (
        None
        if right is None
        else _ensure_row_index_column(right, index_column_name=config.spectrum_id_column)
    )

    left_df = _collect_if_lazy(left)
    right_df = left_df if right_frame is None else _collect_if_lazy(right_frame)

    for label, df in (("left", left_df), ("right", right_df)):
        for col in (config.spectrum_id_column, config.mz_column, config.intensity_column):
            assert col in df.columns, f"{label} missing required column: {col}"

    # Build CSR matrices + global ids from frames.
    left_ids = (
        left_df[config.spectrum_id_column]
        .cast(_polars_dtype_from_numpy(config.dtypes.index_dtype))
        .to_numpy()
        .astype(config.dtypes.index_dtype, copy=False)
    )
    right_ids = (
        right_df[config.spectrum_id_column]
        .cast(_polars_dtype_from_numpy(config.dtypes.index_dtype))
        .to_numpy()
        .astype(config.dtypes.index_dtype, copy=False)
    )

    left_csr = _sparse_bin_spectra_df_to_csr(
        left_df,
        mz_column_name=config.mz_column,
        intensity_column_name=config.intensity_column,
        upper_bound=config.upper_mass_bound,
        bin_size=config.bin_size,
        intensity_power=config.intensity.power,
        intensity_dtype=config.dtypes.intensity_dtype,
        csr_index_dtype=config.dtypes.csr_index_dtype,
    )
    right_csr = (
        left_csr
        if right_frame is None
        else _sparse_bin_spectra_df_to_csr(
            right_df,
            mz_column_name=config.mz_column,
            intensity_column_name=config.intensity_column,
            upper_bound=config.upper_mass_bound,
            bin_size=config.bin_size,
            intensity_power=config.intensity.power,
            intensity_dtype=config.dtypes.intensity_dtype,
            csr_index_dtype=config.dtypes.csr_index_dtype,
        )
    )

    assert left_csr.shape[0] == left_ids.shape[0], "left ids and CSR rows must align"
    assert right_csr.shape[0] == right_ids.shape[0], "right ids and CSR rows must align"

    if left_csr.shape[0] == 0 or right_csr.shape[0] == 0:
        empty = pl.DataFrame(
            {
                "idx_left": np.empty((0,), dtype=config.dtypes.index_dtype),
                "idx_right": np.empty((0,), dtype=config.dtypes.index_dtype),
                "approx_similarity": np.empty((0,), dtype=config.dtypes.similarity_dtype),
            }
        )
        return empty if config.batching.flush_to_parquet_every_n_batches is None else empty.lazy()

    free_mem, _ = cp.cuda.Device(0).mem_info

    max_peaks = _estimate_max_peaks_per_batch(
        free_mem_bytes=int(free_mem),
        target_gpu_memory_usage_ratio=float(config.batching.target_gpu_memory_usage_ratio),
        min_spectra_per_batch=int(config.batching.min_spectra_per_batch),
        csr_index_itemsize=int(np.dtype(config.dtypes.csr_index_dtype).itemsize),
        intensity_itemsize=int(np.dtype(config.dtypes.intensity_dtype).itemsize),
        similarity_itemsize=int(np.dtype(config.dtypes.similarity_dtype).itemsize),
    )
    if config.batching.max_peaks_per_batch is not None:
        max_peaks = min(int(max_peaks), int(config.batching.max_peaks_per_batch))

    left_batches = list(
        _yield_batches_dynamic(
            left_csr,
            left_ids,
            max_peaks=max_peaks,
            min_batch_size=int(config.batching.min_spectra_per_batch),
        )
    )
    right_batches = (
        left_batches
        if right_frame is None
        else list(
            _yield_batches_dynamic(
                right_csr,
                right_ids,
                max_peaks=max_peaks,
                min_batch_size=int(config.batching.min_spectra_per_batch),
            )
        )
    )

    should_write = config.batching.flush_to_parquet_every_n_batches is not None
    if should_write:
        assert config.output_parquet_path is not None, (
            "When flush_to_parquet_every_n_batches is set, config.output_parquet_path must be set"
        )
        out_dir = Path(config.output_parquet_path)
        assert not out_dir.exists(), (
            f"Output path already exists: {out_dir}. Remove it or choose a new one."
        )
        writer = _ParquetPartitionWriter(out_dir)
        write_every = int(config.batching.flush_to_parquet_every_n_batches)
        buffer_parts: list[pl.DataFrame] = []
        part_index = 0
    else:
        collected_parts: list[pl.DataFrame] = []

    batch_counter = 0

    for _, _, right_batch_csr, right_batch_ids in right_batches:
        right_gpu = cp.sparse.csr_matrix(right_batch_csr)
        _normalize_csr_rows_inplace_gpu(right_gpu)

        for _, _, left_batch_csr, left_batch_ids in left_batches:
            left_gpu = cp.sparse.csr_matrix(left_batch_csr)
            _normalize_csr_rows_inplace_gpu(left_gpu)

            pairs = _pairs_above_threshold_from_sparse_dot_gpu(
                left_gpu,
                right_gpu,
                approx_threshold=float(config.approx_threshold),
                similarity_dtype=config.dtypes.similarity_dtype,
                upper_triangle_by_position=(right_frame is None),
                left_global_ids=left_batch_ids,
                right_global_ids=right_batch_ids,
            )

            if pairs.height > 0:
                if should_write:
                    buffer_parts.append(pairs)
                else:
                    collected_parts.append(pairs)

            batch_counter += 1
            if should_write and (batch_counter % write_every == 0) and buffer_parts:
                writer.write_partition(pl.concat(buffer_parts), partition_index=part_index)
                buffer_parts.clear()
                part_index += 1

    if should_write:
        if buffer_parts:
            writer.write_partition(pl.concat(buffer_parts), partition_index=part_index)

        out_dir = Path(config.output_parquet_path)  # type: ignore[arg-type]
        return pl.scan_parquet(str(out_dir / "*.parquet"))

    if not collected_parts:
        return pl.DataFrame(
            {
                "idx_left": np.empty((0,), dtype=config.dtypes.index_dtype),
                "idx_right": np.empty((0,), dtype=config.dtypes.index_dtype),
                "approx_similarity": np.empty((0,), dtype=config.dtypes.similarity_dtype),
            }
        )

    return pl.concat(collected_parts)
# 