from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cupyx.scipy.sparse as cps
import numpy as np
import polars as pl
import scipy.sparse as sp
from numpy.typing import NDArray

from .config import ApproximateGpuBatchedSimilarityConfig

# Why: conservative overhead multipliers for the batching memory model.
GPU_CSR_OVERHEAD_FACTOR: float = 1.25
GPU_SIM_TEMP_OVERHEAD_FACTOR: float = 1.20


class LoggerLike(Protocol):
    def info(self, message: str) -> None: ...


def _log(logger: LoggerLike | None, message: str) -> None:
    # Why: caller-provided loggers can vary; fail-safe logging avoids breaking GPU jobs.
    if logger is None:
        return
    try:
        logger.info(message)
    except Exception:
        return


def _collect_if_lazy(frame: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    return frame.collect() if isinstance(frame, pl.LazyFrame) else frame


def _ensure_row_index_column(
    frame: pl.DataFrame | pl.LazyFrame, *, index_column_name: str
) -> pl.DataFrame | pl.LazyFrame:
    if index_column_name in frame.collect_schema().names():
        return frame
    return frame.with_row_index(index_column_name)


def _apply_intensity_power_if_needed(
    intensity: np.ndarray, *, power: float
) -> np.ndarray:
    # Contract from you:
    #   - power==1.0 => skip extra work (identity)
    #   - power==0.0 => presence-only (set all non-zero to 1.0)
    p = float(power)
    if p == 1.0:
        return intensity
    if p == 0.0:
        # Note: intensity here is already filtered to valid bins and zeros are later dropped after summing.
        return np.ones_like(intensity)
    # Why: avoid implicit float64 upcast.
    return np.power(intensity, p, dtype=intensity.dtype)


def _flatten_spectra_to_numpy(
    df: pl.DataFrame, *, mz_col: str, intensity_col: str, spectrum_index_col: str
) -> tuple[NDArray[np.float64], NDArray[np.float32], NDArray[np.int32], int]:
    """
    Flatten list-valued spectrum columns into NumPy arrays via Polars explode.

    Returns: (flat_mzs, flat_ints, spec_pos, n_spec)
      - flat_mzs: float64 m/z values (exploded)
      - flat_ints: float32 intensities (exploded)
      - spec_pos: int32 position index of the parent spectrum for each peak
      - n_spec: number of spectra (rows in original df)
    """
    n_spec = int(len(df))
    if n_spec == 0:
        return (
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float32),
            np.asarray([], dtype=np.int32),
            0,
        )

    df_idx = df.with_row_index(spectrum_index_col)
    exploded = df_idx.explode([mz_col, intensity_col])
    if len(exploded) == 0:
        return (
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float32),
            np.asarray([], dtype=np.int32),
            n_spec,
        )

    exploded = exploded.with_columns(
        [
            pl.col(mz_col).cast(pl.Float64),
            pl.col(intensity_col).cast(pl.Float32),
            pl.col(spectrum_index_col).cast(pl.Int32),
        ]
    )

    flat_mzs: NDArray[np.float64] = exploded.get_column(mz_col).to_numpy()
    flat_ints: NDArray[np.float32] = exploded.get_column(intensity_col).to_numpy()
    spec_pos: NDArray[np.int32] = exploded.get_column(spectrum_index_col).to_numpy()

    return flat_mzs, flat_ints, spec_pos, n_spec


def _sparse_bin_flat_spectra_to_csr(
    *,
    flat_mzs: NDArray[np.float64],
    flat_ints: NDArray[np.float32],
    spec_pos: NDArray[np.int32],
    n_spec: int,
    upper_bound: float,
    bin_size: float,
    intensity_power: float,
    csr_index_dtype: np.dtype,
    intensity_dtype: np.dtype,
) -> sp.csr_matrix:
    """
    Turn flattened arrays into a sparse CSR matrix (n_spec, nbins).

    Binning uses bin = rint(mz / bin_size) (matches the experiment path).
    Duplicates are summed via COO -> CSR.

    Intensity semantics:
      - power == 1.0 => identity
      - power == 0.0 => presence-only (set to 1)
      - otherwise => intensity ** power
    """
    assert upper_bound > 0.0, "upper_bound must be positive"
    assert bin_size > 0.0, "bin_size must be positive"

    nbins = int(np.floor(float(upper_bound) / float(bin_size))) + 1
    assert nbins > 0, f"Computed nbins must be positive, got {nbins}"

    if n_spec == 0:
        return sp.csr_matrix((0, nbins), dtype=intensity_dtype)
    if flat_mzs.size == 0 or flat_ints.size == 0:
        return sp.csr_matrix((n_spec, nbins), dtype=intensity_dtype)

    mass_bins = np.rint(flat_mzs / float(bin_size)).astype(csr_index_dtype, copy=False)

    # Keep only in-range bins and positive intensities.
    valid_mask = (mass_bins >= 0) & (mass_bins < nbins) & (flat_ints > 0)
    if not np.any(valid_mask):
        return sp.csr_matrix((n_spec, nbins), dtype=intensity_dtype)

    mass_bins = mass_bins[valid_mask].astype(csr_index_dtype, copy=False)
    # Why: COO row indices must be integer; keep as int32 for SciPy compatibility.
    spec_pos = spec_pos[valid_mask].astype(np.int32, copy=False)

    # NumPy 2.x removed `copy=` from `np.asarray(...)` (it's available on `np.array(...)`).
    # Why: we want a C-contiguous float32 view when possible, but must remain compatible across NumPy versions.
    weights = np.array(
        flat_ints[valid_mask], dtype=intensity_dtype, order="C", copy=False
    )
    weights = _apply_intensity_power_if_needed(
        weights, power=float(intensity_power)
    ).astype(intensity_dtype, copy=False)

    coo = sp.coo_matrix(
        (weights, (spec_pos, mass_bins)),
        shape=(n_spec, nbins),
        dtype=intensity_dtype,
    )
    csr = sp.csr_matrix(coo.tocsr())

    # CuPy sparse is most compatible with int32 indices/indptr; fail fast if too large.
    assert csr.nnz <= np.iinfo(np.int32).max, (
        f"CSR nnz too large for int32 indptr (nnz_total={csr.nnz}); "
        "lower batch sizes / reduce nnz or switch approach."
    )

    # Enforce dtypes for GPU transfer.
    assert csr.indices is not None, "CSR indices must not be None"
    assert csr.indptr is not None, "CSR indptr must not be None"
    csr.indices = np.asarray(csr.indices, dtype=csr_index_dtype)
    csr.indptr = np.asarray(csr.indptr, dtype=csr_index_dtype)
    csr.data = np.asarray(csr.data, dtype=intensity_dtype)

    return csr


def _normalize_csr_rows_inplace_gpu(mat_gpu) -> "cp.ndarray":
    """
    GPU version of row-normalization. Returns norms as a CuPy array.

    Why: this mirrors the experiments implementation in
    `experiments/fast_cosine_sim/approximate_similarity.py` and avoids using
    `cp.repeat(..., repeats=cupy_array)` which is not supported on some CuPy versions.
    """
    import cupy as cp
    import cupyx.scipy.sparse as cps

    n_rows = mat_gpu.shape[0]
    if n_rows == 0:
        return cp.zeros(
            (0,), dtype=mat_gpu.data.dtype if mat_gpu.nnz > 0 else cp.float32
        )
    if mat_gpu.nnz == 0:
        return cp.zeros((n_rows,), dtype=cp.float32)

    data_sq = mat_gpu.data**2
    sq = cps.csr_matrix((data_sq, mat_gpu.indices, mat_gpu.indptr), shape=mat_gpu.shape)
    row_sums_sq = sq.sum(axis=1).ravel()
    norms = cp.sqrt(row_sums_sq)

    safe = norms.copy()
    safe[safe == 0.0] = 1.0

    # Fail fast and help type-checkers: sparse stubs may consider `indptr` Optional.
    assert mat_gpu.indptr is not None, "GPU CSR indptr must not be None"
    if mat_gpu.nnz > 0:
        row_idx = (
            cp.searchsorted(
                mat_gpu.indptr,
                cp.arange(mat_gpu.nnz, dtype=mat_gpu.indptr.dtype),
                side="right",
            )
            - 1
        )
        # Keep configured GPU intensity dtype; do not hard-cast to float32.
        mat_gpu.data = mat_gpu.data.astype(mat_gpu.data.dtype, copy=False)
        mat_gpu.data /= safe[row_idx].astype(mat_gpu.data.dtype, copy=False)

    return norms


def _expand_csr_horizontal_adaptive_gpu(
    mat_gpu,
    *,
    bin_size: float,
    ms2_tolerance_ppm: float,
    nbins: int,
    mass_tolerance_cutoff_mz: float,
):
    """
    GPU version of adaptive horizontal expansion using fully vectorized operations.

    Mirrors the experiments implementation in `experiments/fast_cosine_sim/approximate_similarity.py`
    (`_expand_csr_horizontal_adaptive_gpu`): for each non-zero at column j (m/z=j*bin_size),
    compute a ppm-based tolerance in Da (with a low-m/z cutoff), convert to a window size
    in bins, and replicate the entry into all columns within +/- window.

    Contract:
      - mat_gpu is a CuPy CSR matrix (cupyx.scipy.sparse.csr_matrix)
      - Returns a CuPy CSR matrix with the same shape as mat_gpu
    """
    import cupy as cp
    import cupyx.scipy.sparse as cps

    assert float(bin_size) > 0.0, f"bin_size must be positive, got {bin_size}"
    assert float(ms2_tolerance_ppm) > 0.0, (
        f"ms2_tolerance_ppm must be positive, got {ms2_tolerance_ppm}"
    )
    assert int(nbins) > 0, f"nbins must be positive, got {nbins}"
    assert float(mass_tolerance_cutoff_mz) > 0.0, (
        f"mass_tolerance_cutoff_mz must be positive; got {mass_tolerance_cutoff_mz}"
    )

    if mat_gpu.nnz == 0:
        return mat_gpu

    # Why: fail fast; some sparse stubs treat indptr/indices as Optional.
    assert mat_gpu.indptr is not None, "GPU CSR indptr must not be None"
    assert mat_gpu.indices is not None, "GPU CSR indices must not be None"

    col_indices = mat_gpu.indices
    col_mz = col_indices.astype(cp.float64) * float(bin_size)
    eff_mz = cp.maximum(col_mz, float(mass_tolerance_cutoff_mz))
    tol_da = eff_mz * float(ms2_tolerance_ppm) * 1e-6
    windows = cp.ceil(tol_da / float(bin_size)).astype(cp.int32)
    del col_mz, eff_mz, tol_da

    repeats = 2 * windows + 1
    ends = cp.cumsum(repeats)
    total_items = int(ends[-1])
    dest_indices = cp.arange(total_items, dtype=cp.int64)
    source_idxs = cp.searchsorted(ends, dest_indices, side="right")
    del repeats, ends, total_items

    new_data = mat_gpu.data[source_idxs]

    # Start offset per *source nnz element* (length = mat_gpu.nnz), then gather by `source_idxs`.
    starts = cp.zeros((mat_gpu.nnz,), dtype=cp.int64)
    starts[1:] = cp.cumsum(2 * windows + 1, dtype=cp.int64)[:-1]
    start_offsets = starts[source_idxs]
    del starts

    local_offsets = dest_indices - start_offsets
    shifts = local_offsets - windows[source_idxs]
    new_cols = col_indices[source_idxs] + shifts
    del dest_indices, start_offsets, local_offsets, shifts

    mask = (new_cols >= 0) & (new_cols < int(nbins))
    n_valid = int(mask.sum())
    if not n_valid:
        del mask, new_cols, new_data, source_idxs, windows
        return cps.csr_matrix(mat_gpu.shape, dtype=mat_gpu.data.dtype)

    new_cols = new_cols[mask]
    new_data = new_data[mask]
    valid_source_idxs = source_idxs[mask]
    del mask, source_idxs

    source_rows_compact = (
        cp.searchsorted(
            mat_gpu.indptr,
            cp.arange(mat_gpu.nnz, dtype=mat_gpu.indptr.dtype),
            side="right",
        )
        - 1
    )
    new_rows = source_rows_compact[valid_source_idxs]
    del source_rows_compact, valid_source_idxs, windows

    out = cps.coo_matrix((new_data, (new_rows, new_cols)), shape=mat_gpu.shape).tocsr()
    del new_data, new_rows, new_cols
    return out


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

    # Match experiments behavior:
    # - use sparse dot -> CSR `sim`
    # - threshold on `sim.data`
    # - map `data` positions back to (row, col) using `sim.indices` and `sim.indptr`
    sim = (left_gpu @ right_gpu.T).astype(similarity_dtype)

    mask = sim.data >= cp.asarray(float(approx_threshold), dtype=sim.data.dtype)
    if not int(mask.sum()):
        return pl.DataFrame(
            {
                "idx_left": np.empty((0,), dtype=left_global_ids.dtype),
                "idx_right": np.empty((0,), dtype=right_global_ids.dtype),
                "approx_similarity": np.empty((0,), dtype=np.dtype(similarity_dtype)),
            }
        )

    out_data = sim.data[mask]
    out_cols = sim.indices[mask]
    indices_in_data = cp.nonzero(mask)[0]
    if indices_in_data.size == 0:
        return pl.DataFrame(
            {
                "idx_left": np.empty((0,), dtype=left_global_ids.dtype),
                "idx_right": np.empty((0,), dtype=right_global_ids.dtype),
                "approx_similarity": np.empty((0,), dtype=np.dtype(similarity_dtype)),
            }
        )

    out_rows = cp.searchsorted(sim.indptr, indices_in_data, side="right") - 1

    if upper_triangle_by_position:
        # Self mode: keep upper triangle by *position* within the batch ordering.
        # Experiments also remove diagonal/self-matches.
        keep = out_rows < out_cols
        out_rows = out_rows[keep]
        out_cols = out_cols[keep]
        out_data = out_data[keep]

        if out_rows.size == 0:
            return pl.DataFrame(
                {
                    "idx_left": np.empty((0,), dtype=left_global_ids.dtype),
                    "idx_right": np.empty((0,), dtype=right_global_ids.dtype),
                    "approx_similarity": np.empty(
                        (0,), dtype=np.dtype(similarity_dtype)
                    ),
                }
            )

    # Map local to global ids (CPU index arrays).
    left_ids = left_global_ids[cp.asnumpy(out_rows)]
    right_ids = right_global_ids[cp.asnumpy(out_cols)]

    return pl.DataFrame(
        {
            "idx_left": left_ids,
            "idx_right": right_ids,
            "approx_similarity": cp.asnumpy(out_data),
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
    sim_bytes = (
        int(min_spectra_per_batch)
        * int(min_spectra_per_batch)
        * int(similarity_itemsize)
    )
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
    global_ids: NDArray[np.int64]
    | NDArray[np.int32]
    | NDArray[np.uint64]
    | NDArray[np.uint32],
    *,
    max_peaks: int,
    min_batch_size: int,
) -> Iterator[
    tuple[
        int,
        int,
        sp.csr_matrix,
        NDArray[np.int64] | NDArray[np.int32] | NDArray[np.uint64] | NDArray[np.uint32],
    ]
]:
    n_rows = int(csr_matrix.shape[0])

    # Why: some type checkers treat CSR internals as Optional; we explicitly assert and fail fast.
    assert csr_matrix.indptr is not None, "CSR matrix indptr must not be None"
    assert csr_matrix.indices is not None, "CSR matrix indices must not be None"

    # Why: `indptr` is used for indexing and for computing `target_peaks`. Some sparse stacks keep this as
    # int32; we upcast to int64 to avoid overflow in cumulative peak counts for large batches.
    indptr = np.asarray(csr_matrix.indptr, dtype=np.int64)
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
    logger: LoggerLike | None = None,
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
        uses `config.output_parquet.path` as the directory and returns a `pl.LazyFrame`
        via `pl.scan_parquet(dir/*.parquet)`.
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

    _log(logger, "compute_gpu_batched_approximate_similarity_pairs: start")
    left = _ensure_row_index_column(left, index_column_name=config.spectrum_id_column)
    right_frame = (
        None
        if right is None
        else _ensure_row_index_column(
            right, index_column_name=config.spectrum_id_column
        )
    )

    left_df = _collect_if_lazy(left)
    right_df = left_df if right_frame is None else _collect_if_lazy(right_frame)

    for label, df in (("left", left_df), ("right", right_df)):
        for col in (
            config.spectrum_id_column,
            config.mz_column,
            config.intensity_column,
        ):
            assert col in df.columns, f"{label} missing required column: {col}"

    # Build CSR matrices + global ids from frames (use explode/flatten path from experiment for speed).
    left_ids = (
        left_df.get_column(config.spectrum_id_column)
        .to_numpy()
        .astype(config.dtypes.index_dtype, copy=False)
    )
    right_ids = (
        right_df.get_column(config.spectrum_id_column)
        .to_numpy()
        .astype(config.dtypes.index_dtype, copy=False)
    )

    flat_left_mzs, flat_left_ints, left_spec_pos, n_left = _flatten_spectra_to_numpy(
        left_df,
        mz_col=config.mz_column,
        intensity_col=config.intensity_column,
        spectrum_index_col="__spec_pos",
    )
    left_csr = _sparse_bin_flat_spectra_to_csr(
        flat_mzs=flat_left_mzs,
        flat_ints=flat_left_ints,
        spec_pos=left_spec_pos,
        n_spec=n_left,
        upper_bound=float(config.upper_mass_bound),
        bin_size=float(config.bin_size),
        intensity_power=float(config.intensity.power),
        csr_index_dtype=config.dtypes.csr_index_dtype,
        intensity_dtype=config.dtypes.intensity_dtype,
    )

    if right_frame is None:
        right_csr = left_csr
    else:
        flat_right_mzs, flat_right_ints, right_spec_pos, n_right = (
            _flatten_spectra_to_numpy(
                right_df,
                mz_col=config.mz_column,
                intensity_col=config.intensity_column,
                spectrum_index_col="__spec_pos",
            )
        )
        right_csr = _sparse_bin_flat_spectra_to_csr(
            flat_mzs=flat_right_mzs,
            flat_ints=flat_right_ints,
            spec_pos=right_spec_pos,
            n_spec=n_right,
            upper_bound=float(config.upper_mass_bound),
            bin_size=float(config.bin_size),
            intensity_power=float(config.intensity.power),
            csr_index_dtype=config.dtypes.csr_index_dtype,
            intensity_dtype=config.dtypes.intensity_dtype,
        )

    assert left_csr.shape[0] == left_ids.shape[0], "left ids and CSR rows must align"
    assert right_csr.shape[0] == right_ids.shape[0], "right ids and CSR rows must align"

    if left_csr.shape[0] == 0 or right_csr.shape[0] == 0:
        empty = pl.DataFrame(
            {
                "idx_left": np.empty((0,), dtype=config.dtypes.index_dtype),
                "idx_right": np.empty((0,), dtype=config.dtypes.index_dtype),
                "approx_similarity": np.empty(
                    (0,), dtype=config.dtypes.similarity_dtype
                ),
            }
        )
        return (
            empty
            if config.batching.flush_to_parquet_every_n_batches is None
            else empty.lazy()
        )

    free_mem, _ = cp.cuda.Device(0).mem_info

    max_peaks = _estimate_max_peaks_per_batch(
        free_mem_bytes=int(free_mem),
        target_gpu_memory_usage_ratio=float(
            config.batching.target_gpu_memory_usage_ratio
        ),
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

    flush_every_n_batches = config.batching.flush_to_parquet_every_n_batches
    should_write = flush_every_n_batches is not None

    # Predeclare to satisfy static typing / definite assignment analysis.
    buffer_parts: list[pl.DataFrame] = []
    collected_parts: list[pl.DataFrame] = []
    writer: _ParquetPartitionWriter | None = None
    part_index = 0
    write_every = 0

    if should_write:
        assert config.output_parquet.path is not None, (
            "When flush_to_parquet_every_n_batches is set, config.output_parquet.path must be set"
        )
        out_dir = Path(config.output_parquet.path)
        assert not out_dir.exists(), (
            f"Output path already exists: {out_dir}. Remove it or choose a new one."
        )
        writer = _ParquetPartitionWriter(out_dir)
        write_every = int(flush_every_n_batches)
        _log(logger, f"write mode enabled: out_dir={out_dir} write_every={write_every}")
    else:
        _log(logger, "in-memory mode enabled")

    batch_counter = 0

    nbins = int(np.floor(float(config.upper_mass_bound) / float(config.bin_size))) + 1

    # Outer loop: right batches (normalize + expand once, reuse for all left batches).
    for _, _, right_batch_csr, right_batch_ids in right_batches:
        # Match experiments transfer/CSR construction semantics:
        # explicitly allocate device buffers and avoid mutating the host CSR.
        r_data_gpu = cp.asarray(
            np.asarray(right_batch_csr.data).astype(
                config.dtypes.intensity_dtype, copy=False
            )
        )
        r_indices_gpu = cp.asarray(np.asarray(right_batch_csr.indices))
        r_indptr_gpu = cp.asarray(np.asarray(right_batch_csr.indptr))
        right_gpu_base = cps.csr_matrix(
            (r_data_gpu, r_indices_gpu, r_indptr_gpu),
            shape=right_batch_csr.shape,
        )
        del r_data_gpu, r_indices_gpu, r_indptr_gpu

        _normalize_csr_rows_inplace_gpu(right_gpu_base)

        # Adaptive (mass-dependent) horizontal expansion on the RHS (MANDATORY).
        # Why: this is a core part of the approximate candidate generation logic; without it,
        # peaks that are within MS2 tolerance but in adjacent bins would not match.
        right_gpu_base = _expand_csr_horizontal_adaptive_gpu(
            right_gpu_base,
            bin_size=float(config.bin_size),
            ms2_tolerance_ppm=float(config.ms2_tolerance_ppm),
            nbins=nbins,
            mass_tolerance_cutoff_mz=float(config.mass_tolerance_cutoff_mz),
        )

        # Inner loop: left batches
        for _, _, left_batch_csr, left_batch_ids in left_batches:
            l_data_gpu = cp.asarray(
                np.asarray(left_batch_csr.data).astype(
                    config.dtypes.intensity_dtype, copy=False
                )
            )
            l_indices_gpu = cp.asarray(np.asarray(left_batch_csr.indices))
            l_indptr_gpu = cp.asarray(np.asarray(left_batch_csr.indptr))
            left_gpu = cps.csr_matrix(
                (l_data_gpu, l_indices_gpu, l_indptr_gpu), shape=left_batch_csr.shape
            )
            del l_data_gpu, l_indices_gpu, l_indptr_gpu

            _normalize_csr_rows_inplace_gpu(left_gpu)

            pairs = _pairs_above_threshold_from_sparse_dot_gpu(
                left_gpu,
                right_gpu_base,
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
                assert writer is not None, (
                    "writer must be initialized when should_write is True"
                )
                writer.write_partition(
                    pl.concat(buffer_parts), partition_index=part_index
                )
                buffer_parts.clear()
                part_index += 1

    if should_write:
        if buffer_parts:
            assert writer is not None, (
                "writer must be initialized when should_write is True"
            )
            writer.write_partition(pl.concat(buffer_parts), partition_index=part_index)

        out_dir = Path(config.output_parquet.path)  # type: ignore[arg-type]
        _log(
            logger,
            f"compute_gpu_batched_approximate_similarity_pairs: done (wrote parquet to {out_dir})",
        )
        return pl.scan_parquet(str(out_dir / "*.parquet"))

    if not collected_parts:
        return pl.DataFrame(
            {
                "idx_left": np.empty((0,), dtype=config.dtypes.index_dtype),
                "idx_right": np.empty((0,), dtype=config.dtypes.index_dtype),
                "approx_similarity": np.empty(
                    (0,), dtype=config.dtypes.similarity_dtype
                ),
            }
        )

    _log(
        logger,
        (
            "compute_gpu_batched_approximate_similarity_pairs: done "
            f"(rows={sum(p.height for p in collected_parts)})"
        ),
    )
    return pl.concat(collected_parts)
