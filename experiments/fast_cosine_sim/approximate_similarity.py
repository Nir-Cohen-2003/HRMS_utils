# HRMS_utils/experiments/fast_cosine_sim/approximate_similarity.py
"""
Proximate-similarity baseline helpers and a small "all-vs-all" wrapper.

This module implements the proximate (binned dot-product) prefilter using
SciPy sparse matrices and MKL (via `sparse_dot_mkl`) with an optional GPU
implementation via CuPy. The module always uses the sparse path (no dense
NumPy/BLAS option) and fails fast at import time if required packages are missing.

Key points:
- Polars is used as the dataframe abstraction (consistent with project rules).
- SciPy sparse CSR matrices are used for binning and fast sparse matmul.
- MKL-accelerated sparse matmul is provided by `sparse_dot_mkl.dot_product_mkl`.
- Optional GPU path uses CuPy / cupyx.sparse for transfer, normalization, expansion,
  and matmul.
- Adaptive expansion adjusts expansion windows by fragment m/z (preferred mode).
"""

from __future__ import annotations, print_function

import logging
from time import perf_counter
from typing import Dict, Optional

import cupy as cp
import cupyx.scipy.sparse as cps
import numba
import numpy as np
import polars as pl
import scipy.sparse as sp
from numpy.typing import NDArray
from sparse_dot_mkl import dot_product_mkl

import hrms_utils

logger = logging.getLogger(__name__)
MASS_TOLERANCE_CUTOFF = 200.0


@numba.njit
def _extract_indices_and_values_above_threshold_from_csr(
    indptr: NDArray[np.int32],
    indices: NDArray[np.int32],
    data: NDArray[np.float32],
    threshold: float,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float32]]:
    """
    Extract row indices, column indices, and values from a CSR matrix where data >= threshold.

    Why: Avoid CSR->COO conversion and boolean masking which are expensive. Direct CSR
    iteration with a pre-count pass enables exact preallocation and is faster.
    """
    # Count qualifying entries
    count = 0
    for i in range(len(indptr) - 1):
        start = indptr[i]
        end = indptr[i + 1]
        for j in range(start, end):
            if data[j] >= threshold:
                count += 1

    # Allocate outputs
    row_out = np.empty(count, dtype=np.int64)
    col_out = np.empty(count, dtype=np.int64)
    val_out = np.empty(count, dtype=np.float32)

    # Fill arrays
    idx = 0
    for i in range(len(indptr) - 1):
        start = indptr[i]
        end = indptr[i + 1]
        for j in range(start, end):
            if data[j] >= threshold:
                row_out[idx] = i
                col_out[idx] = indices[j]
                val_out[idx] = data[j]
                idx += 1

    return row_out, col_out, val_out


def _flatten_spectra_to_numpy(
    df: pl.DataFrame, mz_col: str, int_col: str
) -> tuple[NDArray[np.float64], NDArray[np.float32], NDArray[np.int64], int]:
    """
    Flatten list-valued spectrum columns from `df` into NumPy arrays.

    Returns: (flat_mzs, flat_ints, spec_idx, n_spec)
      - flat_mzs: np.ndarray[np.float64] of all m/z values
      - flat_ints: np.ndarray[np.float32] of intensities
      - spec_idx: np.ndarray[np.int64] mapping each flattened peak to its spectrum index
      - n_spec: number of spectra
    """
    n_spec = len(df)
    if n_spec == 0:
        return (
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float32),
            np.asarray([], dtype=np.int64),
            0,
        )

    df_idx = df.with_row_index("__spec_idx")
    exploded = df_idx.explode([mz_col, int_col])
    if len(exploded) == 0:
        return (
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float32),
            np.asarray([], dtype=np.int64),
            n_spec,
        )

    exploded = exploded.with_columns(
        [
            pl.col(mz_col).cast(pl.Float32),
            pl.col(int_col).cast(pl.Float32),
            pl.col("__spec_idx").cast(pl.Int64),
        ]
    )

    flat_mzs = exploded.get_column(mz_col).to_numpy()
    flat_ints = exploded.get_column(int_col).to_numpy()
    spec_idx = exploded.get_column("__spec_idx").to_numpy()

    return flat_mzs, flat_ints, spec_idx, n_spec


def _sparse_bin_flat_spectra_to_csr(
    flat_mzs: NDArray[np.float64],
    flat_ints: NDArray[np.float32],
    spec_idx: NDArray[np.int64],
    n_spec: int,
    upper_bound: int = 1000,
    intensity_power: float = 0.5,
    bin_size: float = 1.0,
) -> sp.csr_matrix:
    """
    Turn flattened arrays into a sparse CSR matrix (n_spec, nbins).
    Binning uses bin = np.rint(mz / bin_size). Duplicates are summed via COO -> CSR.
    """
    nbins = int(np.floor(upper_bound / float(bin_size))) + 1
    if n_spec == 0:
        return sp.csr_matrix((n_spec, nbins), dtype=np.float32)
    if flat_mzs.size == 0 or flat_ints.size == 0:
        return sp.csr_matrix((n_spec, nbins), dtype=np.float32)

    mass_bins = np.rint(flat_mzs / float(bin_size)).astype(np.int64)
    valid_mask = (mass_bins >= 0) & (mass_bins < nbins) & (flat_ints > 0)
    if not np.any(valid_mask):
        return sp.csr_matrix((n_spec, nbins), dtype=np.float32)

    mass_bins = mass_bins[valid_mask].astype(np.int64)
    spec_idx = spec_idx[valid_mask].astype(np.int64)
    weights = np.asarray(flat_ints[valid_mask], dtype=np.float32) ** float(
        intensity_power
    )

    coo = sp.coo_matrix(
        (weights, (spec_idx, mass_bins)), shape=(n_spec, nbins), dtype=np.float32
    )
    return coo.tocsr()


def _expand_csr_horizontal(
    mat: "sp.csr_matrix", window: int, nbins: int
) -> "sp.csr_matrix":
    """
    Expand a CSR matrix across columns by adding each entry to neighboring bins in [-window, window].
    This is typically applied to the RIGHT matrix to make peaks fuzzy.
    """
    if window <= 0 or mat.nnz == 0:
        return mat

    coo = mat.tocoo()
    rows_list = []
    cols_list = []
    data_list = []

    for shift in range(-window, window + 1):
        shifted_cols = coo.col + shift
        mask = (shifted_cols >= 0) & (shifted_cols < nbins)
        if not np.any(mask):
            continue
        rows_list.append(coo.row[mask])
        cols_list.append(shifted_cols[mask])
        data_list.append(coo.data[mask])

    if not rows_list:
        return sp.csr_matrix(mat.shape, dtype=np.float32)

    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    data = np.concatenate(data_list)
    new_coo = sp.coo_matrix(
        (data.astype(np.float32), (rows, cols)), shape=mat.shape, dtype=np.float32
    )
    return new_coo.tocsr()


def _expand_csr_horizontal_adaptive(
    mat: "sp.csr_matrix",
    bin_size: float,
    ms2_tolerance_ppm: float,
    nbins: int,
) -> "sp.csr_matrix":
    """
    Expand columns with mass-dependent window sizes. For each non-zero at column j (m/z = j * bin_size)
    compute tolerance (ppm or absolute below cutoff) and expand by that many bins.
    """
    assert bin_size > 0, f"bin_size must be positive, got {bin_size}"
    assert ms2_tolerance_ppm > 0, (
        f"ms2_tolerance_ppm must be positive, got {ms2_tolerance_ppm}"
    )

    if mat.nnz == 0:
        return mat

    coo = mat.tocoo()
    col_mz = coo.col.astype(np.float64) * bin_size
    effective_mz = np.maximum(col_mz, MASS_TOLERANCE_CUTOFF)
    tolerance_da = effective_mz * ms2_tolerance_ppm * 1e-6
    expansion_bins = np.ceil(tolerance_da / bin_size).astype(np.int64)

    unique_windows = np.unique(expansion_bins)

    rows_list = []
    cols_list = []
    data_list = []

    for window in unique_windows:
        mask = expansion_bins == window
        if not np.any(mask):
            continue

        rows_subset = coo.row[mask]
        cols_subset = coo.col[mask]
        data_subset = coo.data[mask]

        for shift in range(-window, window + 1):
            shifted_cols = cols_subset + shift
            valid_mask = (shifted_cols >= 0) & (shifted_cols < nbins)
            if not np.any(valid_mask):
                continue
            rows_list.append(rows_subset[valid_mask])
            cols_list.append(shifted_cols[valid_mask])
            data_list.append(data_subset[valid_mask])

    if not rows_list:
        return sp.csr_matrix(mat.shape, dtype=np.float32)

    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    data = np.concatenate(data_list)
    new_coo = sp.coo_matrix(
        (data.astype(np.float32), (rows, cols)), shape=mat.shape, dtype=np.float32
    )
    return new_coo.tocsr()


def _normalize_csr_rows_inplace(mat: "sp.csr_matrix") -> NDArray[np.float32]:
    """
    In-place L2-normalize each row of `mat`. Returns original row norms.
    """
    n_rows = mat.shape[0]
    if n_rows == 0:
        return np.zeros((0,), dtype=np.float32)
    if mat.nnz == 0:
        return np.zeros((n_rows,), dtype=np.float32)

    sq = mat.multiply(mat)
    row_sums_sq = np.asarray(sq.sum(axis=1)).ravel().astype(np.float32)
    norms = np.sqrt(row_sums_sq)
    safe = norms.copy()
    safe[safe == 0.0] = 1.0

    counts = np.diff(mat.indptr)
    if counts.sum() > 0:
        row_idx = np.repeat(np.arange(n_rows), counts)
        mat.data = mat.data.astype(np.float32, copy=False)
        mat.data /= safe[row_idx]

    return norms


def _sparse_proximate_similarity_pairs_above_threshold(
    left_csr: "sp.csr_matrix",
    right_csr: "sp.csr_matrix",
    threshold: float,
    left_global_idxs: NDArray[np.int64],
    right_global_idxs: NDArray[np.int64],
    return_timings: bool = False,
    right_expansion_window: int = 0,
    nbins: int | None = None,
    ms2_tolerance_ppm: float | None = None,
    bin_size: float | None = None,
) -> (
    tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float32]]
    | tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float32], dict]
):
    """
    Compute row-wise cosine similarities between `left_csr` and `right_csr` using SciPy.
    Returns global index pairs and similarities for entries >= threshold.

    Expansion (adaptive preferred) is applied AFTER normalization.
    """
    # Validate input shapes
    if left_csr.shape[0] == 0 or right_csr.shape[0] == 0:
        raise ValueError("Input CSR matrices must have at least one row each")

    L = left_csr.copy()
    R = right_csr.copy()

    # Normalize
    t_norm0 = perf_counter()
    _ = _normalize_csr_rows_inplace(L)
    _ = _normalize_csr_rows_inplace(R)
    norm_time = perf_counter() - t_norm0

    # Expansion (after normalization)
    if ms2_tolerance_ppm is not None and bin_size is not None:
        assert nbins is not None, "nbins must be provided for adaptive expansion"
        assert bin_size > 0, f"bin_size must be positive, got {bin_size}"
        assert ms2_tolerance_ppm > 0, (
            f"ms2_tolerance_ppm must be positive, got {ms2_tolerance_ppm}"
        )

        t_exp0 = perf_counter()
        R = _expand_csr_horizontal_adaptive(R, bin_size, ms2_tolerance_ppm, nbins)
        expansion_time = perf_counter() - t_exp0
        norm_time += expansion_time
        logger.info(
            "Applied adaptive expansion with ms2_tolerance=%.1f ppm, bin_size=%.4f Da (time=%.3fs)",
            ms2_tolerance_ppm,
            bin_size,
            expansion_time,
        )
    elif right_expansion_window and right_expansion_window > 0:
        assert nbins is not None, "nbins must be provided when using fixed expansion"
        t_exp0 = perf_counter()
        R = _expand_csr_horizontal(R, right_expansion_window, nbins)
        expansion_time = perf_counter() - t_exp0
        norm_time += expansion_time
        logger.info(
            "Applied fixed expansion window=%d bins (time=%.3fs)",
            right_expansion_window,
            expansion_time,
        )

    # Sparse matmul using MKL-backed dot_product_mkl (fast)
    Rt = R.T.tocsr()
    logger.info("Starting sparse matmul (CPU/MKL)")
    t_mat0 = perf_counter()
    sim = dot_product_mkl(L, Rt, cast=True, reorder_output=True)
    if not sp.issparse(sim):
        sim = sp.csr_matrix(sim)
    matmul_time = perf_counter() - t_mat0
    logger.info("Sparse matmul complete (CPU)")

    # Thresholding & extraction via numba-accelerated CSR iteration
    t_idx0 = perf_counter()

    if not isinstance(sim, sp.csr_matrix):
        sim = sim.tocsr()

    indptr = np.ascontiguousarray(sim.indptr, dtype=np.int32)
    indices = np.ascontiguousarray(sim.indices, dtype=np.int32)
    data = np.ascontiguousarray(sim.data, dtype=np.float32)

    li, ri, prox_sims_out = _extract_indices_and_values_above_threshold_from_csr(
        indptr, indices, data, float(threshold)
    )

    if li.size == 0:
        timings = {
            "norm_time": float(norm_time),
            "matmul_time": float(matmul_time),
            "index_time": 0.0,
            "total_approx_time": float(norm_time + matmul_time),
        }
        if return_timings:
            return (
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.float32),
                timings,
            )
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float32),
        )

    left_idxs_out = left_global_idxs[li]
    right_idxs_out = right_global_idxs[ri]

    index_time = perf_counter() - t_idx0

    timings = {
        "norm_time": float(norm_time),
        "matmul_time": float(matmul_time),
        "index_time": float(index_time),
        "total_approx_time": float(norm_time + matmul_time + index_time),
    }

    if return_timings:
        return left_idxs_out, right_idxs_out, prox_sims_out, timings
    return left_idxs_out, right_idxs_out, prox_sims_out


def _sparse_bin_spectra_df_to_csr(
    df: pl.DataFrame,
    mz_col: str = "cleaned_normalized_mz",
    int_col: str = "cleaned_normalized_intensity",
    upper_bound: int = 1000,
    intensity_power: float = 0.5,
    bin_size: float = 1.0,
) -> tuple["sp.csr_matrix", Dict[str, float]]:
    """
    Explode list-valued spectra and bin into a sparse CSR matrix.

    Returns (csr_matrix, timings).
    """
    nbins = int(np.floor(upper_bound / float(bin_size))) + 1
    timings: Dict[str, float] = {
        "flatten_time": 0.0,
        "bin_time": 0.0,
        "n_spec": 0,
        "n_peaks_total": 0,
        "n_peaks_valid": 0,
        "nbins": nbins,
    }

    t_flat0 = perf_counter()
    flat_mzs, flat_ints, spec_idx, n_spec = _flatten_spectra_to_numpy(
        df, mz_col, int_col
    )
    timings["flatten_time"] = float(perf_counter() - t_flat0)
    timings["n_spec"] = int(n_spec)
    timings["n_peaks_total"] = int(flat_mzs.size)

    if n_spec == 0:
        return sp.csr_matrix((0, nbins), dtype=np.float32), timings
    if flat_mzs.size == 0 or flat_ints.size == 0:
        return sp.csr_matrix((n_spec, nbins), dtype=np.float32), timings

    mass_bins = np.rint(flat_mzs / float(bin_size)).astype(np.int64)
    valid_mask = (mass_bins >= 0) & (mass_bins < nbins) & (flat_ints > 0)
    timings["n_peaks_valid"] = int(np.count_nonzero(valid_mask))

    t_bin0 = perf_counter()
    matrix = _sparse_bin_flat_spectra_to_csr(
        flat_mzs, flat_ints, spec_idx, n_spec, upper_bound, intensity_power, bin_size
    )
    timings["bin_time"] = float(perf_counter() - t_bin0)

    return matrix, timings


def _expand_csr_horizontal_gpu(
    mat: "cps.csr_matrix", window: int, nbins: int
) -> "cps.csr_matrix":
    """
    GPU version of horizontal expansion using CuPy.
    """
    if window <= 0 or mat.nnz == 0:
        return mat

    coo = mat.tocoo()
    rows_list = []
    cols_list = []
    data_list = []

    for shift in range(-window, window + 1):
        shifted_cols = coo.col + shift
        mask = (shifted_cols >= 0) & (shifted_cols < nbins)
        if not int(mask.sum()):
            continue
        rows_list.append(coo.row[mask])
        cols_list.append(shifted_cols[mask])
        data_list.append(coo.data[mask])

    if not rows_list:
        return cps.csr_matrix(mat.shape, dtype=cp.float32)

    rows = cp.concatenate(rows_list)
    cols = cp.concatenate(cols_list)
    data = cp.concatenate(data_list)
    new_coo = cps.coo_matrix(
        (data.astype(cp.float32), (rows, cols)), shape=mat.shape, dtype=cp.float32
    )
    return new_coo.tocsr()


def _expand_csr_horizontal_adaptive_gpu(
    mat: "cps.csr_matrix",
    bin_size: float,
    ms2_tolerance_ppm: float,
    nbins: int,
) -> "cps.csr_matrix":
    """
    GPU version of adaptive horizontal expansion using fully vectorized operations.
    """
    if mat.nnz == 0:
        return mat

    col_indices = mat.indices
    col_mz = col_indices.astype(cp.float64) * bin_size
    eff_mz = cp.maximum(col_mz, MASS_TOLERANCE_CUTOFF)
    tol_da = eff_mz * ms2_tolerance_ppm * 1e-6
    windows = cp.ceil(tol_da / bin_size).astype(cp.int32)

    repeats = 2 * windows + 1
    ends = cp.cumsum(repeats)
    total_items = int(ends[-1])
    dest_indices = cp.arange(total_items, dtype=cp.int64)
    source_idxs = cp.searchsorted(ends, dest_indices, side="right")

    new_data = mat.data[source_idxs]

    starts = cp.zeros_like(ends)
    starts[1:] = ends[:-1]
    start_offsets = starts[source_idxs]
    local_offsets = dest_indices - start_offsets
    shifts = local_offsets - windows[source_idxs]
    new_cols = col_indices[source_idxs] + shifts

    mask = (new_cols >= 0) & (new_cols < nbins)
    if not int(mask.sum()):
        return cps.csr_matrix(mat.shape, dtype=cp.float32)

    new_cols = new_cols[mask]
    new_data = new_data[mask]
    valid_source_idxs = source_idxs[mask]

    source_rows_compact = (
        cp.searchsorted(mat.indptr, cp.arange(mat.nnz, dtype=cp.int32), side="right")
        - 1
    )
    new_rows = source_rows_compact[valid_source_idxs]

    return cps.coo_matrix((new_data, (new_rows, new_cols)), shape=mat.shape).tocsr()


def _normalize_csr_rows_inplace_gpu(mat: "cps.csr_matrix") -> "cp.ndarray":
    """
    GPU version of row-normalization. Returns norms as a CuPy array.
    """
    n_rows = mat.shape[0]
    if n_rows == 0:
        return cp.zeros((0,), dtype=cp.float32)
    if mat.nnz == 0:
        return cp.zeros((n_rows,), dtype=cp.float32)

    data_sq = mat.data**2
    sq = cps.csr_matrix((data_sq, mat.indices, mat.indptr), shape=mat.shape)
    row_sums_sq = sq.sum(axis=1).ravel()
    norms = cp.sqrt(row_sums_sq)

    safe = norms.copy()
    safe[safe == 0.0] = 1.0

    if mat.nnz > 0:
        row_idx = (
            cp.searchsorted(
                mat.indptr, cp.arange(mat.nnz, dtype=cp.int32), side="right"
            )
            - 1
        )
        mat.data = mat.data.astype(cp.float32, copy=False)
        mat.data /= safe[row_idx]

    return norms


def _sparse_proximate_similarity_pairs_above_threshold_gpu(
    left_csr: "sp.csr_matrix",
    right_csr: "sp.csr_matrix",
    threshold: float,
    left_global_idxs: NDArray[np.int64],
    right_global_idxs: NDArray[np.int64],
    return_timings: bool = False,
    right_expansion_window: int = 0,
    nbins: int | None = None,
    ms2_tolerance_ppm: float | None = None,
    bin_size: float | None = None,
) -> (
    tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float32]]
    | tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float32], dict]
):
    """
    Compute row-wise cosine similarities using GPU (CuPy). Returns numpy arrays
    for indices and similarities. Timings include transfer times.
    """
    if left_csr.shape[0] == 0 or right_csr.shape[0] == 0:
        raise ValueError("Input CSR matrices must have at least one row each")

    # Transfer to GPU
    t_transfer_in = perf_counter()
    L = cps.csr_matrix(left_csr)
    R = cps.csr_matrix(right_csr)
    transfer_in_time = perf_counter() - t_transfer_in

    # Normalize
    t_norm0 = perf_counter()
    _ = _normalize_csr_rows_inplace_gpu(L)
    _ = _normalize_csr_rows_inplace_gpu(R)
    norm_time = perf_counter() - t_norm0

    # Expansion (after normalization)
    if ms2_tolerance_ppm is not None and bin_size is not None:
        assert nbins is not None
        t_exp0 = perf_counter()
        R = _expand_csr_horizontal_adaptive_gpu(R, bin_size, ms2_tolerance_ppm, nbins)
        expansion_time = perf_counter() - t_exp0
        norm_time += expansion_time
        logger.info(
            "Applied adaptive expansion (GPU) with ms2_tolerance=%.1f ppm, bin_size=%.4f Da (time=%.3fs)",
            ms2_tolerance_ppm,
            bin_size,
            expansion_time,
        )
    elif right_expansion_window and right_expansion_window > 0:
        assert nbins is not None
        t_exp0 = perf_counter()
        R = _expand_csr_horizontal_gpu(R, right_expansion_window, nbins)
        expansion_time = perf_counter() - t_exp0
        norm_time += expansion_time
        logger.info(
            "Applied fixed expansion (GPU) window=%d bins (time=%.3fs)",
            right_expansion_window,
            expansion_time,
        )

    # Matmul (GPU sparse matmul)
    logger.info("Starting sparse matmul (GPU)")
    t_mat0 = perf_counter()
    sim = L.dot(R.T)
    matmul_time = perf_counter() - t_mat0
    logger.info("Sparse matmul complete (GPU)")

    # Thresholding & extraction (fused on GPU)
    t_idx0 = perf_counter()
    mask = sim.data >= threshold

    if not int(mask.sum()):
        index_time = perf_counter() - t_idx0
        timings = {
            "norm_time": float(norm_time),
            "matmul_time": float(matmul_time),
            "index_time": float(index_time),
            "transfer_time": float(transfer_in_time),
            "total_approx_time": float(
                norm_time + matmul_time + index_time + transfer_in_time
            ),
        }
        if return_timings:
            return (
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.float32),
                timings,
            )
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float32),
        )

    out_data = sim.data[mask]
    out_cols = sim.indices[mask]
    indices_in_data = cp.nonzero(mask)[0]
    if indices_in_data.size == 0:
        index_time = perf_counter() - t_idx0
        timings = {
            "norm_time": float(norm_time),
            "matmul_time": float(matmul_time),
            "index_time": float(index_time),
            "transfer_time": float(transfer_in_time),
            "total_approx_time": float(
                norm_time + matmul_time + index_time + transfer_in_time
            ),
        }
        if return_timings:
            return (
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.float32),
                timings,
            )
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float32),
        )

    out_rows = cp.searchsorted(sim.indptr, indices_in_data, side="right") - 1

    # Transfer back to CPU (only the passing elements)
    t_transfer_out = perf_counter()
    li = cp.asnumpy(out_rows).astype(np.int64)
    ri = cp.asnumpy(out_cols).astype(np.int64)
    prox_sims_out = cp.asnumpy(out_data).astype(np.float32)
    transfer_out_time = perf_counter() - t_transfer_out

    index_time = (perf_counter() - t_idx0) - transfer_out_time

    left_idxs_out = left_global_idxs[li]
    right_idxs_out = right_global_idxs[ri]

    timings = {
        "norm_time": float(norm_time),
        "matmul_time": float(matmul_time),
        "index_time": float(index_time),
        "transfer_time": float(transfer_in_time + transfer_out_time),
        "total_approx_time": float(
            norm_time + matmul_time + index_time + transfer_in_time + transfer_out_time
        ),
    }

    if return_timings:
        return left_idxs_out, right_idxs_out, prox_sims_out, timings
    return left_idxs_out, right_idxs_out, prox_sims_out


def _ensure_idx_column(df: pl.DataFrame, idx_col: str = "idx") -> pl.DataFrame:
    """
    Ensure the DataFrame has an integer row-index column named `idx_col`.
    """
    if idx_col in df.columns:
        return df.with_columns(pl.col(idx_col).cast(pl.Int64))
    return df.with_row_index(idx_col).with_columns(pl.col(idx_col).cast(pl.Int64))


def proximate_all_vs_all_pairs(
    left_df: pl.DataFrame,
    right_df: pl.DataFrame,
    threshold: float,
    ms2_tolerance_ppm: Optional[float] = None,
    mz_col: str = "cleaned_normalized_mz",
    int_col: str = "cleaned_normalized_intensity",
    proximate_bin_upper: int = 1000,
    proximate_bin_size: float = 1.0,
    intensity_power: float = 0.5,
    right_expansion_window: int = 0,
    use_adaptive_expansion: bool = True,
    use_gpu: bool = False,
    return_timings: bool = False,
) -> pl.DataFrame | tuple[pl.DataFrame, dict]:
    """
    Compute proximate (binned dot-product) similarities between two collections
    of spectra and return all pairs with similarity >= `threshold`.

    Notes:
      - This implementation always uses the sparse method (SciPy CSR matrices).
      - If `use_gpu=True`, computation is performed on the GPU using CuPy.
      - `use_adaptive_expansion=True` (and providing `ms2_tolerance_ppm`) applies
        mass-dependent expansion windows. Otherwise, `right_expansion_window`
        (fixed integer) is used for expansion.
    """
    assert 0.0 <= threshold <= 1.0, "threshold must be between 0 and 1"

    # Ensure consistent indices
    left = _ensure_idx_column(left_df, "idx")
    right = _ensure_idx_column(right_df, "idx")

    n_left = len(left)
    n_right = len(right)
    if n_left == 0 or n_right == 0:
        return pl.DataFrame(
            {
                "idx": pl.Series([], dtype=pl.Int64),
                "idx_right": pl.Series([], dtype=pl.Int64),
                "proximate_similarity": pl.Series([], dtype=pl.Float32),
            }
        )

    # Number of bins (for expansion)
    nbins = int(np.floor(proximate_bin_upper / float(proximate_bin_size))) + 1

    # Bin both sides into sparse CSR matrices
    left_mat, left_timings = _sparse_bin_spectra_df_to_csr(
        left,
        mz_col,
        int_col,
        upper_bound=proximate_bin_upper,
        intensity_power=intensity_power,
        bin_size=proximate_bin_size,
    )
    right_mat, right_timings = _sparse_bin_spectra_df_to_csr(
        right,
        mz_col,
        int_col,
        upper_bound=proximate_bin_upper,
        intensity_power=intensity_power,
        bin_size=proximate_bin_size,
    )

    left_global_idxs = left["idx"].to_numpy()
    right_global_idxs = right["idx"].to_numpy()

    left_flat = float(left_timings.get("flatten_time", 0.0))
    right_flat = float(right_timings.get("flatten_time", 0.0))
    left_bin = float(left_timings.get("bin_time", 0.0))
    right_bin = float(right_timings.get("bin_time", 0.0))

    preproc_flat_total = left_flat + right_flat
    preproc_bin_total = left_bin + right_bin
    preproc_total = preproc_flat_total + preproc_bin_total

    logger.info(
        "Preprocessing times (s): flat_left=%.3f flat_right=%.3f total_flat=%.3f | bin_left=%.3f bin_right=%.3f total_bin=%.3f | preproc_total=%.3f",
        left_flat,
        right_flat,
        preproc_flat_total,
        left_bin,
        right_bin,
        preproc_bin_total,
        preproc_total,
    )

    # Approximate stage: sparse path (CPU or GPU)
    if use_gpu:
        if use_adaptive_expansion and ms2_tolerance_ppm is not None:
            ret = _sparse_proximate_similarity_pairs_above_threshold_gpu(
                left_mat,
                right_mat,
                threshold * 0.9,
                left_global_idxs,
                right_global_idxs,
                return_timings=True,
                right_expansion_window=0,
                nbins=nbins,
                ms2_tolerance_ppm=ms2_tolerance_ppm,
                bin_size=proximate_bin_size,
            )
        else:
            ret = _sparse_proximate_similarity_pairs_above_threshold_gpu(
                left_mat,
                right_mat,
                threshold * 0.9,
                left_global_idxs,
                right_global_idxs,
                return_timings=True,
                right_expansion_window=right_expansion_window,
                nbins=nbins,
            )
    else:
        if use_adaptive_expansion and ms2_tolerance_ppm is not None:
            ret = _sparse_proximate_similarity_pairs_above_threshold(
                left_mat,
                right_mat,
                threshold * 0.9,
                left_global_idxs,
                right_global_idxs,
                return_timings=True,
                right_expansion_window=0,
                nbins=nbins,
                ms2_tolerance_ppm=ms2_tolerance_ppm,
                bin_size=proximate_bin_size,
            )
        else:
            ret = _sparse_proximate_similarity_pairs_above_threshold(
                left_mat,
                right_mat,
                threshold * 0.9,
                left_global_idxs,
                right_global_idxs,
                return_timings=True,
                right_expansion_window=right_expansion_window,
                nbins=nbins,
            )

    if len(ret) == 4:
        l_idxs, r_idxs, sims, approx_tims = ret
    else:
        l_idxs, r_idxs, sims = ret
        approx_tims = {
            "norm_time": 0.0,
            "matmul_time": 0.0,
            "index_time": 0.0,
            "total_approx_time": 0.0,
        }

    norm_time = float(approx_tims.get("norm_time", 0.0))
    matmul_time = float(approx_tims.get("matmul_time", 0.0))
    index_time = float(approx_tims.get("index_time", 0.0))
    approx_time_total = float(
        approx_tims.get("total_approx_time", norm_time + matmul_time + index_time)
    )

    n_candidates = int(l_idxs.size)
    logger.info(
        "Approximate stage: total=%.3fs (norm=%.3fs, matmul=%.3fs, index=%.3fs) — candidates=%d (threshold_used=%.4f)",
        approx_time_total,
        norm_time,
        matmul_time,
        index_time,
        n_candidates,
        threshold * 0.9,
    )

    if n_candidates == 0:
        empty = pl.DataFrame(
            {
                "idx": pl.Series([], dtype=pl.Int64),
                "idx_right": pl.Series([], dtype=pl.Int64),
                "proximate_similarity": pl.Series([], dtype=pl.Float32),
            }
        )
        timings = {
            "left_flatten_time": left_flat,
            "right_flatten_time": right_flat,
            "left_bin_time": left_bin,
            "right_bin_time": right_bin,
            "approximate_time": approx_time_total,
            "approx_norm_time": norm_time,
            "approx_matmul_time": matmul_time,
            "approx_index_time": index_time,
            "exact_time": 0.0,
            "n_candidates": 0,
            "n_final_pairs": 0,
        }
        if return_timings:
            return empty, timings
        return empty

    out = pl.DataFrame(
        {
            "idx": l_idxs.astype(np.int64),
            "idx_right": r_idxs.astype(np.int64),
            "proximate_similarity": sims.astype(np.float32),
        }
    )

    # Exact stage: join candidate pairs and compute precise dot-product similarity
    joined = (
        out.lazy()
        .join(left.lazy(), on="idx")
        .join(right.lazy(), left_on="idx_right", right_on="idx", suffix="_right")
    )

    joined = joined.with_columns(
        spectra=pl.struct(
            mz1=pl.col("cleaned_normalized_mz"),
            intensities1=pl.col("cleaned_normalized_intensity"),
            mz2=pl.col("cleaned_normalized_mz_right"),
            intensities2=pl.col("cleaned_normalized_intensity_right"),
            precursor_mz1=pl.col("precursor_mz"),
            precursor_mz2=pl.col("precursor_mz_right"),
        )
    )

    assert ms2_tolerance_ppm is not None, (
        "ms2_tolerance_ppm must be provided to compute exact dotprod similarity"
    )

    joined = joined.with_columns(
        dotprod_similarity=pl.col("spectra").spectral_similarity.dotprod_similarity(  # type: ignore
            ms2_tolerance_in_ppm=float(ms2_tolerance_ppm),
            clean_spectra_first=False,
            ignore_precursor=True,
        )
    )

    joined = joined.filter(
        pl.col("dotprod_similarity").is_not_null(),
        pl.col("dotprod_similarity").ge(threshold),
    )

    sel_cols = ["idx", "idx_right", "proximate_similarity", "dotprod_similarity"]
    for col in ("base_inchikey", "smiles", "spectral_information_score"):
        if col in left.columns:
            sel_cols.append(col)
        if col + "_right" in right.columns:
            sel_cols.append(col + "_right")

    t_plan0 = perf_counter()
    joined = joined
    plan_build_time = perf_counter() - t_plan0

    t_collect0 = perf_counter()
    results = joined.select(sel_cols).collect()
    collect_time = perf_counter() - t_collect0

    exact_time_total = plan_build_time + collect_time
    n_final_pairs = int(len(results))

    timings = {
        "left_flatten_time": left_flat,
        "right_flatten_time": right_flat,
        "left_bin_time": left_bin,
        "right_bin_time": right_bin,
        "approximate_time": approx_time_total,
        "approx_norm_time": norm_time,
        "approx_matmul_time": matmul_time,
        "approx_index_time": index_time,
        "exact_time": exact_time_total,
        "exact_plan_time": plan_build_time,
        "exact_collect_time": collect_time,
        "n_candidates": n_candidates,
        "n_final_pairs": n_final_pairs,
    }

    results = results.with_columns(
        [
            pl.col("idx").cast(pl.Int64),
            pl.col("idx_right").cast(pl.Int64),
            pl.col("proximate_similarity").cast(pl.Float32),
            pl.col("dotprod_similarity").cast(pl.Float32),
        ]
    )

    if return_timings:
        return results, timings
    return results


def calculate_recommended_bin_parameters(
    ms2_tolerance_ppm: float,
    reference_mz: float = 500.0,
    bins_per_tolerance: int = 5,
) -> tuple[float, int]:
    """
    Calculate recommended proximate_bin_size and right_expansion_window for fixed-window expansion.
    """
    effective_mz = max(reference_mz, MASS_TOLERANCE_CUTOFF)
    tolerance_da = effective_mz * ms2_tolerance_ppm * 1e-6
    bin_size = tolerance_da / float(bins_per_tolerance)
    expansion_window = (bins_per_tolerance - 1) // 2
    return bin_size, expansion_window


def proximate_all_vs_all(
    left_df: pl.DataFrame,
    right_df: pl.DataFrame,
    threshold: float,
    mass_tolerance_ppm: float,
    return_timings: bool = False,
    use_gpu: bool = False,
) -> pl.DataFrame | tuple[pl.DataFrame, dict]:
    """
    Compatibility wrapper mapping `mass_tolerance_ppm` to `ms2_tolerance_ppm`.
    """
    return proximate_all_vs_all_pairs(
        left_df,
        right_df,
        threshold,
        ms2_tolerance_ppm=mass_tolerance_ppm,
        return_timings=return_timings,
        use_gpu=use_gpu,
    )


if __name__ == "__main__":
    spectra = (
        pl.scan_parquet(
            "/home/analytit_admin/Data/spectral_libs/fraghub/fraghub.parquet"
        )
        .head(40_000)
        .collect()
    )

    # Example 1: Using adaptive expansion (RECOMMENDED)
    # Why: Automatically adjusts expansion window based on fragment m/z
    print("\n=== Example 1: Adaptive Mass-Dependent Expansion ===")
    ms2_tol = 5.0  # ppm
    bin_size = 0.0001
    print(f"Using ms2_tolerance_ppm={ms2_tol} ppm, bin_size={bin_size:.6f} Da")
    print("Expansion windows will be calculated adaptively per fragment m/z")
    similarity_threhsold = 0.65
    results_adaptive_df, timings_adaptive = proximate_all_vs_all_pairs(
        spectra,
        spectra,
        threshold=similarity_threhsold,
        ms2_tolerance_ppm=ms2_tol,
        proximate_bin_size=bin_size,
        use_adaptive_expansion=True,  # Enable adaptive expansion
        return_timings=True,
        use_gpu=True,
    )
    results_adaptive_df, timings_adaptive = proximate_all_vs_all_pairs(
        spectra,
        spectra,
        threshold=similarity_threhsold,
        ms2_tolerance_ppm=ms2_tol,
        proximate_bin_size=bin_size,
        use_adaptive_expansion=True,  # Enable adaptive expansion
        return_timings=True,
        use_gpu=True,
    )
    print("with gpu:")
    print(timings_adaptive)
    results_adaptive_df, timings_adaptive = proximate_all_vs_all_pairs(
        spectra,
        spectra,
        threshold=similarity_threhsold,
        ms2_tolerance_ppm=ms2_tol,
        proximate_bin_size=bin_size,
        use_adaptive_expansion=True,  # Enable adaptive expansion
        return_timings=True,
        use_gpu=False,
    )
    print("without gpu:")
    print(timings_adaptive)

    # Example 2: Using fixed expansion window (legacy mode)
    print("\n=== Example 2: Fixed Expansion Window (Legacy) ===")
    bin_size_fixed, window_fixed = calculate_recommended_bin_parameters(
        ms2_tol, bins_per_tolerance=20
    )
    print(f"Using bin_size={bin_size_fixed:.4f} Da, expansion_window={window_fixed}")

    results_fixed_df, timings_fixed = proximate_all_vs_all_pairs(
        spectra,
        spectra,
        threshold=similarity_threhsold,
        ms2_tolerance_ppm=ms2_tol,
        proximate_bin_size=bin_size_fixed,
        right_expansion_window=window_fixed,
        use_adaptive_expansion=False,  # Disable adaptive expansion
        return_timings=True,
        use_gpu=True,
    )
    print(timings_fixed)
    # Calculate statistics for adaptive expansion
    abs_diff_adaptive = np.abs(
        results_adaptive_df["proximate_similarity"].to_numpy()
        - results_adaptive_df["dotprod_similarity"].to_numpy()
    )
    total_pairs_adaptive = len(abs_diff_adaptive)
    avg_abs_diff_adaptive = float(np.mean(abs_diff_adaptive))
    frac_over_0_1_adaptive = (
        float(np.sum(abs_diff_adaptive > 0.1)) / total_pairs_adaptive
    )
    frac_over_0_01_adaptive = (
        float(np.sum(abs_diff_adaptive > 0.01)) / total_pairs_adaptive
    )
    frac_over_0_001_adaptive = (
        float(np.sum(abs_diff_adaptive > 0.001)) / total_pairs_adaptive
    )

    # Calculate statistics for fixed expansion
    abs_diff_fixed = np.abs(
        results_fixed_df["proximate_similarity"].to_numpy()
        - results_fixed_df["dotprod_similarity"].to_numpy()
    )
    total_pairs_fixed = len(abs_diff_fixed)
    avg_abs_diff_fixed = float(np.mean(abs_diff_fixed))
    frac_over_0_1_fixed = float(np.sum(abs_diff_fixed > 0.1)) / total_pairs_fixed
    frac_over_0_01_fixed = float(np.sum(abs_diff_fixed > 0.01)) / total_pairs_fixed
    frac_over_0_001_fixed = float(np.sum(abs_diff_fixed > 0.001)) / total_pairs_fixed

    # Print side-by-side comparison
    print("\n" + "=" * 80)
    print("=== SIMILARITY COMPARISON: ADAPTIVE vs FIXED EXPANSION ===")
    print("=" * 80)
    print(f"\n{'Metric':<40} {'Adaptive':<20} {'Fixed':<20}")
    print("-" * 80)
    print(
        f"{'Total pairs found':<40} {total_pairs_adaptive:<20} {total_pairs_fixed:<20}"
    )
    print(
        f"{'Average absolute difference':<40} {avg_abs_diff_adaptive:<20.6f} {avg_abs_diff_fixed:<20.6f}"
    )
    print(
        f"{'Fraction with |diff| > 0.1':<40} {frac_over_0_1_adaptive:<20.6f} {frac_over_0_1_fixed:<20.6f}"
    )
    print(
        f"{'Fraction with |diff| > 0.01':<40} {frac_over_0_01_adaptive:<20.6f} {frac_over_0_01_fixed:<20.6f}"
    )
    print(
        f"{'Fraction with |diff| > 0.001':<40} {frac_over_0_001_adaptive:<20.6f} {frac_over_0_001_fixed:<20.6f}"
    )
    print(
        f"{'Count with |diff| > 0.1':<40} {int(frac_over_0_1_adaptive * total_pairs_adaptive):<20} {int(frac_over_0_1_fixed * total_pairs_fixed):<20}"
    )
    print(
        f"{'Count with |diff| > 0.01':<40} {int(frac_over_0_01_adaptive * total_pairs_adaptive):<20} {int(frac_over_0_01_fixed * total_pairs_fixed):<20}"
    )
    print(
        f"{'Count with |diff| > 0.001':<40} {int(frac_over_0_001_adaptive * total_pairs_adaptive):<20} {int(frac_over_0_001_fixed * total_pairs_fixed):<20}"
    )
    print("=" * 80)
