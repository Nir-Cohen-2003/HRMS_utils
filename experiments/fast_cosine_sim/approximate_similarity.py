# HRMS_utils/experiments/fast_cosine_sim/approximate_similarity.py
"""
Proximate-similarity baseline helpers and a small "all-vs-all" wrapper.

This module extracts the proximate-similarity pieces from the larger
`experiments/spectral_information/proximate_simialrity.py` implementation and
exposes a single, small, deterministic interface suitable for comparing
different similarity methods:

  proximate_all_vs_all_pairs(left_df, right_df, threshold, ms2_tolerance_ppm, ...)

Design notes / decisions:
- Polars is used as the dataframe abstraction (consistent with project rules).
- NumPy / BLAS is used to compute the dense dot-product similarity ("proximate
  stage") as in the original code. This is intentionally memory-heavy; the
  wrapper is targeted at relatively small comparisons (the user stated batching
  is not needed).
- The functions are small and explicit; helpers are private and the main
  exported function has a simple uniform interface.

Adaptive Expansion (NEW):
- Mass-dependent expansion windows automatically adjust based on fragment m/z values
- Handles the transition from ppm to absolute tolerance below MASS_TOLERANCE_CUTOFF (200 Da)
- Minimizes both false negatives (misses) and false positives by using appropriate
  windows at each mass: narrow windows for low-mass fragments, wider for high-mass
- Enable with use_adaptive_expansion=True (default) in proximate_all_vs_all_pairs()

Parameter Selection Guide:
  For adaptive expansion (RECOMMENDED):
    - Set proximate_bin_size based on desired resolution (e.g., 0.001 Da for 5 bins per
      tolerance window at 500 m/z with 10 ppm tolerance)
    - Set use_adaptive_expansion=True (default)
    - The system automatically calculates per-fragment expansion windows
    - Use calculate_recommended_bin_parameters() helper for bin_size suggestion

  For fixed expansion (legacy):
    - Set both proximate_bin_size and right_expansion_window
    - Set use_adaptive_expansion=False
    - Use calculate_recommended_bin_parameters() to get both values
    - Less accurate at mass extremes but computationally simpler
"""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Dict, Literal, Optional, cast

import cupy as cp
import cupyx.scipy.sparse as cps
import numba
import numpy as np
import polars as pl

# Optional dependency for sparse computations. Import guardedly so the module
# remains importable even if SciPy isn't available; we fail fast only when the
# sparse method is requested.
import scipy.sparse as sp
from numpy.dtypes import UShortDType
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
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

    Why: This avoids the expensive CSR->COO conversion and mask operations.
    Direct iteration through CSR structure is much faster than tocoo() + boolean masking.

    Args:
        indptr: CSR indptr array (row pointers)
        indices: CSR indices array (column indices)
        data: CSR data array (values)
        threshold: Minimum value to include

    Returns:
        row_indices: Array of row indices where data >= threshold
        col_indices: Array of column indices where data >= threshold
        values: Array of values where data >= threshold
    """
    # First pass: count how many entries meet the threshold
    # Why: Pre-allocating exact size is faster than dynamic growth
    count = 0
    for i in range(len(indptr) - 1):
        start = indptr[i]
        end = indptr[i + 1]
        for j in range(start, end):
            if data[j] >= threshold:
                count += 1

    # Allocate output arrays with exact size
    row_out = np.empty(count, dtype=np.int64)
    col_out = np.empty(count, dtype=np.int64)
    val_out = np.empty(count, dtype=np.float32)

    # Second pass: fill the arrays
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


def _numpy_proximate_similarity_pairs_above_threshold(
    left_mat: NDArray[np.float32],
    right_mat: NDArray[np.float32],
    threshold: float,
    left_global_idxs: NDArray[np.int64],
    right_global_idxs: NDArray[np.int64],
    return_timings: bool = False,
) -> (
    tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float32]]
    | tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float32], dict]
):
    """
    Compute row-wise cosine similarities between `left_mat` and `right_mat` using
    BLAS matmul and return indices (global) of pairs with similarity >= threshold.

    Returns (left_idxs_out, right_idxs_out, similarities).

    When `return_timings=True` also returns a timings dict with keys:
      - 'norm_time': time spent computing per-row norms and normalizing rows
      - 'matmul_time': time spent in the BLAS matmul
      - 'index_time': time spent thresholding and converting to index arrays
      - 'total_approx_time': sum of the three times above
    """
    # Early exits with zero timings
    if left_mat.size == 0 or right_mat.size == 0:
        timings = {
            "norm_time": 0.0,
            "matmul_time": 0.0,
            "index_time": 0.0,
            "total_approx_time": 0.0,
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

    assert left_mat.ndim == 2, f"expected left_mat to be 2D, got shape={left_mat.shape}"
    assert right_mat.ndim == 2, (
        f"expected right_mat to be 2D, got shape={right_mat.shape}"
    )

    L = left_mat
    R = right_mat

    # Row-wise L2 normalization (safe for zero rows) -> timed
    t_norm0 = perf_counter()
    lnorm = np.linalg.norm(L, axis=1, keepdims=True)
    rnorm = np.linalg.norm(R, axis=1, keepdims=True)
    lnorm_safe = np.where(lnorm > 0.0, lnorm, 1.0)
    rnorm_safe = np.where(rnorm > 0.0, rnorm, 1.0)

    Ln = L / lnorm_safe
    Rn = R / rnorm_safe
    norm_time = perf_counter() - t_norm0

    # BLAS matmul -> timed
    t_mat0 = perf_counter()
    sim_matrix = (Ln @ Rn.T).astype(np.float32, copy=False)
    matmul_time = perf_counter() - t_mat0

    # Thresholding & conversion to index arrays -> timed
    t_idx0 = perf_counter()
    li, ri = np.where(sim_matrix >= np.float32(threshold))
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
    prox_sims_out = sim_matrix[li, ri].astype(np.float32, copy=False)
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


def _flatten_spectra_to_numpy(
    df: pl.DataFrame, mz_col: str, int_col: str
) -> tuple[NDArray[np.float64], NDArray[np.float32], NDArray[np.int64], int]:
    """
    Flatten list-valued spectrum columns from `df` into NumPy arrays:

    - flat_mzs: np.ndarray[np.float64] of all m/z values
    - flat_ints: np.ndarray[np.float32] of corresponding intensities
    - spec_idx: np.ndarray[np.int64] mapping each flattened peak to its spectrum index
    - n_spec: number of spectra in `df`

    This uses Polars' explode to avoid Python loops and then exports views into
    the underlying buffers where possible.
    """
    n_spec = len(df)
    if n_spec == 0:
        return (
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float32),
            np.asarray([], dtype=np.int64),
            0,
        )

    # Add a row index so each flattened peak has a spectrum id
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


def _bin_flat_spectra_to_matrix(
    flat_mzs: NDArray[np.float64],
    flat_ints: NDArray[np.float32],
    spec_idx: NDArray[np.int64],
    n_spec: int,
    upper_bound: int = 1000,
    intensity_power: float = 0.5,
) -> NDArray[np.float32]:
    """
    Turn flattened (mz, intensity, spec_idx) arrays into a dense matrix of shape
    (n_spec, nbins) using integer mass bins (nearest integer).

    Returns a float32 matrix suitable for BLAS operations.
    """
    nbins = int(upper_bound) + 1
    if n_spec == 0:
        return np.zeros((0, nbins), dtype=np.float32)
    if flat_mzs.size == 0 or flat_ints.size == 0:
        return np.zeros((n_spec, nbins), dtype=np.float32)

    mass_bins = np.rint(flat_mzs).astype(np.int32)
    valid_mask = (mass_bins >= 0) & (mass_bins <= upper_bound) & (flat_ints > 0)
    if not np.any(valid_mask):
        return np.zeros((n_spec, nbins), dtype=np.float32)

    mass_bins = mass_bins[valid_mask]
    spec_idx = spec_idx[valid_mask]
    weights = np.asarray(flat_ints[valid_mask], dtype=np.float32) ** float(
        intensity_power
    )

    # Accumulate per-(spec,bin) using bincount on flat keys
    flat_keys = spec_idx * nbins + mass_bins
    accum = np.bincount(flat_keys, weights=weights, minlength=n_spec * nbins).astype(
        np.float32
    )
    matrix = accum.reshape((n_spec, nbins))
    return matrix


def _bin_spectra_df_to_matrix(
    df: pl.DataFrame,
    mz_col: str = "cleaned_normalized_mz",
    int_col: str = "cleaned_normalized_intensity",
    upper_bound: int = 1000,
    intensity_power: float = 0.5,
) -> tuple[NDArray[np.float32], Dict[str, float]]:
    """
    Explode list-valued spectra columns in `df` and bin them into a dense matrix
    of shape (n_spectra, nbins).

    Returns (matrix, timings), where timings contains simple profiling info:
      - flatten_time
      - bin_time
      - n_spec
      - n_peaks_total
      - n_peaks_valid
      - nbins
    """
    nbins = int(upper_bound) + 1
    timings: Dict[str, float] = {
        "flatten_time": 0.0,
        "bin_time": 0.0,
        "n_spec": 0,
        "n_peaks_total": 0,
        "n_peaks_valid": 0,
        "nbins": nbins,
    }

    # Flatten
    flat_mzs, flat_ints, spec_idx, n_spec = _flatten_spectra_to_numpy(
        df, mz_col, int_col
    )
    timings["n_spec"] = int(n_spec)
    timings["n_peaks_total"] = int(flat_mzs.size)

    if n_spec == 0:
        return np.zeros((0, nbins), dtype=np.float32), timings
    if flat_mzs.size == 0 or flat_ints.size == 0:
        return np.zeros((n_spec, nbins), dtype=np.float32), timings

    mass_bins = np.rint(flat_mzs).astype(np.int64)
    valid_mask = (mass_bins >= 0) & (mass_bins <= upper_bound) & (flat_ints > 0)
    timings["n_peaks_valid"] = int(np.count_nonzero(valid_mask))

    matrix = _bin_flat_spectra_to_matrix(
        flat_mzs, flat_ints, spec_idx, n_spec, upper_bound, intensity_power
    )
    return matrix, timings


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
    Turn flattened (mz, intensity, spec_idx) arrays into a sparse CSR matrix
    of shape (n_spec, nbins), using bins computed as
    `bin = np.rint(mz / bin_size)`. Returns a CSR matrix with dtype float32.

    Note: SciPy (scipy.sparse) is required for this function; an assertion will
    fail if it's unavailable.
    """
    assert sp is not None, "scipy.sparse is required for sparse binning (install scipy)"

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

    # Use COO accumulation (duplicates are summed during conversion to CSR)
    coo = sp.coo_matrix(
        (weights, (spec_idx, mass_bins)), shape=(n_spec, nbins), dtype=np.float32
    )
    return coo.tocsr()


def _expand_csr_horizontal(
    mat: "sp.csr_matrix", window: int, nbins: int
) -> "sp.csr_matrix":
    """
    Expand a CSR matrix across columns by adding each entry to neighboring bins
    in the inclusive range [-window, window]. This is intended for expanding
    the RIGHT matrix only (making each peak influence adjacent bins).

    Returns a CSR matrix with the same shape as `mat`.
    """
    assert sp is not None, "scipy.sparse is required for expansion (install scipy)"
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
    Expand a CSR matrix across columns with mass-dependent window sizes.

    For each non-zero entry at column index j (representing m/z = j * bin_size),
    calculate the appropriate tolerance window and expand to neighboring bins
    within that window. This ensures proper matching behavior across the full
    mass range while minimizing both false positives and false negatives.

    Why: PPM tolerance is mass-dependent, so high-mass fragments need larger
    windows than low-mass fragments. Below MASS_TOLERANCE_CUTOFF, we use absolute
    tolerance to avoid unrealistically tight windows at very low masses.

    Args:
        mat: CSR matrix to expand (typically the RIGHT matrix after normalization)
        bin_size: Width of each m/z bin in Daltons
        ms2_tolerance_ppm: MS2 fragment tolerance in ppm
        nbins: Total number of bins (for bounds checking)

    Returns:
        Expanded CSR matrix with the same shape as input
    """
    assert sp is not None, "scipy.sparse is required for expansion (install scipy)"
    assert bin_size > 0, f"bin_size must be positive, got {bin_size}"
    assert ms2_tolerance_ppm > 0, (
        f"ms2_tolerance_ppm must be positive, got {ms2_tolerance_ppm}"
    )

    if mat.nnz == 0:
        return mat

    # Convert to COO for easier column-wise processing
    coo = mat.tocoo()

    # Calculate m/z value for each column index
    # Why: column j represents m/z bin centered at j * bin_size
    col_mz = coo.col.astype(np.float64) * bin_size

    # Calculate mass-dependent tolerance in Daltons for each peak
    # Why: Below MASS_TOLERANCE_CUTOFF, use absolute tolerance to avoid overly tight windows
    effective_mz = np.maximum(col_mz, MASS_TOLERANCE_CUTOFF)
    tolerance_da = effective_mz * ms2_tolerance_ppm * 1e-6

    # Calculate required expansion window (in bins) for each peak
    # Why: We need to expand by enough bins to cover the tolerance window on each side
    expansion_bins = np.ceil(tolerance_da / bin_size).astype(np.int64)

    # Get unique expansion windows to process
    # Why: Many peaks will have the same window size, so we can batch process them
    unique_windows = np.unique(expansion_bins)

    rows_list = []
    cols_list = []
    data_list = []

    # Process each unique window size separately
    for window in unique_windows:
        # Find all peaks that need this window size
        mask = expansion_bins == window
        if not np.any(mask):
            continue

        rows_subset = coo.row[mask]
        cols_subset = coo.col[mask]
        data_subset = coo.data[mask]

        # Expand each peak to neighboring bins within [-window, +window]
        for shift in range(-window, window + 1):
            shifted_cols = cols_subset + shift
            # Only keep shifts that stay within valid bin range
            valid_mask = (shifted_cols >= 0) & (shifted_cols < nbins)
            if not np.any(valid_mask):
                continue

            rows_list.append(rows_subset[valid_mask])
            cols_list.append(shifted_cols[valid_mask])
            data_list.append(data_subset[valid_mask])

    if not rows_list:
        return sp.csr_matrix(mat.shape, dtype=np.float32)

    # Combine all expanded entries
    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    data = np.concatenate(data_list)

    # Build new sparse matrix (COO accumulates duplicates during conversion to CSR)
    new_coo = sp.coo_matrix(
        (data.astype(np.float32), (rows, cols)), shape=mat.shape, dtype=np.float32
    )
    return new_coo.tocsr()


def _normalize_csr_rows_inplace(mat: "sp.csr_matrix") -> NDArray[np.float32]:
    """
    In-place L2-normalize each row of `mat`. Returns the original (pre-normalization)
    row norms as a float32 NumPy array with shape (n_rows,).

    This avoids creating dense intermediate matrices and operates directly on
    the CSR data buffer.
    """
    assert sp is not None, "scipy.sparse is required for normalization (install scipy)"
    n_rows = mat.shape[0]
    if n_rows == 0:
        return np.zeros((0,), dtype=np.float32)
    if mat.nnz == 0:
        return np.zeros((n_rows,), dtype=np.float32)

    # Sum of squares per row
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
    Compute row-wise cosine similarities between `left_csr` and `right_csr`
    using sparse matrix operations (SciPy). Returns global index pairs and
    similarities for entries >= threshold.

    The timings dictionary (when requested) contains the same keys as the
    dense/Numpy variant: 'norm_time', 'matmul_time', 'index_time', and
    'total_approx_time'.

    Expansion behavior (applied AFTER normalization to preserve similarity values):
    - If ms2_tolerance_ppm and bin_size are provided: uses adaptive mass-dependent
      expansion windows that automatically adjust based on fragment m/z values
    - Else if right_expansion_window > 0: uses fixed expansion window (legacy behavior)
    - Else: no expansion applied

    Args:
        ms2_tolerance_ppm: MS2 tolerance in ppm for adaptive expansion (optional)
        bin_size: Bin size in Daltons for adaptive expansion (optional)
    """
    assert sp is not None, "scipy.sparse is required for sparse approximate stage"

    # Early exits with zero timings
    if left_csr.shape[0] == 0 or right_csr.shape[0] == 0:
        raise ValueError("Input CSR matrices must have at least one row each")

    L = left_csr.copy()
    R = right_csr.copy()

    # Row-wise normalization (safe for zero rows)
    t_norm0 = perf_counter()
    _ = _normalize_csr_rows_inplace(L)
    _ = _normalize_csr_rows_inplace(R)
    norm_time = perf_counter() - t_norm0

    # Optionally expand right matrix AFTER normalization to preserve similarity magnitudes
    # Why: Expansion after normalization ensures similarity scores remain meaningful
    if ms2_tolerance_ppm is not None and bin_size is not None:
        # Use adaptive mass-dependent expansion (preferred method)
        assert nbins is not None, "nbins must be provided for adaptive expansion"
        assert bin_size > 0, f"bin_size must be positive, got {bin_size}"
        assert ms2_tolerance_ppm > 0, (
            f"ms2_tolerance_ppm must be positive, got {ms2_tolerance_ppm}"
        )

        t_exp0 = perf_counter()
        R = _expand_csr_horizontal_adaptive(R, bin_size, ms2_tolerance_ppm, nbins)
        expansion_time = perf_counter() - t_exp0
        norm_time += (
            expansion_time  # Include expansion in norm_time for backwards compatibility
        )
        logger.info(
            "Applied adaptive expansion with ms2_tolerance=%.1f ppm, bin_size=%.4f Da (time=%.3fs)",
            ms2_tolerance_ppm,
            bin_size,
            expansion_time,
        )
    elif right_expansion_window and right_expansion_window > 0:
        # Use fixed expansion window (legacy behavior)
        assert nbins is not None, (
            "nbins must be provided when right_expansion_window > 0"
        )
        t_exp0 = perf_counter()
        R = _expand_csr_horizontal(R, right_expansion_window, nbins)
        expansion_time = perf_counter() - t_exp0
        norm_time += (
            expansion_time  # Include expansion in norm_time for backwards compatibility
        )
        logger.info(
            "Applied fixed expansion window=%d bins (time=%.3fs)",
            right_expansion_window,
            expansion_time,
        )

    Rt = R.T.tocsr()
    print("got matrices, starting sparse matmul")
    # Sparse matrix multiplication
    t_mat0 = perf_counter()
    sim = dot_product_mkl(L, Rt, cast=True, reorder_output=True)  # uses MKL sparse BLAS
    # Fallback to standard SciPy sparse matmul
    # sim = L @ Rt
    # Ensure we have a sparse matrix to work with
    if not sp.issparse(sim):
        sim = sp.csr_matrix(sim)
    matmul_time = perf_counter() - t_mat0
    print("sparse matmul done, starting thresholding")

    # Thresholding & conversion to index arrays
    # Why: Use numba-accelerated direct CSR index extraction instead of expensive CSR->COO conversion
    # We only need indices (not values) since everything goes to exact comparison anyway
    t_idx0 = perf_counter()

    # Ensure we have CSR format and convert data to float32 for numba compatibility
    if not isinstance(sim, sp.csr_matrix):
        sim = sim.tocsr()

    # Convert to contiguous arrays with correct dtypes for numba
    indptr = np.ascontiguousarray(sim.indptr, dtype=np.int32)
    indices = np.ascontiguousarray(sim.indices, dtype=np.int32)
    data = np.ascontiguousarray(sim.data, dtype=np.float32)
    print("prepared csr arrays, calling numba extraction")
    # Extract indices and values directly from CSR structure using numba
    # Why: Direct CSR iteration is ~10x faster than tocoo() + boolean masking
    li, ri, prox_sims_out = _extract_indices_and_values_above_threshold_from_csr(
        indptr, indices, data, float(threshold)
    )
    print("extraction done")
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

    # Map local indices to global indices
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
    Explode list-valued spectra in `df` and bin them into a sparse CSR matrix.

    Returns (csr_matrix, timings), where timings mirrors the dense variant and
    includes:
      - flatten_time
      - bin_time
      - n_spec
      - n_peaks_total
      - n_peaks_valid
      - nbins
    """
    assert sp is not None, "scipy.sparse is required for sparse binning"

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
    GPU version of _expand_csr_horizontal using CuPy.
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
    GPU version of _expand_csr_horizontal_adaptive using CuPy.
    """
    if mat.nnz == 0:
        return mat

    coo = mat.tocoo()
    col_mz = coo.col.astype(cp.float64) * bin_size
    effective_mz = cp.maximum(col_mz, MASS_TOLERANCE_CUTOFF)
    tolerance_da = effective_mz * ms2_tolerance_ppm * 1e-6
    expansion_bins = cp.ceil(tolerance_da / bin_size).astype(cp.int64)

    # Bring unique windows to CPU for iteration
    unique_windows = cp.asnumpy(cp.unique(expansion_bins))

    rows_list = []
    cols_list = []
    data_list = []

    for window in unique_windows:
        window = int(window)
        mask = expansion_bins == window
        if not int(mask.sum()):
            continue

        rows_subset = coo.row[mask]
        cols_subset = coo.col[mask]
        data_subset = coo.data[mask]

        for shift in range(-window, window + 1):
            shifted_cols = cols_subset + shift
            valid_mask = (shifted_cols >= 0) & (shifted_cols < nbins)
            if not int(valid_mask.sum()):
                continue

            rows_list.append(rows_subset[valid_mask])
            cols_list.append(shifted_cols[valid_mask])
            data_list.append(data_subset[valid_mask])

    if not rows_list:
        return cps.csr_matrix(mat.shape, dtype=cp.float32)

    rows = cp.concatenate(rows_list)
    cols = cp.concatenate(cols_list)
    data = cp.concatenate(data_list)

    new_coo = cps.coo_matrix(
        (data.astype(cp.float32), (rows, cols)), shape=mat.shape, dtype=cp.float32
    )
    return new_coo.tocsr()


def _normalize_csr_rows_inplace_gpu(mat: "cps.csr_matrix") -> "cp.ndarray":
    """
    GPU version of _normalize_csr_rows_inplace using CuPy.
    Returns the norms as a CuPy array to avoid expensive CPU transfer/sync.
    """
    n_rows = mat.shape[0]
    if n_rows == 0:
        return cp.zeros((0,), dtype=cp.float32)
    if mat.nnz == 0:
        return cp.zeros((n_rows,), dtype=cp.float32)

    # Optimization: Square data directly to avoid full CSR allocation in multiply
    # Create a lightweight wrapper sharing structure to sum rows
    data_sq = mat.data**2
    sq = cps.csr_matrix((data_sq, mat.indices, mat.indptr), shape=mat.shape)

    # Sum along rows (returns matrix, ravel to array)
    row_sums_sq = sq.sum(axis=1).ravel()
    norms = cp.sqrt(row_sums_sq)

    # Create safe divisor
    safe = norms.copy()
    safe[safe == 0.0] = 1.0

    if mat.nnz > 0:
        # Generate row indices from indptr using searchsorted
        # Why: This is robust, handles empty rows correctly, and stays entirely on GPU.
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
    Compute row-wise cosine similarities using GPU (CuPy).
    """
    assert cp is not None and cps is not None, "cupy is required for GPU method"

    if left_csr.shape[0] == 0 or right_csr.shape[0] == 0:
        raise ValueError("Input CSR matrices must have at least one row each")

    # Transfer to GPU
    t_transfer_in = perf_counter()
    L = cps.csr_matrix(left_csr)
    R = cps.csr_matrix(right_csr)
    transfer_in_time = perf_counter() - t_transfer_in

    # Normalization
    t_norm0 = perf_counter()
    _ = _normalize_csr_rows_inplace_gpu(L)
    _ = _normalize_csr_rows_inplace_gpu(R)
    norm_time = perf_counter() - t_norm0

    # Expansion
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

    # Matmul
    print("starting gpu sparse matmul")
    t_mat0 = perf_counter()
    sim = L.dot(R.T)
    matmul_time = perf_counter() - t_mat0
    print("gpu sparse matmul done")

    # Thresholding & Extraction
    t_idx0 = perf_counter()

    # Convert to COO on GPU
    sim_coo = sim.tocoo()

    # Boolean mask on GPU
    mask = sim_coo.data >= threshold

    # Apply mask (GPU)
    rows_gpu = sim_coo.row[mask]
    cols_gpu = sim_coo.col[mask]
    vals_gpu = sim_coo.data[mask]

    # Transfer back to CPU
    t_transfer_out = perf_counter()
    li = cp.asnumpy(rows_gpu).astype(np.int64)
    ri = cp.asnumpy(cols_gpu).astype(np.int64)
    prox_sims_out = cp.asnumpy(vals_gpu).astype(np.float32)
    transfer_out_time = perf_counter() - t_transfer_out

    index_time = (perf_counter() - t_idx0) - transfer_out_time

    if li.size == 0:
        timings = {
            "norm_time": float(norm_time),
            "matmul_time": float(matmul_time),
            "index_time": float(index_time),
            "transfer_time": float(transfer_in_time + transfer_out_time),
            "total_approx_time": float(
                norm_time
                + matmul_time
                + index_time
                + transfer_in_time
                + transfer_out_time
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
    Ensure the DataFrame has an integer row-index column with name `idx_col`.
    If it already exists, ensure it is Int64 typed; otherwise create it with
    Polars' `with_row_index`.
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
    method: Literal["numpy", "sparse"] = "numpy",
    right_expansion_window: int = 0,
    use_adaptive_expansion: bool = True,
    use_gpu: bool = False,
    return_timings: bool = False,
) -> pl.DataFrame | tuple[pl.DataFrame, dict]:
    """
    Compute proximate (binned dot-product) similarities between two collections
    of spectra and return all pairs with similarity >= `threshold`.

    Additional options:
      - `method`: either 'numpy' (the original dense BLAS matmul path) or
        'sparse' (use SciPy sparse matrices and sparse matrix multiply).
        If 'sparse' is selected, SciPy (scipy.sparse) must be installed.
      - `proximate_bin_size`: bin width in m/z units (e.g., 1.0, 0.01). Peaks
        are assigned with `np.rint(mz / bin_size)`.
      - `use_adaptive_expansion`: if True and method='sparse' and ms2_tolerance_ppm
        is provided, uses mass-dependent expansion windows that automatically adjust
        based on fragment m/z (handles low-mass regions with MASS_TOLERANCE_CUTOFF).
        If False, falls back to fixed `right_expansion_window` (default: True).
      - `right_expansion_window`: integer W >= 0; if >0 and adaptive expansion is
        disabled, the RIGHT matrix is expanded so that each binned value is added
        into neighboring bins in +/- W (useful to make the right-side peaks "fuzzy"
        for approximate matching). Ignored when use_adaptive_expansion=True.

    If `return_timings` is True the function returns a tuple
    `(results_df, timings)` where `timings` contains:
      - left/right flatten_time (conversion of Polars -> NumPy)
      - left/right bin_time
      - approximate_time (BLAS matmul stage)
      - exact_time (Polars dotprod compute/collect stage)
      - n_candidates (number of pairs produced by the approximate stage)
      - n_final_pairs (number surviving exact filtering)

    Interface (uniform for comparing similarity methods):
      - left_df, right_df: polars.DataFrame containing at minimum spectrum columns
          `mz_col` and `int_col`.
      - threshold: similarity threshold (float). Pairs with similarity >= threshold
          are kept.
      - ms2_tolerance_ppm: precursor m/z tolerance (ppm) REQUIRED for the exact
          dot-product similarity computation. This value is passed to the
          `dotprod_similarity` expression and controls how fragment m/z values
          are matched during the exact stage.

    Returns a Polars DataFrame with columns:
      - 'idx': left-spectrum index (Int64)
      - 'idx_right': right-spectrum index (Int64)
      - 'proximate_similarity': float32 similarity value
    """
    assert threshold >= 0.0 and threshold <= 1.0, "threshold must be between 0 and 1"

    # Ensure consistent index columns
    left = _ensure_idx_column(left_df, "idx")
    right = _ensure_idx_column(right_df, "idx")

    n_left = len(left)
    n_right = len(right)
    if n_left == 0 or n_right == 0:
        # Return empty frame with expected schema
        return pl.DataFrame(
            {
                "idx": pl.Series([], dtype=pl.Int64),
                "idx_right": pl.Series([], dtype=pl.Int64),
                "proximate_similarity": pl.Series([], dtype=pl.Float32),
            }
        )

    # Bin spectra into matrices (support both dense and sparse approaches)
    method = method.lower()
    assert method in ("numpy", "sparse"), "method must be either 'numpy' or 'sparse'"

    # Initialize nbins for sparse expansion (used later if method == "sparse")
    nbins = int(np.floor(proximate_bin_upper / float(proximate_bin_size))) + 1

    if method == "numpy":
        left_mat, left_timings = _bin_spectra_df_to_matrix(
            left,
            mz_col,
            int_col,
            upper_bound=proximate_bin_upper,
            intensity_power=intensity_power,
        )
        right_mat, right_timings = _bin_spectra_df_to_matrix(
            right,
            mz_col,
            int_col,
            upper_bound=proximate_bin_upper,
            intensity_power=intensity_power,
        )
    else:
        assert sp is not None, "scipy is required for the sparse method (install scipy)"
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

    # Conversion timings (flatten step = Polars -> NumPy conversion)
    left_flat = float(left_timings.get("flatten_time", 0.0))
    right_flat = float(right_timings.get("flatten_time", 0.0))

    left_bin = float(left_timings.get("bin_time", 0.0))
    right_bin = float(right_timings.get("bin_time", 0.0))

    # Print / log conversion & binning timings and totals
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
    print(
        f"Preprocessing: flat_left={left_flat:.3f}s flat_right={right_flat:.3f}s total_flat={preproc_flat_total:.3f}s | "
        f"bin_left={left_bin:.3f}s bin_right={right_bin:.3f}s total_bin={preproc_bin_total:.3f}s | preproc_total={preproc_total:.3f}s"
    )

    # Approximate stage (detailed timings) - use dense or sparse implementation
    if method == "numpy":
        # Narrow types for the type checker / clarity
        left_np = cast(NDArray[np.float32], left_mat)
        right_np = cast(NDArray[np.float32], right_mat)
        ret = _numpy_proximate_similarity_pairs_above_threshold(
            left_np,
            right_np,
            threshold - 0.15,
            left_global_idxs,
            right_global_idxs,
            return_timings=True,
        )  # Use 0.9*threshold to avoid missing borderline pairs
    else:
        # Narrow to SciPy CSR for the sparse path
        left_sp = cast(sp.csr_matrix, left_mat)
        right_sp = cast(sp.csr_matrix, right_mat)
        # Determine expansion strategy
        if use_gpu and cp is not None:
            if use_adaptive_expansion and ms2_tolerance_ppm is not None:
                # Use adaptive mass-dependent expansion (preferred)
                ret = _sparse_proximate_similarity_pairs_above_threshold_gpu(
                    left_sp,
                    right_sp,
                    threshold * 0.9,
                    left_global_idxs,
                    right_global_idxs,
                    return_timings=True,
                    right_expansion_window=0,  # Disable fixed window
                    nbins=nbins,
                    ms2_tolerance_ppm=ms2_tolerance_ppm,
                    bin_size=proximate_bin_size,
                )
            else:
                # Use fixed expansion window (legacy behavior)
                ret = _sparse_proximate_similarity_pairs_above_threshold_gpu(
                    left_sp,
                    right_sp,
                    threshold * 0.9,
                    left_global_idxs,
                    right_global_idxs,
                    return_timings=True,
                    right_expansion_window=right_expansion_window,
                    nbins=nbins,
                )
        elif use_adaptive_expansion and ms2_tolerance_ppm is not None:
            # Use adaptive mass-dependent expansion (preferred)
            ret = _sparse_proximate_similarity_pairs_above_threshold(
                left_sp,
                right_sp,
                threshold * 0.9,
                left_global_idxs,
                right_global_idxs,
                return_timings=True,
                right_expansion_window=0,  # Disable fixed window
                nbins=nbins,
                ms2_tolerance_ppm=ms2_tolerance_ppm,
                bin_size=proximate_bin_size,
            )
        else:
            # Use fixed expansion window (legacy behavior)
            ret = _sparse_proximate_similarity_pairs_above_threshold(
                left_sp,
                right_sp,
                threshold * 0.9,
                left_global_idxs,
                right_global_idxs,
                return_timings=True,
                right_expansion_window=right_expansion_window,
                nbins=nbins,
            )
        # Use 0.9*threshold to avoid missing borderline pairs

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

    # Breakdown of approximate stage timings
    norm_time = float(approx_tims.get("norm_time", 0.0))
    matmul_time = float(approx_tims.get("matmul_time", 0.0))
    index_time = float(approx_tims.get("index_time", 0.0))
    approx_time_total = float(
        approx_tims.get("total_approx_time", norm_time + matmul_time + index_time)
    )

    n_candidates = int(l_idxs.size)
    logger.info(
        f"Approximate stage: total={approx_time_total:.3f}s (norm={norm_time:.3f}s, matmul={matmul_time:.3f}s, index={index_time:.3f}s) — candidates={n_candidates} (threshold_used={threshold * 0.9:.4f})"
    )
    print(
        f"Approximate stage: total={approx_time_total:.3f}s (norm={norm_time:.3f}s, matmul={matmul_time:.3f}s, index={index_time:.3f}s) — candidates={n_candidates} (threshold_used={threshold * 0.9:.4f})"
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

    # Join candidates with original spectra (lazy) for the exact stage
    joined = (
        out.lazy()
        .join(left.lazy(), on="idx")
        .join(right.lazy(), left_on="idx_right", right_on="idx", suffix="_right")
    )

    # Build spectra struct for dot-product computation; require mz/intensity columns exist
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

    # ms2_tolerance_ppm is required to compute the exact dot-product similarity
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

    # Filter final pairs by dotprod similarity threshold
    joined = joined.filter(
        pl.col("dotprod_similarity").is_not_null(),
        pl.col("dotprod_similarity").ge(threshold),
    )

    # Build selection columns dynamically (include common metadata when present)
    sel_cols = ["idx", "idx_right", "proximate_similarity", "dotprod_similarity"]
    for col in ("base_inchikey", "smiles", "spectral_information_score"):
        if col in left.columns:
            sel_cols.append(col)
        if col + "_right" in right.columns:
            sel_cols.append(col + "_right")

    # Exact stage timing: split into (plan build) and (collect / execution)
    t_plan0 = perf_counter()
    # The following transform-building is lazy; measure the time to assemble the plan.
    joined = joined
    plan_build_time = perf_counter() - t_plan0

    t_collect0 = perf_counter()
    results = joined.select(sel_cols).collect()
    collect_time = perf_counter() - t_collect0

    exact_time_total = plan_build_time + collect_time

    n_final_pairs = int(len(results))
    logger.info(
        "Exact stage: plan_build=%.3fs collect=%.3fs total=%.3fs final_pairs=%d",
        plan_build_time,
        collect_time,
        exact_time_total,
        n_final_pairs,
    )
    print(
        f"Exact stage: plan_build={plan_build_time:.3f}s collect={collect_time:.3f}s total={exact_time_total:.3f}s final_pairs={n_final_pairs}"
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
        "exact_time": exact_time_total,
        "exact_plan_time": plan_build_time,
        "exact_collect_time": collect_time,
        "n_candidates": n_candidates,
        "n_final_pairs": n_final_pairs,
    }

    # Ensure dtypes for the canonical numeric columns
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

    Note: This is for the legacy fixed-window expansion mode. When using adaptive expansion
    (use_adaptive_expansion=True), you only need to set proximate_bin_size; the expansion
    window is calculated automatically for each fragment based on its m/z.

    Args:
        ms2_tolerance_ppm: MS2 fragment tolerance in ppm
        reference_mz: Reference m/z for calculating bin size (default 500 Da, typical fragment mass)
        bins_per_tolerance: How many bins should span the tolerance window (default 5)
            Higher values = finer resolution but more memory/computation

    Returns:
        (proximate_bin_size, right_expansion_window)

    Example:
        >>> # For 10 ppm tolerance at 500 m/z
        >>> bin_size, window = calculate_recommended_bin_parameters(10.0)
        >>> # bin_size ≈ 0.001 Da, window = 2
        >>> # Total window coverage = 5 bins * 0.001 Da = 0.005 Da
        >>> # At 500 m/z: 10 ppm = 0.005 Da ✓
    """
    # Calculate tolerance in Da at reference mass
    # Why: Below MASS_TOLERANCE_CUTOFF, use absolute tolerance
    effective_mz = max(reference_mz, MASS_TOLERANCE_CUTOFF)
    tolerance_da = effective_mz * ms2_tolerance_ppm * 1e-6

    # Calculate bin size to achieve desired resolution
    bin_size = tolerance_da / float(bins_per_tolerance)

    # Calculate expansion window (bins on each side)
    # Why: bins_per_tolerance bins total = (2*window + 1) bins
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
    Thin compatibility wrapper with the uniform signature expected by comparison
    experiments: (left_df, right_df, threshold, mass_tolerance).

    This forwards to `proximate_all_vs_all_pairs` and maps the `mass_tolerance_ppm`
    to the underlying `ms2_tolerance_ppm` argument.

    Args:
      - return_timings: if True, the function returns (results_df, timings)
        timing dict contains conversion/approx/exact times and pair counts.
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
    # Handles low-mass fragments correctly with MASS_TOLERANCE_CUTOFF
    print("\n=== Example 1: Adaptive Mass-Dependent Expansion ===")
    ms2_tol = 5.0  # ppm
    # bin_size, _ = calculate_recommended_bin_parameters(ms2_tol, bins_per_tolerance=5)
    bin_size = 0.0001
    print(f"Using ms2_tolerance_ppm={ms2_tol} ppm, bin_size={bin_size:.6f} Da")
    print("Expansion windows will be calculated adaptively per fragment m/z")

    results_adaptive = proximate_all_vs_all_pairs(
        spectra,
        spectra,
        threshold=0.7,
        ms2_tolerance_ppm=ms2_tol,
        method="sparse",
        proximate_bin_size=bin_size,
        use_adaptive_expansion=True,  # Enable adaptive expansion
        return_timings=True,
        # use_gpu=True,
    )

    # Example 2: Using fixed expansion window (legacy mode)
    print("\n=== Example 2: Fixed Expansion Window (Legacy) ===")
    bin_size_fixed, window_fixed = calculate_recommended_bin_parameters(
        ms2_tol, bins_per_tolerance=20
    )
    print(f"Using bin_size={bin_size_fixed:.4f} Da, expansion_window={window_fixed}")

    results_fixed = proximate_all_vs_all_pairs(
        spectra,
        spectra,
        threshold=0.7,
        ms2_tolerance_ppm=ms2_tol,
        method="sparse",
        proximate_bin_size=bin_size_fixed,
        right_expansion_window=window_fixed,
        use_adaptive_expansion=False,  # Disable adaptive expansion
        return_timings=True,
        # use_gpu=True,
    )

    # Extract results and timings for both methods
    if isinstance(results_adaptive, tuple):
        results_adaptive_df, timings_adaptive = results_adaptive
    else:
        results_adaptive_df = results_adaptive
        timings_adaptive = {}

    if isinstance(results_fixed, tuple):
        results_fixed_df, timings_fixed = results_fixed
    else:
        results_fixed_df = results_fixed
        timings_fixed = {}

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
