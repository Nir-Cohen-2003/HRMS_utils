"""
GPU kernel operations for sparse matrix processing.

This module contains all GPU-side operations:
- CSR row normalization (L2)
- Adaptive horizontal expansion (tolerance windows)
- SpMM expansion matrix construction

Why separate module:
- Isolates CuPy/CUDA dependencies
- Can be tested with GPU fixtures
- Clear boundary between CPU preprocessing and GPU computation
"""

from __future__ import annotations

import logging
from typing import Optional

import cupy as cp
import cupyx.scipy.sparse as cps
import numba
import numpy as np
from numba import cuda
from numpy.typing import NDArray

from .config import MASS_TOLERANCE_CUTOFF


# =============================================================================
# CSR Row Normalization
# =============================================================================


def normalize_csr_rows_inplace_gpu(mat: cps.csr_matrix) -> cp.ndarray:
    """
    L2-normalize each row of a CuPy CSR matrix in-place.

    Why: Cosine similarity requires normalized vectors. Doing this on GPU
    before matmul is more efficient than normalizing after.

    Args:
        mat: CuPy CSR matrix to normalize (modified in-place)

    Returns:
        CuPy array of original row norms
    """
    n_rows = mat.shape[0]
    if n_rows == 0:
        return cp.zeros((0,), dtype=cp.float32)
    if mat.nnz == 0:
        return cp.zeros((n_rows,), dtype=cp.float32)

    # Compute row-wise L2 norms
    data_sq = mat.data**2
    sq = cps.csr_matrix((data_sq, mat.indices, mat.indptr), shape=mat.shape)
    row_sums_sq = sq.sum(axis=1).ravel()
    norms = cp.sqrt(row_sums_sq)

    # Safe division (avoid divide by zero)
    safe = norms.copy()
    safe[safe == 0.0] = 1.0

    # Normalize in-place
    assert mat.indptr is not None, "CSR matrix indptr must not be None"
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


# =============================================================================
# Adaptive Horizontal Expansion (Element-wise Fallback)
# =============================================================================


def expand_csr_horizontal_adaptive_gpu(
    mat: cps.csr_matrix,
    bin_size: float,
    ms2_tolerance_ppm: float,
    nbins: int,
) -> cps.csr_matrix:
    """
    Expand CSR matrix columns with mass-dependent window sizes (GPU version).

    Why: MS2 tolerance creates a window around each peak. For each non-zero at
    column j (m/z = j * bin_size), compute the tolerance-based window and expand
    by that many bins. This is fully vectorized on GPU for efficiency.

    The expansion creates a "fuzzy" binning that accounts for instrument precision.

    Args:
        mat: CuPy CSR matrix
        bin_size: Bin width in Da
        ms2_tolerance_ppm: MS2 tolerance in ppm
        nbins: Total number of bins

    Returns:
        Expanded CuPy CSR matrix (may have more non-zeros)
    """
    if mat.nnz == 0:
        return mat

    # Compute m/z for each column index
    col_indices = mat.indices
    col_mz = col_indices.astype(cp.float64) * bin_size

    # Adaptive tolerance (ppm-based above 200 Da, absolute below)
    eff_mz = cp.maximum(col_mz, MASS_TOLERANCE_CUTOFF)
    tol_da = eff_mz * ms2_tolerance_ppm * 1e-6
    windows = cp.ceil(tol_da / bin_size).astype(cp.int32)

    # Free intermediate arrays ASAP
    del col_mz, eff_mz, tol_da

    # Expand: for each element, create (2*window + 1) copies with shifted columns
    repeats = 2 * windows + 1
    ends = cp.cumsum(repeats)
    total_items = int(ends[-1])

    # Map each output element back to its source element
    dest_indices = cp.arange(total_items, dtype=cp.int64)
    source_idxs = cp.searchsorted(ends, dest_indices, side="right")
    del repeats, ends

    # Gather data from source elements
    new_data = mat.data[source_idxs]

    # Compute local offset within each expansion group
    starts = cp.zeros((mat.nnz,), dtype=cp.int64)
    starts[1:] = cp.cumsum(2 * windows + 1, dtype=cp.int64)[:-1]
    start_offsets = starts[source_idxs]
    del starts

    local_offsets = dest_indices - start_offsets
    shifts = local_offsets - windows[source_idxs]
    new_cols = col_indices[source_idxs] + shifts
    del dest_indices, start_offsets, local_offsets, shifts

    # Filter to valid column range
    mask = (new_cols >= 0) & (new_cols < nbins)
    n_valid = int(mask.sum())
    if not n_valid:
        del mask, new_cols, new_data, source_idxs, windows
        return cps.csr_matrix(mat.shape, dtype=cp.float32)

    new_cols = new_cols[mask]
    new_data = new_data[mask]
    valid_source_idxs = source_idxs[mask]
    del mask, source_idxs

    # Map source indices to row indices
    source_rows_compact = (
        cp.searchsorted(mat.indptr, cp.arange(mat.nnz, dtype=cp.int32), side="right")
        - 1
    )
    new_rows = source_rows_compact[valid_source_idxs]
    del source_rows_compact, valid_source_idxs, windows

    # Build expanded matrix (COO -> CSR sums duplicates)
    out = cps.coo_matrix((new_data, (new_rows, new_cols)), shape=mat.shape).tocsr()
    del new_data, new_rows, new_cols

    return out


# =============================================================================
# SpMM Expansion Matrix Construction
# =============================================================================


@numba.njit(parallel=True, cache=True)
def _expansion_matrix_get_row_lengths(
    nbins: int,
    bin_size: float,
    ms2_tolerance_ppm: float,
    mass_tol_cutoff: float,
) -> NDArray[np.int32]:
    """Compute number of non-zeros per row for expansion matrix."""
    row_lengths = np.zeros(nbins, dtype=np.int32)
    for i in numba.prange(nbins):
        mz = float(i) * bin_size
        eff_mz = max(mz, mass_tol_cutoff)
        tol_da = eff_mz * ms2_tolerance_ppm * 1e-6
        window = int(np.ceil(tol_da / bin_size))

        start = max(0, i - window)
        end = min(nbins - 1, i + window)
        row_lengths[i] = end - start + 1
    return row_lengths


@cuda.jit
def _expansion_matrix_fill_indices_cuda(
    indptr: NDArray[np.int64],
    indices: NDArray[np.int32],
    nbins: int,
    bin_size: float,
    ms2_tolerance_ppm: float,
    mass_tol_cutoff: float,
) -> None:
    """CUDA kernel to fill expansion matrix indices."""
    row = cuda.grid(1)
    if row >= nbins:
        return

    start_idx = indptr[row]

    mz = float(row) * bin_size
    eff_mz = max(mz, mass_tol_cutoff)
    tol_da = eff_mz * ms2_tolerance_ppm * 1e-6
    window = int(cuda.libdevice.ceil(tol_da / bin_size))

    col_start = max(0, row - window)
    col_end = min(nbins - 1, row + window)
    count = col_end - col_start + 1

    for k in range(count):
        indices[start_idx + k] = col_start + k


def construct_expansion_matrix_gpu(
    bin_size: float,
    ms2_tolerance_ppm: float,
    nbins: int,
    upper_mass_bound: float,
    logger: Optional[logging.Logger] = None,
) -> Optional[cps.csr_matrix]:
    """
    Construct the sparse expansion matrix on GPU.

    The expansion matrix E is a square (nbins, nbins) matrix where E[i,j] = 1
    if column j is within the tolerance window of column i. Multiplying a
    spectrum matrix by E performs the tolerance expansion via SpMM.

    Why SpMM expansion:
    - Single matrix multiplication instead of per-element expansion
    - Better GPU utilization for large matrices
    - Expansion matrix is computed once and reused across all batches

    Args:
        bin_size: Bin width in Da
        ms2_tolerance_ppm: MS2 tolerance in ppm
        nbins: Number of bins
        upper_mass_bound: Maximum m/z (for fallback recommendations)
        logger: Optional logger for warnings

    Returns:
        Square CuPy CSR matrix (nbins, nbins) or None if OOM predicted.
    """
    try:
        # Calculate lengths on CPU (fast and low memory)
        lengths_cpu = _expansion_matrix_get_row_lengths(
            nbins, bin_size, ms2_tolerance_ppm, MASS_TOLERANCE_CUTOFF
        )
        total_nnz = int(np.sum(lengths_cpu))

        # Estimate size: nnz*(4+4) + (nbins+1)*8 bytes
        # float32 data + int32 indices + int64 indptr
        size_bytes = total_nnz * 8 + (nbins + 1) * 8
        size_gb = size_bytes / 1e9

        free_mem, _ = cp.cuda.Device(0).mem_info
        free_mem_gb = free_mem / 1e9

        # Debug print (always print for diagnosis)
        print(
            f"DEBUG: SpMM Matrix Construction: nbins={nbins}, nnz={total_nnz}, "
            f"size={size_gb:.4f}GB, free={free_mem_gb:.2f}GB"
        )

        # Require that matrix leaves at least 2GB of free memory for batches
        # AND consumes no more than 85% of available memory (relaxed from 60%)
        # This allows running on 16GB cards with 0.0001 bin size (req ~9.7GB)
        remaining_mem_gb = free_mem_gb - size_gb
        if size_bytes > free_mem * 0.85 or remaining_mem_gb < 2.0:
            # Calculate hypothetical sizes for recommendations
            # 1. Reduced mass bound (500 Da)
            nbins_500 = int(np.floor(500.0 / float(bin_size))) + 1
            lengths_500 = _expansion_matrix_get_row_lengths(
                nbins_500, bin_size, ms2_tolerance_ppm, MASS_TOLERANCE_CUTOFF
            )
            nnz_500 = int(np.sum(lengths_500))
            size_500_gb = (nnz_500 * 8 + (nbins_500 + 1) * 8) / 1e9

            # 2. Coarser binning (0.001 Da)
            nbins_coarse = int(np.floor(upper_mass_bound / 0.001)) + 1
            lengths_coarse = _expansion_matrix_get_row_lengths(
                nbins_coarse, 0.001, ms2_tolerance_ppm, MASS_TOLERANCE_CUTOFF
            )
            nnz_coarse = int(np.sum(lengths_coarse))
            size_coarse_gb = (nnz_coarse * 8 + (nbins_coarse + 1) * 8) / 1e9

            msg = (
                f"Expansion matrix (~{size_gb:.2f} GB) too large for available GPU memory "
                f"({free_mem_gb:.2f} GB). "
                "Falling back to slower kernel-based expansion. Performance will be reduced.\n"
                "To enable fast SpMM expansion, consider:\n"
                f"1. Reducing upper_mass_bound (current: {upper_mass_bound}). "
                f"E.g., 500 Da requires ~{size_500_gb:.2f} GB.\n"
                f"2. Increasing bin_size (current: {bin_size}). "
                f"E.g., 0.001 Da requires ~{size_coarse_gb:.2f} GB."
            )

            print(f"DEBUG: Fallback triggered. {msg}")

            if logger:
                logger.warning(msg)
            return None

        lengths_gpu = cp.asarray(lengths_cpu)

        indptr_gpu = cp.zeros(nbins + 1, dtype=cp.int64)
        indptr_gpu[1:] = cp.cumsum(lengths_gpu)

        indices_gpu = cp.zeros(total_nnz, dtype=cp.int32)

        threads_per_block = 256
        blocks = (nbins + threads_per_block - 1) // threads_per_block

        _expansion_matrix_fill_indices_cuda[blocks, threads_per_block](
            indptr_gpu,
            indices_gpu,
            nbins,
            bin_size,
            ms2_tolerance_ppm,
            MASS_TOLERANCE_CUTOFF,
        )

        data_gpu = cp.ones(total_nnz, dtype=cp.float32)

        return cps.csr_matrix(
            (data_gpu, indices_gpu, indptr_gpu), shape=(nbins, nbins), dtype=cp.float32
        )
    except cp.cuda.memory.OutOfMemoryError:
        if logger:
            logger.warning(
                "OOM while constructing expansion matrix. Falling back to kernel expansion."
            )
        return None
    except Exception as e:
        if logger:
            logger.warning(
                f"Failed to construct expansion matrix: {e}. Falling back to kernel expansion."
            )
        return None
