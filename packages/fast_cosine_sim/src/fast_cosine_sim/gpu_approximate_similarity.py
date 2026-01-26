#!/usr/bin/env python
"""
Single-file GPU-accelerated batched approximate similarity computation.

This module provides a self-contained implementation for computing pairwise
approximate (binned dot-product) similarities on GPU with efficient memory
management and batching.

Key features:
- GPU-only computation using CuPy
- Self-comparison mode with upper-triangular optimization (exploits ij=ji symmetry)
- Cross-library comparison mode (full NxM)
- Dynamic batching based on GPU memory and peak counts
- Configurable via GPUApproximateConfig dataclass
- Optional async parquet writing with queue + writer thread
- Efficient polars → numpy → GPU arrays → CSR → COO pipeline

Usage:
    config = GPUApproximateConfig(
        upper_mass_bound=1000.0,
        bin_size=0.0001,
        ms2_tolerance_ppm=10.0,
        approx_threshold=0.65,
        target_gpu_mem_ratio=0.6,
    )

    # Self-comparison (upper triangular)
    result = batched_approximate_similarity_gpu(df, config)

    # Cross-library comparison
    result = batched_approximate_similarity_gpu(df1, config, right_df=df2)

    # Write to parquet with async I/O
    batched_approximate_similarity_gpu(df, config, output_path="output.parquet")

Why this module exists:
- Consolidates GPU approximate similarity into a single, dependency-free file
- Avoids circular imports and module coupling
- Provides a simple, well-documented API for GPU similarity computation
- All GPU operations (binning, expansion, matmul, extraction) in one place
"""

from __future__ import annotations

import gc
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from time import perf_counter
from typing import Iterator, Literal, Optional

import cupy as cp
import cupyx.scipy.sparse as cps
import numba
import numpy as np
import polars as pl
import scipy.sparse as sp
from numba import cuda
from numpy.typing import NDArray

# =============================================================================
# Constants
# =============================================================================

INDEX_DTYPE_NP = np.int32
INDEX_DTYPE_PL = pl.Int32
APPROX_INTENSITY_DTYPE_NP = np.float32
CSR_DATA_DTYPE_CPU = np.float32
GPU_CSR_INDEX_DTYPE_NP = np.int32
GPU_CSR_INDPTR_DTYPE_NP = np.int32
GPU_SIM_DTYPE_NP = np.float32
GPU_CSR_OVERHEAD_FACTOR = 1.25
GPU_SIM_TEMP_OVERHEAD_FACTOR = 1.20
MASS_TOLERANCE_CUTOFF = 200.0


# =============================================================================
# Configuration Dataclass
# =============================================================================



@dataclass
class AggregatedKernelTimings:
    """Aggregated timings for GPU kernel operations."""
    transfer_to_gpu_ms: float = 0.0
    normalize_left_ms: float = 0.0
    normalize_right_ms: float = 0.0
    expand_ms: float = 0.0
    spmm_ms: float = 0.0
    threshold_and_extract_ms: float = 0.0
    transfer_to_cpu_ms: float = 0.0
    total_ms: float = 0.0
    
    # Percentages (computed by compute_percentages)
    transfer_to_gpu_pct: float = 0.0
    normalize_left_pct: float = 0.0
    normalize_right_pct: float = 0.0
    expand_pct: float = 0.0
    expand_right_pct: float = 0.0
    spmm_pct: float = 0.0
    threshold_and_extract_pct: float = 0.0
    transfer_to_cpu_pct: float = 0.0
    
    def __add__(self, other: AggregatedKernelTimings) -> AggregatedKernelTimings:
        return AggregatedKernelTimings(
            transfer_to_gpu_ms=self.transfer_to_gpu_ms + other.transfer_to_gpu_ms,
            normalize_left_ms=self.normalize_left_ms + other.normalize_left_ms,
            normalize_right_ms=self.normalize_right_ms + other.normalize_right_ms,
            expand_ms=self.expand_ms + other.expand_ms,
            spmm_ms=self.spmm_ms + other.spmm_ms,
            threshold_and_extract_ms=self.threshold_and_extract_ms + other.threshold_and_extract_ms,
            transfer_to_cpu_ms=self.transfer_to_cpu_ms + other.transfer_to_cpu_ms,
            total_ms=self.total_ms + other.total_ms
        )

    def compute_percentages(self) -> None:
        """Compute percentage of total time for each operation."""
        if self.total_ms <= 0:
            return
            
        self.transfer_to_gpu_pct = (self.transfer_to_gpu_ms / self.total_ms) * 100.0
        self.normalize_left_pct = (self.normalize_left_ms / self.total_ms) * 100.0
        self.normalize_right_pct = (self.normalize_right_ms / self.total_ms) * 100.0
        self.expand_pct = (self.expand_ms / self.total_ms) * 100.0
        self.expand_right_pct = self.expand_pct
        self.spmm_pct = (self.spmm_ms / self.total_ms) * 100.0
        self.threshold_and_extract_pct = (self.threshold_and_extract_ms / self.total_ms) * 100.0
        self.transfer_to_cpu_pct = (self.transfer_to_cpu_ms / self.total_ms) * 100.0


@dataclass
class GPUApproximateConfig:
    """
    Configuration for GPU approximate similarity computation.

    This dataclass centralizes all parameters needed for binning, expansion,
    similarity computation, and memory management.

    Attributes:
        # Binning parameters
        upper_mass_bound: Maximum m/z to consider for binning (Da)
        bin_size: Bin width in Da
        ms2_tolerance_ppm: MS2 tolerance for adaptive expansion (ppm)
        intensity_power: Power applied to intensities during binning
        approx_threshold: Minimum similarity score to return

        # Centroiding (enabled by default to prevent similarities > 1.0)
        centroiding_enabled: Enable centroiding preprocessing (default: True)
        mass_tolerance_cutoff_mz: Minimum m/z for ppm tolerance calculation (default: 200 Da)

        # Experimental optimization
        enable_spmm_expansion: Use SpMM expansion matrix for better performance (default: True)

        # Comparison mode
        comparison_mode: "self" for upper-triangular, "cross" for full NxM

        # Memory management
        target_gpu_mem_ratio: Fraction (0, 1] of free GPU memory to target
        max_peaks_per_batch: Optional user limit on peaks per batch (None = auto)
        safety_factor: Conservative multiplier applied to memory estimates (0.5 = safe)

        # Column names (configurable for different input schemas)
        spectrum_id_col: Column name for spectrum IDs (default: "idx")
        mz_col: Column name for m/z values (list of floats)
        intensity_col: Column name for intensity values (list of floats)

        # Data types (for memory estimation and GPU transfer)
        csr_data_dtype: CSR data dtype (float32 recommended)
        csr_index_dtype: CSR indices dtype (int32 sufficient for most cases)
        similarity_dtype: Similarity matrix dtype (float32 recommended)

        # Buffering for async writes
        write_buffer_batches: Number of GPU batches to accumulate before flushing to writer

        # Computed in __post_init__
        nbins: Number of bins (computed from upper_mass_bound / bin_size)
    """

    # Binning parameters
    upper_mass_bound: float = 1000.0
    bin_size: float = 0.0001
    ms2_tolerance_ppm: float = 10.0
    intensity_power: float = 0.5
    approx_threshold: float = 0.65

    # Centroiding (enabled by default)
    centroiding_enabled: bool = True
    mass_tolerance_cutoff_mz: float = 200.0

    # Experimental: SpMM Expansion (uses precomputed matrix)
    enable_spmm_expansion: bool = True

    # Comparison mode
    comparison_mode: Literal["self", "cross"] = "self"

    # Memory management
    target_gpu_mem_ratio: float = 0.6
    max_peaks_per_batch: Optional[int] = None
    safety_factor: float = 0.5

    # Column names
    spectrum_id_col: str = "idx"
    mz_col: str = "mz"
    intensity_col: str = "intensity"

    # Data types
    csr_data_dtype: np.dtype = np.dtype(np.float32)
    csr_index_dtype: np.dtype = np.dtype(np.int32)
    similarity_dtype: np.dtype = np.dtype(np.float32)

    # Buffering
    write_buffer_batches: int = 50

    # Computed
    nbins: int = 0

    def __post_init__(self) -> None:
        """Validate parameters and compute derived values."""
        assert self.upper_mass_bound > 0.0, "upper_mass_bound must be positive"
        assert self.bin_size > 0.0, "bin_size must be positive"
        assert self.ms2_tolerance_ppm >= 0.0, "ms2_tolerance_ppm must be non-negative"
        assert 0.0 <= self.approx_threshold <= 1.0, "approx_threshold must be in [0, 1]"
        assert 0.0 < self.target_gpu_mem_ratio <= 1.0, (
            "target_gpu_mem_ratio must be in (0, 1]"
        )
        assert 0.0 < self.safety_factor <= 1.0, (
            "safety_factor must be in (0, 1] (0.5 = conservative, 1.0 = aggressive)"
        )
        if self.max_peaks_per_batch is not None:
            assert self.max_peaks_per_batch > 0, (
                "max_peaks_per_batch must be positive if provided"
            )
        assert self.comparison_mode in ("self", "cross"), (
            f"comparison_mode must be 'self' or 'cross', got {self.comparison_mode}"
        )
        assert self.mass_tolerance_cutoff_mz > 0.0, (
            f"mass_tolerance_cutoff_mz must be positive, got {self.mass_tolerance_cutoff_mz}"
        )

        # Compute number of bins
        self.nbins = int(np.floor(self.upper_mass_bound / float(self.bin_size))) + 1
        assert self.nbins > 0, (
            f"Computed nbins={self.nbins} must be positive. "
            f"Check upper_mass_bound={self.upper_mass_bound} and bin_size={self.bin_size}."
        )


# =============================================================================
# Helper Functions: Flattening and Binning
# =============================================================================


def select_and_collect(
    frame: pl.DataFrame | pl.LazyFrame, config: GPUApproximateConfig
) -> pl.DataFrame:
    """
    Collect a LazyFrame if needed, otherwise return DataFrame, anyway returns only necessery columns.

    Why: Accept both DataFrame and LazyFrame inputs for flexibility.
    """
    frame = frame.select(config.mz_col, config.intensity_col, config.spectrum_id_col)
    return (
        frame.collect(engine="streaming") if isinstance(frame, pl.LazyFrame) else frame
    )


def _flatten_spectra_to_numpy(
    df: pl.DataFrame, mz_col: str, int_col: str
) -> tuple[NDArray[np.float64], NDArray[np.float32], NDArray[np.int32], int]:
    """
    Flatten list-valued spectrum columns from DataFrame into NumPy arrays.

    Why: Polars list columns are efficient for storage but need to be exploded
    for binning. This function explodes and converts to numpy arrays in one pass.

    Args:
        df: DataFrame with list columns
        mz_col: Name of m/z column (list of floats)
        int_col: Name of intensity column (list of floats)

    Returns:
        (flat_mzs, flat_ints, spec_idx, n_spec)
        - flat_mzs: np.ndarray[np.float64] of all m/z values
        - flat_ints: np.ndarray[np.float32] of all intensities
        - spec_idx: np.ndarray[np.int32] mapping each peak to its spectrum index
        - n_spec: number of spectra
    """
    n_spec = len(df)
    if n_spec == 0:
        return (
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float32),
            np.asarray([], dtype=np.int32),
            0,
        )

    # Add temporary row index and explode
    df_idx = df.with_row_index("__spec_idx")
    exploded = df_idx.explode([mz_col, int_col])

    if len(exploded) == 0:
        return (
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float32),
            np.asarray([], dtype=np.int32),
            n_spec,
        )

    # Cast and extract
    exploded = exploded.with_columns(
        [
            pl.col(mz_col).cast(pl.Float32),
            pl.col(int_col).cast(pl.Float32),
            pl.col("__spec_idx").cast(INDEX_DTYPE_PL),
        ]
    )

    flat_mzs = exploded.get_column(mz_col).to_numpy()
    flat_ints = exploded.get_column(int_col).to_numpy()
    spec_idx = exploded.get_column("__spec_idx").to_numpy()

    return flat_mzs, flat_ints, spec_idx, n_spec


def _sparse_bin_flat_spectra_to_csr(
    flat_mzs: NDArray[np.float64],
    flat_ints: NDArray[np.float32],
    spec_idx: NDArray[np.int32],
    n_spec: int,
    upper_bound: float,
    intensity_power: float,
    bin_size: float,
) -> sp.csr_matrix:
    """
    Turn flattened arrays into a sparse CSR matrix (n_spec, nbins).

    Why: Binning reduces dimensionality and enables fast sparse matmul for
    approximate similarity. COO construction with duplicates summed is the
    most efficient path in SciPy.

    Binning uses: bin = np.rint(mz / bin_size)
    Duplicates are summed via COO -> CSR conversion.

    Args:
        flat_mzs: All m/z values
        flat_ints: All intensity values
        spec_idx: Spectrum index for each peak
        n_spec: Total number of spectra
        upper_bound: Maximum m/z
        intensity_power: Power to apply to intensities
        bin_size: Bin width

    Returns:
        scipy.sparse.csr_matrix of shape (n_spec, nbins)
    """
    nbins = int(np.floor(upper_bound / float(bin_size))) + 1

    if n_spec == 0 or flat_mzs.size == 0 or flat_ints.size == 0:
        return sp.csr_matrix((n_spec, nbins), dtype=CSR_DATA_DTYPE_CPU)

    # Bin m/z values
    mass_bins = np.rint(flat_mzs / float(bin_size)).astype(np.int32)

    # Filter valid bins
    valid_mask = (mass_bins >= 0) & (mass_bins < nbins) & (flat_ints > 0)
    if not np.any(valid_mask):
        return sp.csr_matrix((n_spec, nbins), dtype=CSR_DATA_DTYPE_CPU)

    mass_bins = mass_bins[valid_mask].astype(np.int32)
    spec_idx = spec_idx[valid_mask].astype(np.int32)
    weights = np.asarray(flat_ints[valid_mask], dtype=np.float32) ** float(
        intensity_power
    )

    # Build COO matrix (duplicates are summed in tocsr())
    coo = sp.coo_matrix(
        (weights.astype(CSR_DATA_DTYPE_CPU, copy=False), (spec_idx, mass_bins)),
        shape=(n_spec, nbins),
        dtype=CSR_DATA_DTYPE_CPU,
    )

    # Convert to CSR (SciPy sums duplicates automatically)
    return sp.csr_matrix(coo.tocsr())


def _sparse_bin_spectra_df_to_csr(
    df: pl.DataFrame,
    mz_col: str,
    int_col: str,
    upper_bound: float,
    intensity_power: float,
    bin_size: float,
    *,
    apply_centroiding: bool = False,
    tolerance_ppm: float = 10.0,
    mass_tolerance_cutoff_mz: float = 200.0,
) -> sp.csr_matrix:
    """
    Explode list-valued spectra and bin into a sparse CSR matrix.

    Why: This is the main entry point for converting a Polars DataFrame of
    spectra into a binned sparse matrix ready for GPU transfer.

    Optionally applies centroiding before binning to prevent one-to-many
    peak matching (which causes similarities > 1.0).

    Args:
        df: DataFrame with list columns
        mz_col: Name of m/z column
        int_col: Name of intensity column
        upper_bound: Maximum m/z
        intensity_power: Power to apply to intensities
        bin_size: Bin width
        apply_centroiding: If True, centroid peaks before binning
        tolerance_ppm: PPM tolerance for centroiding (if enabled)
        mass_tolerance_cutoff_mz: m/z cutoff for centroiding (if enabled)

    Returns:
        scipy.sparse.csr_matrix of shape (len(df), nbins)
    """
    nbins = int(np.floor(upper_bound / float(bin_size))) + 1

    flat_mzs, flat_ints, spec_idx, n_spec = _flatten_spectra_to_numpy(
        df, mz_col, int_col
    )

    if n_spec == 0:
        return sp.csr_matrix((0, nbins), dtype=CSR_DATA_DTYPE_CPU)
    if flat_mzs.size == 0 or flat_ints.size == 0:
        return sp.csr_matrix((n_spec, nbins), dtype=CSR_DATA_DTYPE_CPU)

    # Apply centroiding if enabled
    # Why: Prevents one-to-many peak matching which causes similarities > 1.0
    if apply_centroiding:
        from .centroiding import centroid_flat_spectra

        flat_mzs, flat_ints, spec_idx, n_spec = centroid_flat_spectra(
            flat_mzs,
            flat_ints,
            spec_idx,
            n_spec,
            tolerance_ppm=tolerance_ppm,
            mass_tolerance_cutoff_mz=mass_tolerance_cutoff_mz,
        )

    return _sparse_bin_flat_spectra_to_csr(
        flat_mzs, flat_ints, spec_idx, n_spec, upper_bound, intensity_power, bin_size
    )


# =============================================================================
# GPU Operations: Normalization and Expansion
# =============================================================================


def _normalize_csr_rows_inplace_gpu(mat: cps.csr_matrix) -> cp.ndarray:
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


def _expand_csr_horizontal_adaptive_gpu(
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
        print(f"DEBUG: SpMM Matrix Construction: nbins={nbins}, nnz={total_nnz}, size={size_gb:.4f}GB, free={free_mem_gb:.2f}GB")
        
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
                f"Expansion matrix (~{size_gb:.2f} GB) too large for available GPU memory ({free_mem_gb:.2f} GB). "
                "Falling back to slower kernel-based expansion. Performance will be reduced.\n"
                "To enable fast SpMM expansion, consider:\n"
                f"1. Reducing upper_mass_bound (current: {upper_mass_bound}). E.g., 500 Da requires ~{size_500_gb:.2f} GB.\n"
                f"2. Increasing bin_size (current: {bin_size}). E.g., 0.001 Da requires ~{size_coarse_gb:.2f} GB."
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


# =============================================================================
# Batching Logic
# =============================================================================


def _yield_batches_dynamic(
    csr_matrix: sp.csr_matrix,
    global_idxs: NDArray[np.int32],
    max_peaks: int,
    min_batch_size: int = 100,
) -> Iterator[tuple[int, int, sp.csr_matrix, NDArray[np.int32]]]:
    """
    Yield batches (start_idx, end_idx, csr_batch, idxs_batch) based on peak counts.

    Why: GPU memory is limited. By batching based on total peak counts (non-zero
    entries), we can process large libraries without OOM while maintaining high
    GPU utilization.

    This generator greedily accumulates rows until the total non-zero count
    approaches max_peaks, then yields a batch.

    Args:
        csr_matrix: Input CSR matrix where each row = one spectrum
        global_idxs: Global indices corresponding to rows (int32)
        max_peaks: Target maximum peaks (non-zeros) per batch
        min_batch_size: Minimum spectra per batch

    Yields:
        (start_idx, end_idx, csr_matrix[start:end], global_idxs[start:end])
    """
    n_spectra = csr_matrix.shape[0]
    if n_spectra == 0:
        return

    indptr = csr_matrix.indptr
    assert indptr is not None, "CSR matrix indptr must not be None"
    indptr = np.asarray(indptr)
    assert indptr.ndim == 1 and indptr.size >= n_spectra + 1, (
        f"CSR indptr must be 1D with length >= n_spectra + 1, got shape {indptr.shape}"
    )

    start_idx = 0
    while start_idx < n_spectra:
        # Find end index where cumulative peaks <= max_peaks
        target_peaks = indptr[start_idx] + max_peaks
        candidate_end = np.searchsorted(indptr, target_peaks, side="right") - 1

        # Ensure at least one row per batch
        if candidate_end <= start_idx:
            candidate_end = start_idx + 1

        # Enforce minimum batch size if possible
        if candidate_end - start_idx < min_batch_size:
            candidate_end = min(start_idx + min_batch_size, n_spectra)

        end_idx = min(candidate_end, n_spectra)

        batch_csr = csr_matrix[start_idx:end_idx]
        batch_idxs = global_idxs[start_idx:end_idx].astype(INDEX_DTYPE_NP, copy=False)

        yield start_idx, end_idx, batch_csr, batch_idxs

        start_idx = end_idx


def _compute_dynamic_max_peaks(
    config: GPUApproximateConfig,
    avg_peaks_per_spectrum: float,
) -> int:
    """
    Estimate maximum peaks per batch based on GPU memory and config.

    Why: GPU memory usage is dominated by:
    1. CSR matrices (left and right batches)
    2. Expanded right matrix (after tolerance window expansion)
    3. Similarity matrix (N_left × N_right dense/semi-dense)
    4. Temporary arrays during thresholding

    We solve a quadratic equation to find optimal batch size N that fits in
    target_mem * safety_factor.

    Memory model:
    target_mem * safety_factor = N² * (sim_bytes * temp_overhead) + N * (bytes_expanded + bytes_csr)

    Args:
        config: GPUApproximateConfig with memory parameters
        avg_peaks_per_spectrum: Average peaks per spectrum (from data)

    Returns:
        Maximum peaks per batch (at least 100k)
    """
    free_mem, total_mem = cp.cuda.Device(0).mem_info
    target_mem = free_mem * config.target_gpu_mem_ratio

    # Bytes per peak in CSR (data + indices + indptr, with overhead)
    bytes_per_data = int(np.dtype(config.csr_data_dtype).itemsize)
    bytes_per_index = int(np.dtype(config.csr_index_dtype).itemsize)
    bytes_per_indptr = int(np.dtype(config.csr_index_dtype).itemsize)
    bytes_per_peak = (
        bytes_per_data + bytes_per_index + bytes_per_indptr
    ) * GPU_CSR_OVERHEAD_FACTOR

    # Expansion factor from tolerance window
    expansion_factor = 1.0
    if config.ms2_tolerance_ppm > 0:
        window_da = config.upper_mass_bound * config.ms2_tolerance_ppm * 1e-6
        expansion_factor = max(1.0, (2 * window_da) / config.bin_size)
        expansion_factor *= 2.0  # Safety margin for density increase

    bytes_per_spectrum_csr = avg_peaks_per_spectrum * bytes_per_peak
    bytes_per_spectrum_expanded = bytes_per_spectrum_csr * expansion_factor

    # Similarity matrix cost
    sim_bytes_per_element = float(np.dtype(config.similarity_dtype).itemsize)

    # Solve quadratic: a*N² + b*N - c = 0
    # Why quadratic: similarity matrix is N×M (quadratic in batch size)
    a = sim_bytes_per_element * GPU_SIM_TEMP_OVERHEAD_FACTOR
    b = bytes_per_spectrum_expanded + bytes_per_spectrum_csr
    c = target_mem * config.safety_factor

    discriminant = b**2 + 4 * a * c
    if discriminant > 0:
        estimated_spectra_per_batch = int((-b + np.sqrt(discriminant)) / (2 * a))
    else:
        # Fallback: very conservative
        estimated_spectra_per_batch = int(
            target_mem / (bytes_per_spectrum_expanded * 10)
        )

    # Ensure reasonable minimum
    estimated_spectra_per_batch = max(100, estimated_spectra_per_batch)

    # Convert to peak count
    estimated_max_peaks = int(estimated_spectra_per_batch * avg_peaks_per_spectrum)

    # Apply user limit if provided
    if config.max_peaks_per_batch is not None:
        max_peaks = min(estimated_max_peaks, config.max_peaks_per_batch)
    else:
        max_peaks = estimated_max_peaks

    # Ensure minimum (100k peaks)
    return max(max_peaks, 100_000)


# =============================================================================
# Async Writer for Non-Blocking I/O
# =============================================================================


class AsyncParquetWriter:
    """
    Thread-based writer for non-blocking parquet writes.

    Why: GPU computation is fast, but parquet writes can be slow (especially with
    compression). Using a separate writer thread with a queue allows the GPU to
    continue computing while previous results are being written.

    The writer accumulates chunks and appends to the parquet file in batches.
    """

    def __init__(self, output_path: Path, logger: Optional[logging.Logger] = None):
        """
        Initialize async writer.

        Args:
            output_path: Path to output parquet file
            logger: Optional logger for progress reporting
        """
        self.output_path = output_path
        self.logger = logger
        self.queue: Queue = Queue(maxsize=5)  # Limit memory usage
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
        Submit a batch for writing (blocks if queue is full).

        Args:
            data: Dictionary with keys 'idx_left', 'idx_right', 'similarity'
        """
        if self._exception is not None:
            raise RuntimeError(f"Writer thread failed: {self._exception}")
        self.queue.put(data)

    def stop(self) -> None:
        """Signal the writer thread to stop and wait for completion."""
        self._stop_event.set()
        self.thread.join()
        if self._exception is not None:
            raise RuntimeError(f"Writer thread failed: {self._exception}")

    def _writer_loop(self) -> None:
        """Writer thread main loop."""
        try:
            chunks = []

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


class ResultBuffer:
    """
    Thread-safe accumulator for pair results before writing.

    Why: Collecting multiple GPU batch results before submitting to the writer
    reduces write frequency and improves throughput.
    """

    def __init__(self):
        self.left_idxs: list[np.ndarray] = []
        self.right_idxs: list[np.ndarray] = []
        self.similarities: list[np.ndarray] = []
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
# Main Function
# =============================================================================


def batched_approximate_similarity_gpu(
    left_df: pl.DataFrame | pl.LazyFrame,
    config: GPUApproximateConfig,
    right_df: Optional[pl.DataFrame | pl.LazyFrame] = None,
    output_path: Optional[Path | str] = None,
    logger: Optional[logging.Logger] = None,
    log_timings: bool = False,
) -> pl.DataFrame | pl.LazyFrame | tuple[pl.DataFrame | pl.LazyFrame, AggregatedKernelTimings]:
    """
    Compute batched approximate similarity on GPU with optimized memory management.

    This function performs binned dot-product similarity (cosine similarity on
    binned spectra) using GPU acceleration. It supports two modes:

    1. Self-comparison (right_df=None, config.comparison_mode="self"):
       Computes upper-triangular similarity matrix, exploiting symmetry (ij = ji)
       to reduce computation by ~2x.

    2. Cross-library comparison (right_df provided, config.comparison_mode="cross"):
       Computes full NxM similarity matrix between two libraries.

    The algorithm:
    1. Optional: Centroid spectra (merge peaks within tolerance, prevents similarities > 1.0)
    2. Convert DataFrames to binned CSR matrices (CPU)
    3. Dynamically batch based on GPU memory and peak counts
    4. For each batch pair:
       - Transfer to GPU
       - Expand left matrix (tolerance window) - done once per left batch, reused
       - Normalize right matrix rows (L2)
       - Compute expanded_left @ normalized_right.T (sparse matmul)
       - Threshold and extract pairs above threshold
       - Accumulate results
    5. Either return DataFrame or write to parquet with async I/O

    Why expand left:
    - The expanded matrix is larger (2-3x more non-zeros due to tolerance windows)
    - Streaming the expanded matrix row-wise in SpMM is more cache-friendly
    - The smaller normalized matrix is accessed repeatedly in inner loop and stays in cache
    - This reduces memory bandwidth pressure on the sparse matmul operation

    Args:
        left_df: DataFrame or LazyFrame with list columns specified by
                 config.mz_col and config.intensity_col
        config: GPUApproximateConfig instance with all parameters
        right_df: Optional second library for cross-comparison (None = self-comparison)
        output_path: Optional path for parquet output (None = return DataFrame)
        logger: Optional logger for progress reporting
        log_timings: If True, return (result, timings) tuple with detailed GPU profiling data

    Returns:
        If log_timings is False:
            DataFrame with columns ['idx_left', 'idx_right', 'similarity'] if output_path is None,
            LazyFrame (scan of written parquet) if output_path is provided
        If log_timings is True:
            Tuple of (DataFrame/LazyFrame, AggregatedKernelTimings)

    Raises:
        AssertionError: If inputs are invalid (with detailed messages)
        RuntimeError: If GPU operations fail
    """
    # =========================================================================
    # 1. Validate Inputs
    # =========================================================================

    # Collect LazyFrames if needed
    left_df = select_and_collect(left_df, config)
    if right_df is not None:
        right_df = select_and_collect(right_df, config)

    assert len(left_df) > 0, "left_df is empty. Provide at least one spectrum."

    assert config.mz_col in left_df.columns, (
        f"Column '{config.mz_col}' not found in left_df. "
        f"Available columns: {left_df.columns}. "
        f"Set config.mz_col to the correct m/z column name."
    )

    assert config.intensity_col in left_df.columns, (
        f"Column '{config.intensity_col}' not found in left_df. "
        f"Available columns: {left_df.columns}. "
        f"Set config.intensity_col to the correct intensity column name."
    )

    # Determine mode and validate consistency
    is_cross_library = right_df is not None

    if is_cross_library:
        assert config.comparison_mode == "cross", (
            f"When right_df is provided, config.comparison_mode must be 'cross', "
            f"got '{config.comparison_mode}'"
        )
        assert right_df is not None, (
            "right_df must not be None in cross-library mode (logic error)"
        )
        assert len(right_df) > 0, (
            "right_df is empty. Provide at least one spectrum for cross-comparison."
        )
        assert config.mz_col in right_df.columns, (
            f"Column '{config.mz_col}' not found in right_df. "
            f"Available columns: {right_df.columns}."
        )
        assert config.intensity_col in right_df.columns, (
            f"Column '{config.intensity_col}' not found in right_df. "
            f"Available columns: {right_df.columns}."
        )
    else:
        assert config.comparison_mode == "self", (
            f"When right_df is None, config.comparison_mode must be 'self', "
            f"got '{config.comparison_mode}'"
        )

    if logger:
        mode_str = (
            "cross-library"
            if is_cross_library
            else "self-comparison (upper triangular)"
        )
        logger.info(f"Starting batched GPU approximate similarity: mode={mode_str}")
        logger.info(
            f"  Config: threshold={config.approx_threshold}, "
            f"bin_size={config.bin_size}, tolerance={config.ms2_tolerance_ppm} ppm, "
            f"centroiding={'enabled' if config.centroiding_enabled else 'disabled'}"
        )

    # =========================================================================
    # 2. Add Row Indices (if not present)
    # =========================================================================

    # Add 0..N-1 indices to left if needed
    if config.spectrum_id_col not in left_df.columns:
        left_df_idx = left_df.with_row_index(config.spectrum_id_col).with_columns(
            pl.col(config.spectrum_id_col).cast(pl.Int32)
        )
    else:
        left_df_idx = left_df.with_columns(
            pl.col(config.spectrum_id_col).cast(pl.Int32)
        )

    # Validate int32 range
    idx_max = left_df_idx.select(pl.col(config.spectrum_id_col).max()).item()
    assert idx_max is not None, (
        f"{config.spectrum_id_col} max was None; left_df appears empty unexpectedly"
    )
    assert int(idx_max) <= np.iinfo(np.int32).max, (
        f"Index overflow: max {config.spectrum_id_col}={idx_max} exceeds int32 limit ({np.iinfo(np.int32).max}). "
        f"Reduce library size or change index dtype policy. "
        f"Current library size: {len(left_df_idx)} spectra."
    )

    n_spectra_left = len(left_df_idx)

    # Add indices to right if cross-library
    if is_cross_library:
        assert right_df is not None
        if config.spectrum_id_col not in right_df.columns:
            right_df_idx = right_df.with_row_index(config.spectrum_id_col).with_columns(
                pl.col(config.spectrum_id_col).cast(pl.Int32)
            )
        else:
            right_df_idx = right_df.with_columns(
                pl.col(config.spectrum_id_col).cast(pl.Int32)
            )

        idx_max_right = right_df_idx.select(pl.col(config.spectrum_id_col).max()).item()
        assert idx_max_right is not None, (
            f"{config.spectrum_id_col} max was None; right_df appears empty unexpectedly"
        )
        assert int(idx_max_right) <= np.iinfo(np.int32).max, (
            f"Index overflow in right_df: max {config.spectrum_id_col}={idx_max_right} exceeds int32 limit. "
            f"Reduce library size. Current library size: {len(right_df_idx)} spectra."
        )

        n_spectra_right = len(right_df_idx)
    else:
        right_df_idx = None
        n_spectra_right = n_spectra_left

    if logger:
        if is_cross_library:
            logger.info(f"  Left library: {n_spectra_left} spectra")
            logger.info(f"  Right library: {n_spectra_right} spectra")
        else:
            logger.info(f"  Library: {n_spectra_left} spectra")

    # =========================================================================
    # 3. Initialize Resources
    # =========================================================================

    # SpMM Expansion Matrix (initialized lazily if enabled)
    expansion_matrix: Optional[cps.csr_matrix] = None

    # Construct expansion matrix if enabled and tolerance > 0
    if config.enable_spmm_expansion and config.ms2_tolerance_ppm > 0.0:
        if logger:
            logger.info("Attempting to construct SpMM expansion matrix...")
        
        expansion_matrix = construct_expansion_matrix_gpu(
            bin_size=config.bin_size,
            ms2_tolerance_ppm=config.ms2_tolerance_ppm,
            nbins=config.nbins,
            upper_mass_bound=config.upper_mass_bound,
            logger=logger
        )
        
        if expansion_matrix is None:
            # Fallback message already logged by construct_expansion_matrix_gpu
            if logger:
                logger.info("Using element-wise adaptive expansion (fallback).")
        else:
            if logger:
                logger.info(f"SpMM expansion matrix constructed successfully ({expansion_matrix.nnz} elements).")

    # =========================================================================
    # 4. Convert to CSR Matrices (with optional centroiding)
    # =========================================================================

    t_bin = perf_counter()

    if logger:
        logger.info("  Binning left library...")

    left_csr_matrix = _sparse_bin_spectra_df_to_csr(
        left_df_idx,
        config.mz_col,
        config.intensity_col,
        upper_bound=config.upper_mass_bound,
        intensity_power=config.intensity_power,
        bin_size=config.bin_size,
        apply_centroiding=config.centroiding_enabled,
        tolerance_ppm=config.ms2_tolerance_ppm,
        mass_tolerance_cutoff_mz=config.mass_tolerance_cutoff_mz,
    )

    left_global_idxs = (
        left_df_idx[config.spectrum_id_col]
        .to_numpy()
        .astype(INDEX_DTYPE_NP, copy=False)
    )

    if is_cross_library:
        if logger:
            logger.info("  Binning right library...")

        assert right_df_idx is not None
        right_csr_matrix = _sparse_bin_spectra_df_to_csr(
            right_df_idx,
            config.mz_col,
            config.intensity_col,
            upper_bound=config.upper_mass_bound,
            intensity_power=config.intensity_power,
            bin_size=config.bin_size,
            apply_centroiding=config.centroiding_enabled,
            tolerance_ppm=config.ms2_tolerance_ppm,
            mass_tolerance_cutoff_mz=config.mass_tolerance_cutoff_mz,
        )

        right_global_idxs = (
            right_df_idx[config.spectrum_id_col]
            .to_numpy()
            .astype(INDEX_DTYPE_NP, copy=False)
        )
    else:
        right_csr_matrix = left_csr_matrix
        right_global_idxs = left_global_idxs

    bin_time = perf_counter() - t_bin

    if logger:
        logger.info(
            f"  Binning complete in {bin_time:.3f}s. "
            f"Left: {left_csr_matrix.shape}, Right: {right_csr_matrix.shape}"
        )
        logger.info(
            f"  Left nnz: {left_csr_matrix.nnz:_}, Right nnz: {right_csr_matrix.nnz:_}"
        )

    # =========================================================================
    # 5. Dynamic Batching
    # =========================================================================

    avg_peaks_left = left_csr_matrix.nnz / max(n_spectra_left, 1)
    max_peaks = _compute_dynamic_max_peaks(config, avg_peaks_left)

    if logger:
        free_mem, total_mem = cp.cuda.Device(0).mem_info
        logger.info(
            f"  GPU Memory: {free_mem / 1e9:.2f} GB free / {total_mem / 1e9:.2f} GB total"
        )
        logger.info(
            f"  Target usage: {config.target_gpu_mem_ratio:.0%} × {config.safety_factor} safety = "
            f"{free_mem * config.target_gpu_mem_ratio * config.safety_factor / 1e9:.2f} GB"
        )
        logger.info(f"  Avg peaks/spectrum: {avg_peaks_left:.1f}")
        logger.info(f"  Max peaks/batch: {max_peaks:_}")

    # Generate batches
    batches_left = list(
        _yield_batches_dynamic(
            left_csr_matrix, left_global_idxs, max_peaks, min_batch_size=100
        )
    )

    if is_cross_library:
        avg_peaks_right = right_csr_matrix.nnz / max(n_spectra_right, 1)
        max_peaks_right = _compute_dynamic_max_peaks(config, avg_peaks_right)
        batches_right = list(
            _yield_batches_dynamic(
                right_csr_matrix, right_global_idxs, max_peaks_right, min_batch_size=100
            )
        )
    else:
        batches_right = batches_left

    num_batches_left = len(batches_left)
    num_batches_right = len(batches_right)

    if logger:
        logger.info(
            f"  Created {num_batches_left} left batches, {num_batches_right} right batches"
        )
        total_batch_pairs = num_batches_left * num_batches_right
        if not is_cross_library:
            # Upper triangular: sum(1..N) = N(N+1)/2
            total_batch_pairs = num_batches_left * (num_batches_left + 1) // 2
        logger.info(f"  Total batch pairs to process: {total_batch_pairs:_}")

    # =========================================================================
    # 6. Setup Output (Writer or Buffer)
    # =========================================================================

    writer: Optional[AsyncParquetWriter] = None
    if output_path is not None:
        output_path = Path(output_path)
        writer = AsyncParquetWriter(output_path, logger=logger)
        writer.start()
        if logger:
            logger.info(f"  Writing results to: {output_path}")

    buffer = ResultBuffer()

    # =========================================================================
    # 7. Batch Processing Loop
    # =========================================================================

    total_pairs = 0
    gpu_batch_count = 0
    aggregated_timings = AggregatedKernelTimings()
    t_compute_start = perf_counter()

    # Outer loop: Left batches (expanded and reused)
    # Why left-outer: expand the larger matrix (expanded) once and stream it in SpMM
    for i, (l_start, l_end, l_csr, l_idxs) in enumerate(batches_left):
        t_left_batch = perf_counter()

        # Log GPU memory before processing
        free_before, total = cp.cuda.Device(0).mem_info
        if logger:
            logger.info(
                f"  [Left batch {i + 1}/{num_batches_left}] GPU mem before: "
                f"{free_before / 1e9:.2f} GB free"
            )

        # Create events for timing
        evt_start = cp.cuda.Event() if log_timings else None
        evt_xfer = cp.cuda.Event() if log_timings else None
        evt_norm = cp.cuda.Event() if log_timings else None
        evt_expand = cp.cuda.Event() if log_timings else None

        if log_timings:
            evt_start.record()

        # Transfer left batch to GPU
        l_data_gpu = cp.asarray(
            np.asarray(l_csr.data).astype(APPROX_INTENSITY_DTYPE_NP, copy=False)
        )
        l_indices_gpu = cp.asarray(np.asarray(l_csr.indices))
        l_indptr_gpu = cp.asarray(np.asarray(l_csr.indptr))
        L_gpu = cps.csr_matrix(
            (l_data_gpu, l_indices_gpu, l_indptr_gpu), shape=l_csr.shape
        )
        del l_data_gpu, l_indices_gpu, l_indptr_gpu

        if log_timings:
            evt_xfer.record()

        # Normalize in-place
        _ = _normalize_csr_rows_inplace_gpu(L_gpu)

        if log_timings:
            evt_norm.record()

        # Expand Left (if using tolerance)
        if config.ms2_tolerance_ppm > 0.0:
            if expansion_matrix is not None:
                # Use fast SpMM expansion
                L_gpu = L_gpu.dot(expansion_matrix)
            else:
                # Fallback to element-wise expansion
                L_gpu = _expand_csr_horizontal_adaptive_gpu(
                    L_gpu,
                    config.bin_size,
                    config.ms2_tolerance_ppm,
                    config.nbins,
                )

        if log_timings:
            evt_expand.record()
            evt_expand.synchronize()
            # Accumulate outer loop timings immediately (will happen once per outer loop)
            # Note: We multiply by number of inner loop iterations later? 
            # No, these are one-time costs per left batch. 
            # BUT, we need to be careful. The aggregation is global.
            aggregated_timings.transfer_to_gpu_ms += cp.cuda.get_elapsed_time(evt_start, evt_xfer)
            aggregated_timings.normalize_left_ms += cp.cuda.get_elapsed_time(evt_xfer, evt_norm)
            aggregated_timings.expand_ms += cp.cuda.get_elapsed_time(evt_norm, evt_expand)
            aggregated_timings.total_ms += cp.cuda.get_elapsed_time(evt_start, evt_expand)

        # Inner loop: Right batches
        for j, (r_start, r_end, r_csr, r_idxs) in enumerate(batches_right):
            # Triangular check for self-comparison
            if not is_cross_library and j > i:
                continue

            # Timing events for inner loop
            evt_inner_start = cp.cuda.Event() if log_timings else None
            evt_r_xfer = cp.cuda.Event() if log_timings else None
            evt_r_norm = cp.cuda.Event() if log_timings else None
            evt_spmm = cp.cuda.Event() if log_timings else None
            evt_thresh = cp.cuda.Event() if log_timings else None
            evt_cpu = cp.cuda.Event() if log_timings else None

            if log_timings:
                evt_inner_start.record()

            # Transfer right batch to GPU
            r_data_gpu = cp.asarray(
                np.asarray(r_csr.data).astype(APPROX_INTENSITY_DTYPE_NP, copy=False)
            )
            r_indices_gpu = cp.asarray(np.asarray(r_csr.indices))
            r_indptr_gpu = cp.asarray(np.asarray(r_csr.indptr))
            R_gpu = cps.csr_matrix(
                (r_data_gpu, r_indices_gpu, r_indptr_gpu), shape=r_csr.shape
            )
            del r_data_gpu, r_indices_gpu, r_indptr_gpu

            if log_timings:
                evt_r_xfer.record()

            # Normalize
            _ = _normalize_csr_rows_inplace_gpu(R_gpu)

            if log_timings:
                evt_r_norm.record()

            # Matmul: L @ R.T (expanded_left @ normalized_right)
            sim = L_gpu.dot(R_gpu.T)

            if log_timings:
                evt_spmm.record()

            # 3. Apply Threshold In-Place
            # We mask the data array directly. This effectively turns low values into explicit zeros.
            sim.data[sim.data < config.approx_threshold] = 0

            # 4. Prune Zeros
            # This shrinks the underlying data/indices arrays on the GPU, freeing memory.
            sim.eliminate_zeros()
            
            if log_timings:
                evt_thresh.record()

            if sim.nnz > 0:
                # 5. Convert to COO
                # Now we convert only the surviving elements to COO format.
                sim_coo = sim.tocoo()

                # Transfer to CPU
                # Using copy=False is safe if we consume immediately
                rows_cpu = cp.asnumpy(sim_coo.row)
                cols_cpu = cp.asnumpy(sim_coo.col)
                data_cpu = cp.asnumpy(sim_coo.data)
                
                if log_timings:
                    evt_cpu.record()

                # Map local indices to global indices
                # Note: rows are indices into L_gpu (0..batch_size), cols are indices into R_gpu
                global_left = l_idxs[rows_cpu]
                global_right = r_idxs[cols_cpu]

                # Accumulate
                buffer.add(global_left, global_right, data_cpu)
                total_pairs += len(data_cpu)

                del rows_cpu, cols_cpu, data_cpu, global_left, global_right, sim_coo
            else:
                if log_timings:
                    evt_cpu.record()

            del sim, R_gpu
            gpu_batch_count += 1
            
            if log_timings:
                evt_cpu.synchronize()
                aggregated_timings.transfer_to_gpu_ms += cp.cuda.get_elapsed_time(evt_inner_start, evt_r_xfer)
                aggregated_timings.normalize_right_ms += cp.cuda.get_elapsed_time(evt_r_xfer, evt_r_norm)
                aggregated_timings.spmm_ms += cp.cuda.get_elapsed_time(evt_r_norm, evt_spmm)
                aggregated_timings.threshold_and_extract_ms += cp.cuda.get_elapsed_time(evt_spmm, evt_thresh)
                aggregated_timings.transfer_to_cpu_ms += cp.cuda.get_elapsed_time(evt_thresh, evt_cpu)
                aggregated_timings.total_ms += cp.cuda.get_elapsed_time(evt_inner_start, evt_cpu)

            # Flush buffer to writer periodically
            if writer and not buffer.is_empty():
                if gpu_batch_count % config.write_buffer_batches == 0:
                    data = buffer.flush()
                    if data:
                        writer.write_batch(data)

        # Cleanup left batch
        del L_gpu
        # Force garbage collection to prevent fragmentation
        # (cp.get_default_memory_pool().free_all_blocks() is faster than gc.collect())
        cp.get_default_memory_pool().free_all_blocks()

        if logger:
            elapsed = perf_counter() - t_left_batch
            rate = l_csr.shape[0] / elapsed
            logger.info(
                f"  [Left batch {i + 1}/{num_batches_left}] Done in {elapsed:.2f}s "
                f"({rate:.1f} spectra/s)"
            )

    t_compute = perf_counter() - t_compute_start

    # =========================================================================
    # 8. Finalize Output
    # =========================================================================

    if logger:
        logger.info(f"Computation complete in {t_compute:.2f}s")
        logger.info(f"Total matching pairs found: {total_pairs:_}")

    # Flush remaining buffer
    final_data = buffer.flush()

    if writer:
        if final_data:
            writer.write_batch(final_data)
        writer.stop()
        if logger:
            logger.info("Writer stopped.")
        
        # Return LazyFrame scan
        result = pl.scan_parquet(output_path)
    else:
        # Return in-memory DataFrame
        if final_data:
            result = pl.DataFrame(final_data)
        else:
            # Empty result
            result = pl.DataFrame(
                {
                    "idx_left": pl.Series([], dtype=pl.Int32),
                    "idx_right": pl.Series([], dtype=pl.Int32),
                    "similarity": pl.Series([], dtype=pl.Float32),
                }
            )
            
    if log_timings:
        return result, aggregated_timings
    else:
        return result
