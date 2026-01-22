#!/usr/bin/env python
"""
Benchmark script comparing different expansion algorithm implementations.

This script compares the performance of various expansion algorithms for
mass spectrometry similarity computation, measuring both expansion time
and transfer overhead for CPU and GPU implementations.

Usage:
    python benchmarks/benchmark_expansion_algorithms.py --batch-sizes 1000,5000,10000

    pixi run -e testing benchmark-expansion-algorithms -- --batch-sizes 1000,5000,10000

    pixi run benchmark-expansion-algorithms --batch-sizes 100,500,1000,5000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median, stdev, mean
from typing import Callable, Optional

import cupy as cp
import cupyx.scipy.sparse as cps
import numba
import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fast_cosine_sim.gpu_approximate_similarity import (
    MASS_TOLERANCE_CUTOFF,
    _expand_csr_horizontal_adaptive_gpu,
)

os.environ["NUMBA_NUM_THREADS"] = str(os.cpu_count() or 1)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark parameters."""
    n_peaks_per_spectrum: int = 50
    bin_size: float = 0.0001
    ms2_tolerance_ppm: float = 10.0
    upper_mass_bound: float = 1000.0
    warmup_runs: int = 3
    timed_runs: int = 10
    batch_sizes: list[int] = field(default_factory=lambda: [1000, 5000, 10000])
    output_dir: Optional[Path] = None
    skip_cpu: bool = False
    skip_gpu: bool = False

    @property
    def nbins(self) -> int:
        return int(self.upper_mass_bound / self.bin_size) + 1


def generate_synthetic_csr(
    n_spectra: int,
    n_peaks: int,
    nbins: int,
    seed: int = 42,
) -> sp.csr_matrix:
    """
    Generate synthetic CSR matrix with random peaks.

    Args:
        n_spectra: Number of spectra (rows)
        n_peaks: Peaks per spectrum
        nbins: Number of bins (columns)
        seed: Random seed for reproducibility

    Returns:
        SciPy CSR matrix with shape (n_spectra, nbins)
    """
    rng = np.random.default_rng(seed)

    rows = np.repeat(np.arange(n_spectra), n_peaks)
    cols = rng.integers(0, nbins, size=n_spectra * n_peaks)
    data = rng.random(n_spectra * n_peaks, dtype=np.float32)

    return sp.csr_matrix(
        (data, (rows, cols)),
        shape=(n_spectra, nbins),
        dtype=np.float32,
    )


def expand_scipy_vectorized(
    mat: sp.csr_matrix,
    bin_size: float,
    ms2_tolerance_ppm: float,
    nbins: int,
) -> sp.csr_matrix:
    """
    Expand CSR matrix using vectorized NumPy/SciPy operations.

    This implementation uses pure vectorized operations without explicit
    Python loops. It constructs a COO matrix and converts to CSR.

    Args:
        mat: Input CSR matrix
        bin_size: Bin width in Da
        ms2_tolerance_ppm: MS2 tolerance in ppm
        nbins: Total number of bins

    Returns:
        Expanded CSR matrix
    """
    if mat.nnz == 0:
        return mat

    col_indices = mat.indices
    col_mz = col_indices.astype(np.float64) * bin_size
    eff_mz = np.maximum(col_mz, MASS_TOLERANCE_CUTOFF)
    tol_da = eff_mz * ms2_tolerance_ppm * 1e-6
    windows = np.ceil(tol_da / bin_size).astype(np.int32)

    repeats = 2 * windows + 1
    ends = np.cumsum(repeats)
    total_items = int(ends[-1])

    dest_indices = np.arange(total_items, dtype=np.int64)
    source_idxs = np.searchsorted(ends, dest_indices, side="right")

    new_data = mat.data[source_idxs]

    starts = np.zeros(mat.nnz, dtype=np.int64)
    starts[1:] = np.cumsum(2 * windows + 1, dtype=np.int64)[:-1]
    start_offsets = starts[source_idxs]

    local_offsets = dest_indices - start_offsets
    shifts = local_offsets - windows[source_idxs]
    new_cols = col_indices[source_idxs] + shifts

    mask = (new_cols >= 0) & (new_cols < nbins)
    new_cols = new_cols[mask]
    new_data = new_data[mask]
    valid_source_idxs = source_idxs[mask]

    source_rows_compact = (
        np.searchsorted(mat.indptr, np.arange(mat.nnz, dtype=np.int32), side="right") - 1
    )
    new_rows = source_rows_compact[valid_source_idxs]

    out = sp.coo_matrix(
        (new_data, (new_rows, new_cols)),
        shape=mat.shape,
    ).tocsr()

    return out


@numba.njit(parallel=True, cache=True)
def _expand_count_elements_numba(
    indices: NDArray[np.int32],
    nnz: int,
    bin_size: float,
    ms2_tolerance_ppm: float,
    nbins: int,
) -> NDArray[np.int32]:
    """
    Count expanded elements per peak using parallel Numba.

    Args:
        indices: CSR column indices
        nnz: Number of non-zero elements
        bin_size: Bin width in Da
        ms2_tolerance_ppm: MS2 tolerance in ppm
        nbins: Total number of bins

    Returns:
        Array of counts per peak
    """
    counts = np.zeros(nnz, dtype=np.int32)

    for i in numba.prange(nnz):
        col_idx = indices[i]
        col_mz = float(col_idx) * bin_size
        eff_mz = max(col_mz, MASS_TOLERANCE_CUTOFF)
        tol_da = eff_mz * ms2_tolerance_ppm * 1e-6
        window = int(np.ceil(tol_da / bin_size))

        count = 0
        for shift in range(-window, window + 1):
            new_col = col_idx + shift
            if 0 <= new_col < nbins:
                count += 1

        counts[i] = count

    return counts


@numba.njit(parallel=True, cache=True)
def _expand_fill_output_numba(
    data: NDArray[np.float32],
    indices: NDArray[np.int32],
    nnz: int,
    bin_size: float,
    ms2_tolerance_ppm: float,
    nbins: int,
    output_offsets: NDArray[np.int32],
    out_data: NDArray[np.float32],
    out_rows: NDArray[np.int32],
    out_cols: NDArray[np.int32],
) -> None:
    """
    Fill output arrays with expanded elements using parallel Numba.

    Args:
        data: CSR data array
        indices: CSR column indices
        nnz: Number of non-zero elements
        bin_size: Bin width in Da
        ms2_tolerance_ppm: MS2 tolerance in ppm
        nbins: Total number of bins
        output_offsets: Starting positions for each peak
        out_data: Output data array (pre-allocated)
        out_rows: Output row indices (pre-allocated)
        out_cols: Output column indices (pre-allocated)
    """
    for i in numba.prange(nnz):
        intensity = data[i]
        col_idx = indices[i]

        col_mz = float(col_idx) * bin_size
        eff_mz = max(col_mz, MASS_TOLERANCE_CUTOFF)
        tol_da = eff_mz * ms2_tolerance_ppm * 1e-6
        window = int(np.ceil(tol_da / bin_size))

        out_idx = output_offsets[i]
        for shift in range(-window, window + 1):
            new_col = col_idx + shift
            if 0 <= new_col < nbins:
                out_data[out_idx] = intensity
                out_rows[out_idx] = 0
                out_cols[out_idx] = new_col
                out_idx += 1


def _compute_row_indices_csr(indptr: NDArray[np.int32], nnz: int) -> NDArray[np.int32]:
    """
    Compute row index for each element in CSR matrix.

    Args:
        indptr: CSR indptr array
        nnz: Number of non-zero elements

    Returns:
        Array of row indices for each element
    """
    row_indices = np.zeros(nnz, dtype=np.int32)
    for i in range(nnz):
        row_indices[i] = np.searchsorted(indptr, i, side="right") - 1
    return row_indices


def expand_numba_cpu_parallel(
    mat: sp.csr_matrix,
    bin_size: float,
    ms2_tolerance_ppm: float,
    nbins: int,
) -> sp.csr_matrix:
    """
    Expand CSR matrix using parallel Numba JIT compilation.

    Uses element-level parallelism with numba.prange for maximum
    CPU utilization.

    Args:
        mat: Input CSR matrix
        bin_size: Bin width in Da
        ms2_tolerance_ppm: MS2 tolerance in ppm
        nbins: Total number of bins

    Returns:
        Expanded CSR matrix
    """
    if mat.nnz == 0:
        return mat

    nnz = mat.nnz

    peak_expansion_counts = _expand_count_elements_numba(
        mat.indices,
        nnz,
        bin_size,
        ms2_tolerance_ppm,
        nbins,
    )

    output_offsets = np.zeros(nnz + 1, dtype=np.int32)
    output_offsets[1:] = np.cumsum(peak_expansion_counts)
    total_expanded = int(output_offsets[-1])

    if total_expanded == 0:
        return sp.csr_matrix(mat.shape, dtype=np.float32)

    out_data = np.zeros(total_expanded, dtype=np.float32)
    out_rows = np.zeros(total_expanded, dtype=np.int32)
    out_cols = np.zeros(total_expanded, dtype=np.int32)

    _expand_fill_output_numba(
        mat.data,
        mat.indices,
        nnz,
        bin_size,
        ms2_tolerance_ppm,
        nbins,
        output_offsets,
        out_data,
        out_rows,
        out_cols,
    )

    row_indices = _compute_row_indices_csr(mat.indptr, nnz)
    out_rows = np.repeat(row_indices, peak_expansion_counts)

    out_coo = sp.coo_matrix(
        (out_data, (out_rows, out_cols)),
        shape=mat.shape,
        dtype=np.float32,
    )

    return sp.csr_matrix(out_coo)


def expand_cupy_vectorized(
    mat_gpu: cps.csr_matrix,
    bin_size: float,
    ms2_tolerance_ppm: float,
    nbins: int,
) -> cps.csr_matrix:
    """
    Expand CSR matrix using vectorized CuPy operations.

    Wrapper around the existing GPU implementation.

    Args:
        mat_gpu: Input CuPy CSR matrix
        bin_size: Bin width in Da
        ms2_tolerance_ppm: MS2 tolerance in ppm
        nbins: Total number of bins

    Returns:
        Expanded CuPy CSR matrix
    """
    return _expand_csr_horizontal_adaptive_gpu(
        mat_gpu, bin_size, ms2_tolerance_ppm, nbins
    )


@numba.cuda.jit
def _expand_count_elements_cuda(
    indices: NDArray[np.int32],
    nnz: int,
    bin_size: float,
    ms2_tolerance_ppm: float,
    mass_tol_cutoff: float,
    nbins: int,
    peak_expansion_counts: NDArray[np.int32],
) -> None:
    """
    CUDA kernel to count expanded elements per peak.

    Args:
        indices: CSR column indices (nnz,)
        nnz: Number of non-zero elements
        bin_size: Bin width in Da
        ms2_tolerance_ppm: MS2 tolerance in ppm
        mass_tol_cutoff: Minimum effective m/z (200.0 Da)
        nbins: Total number of bins
        peak_expansion_counts: Output counts per peak (nnz,)
    """
    elem_idx = numba.cuda.grid(1)
    if elem_idx >= nnz:
        return

    col_idx = indices[elem_idx]
    col_mz = float(col_idx) * bin_size
    eff_mz = max(col_mz, mass_tol_cutoff)
    tol_da = eff_mz * ms2_tolerance_ppm * 1e-6
    window = int(numba.cuda.libdevice.ceil(tol_da / bin_size))

    count = 0
    for shift in range(-window, window + 1):
        new_col = col_idx + shift
        if 0 <= new_col < nbins:
            count += 1

    peak_expansion_counts[elem_idx] = count


@numba.cuda.jit
def _expand_fill_output_cuda(
    data: NDArray[np.float32],
    indices: NDArray[np.int32],
    row_indices: NDArray[np.int32],
    nnz: int,
    bin_size: float,
    ms2_tolerance_ppm: float,
    mass_tol_cutoff: float,
    nbins: int,
    output_offsets: NDArray[np.int32],
    out_data: NDArray[np.float32],
    out_rows: NDArray[np.int32],
    out_cols: NDArray[np.int32],
) -> None:
    """
    CUDA kernel to fill output arrays with expanded elements.

    Args:
        data: CSR data array (nnz,)
        indices: CSR column indices (nnz,)
        row_indices: Row index for each element (nnz,)
        nnz: Number of non-zero elements
        bin_size: Bin width in Da
        ms2_tolerance_ppm: MS2 tolerance in ppm
        mass_tol_cutoff: Minimum effective m/z (200.0 Da)
        nbins: Total number of bins
        output_offsets: Starting position in output for each peak (nnz+1,)
        out_data: Output data array (pre-allocated)
        out_rows: Output row indices (pre-allocated)
        out_cols: Output column indices (pre-allocated)
    """
    elem_idx = numba.cuda.grid(1)
    if elem_idx >= nnz:
        return

    intensity = data[elem_idx]
    col_idx = indices[elem_idx]
    row_idx = row_indices[elem_idx]

    col_mz = float(col_idx) * bin_size
    eff_mz = max(col_mz, mass_tol_cutoff)
    tol_da = eff_mz * ms2_tolerance_ppm * 1e-6
    window = int(numba.cuda.libdevice.ceil(tol_da / bin_size))

    out_idx = output_offsets[elem_idx]
    for shift in range(-window, window + 1):
        new_col = col_idx + shift
        if 0 <= new_col < nbins:
            out_data[out_idx] = intensity
            out_rows[out_idx] = row_idx
            out_cols[out_idx] = new_col
            out_idx += 1


def expand_numba_cuda(
    mat_gpu: cps.csr_matrix,
    bin_size: float,
    ms2_tolerance_ppm: float,
    nbins: int,
) -> cps.csr_matrix:
    """
    Expand CSR matrix using Numba CUDA kernels.

    Uses element-level parallelism with separate counting and filling kernels.

    Args:
        mat_gpu: Input CuPy CSR matrix
        bin_size: Bin width in Da
        ms2_tolerance_ppm: MS2 tolerance in ppm
        nbins: Total number of bins

    Returns:
        Expanded CuPy CSR matrix
    """
    if mat_gpu.nnz == 0:
        return mat_gpu

    nnz = mat_gpu.nnz
    n_rows = mat_gpu.shape[0]

    threads_per_block = 256
    blocks = (nnz + threads_per_block - 1) // threads_per_block

    peak_expansion_counts = cp.zeros(nnz, dtype=cp.int32)

    _expand_count_elements_cuda[blocks, threads_per_block](
        cp.asarray(mat_gpu.indices),
        nnz,
        bin_size,
        ms2_tolerance_ppm,
        MASS_TOLERANCE_CUTOFF,
        nbins,
        peak_expansion_counts,
    )

    output_offsets = cp.zeros(nnz + 1, dtype=cp.int32)
    output_offsets[1:] = cp.cumsum(peak_expansion_counts)
    total_expanded = int(output_offsets[-1])

    if total_expanded == 0:
        return cps.csr_matrix(mat_gpu.shape, dtype=cp.float32)

    out_data = cp.zeros(total_expanded, dtype=cp.float32)
    out_rows = cp.zeros(total_expanded, dtype=cp.int32)
    out_cols = cp.zeros(total_expanded, dtype=cp.int32)

    element_indices = cp.arange(nnz, dtype=cp.int32)
    row_indices = cp.searchsorted(mat_gpu.indptr, element_indices, side="right") - 1
    row_indices = row_indices.astype(cp.int32)

    _expand_fill_output_cuda[blocks, threads_per_block](
        cp.asarray(mat_gpu.data),
        cp.asarray(mat_gpu.indices),
        row_indices,
        nnz,
        bin_size,
        ms2_tolerance_ppm,
        MASS_TOLERANCE_CUTOFF,
        nbins,
        output_offsets,
        out_data,
        out_rows,
        out_cols,
    )

    out_coo = cps.coo_matrix(
        (out_data, (out_rows, out_cols)),
        shape=mat_gpu.shape,
        dtype=cp.float32,
    )

    return out_coo.tocsr()


@dataclass
class BenchmarkResult:
    """Results from benchmarking a single method."""
    method_name: str
    expansion_times_ms: list[float]
    transfer_times_ms: list[float]
    total_times_ms: list[float]
    output_nnz: int = 0
    setup_time_ms: float = 0.0

    @property
    def expansion_median(self) -> float:
        return median(self.expansion_times_ms)

    @property
    def expansion_std(self) -> float:
        return stdev(self.expansion_times_ms) if len(self.expansion_times_ms) > 1 else 0.0

    @property
    def transfer_median(self) -> float:
        return median(self.transfer_times_ms)

    @property
    def total_median(self) -> float:
        return median(self.total_times_ms)

    def to_dict(self) -> dict:
        result = {
            "method": self.method_name,
            "expansion_ms": {
                "median": round(self.expansion_median, 4),
                "std": round(self.expansion_std, 4),
                "min": round(min(self.expansion_times_ms), 4),
                "max": round(max(self.expansion_times_ms), 4),
                "runs": [round(t, 4) for t in self.expansion_times_ms],
            },
            "transfer_ms": {
                "median": round(self.transfer_median, 4),
                "min": round(min(self.transfer_times_ms), 4),
                "max": round(max(self.transfer_times_ms), 4),
            },
            "total_ms": {
                "median": round(self.total_median, 4),
                "min": round(min(self.total_times_ms), 4),
                "max": round(max(self.total_times_ms), 4),
            },
            "output_nnz": self.output_nnz,
        }
        if self.setup_time_ms > 0:
            result["setup_ms"] = round(self.setup_time_ms, 2)
        return result


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


@numba.njit(parallel=True, cache=True)
def _expansion_matrix_fill_indices(
    indptr: NDArray[np.int64],
    indices: NDArray[np.int32],
    nbins: int,
    bin_size: float,
    ms2_tolerance_ppm: float,
    mass_tol_cutoff: float,
) -> None:
    """Fill column indices for expansion matrix."""
    for i in numba.prange(nbins):
        start_idx = indptr[i]
        
        mz = float(i) * bin_size
        eff_mz = max(mz, mass_tol_cutoff)
        tol_da = eff_mz * ms2_tolerance_ppm * 1e-6
        window = int(np.ceil(tol_da / bin_size))
        
        col_start = max(0, i - window)
        col_end = min(nbins - 1, i + window)
        count = col_end - col_start + 1
        
        for k in range(count):
            indices[start_idx + k] = col_start + k


def construct_expansion_matrix_cpu(
    bin_size: float,
    ms2_tolerance_ppm: float,
    nbins: int,
) -> sp.csr_matrix:
    """
    Construct the sparse expansion matrix on CPU.
    
    Returns:
        Square CSR matrix (nbins, nbins)
    """
    lengths = _expansion_matrix_get_row_lengths(
        nbins, bin_size, ms2_tolerance_ppm, MASS_TOLERANCE_CUTOFF
    )
    
    indptr = np.zeros(nbins + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(lengths)
    nnz = indptr[-1]
    
    indices = np.zeros(nnz, dtype=np.int32)
    _expansion_matrix_fill_indices(
        indptr, indices, nbins, bin_size, ms2_tolerance_ppm, MASS_TOLERANCE_CUTOFF
    )
    
    data = np.ones(nnz, dtype=np.float32)
    
    return sp.csr_matrix(
        (data, indices, indptr),
        shape=(nbins, nbins),
        dtype=np.float32
    )


@numba.cuda.jit
def _expansion_matrix_fill_indices_cuda(
    indptr: NDArray[np.int64],
    indices: NDArray[np.int32],
    nbins: int,
    bin_size: float,
    ms2_tolerance_ppm: float,
    mass_tol_cutoff: float,
) -> None:
    """CUDA kernel to fill expansion matrix indices."""
    row = numba.cuda.grid(1)
    if row >= nbins:
        return
        
    start_idx = indptr[row]
    
    mz = float(row) * bin_size
    eff_mz = max(mz, mass_tol_cutoff)
    tol_da = eff_mz * ms2_tolerance_ppm * 1e-6
    window = int(numba.cuda.libdevice.ceil(tol_da / bin_size))
    
    col_start = max(0, row - window)
    col_end = min(nbins - 1, row + window)
    count = col_end - col_start + 1
    
    for k in range(count):
        indices[start_idx + k] = col_start + k


def construct_expansion_matrix_gpu(
    bin_size: float,
    ms2_tolerance_ppm: float,
    nbins: int,
) -> cps.csr_matrix:
    """
    Construct the sparse expansion matrix on GPU.
    
    Returns:
        Square CuPy CSR matrix (nbins, nbins)
    """
    lengths_cpu = _expansion_matrix_get_row_lengths(
        nbins, bin_size, ms2_tolerance_ppm, MASS_TOLERANCE_CUTOFF
    )
    lengths_gpu = cp.asarray(lengths_cpu)
    
    indptr_gpu = cp.zeros(nbins + 1, dtype=cp.int64)
    indptr_gpu[1:] = cp.cumsum(lengths_gpu)
    nnz = int(indptr_gpu[-1])
    
    indices_gpu = cp.zeros(nnz, dtype=cp.int32)
    
    threads_per_block = 256
    blocks = (nbins + threads_per_block - 1) // threads_per_block
    
    _expansion_matrix_fill_indices_cuda[blocks, threads_per_block](
        indptr_gpu, indices_gpu, nbins, bin_size, ms2_tolerance_ppm, MASS_TOLERANCE_CUTOFF
    )
    
    data_gpu = cp.ones(nnz, dtype=cp.float32)
    
    return cps.csr_matrix(
        (data_gpu, indices_gpu, indptr_gpu),
        shape=(nbins, nbins),
        dtype=cp.float32
    )


def benchmark_cpu_scipy(
    mat: sp.csr_matrix,
    config: BenchmarkConfig,
) -> BenchmarkResult:
    """Benchmark scipy vectorized expansion."""
    expansion_times = []
    transfer_times = []
    total_times = []

    for _ in range(config.warmup_runs):
        _ = expand_scipy_vectorized(mat, config.bin_size, config.ms2_tolerance_ppm, config.nbins)

    for _ in range(config.timed_runs):
        t_start = time.perf_counter()
        result = expand_scipy_vectorized(mat, config.bin_size, config.ms2_tolerance_ppm, config.nbins)
        t_end = time.perf_counter()

        expansion_times.append((t_end - t_start) * 1000)
        transfer_times.append(0.01)
        total_times.append((t_end - t_start) * 1000 + 0.01)

    return BenchmarkResult(
        method_name="scipy_vectorized",
        expansion_times_ms=expansion_times,
        transfer_times_ms=transfer_times,
        total_times_ms=total_times,
        output_nnz=result.nnz,
    )


def benchmark_cpu_numba(
    mat: sp.csr_matrix,
    config: BenchmarkConfig,
) -> BenchmarkResult:
    """Benchmark numba CPU parallel expansion."""
    expansion_times = []
    transfer_times = []
    total_times = []

    for _ in range(config.warmup_runs):
        _ = expand_numba_cpu_parallel(mat, config.bin_size, config.ms2_tolerance_ppm, config.nbins)

    for _ in range(config.timed_runs):
        t_start = time.perf_counter()
        result = expand_numba_cpu_parallel(mat, config.bin_size, config.ms2_tolerance_ppm, config.nbins)
        t_end = time.perf_counter()

        expansion_times.append((t_end - t_start) * 1000)
        transfer_times.append(0.01)
        total_times.append((t_end - t_start) * 1000 + 0.01)

    return BenchmarkResult(
        method_name="numba_cpu_parallel",
        expansion_times_ms=expansion_times,
        transfer_times_ms=transfer_times,
        total_times_ms=total_times,
        output_nnz=result.nnz,
    )


def benchmark_gpu_cupy(
    mat: sp.csr_matrix,
    config: BenchmarkConfig,
) -> BenchmarkResult:
    """Benchmark CuPy vectorized expansion."""
    expansion_times = []
    transfer_times = []
    total_times = []

    mat_gpu = cps.csr_matrix(mat)

    for _ in range(config.warmup_runs):
        _ = expand_cupy_vectorized(mat_gpu, config.bin_size, config.ms2_tolerance_ppm, config.nbins)
        cp.get_default_memory_pool().free_all_blocks()

    for _ in range(config.timed_runs):
        mat_gpu = cps.csr_matrix(mat)

        t_transfer_start = time.perf_counter()
        _ = cp.asarray(mat_gpu.data)
        _ = cp.asarray(mat_gpu.indices)
        _ = cp.asarray(mat_gpu.indptr)
        t_transfer_end = time.perf_counter()

        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()

        start_event.record()
        result = expand_cupy_vectorized(mat_gpu, config.bin_size, config.ms2_tolerance_ppm, config.nbins)
        end_event.record()
        end_event.synchronize()

        expansion_times.append(cp.cuda.get_elapsed_time(start_event, end_event))

        transfer_times.append((t_transfer_end - t_transfer_start) * 1000)
        total_times.append((t_transfer_end - t_transfer_start) * 1000 + cp.cuda.get_elapsed_time(start_event, end_event))

        del mat_gpu
        cp.get_default_memory_pool().free_all_blocks()

    return BenchmarkResult(
        method_name="cupy_vectorized",
        expansion_times_ms=expansion_times,
        transfer_times_ms=transfer_times,
        total_times_ms=total_times,
        output_nnz=int(result.nnz),
    )


def benchmark_gpu_numba(
    mat: sp.csr_matrix,
    config: BenchmarkConfig,
) -> BenchmarkResult:
    """Benchmark Numba CUDA expansion."""
    expansion_times = []
    transfer_times = []
    total_times = []

    mat_gpu = cps.csr_matrix(mat)

    for _ in range(config.warmup_runs):
        _ = expand_numba_cuda(mat_gpu, config.bin_size, config.ms2_tolerance_ppm, config.nbins)
        cp.get_default_memory_pool().free_all_blocks()

    for _ in range(config.timed_runs):
        mat_gpu = cps.csr_matrix(mat)

        t_transfer_start = time.perf_counter()
        _ = cp.asarray(mat_gpu.data)
        _ = cp.asarray(mat_gpu.indices)
        _ = cp.asarray(mat_gpu.indptr)
        t_transfer_end = time.perf_counter()

        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()

        start_event.record()
        result = expand_numba_cuda(mat_gpu, config.bin_size, config.ms2_tolerance_ppm, config.nbins)
        end_event.record()
        end_event.synchronize()

        expansion_times.append(cp.cuda.get_elapsed_time(start_event, end_event))

        transfer_times.append((t_transfer_end - t_transfer_start) * 1000)
        total_times.append((t_transfer_end - t_transfer_start) * 1000 + cp.cuda.get_elapsed_time(start_event, end_event))

        del mat_gpu
        cp.get_default_memory_pool().free_all_blocks()

    return BenchmarkResult(
        method_name="numba_cuda",
        expansion_times_ms=expansion_times,
        transfer_times_ms=transfer_times,
        total_times_ms=total_times,
        output_nnz=int(result.nnz),
    )



def benchmark_cpu_spmm(
    mat: sp.csr_matrix,
    config: BenchmarkConfig,
) -> BenchmarkResult:
    """Benchmark SpMM expansion on CPU."""
    print("  Constructing expansion matrix (CPU)...")
    t0 = time.perf_counter()
    exp_mat = construct_expansion_matrix_cpu(config.bin_size, config.ms2_tolerance_ppm, config.nbins)
    setup_time = (time.perf_counter() - t0) * 1000
    print(f"  Matrix constructed in {setup_time:.1f} ms, nnz={exp_mat.nnz:,}")

    expansion_times = []
    transfer_times = []
    total_times = []
    
    # Warmup
    for _ in range(config.warmup_runs):
        _ = mat.dot(exp_mat)
        
    for _ in range(config.timed_runs):
        t_start = time.perf_counter()
        result = mat.dot(exp_mat)
        t_end = time.perf_counter()
        
        expansion_times.append((t_end - t_start) * 1000)
        transfer_times.append(0.0)
        total_times.append((t_end - t_start) * 1000)
        
    return BenchmarkResult(
        method_name="cpu_spmm",
        expansion_times_ms=expansion_times,
        transfer_times_ms=transfer_times,
        total_times_ms=total_times,
        output_nnz=result.nnz,
        setup_time_ms=setup_time
    )


def benchmark_gpu_spmm(
    mat: sp.csr_matrix,
    config: BenchmarkConfig,
) -> BenchmarkResult:
    """Benchmark SpMM expansion on GPU."""
    print("  Constructing expansion matrix (GPU)...")
    # Clean up before allocation
    cp.get_default_memory_pool().free_all_blocks()
    
    t0 = time.perf_counter()
    exp_mat = construct_expansion_matrix_gpu(config.bin_size, config.ms2_tolerance_ppm, config.nbins)
    cp.cuda.Stream.null.synchronize()
    setup_time = (time.perf_counter() - t0) * 1000
    print(f"  Matrix constructed in {setup_time:.1f} ms, nnz={exp_mat.nnz:,}")
    
    expansion_times = []
    transfer_times = []
    total_times = []
    
    mat_gpu = cps.csr_matrix(mat)
    
    # Warmup
    for _ in range(config.warmup_runs):
        _ = mat_gpu.dot(exp_mat)
    
    for _ in range(config.timed_runs):
        del mat_gpu
        mat_gpu_cpu = mat
        
        t_transfer_start = time.perf_counter()
        mat_gpu = cps.csr_matrix(mat_gpu_cpu)
        t_transfer_end = time.perf_counter()
        
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()
        
        start_event.record()
        result = mat_gpu.dot(exp_mat)
        end_event.record()
        end_event.synchronize()
        
        expansion_times.append(cp.cuda.get_elapsed_time(start_event, end_event))
        transfer_times.append((t_transfer_end - t_transfer_start) * 1000)
        total_times.append((t_transfer_end - t_transfer_start) * 1000 + cp.cuda.get_elapsed_time(start_event, end_event))
        
    del exp_mat
    del mat_gpu
    cp.get_default_memory_pool().free_all_blocks()
    
    return BenchmarkResult(
        method_name="gpu_spmm",
        expansion_times_ms=expansion_times,
        transfer_times_ms=transfer_times,
        total_times_ms=total_times,
        output_nnz=int(result.nnz),
        setup_time_ms=setup_time
    )


def run_benchmark(
    batch_size: int,
    config: BenchmarkConfig,
) -> dict:
    """
    Run complete benchmark for a single batch size.

    Args:
        batch_size: Number of spectra to process
        config: Benchmark configuration

    Returns:
        Dictionary with benchmark results
    """
    nbins = config.nbins
    n_peaks = config.n_peaks_per_spectrum
    n_spectra = batch_size

    print(f"\n{'=' * 80}")
    print(f"Batch size: {n_spectra:,} spectra ({n_spectra * n_peaks:,} peaks)")
    print(f"{'=' * 80}")

    mat = generate_synthetic_csr(n_spectra, n_peaks, nbins, seed=42)
    print(f"Input matrix: {mat.shape}, nnz={mat.nnz:,}")

    results = {}

    if not config.skip_cpu:
        print("\n[CPU] SciPy vectorized...")
        result = benchmark_cpu_scipy(mat, config)
        results["scipy_vectorized"] = result
        print(f"       Expansion: {result.expansion_median:.2f} ms (std: {result.expansion_std:.2f})")
        print(f"       Output nnz: {result.output_nnz:,}")

        print("\n[CPU] Numba CPU parallel...")
        result = benchmark_cpu_numba(mat, config)
        results["numba_cpu_parallel"] = result
        print(f"       Expansion: {result.expansion_median:.2f} ms (std: {result.expansion_std:.2f})")
        print(f"       Output nnz: {result.output_nnz:,}")
        
        print("\n[CPU] SpMM (Precomputed Matrix)...")
        result = benchmark_cpu_spmm(mat, config)
        results["cpu_spmm"] = result
        print(f"       Expansion: {result.expansion_median:.2f} ms")
        print(f"       Setup:     {result.setup_time_ms:.2f} ms")
        print(f"       Output nnz: {result.output_nnz:,}")

    if not config.skip_gpu:
        print("\n[GPU] CuPy vectorized...")
        result = benchmark_gpu_cupy(mat, config)
        results["cupy_vectorized"] = result
        print(f"       Expansion: {result.expansion_median:.2f} ms")
        print(f"       Transfer:  {result.transfer_median:.2f} ms")
        print(f"       Output nnz: {result.output_nnz:,}")

        print("\n[GPU] Numba CUDA...")
        result = benchmark_gpu_numba(mat, config)
        results["numba_cuda"] = result
        print(f"       Expansion: {result.expansion_median:.2f} ms")
        print(f"       Transfer:  {result.transfer_median:.2f} ms")
        print(f"       Output nnz: {result.output_nnz:,}")
        
        print("\n[GPU] SpMM (Precomputed Matrix)...")
        result = benchmark_gpu_spmm(mat, config)
        results["gpu_spmm"] = result
        print(f"       Expansion: {result.expansion_median:.2f} ms")
        print(f"       Setup:     {result.setup_time_ms:.2f} ms")
        print(f"       Output nnz: {result.output_nnz:,}")

    return results


def print_summary(results_by_batch: dict, config: BenchmarkConfig) -> None:
    """Print formatted summary table of all results."""
    print("\n" + "=" * 120)
    print("SUMMARY - Expansion Algorithm Benchmark")
    print("=" * 120)
    print(f"\nConfiguration:")
    print(f"  Peaks per spectrum: {config.n_peaks_per_spectrum}")
    print(f"  Bin size: {config.bin_size} Da")
    print(f"  MS2 tolerance: {config.ms2_tolerance_ppm} ppm")
    print(f"  Warmup runs: {config.warmup_runs}")
    print(f"  Timed runs: {config.timed_runs}")
    print(f"  Batch sizes tested: {config.batch_sizes}")

    for batch_size, results in sorted(results_by_batch.items()):
        n_spectra = batch_size
        n_peaks = config.n_peaks_per_spectrum
        total_peaks = n_spectra * n_peaks

        print(f"\n{'─' * 120}")
        print(f"Batch: {n_spectra:,} spectra ({total_peaks:,} peaks)")
        print(f"{'─' * 120}")

        if not results:
            print("  No results (CPU and/or GPU benchmarks skipped)")
            continue

        baseline_total = None
        method_results = list(results.values())

        for r in method_results:
            if baseline_total is None or r.method_name in ("scipy_vectorized", "numba_cpu_parallel"):
                if r.total_median > 0:
                    baseline_total = r.total_median

        print(f"\n{'Method':<24} | {'Setup (ms)':>10} | {'Expansion (ms)':>14} | {'Transfer (ms)':>12} | {'Total (ms)':>12} | {'Speedup':>10} | {'Output NNZ':>12}")
        print(f"{'─' * 24}─{'─' * 12}─{'─' * 16}─{'─' * 14}─{'─' * 14}─{'─' * 12}─{'─' * 14}")

        for r in method_results:
            speedup = baseline_total / r.total_median if baseline_total and r.total_median > 0 else 0.0
            setup_str = f"{r.setup_time_ms:>10.1f}" if r.setup_time_ms > 0 else f"{'-':>10}"
            print(f"{r.method_name:<24} | {setup_str} | {r.expansion_median:>12.3f}  | {r.transfer_median:>10.3f}  | {r.total_median:>10.3f}  | {speedup:>8.2f}x | {r.output_nnz:>12,}")

        print(f"\n{'─' * 24}─{'─' * 12}─{'─' * 16}─{'─' * 14}─{'─' * 14}─{'─' * 12}─{'─' * 14}")

        cpu_methods = [r for r in method_results if "cpu" in r.method_name or "scipy" in r.method_name]
        gpu_methods = [r for r in method_results if "cuda" in r.method_name or "cupy" in r.method_name or "gpu" in r.method_name]

        if cpu_methods:
            fastest_cpu = min(cpu_methods, key=lambda r: r.total_median)
            print(f"\nFastest CPU: {fastest_cpu.method_name} ({fastest_cpu.total_median:.3f} ms)")

        if gpu_methods:
            fastest_gpu = min(gpu_methods, key=lambda r: r.total_median)
            print(f"Fastest GPU: {fastest_gpu.method_name} ({fastest_gpu.total_median:.3f} ms)")

            if cpu_methods:
                cpu_baseline = min(cpu_methods, key=lambda r: r.total_median)
                speedup = cpu_baseline.total_median / fastest_gpu.total_median
                print(f"GPU vs CPU: {speedup:.2f}x faster")


def save_results(results_by_batch: dict, config: BenchmarkConfig, output_dir: Path) -> None:
    """Save benchmark results to JSON file."""
    output_data = {
        "config": {
            "n_peaks_per_spectrum": config.n_peaks_per_spectrum,
            "bin_size": config.bin_size,
            "ms2_tolerance_ppm": config.ms2_tolerance_ppm,
            "upper_mass_bound": config.upper_mass_bound,
            "warmup_runs": config.warmup_runs,
            "timed_runs": config.timed_runs,
            "batch_sizes": config.batch_sizes,
        },
        "results": {},
    }

    for batch_size, results in results_by_batch.items():
        output_data["results"][str(batch_size)] = {
            method: r.to_dict() for method, r in results.items()
        }

    output_file = output_dir / "benchmark_expansion_results.json"
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")


def parse_args() -> BenchmarkConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark expansion algorithms for mass spectrometry similarity computation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default settings
    python benchmarks/benchmark_expansion_algorithms.py

    # Test specific batch sizes
    python benchmarks/benchmark_expansion_algorithms.py --batch-sizes 100,500,1000

    # Skip GPU benchmarks
    python benchmarks/benchmark_expansion_algorithms.py --skip-gpu

    # More runs for statistical significance
    python benchmarks/benchmark_expansion_algorithms.py --runs 20 --warmup 5

    # Run with pixi
    pixi run benchmark-expansion-algorithms -- --batch-sizes 1000,5000,10000
        """,
    )

    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1000,5000,10000",
        help="Comma-separated list of batch sizes to test (default: '1000,5000,10000')",
    )

    parser.add_argument(
        "--n-peaks",
        type=int,
        default=50,
        help="Number of peaks per spectrum (default: 50)",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup runs before timing (default: 3)",
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of timed runs for statistics (default: 10)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save JSON results (default: no file output)",
    )

    parser.add_argument(
        "--skip-cpu",
        action="store_true",
        help="Skip CPU benchmarks",
    )

    parser.add_argument(
        "--skip-gpu",
        action="store_true",
        help="Skip GPU benchmarks (for systems without GPU)",
    )

    args = parser.parse_args()

    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]

    config = BenchmarkConfig(
        n_peaks_per_spectrum=args.n_peaks,
        warmup_runs=args.warmup,
        timed_runs=args.runs,
        batch_sizes=batch_sizes,
        skip_cpu=args.skip_cpu,
        skip_gpu=args.skip_gpu,
    )

    if args.output_dir:
        config.output_dir = Path(args.output_dir)
        config.output_dir.mkdir(parents=True, exist_ok=True)

    return config


def main() -> None:
    """Main entry point."""
    print("=" * 80)
    print("Expansion Algorithm Benchmark")
    print("=" * 80)

    config = parse_args()

    if config.skip_cpu and config.skip_gpu:
        print("Error: Both --skip-cpu and --skip-gpu specified. Nothing to benchmark.")
        sys.exit(1)

    print(f"\nConfiguration:")
    print(f"  Peaks per spectrum: {config.n_peaks_per_spectrum}")
    print(f"  Bin size: {config.bin_size} Da")
    print(f"  MS2 tolerance: {config.ms2_tolerance_ppm} ppm")
    print(f"  Upper mass bound: {config.upper_mass_bound} Da")
    print(f"  Bins: {config.nbins:,}")
    print(f"  Warmup runs: {config.warmup_runs}")
    print(f"  Timed runs: {config.timed_runs}")
    print(f"  Batch sizes: {config.batch_sizes}")
    print(f"  CPU benchmarks: {'Enabled' if not config.skip_cpu else 'Skipped'}")
    print(f"  GPU benchmarks: {'Enabled' if not config.skip_gpu else 'Skipped'}")

    if not config.skip_gpu:
        try:
            device_info = cp.cuda.Device(0)
            print(f"\nGPU Device: {device_info.name}")
            free_mem, total_mem = device_info.mem_info
            print(f"  Memory: {free_mem / 1e9:.2f} GB free / {total_mem / 1e9:.2f} GB total")
        except Exception as e:
            print(f"\nWarning: Could not query GPU info: {e}")
            print("GPU benchmarks may fail if no GPU is available.")

    results_by_batch = {}

    for batch_size in config.batch_sizes:
        results_by_batch[batch_size] = run_benchmark(batch_size, config)

    print_summary(results_by_batch, config)

    if config.output_dir:
        save_results(results_by_batch, config, config.output_dir)

    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
