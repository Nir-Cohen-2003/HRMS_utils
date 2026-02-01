"""
Dynamic batching logic for GPU similarity computation.

This module handles:
- Computing optimal batch sizes based on GPU memory
- Yielding batches of spectra for processing

Why separate module:
- Batching logic is independent of both binning and GPU ops
- Can be tested with mock memory values
- Clean interface for batch generation
"""

from __future__ import annotations

from typing import Iterator

import cupy as cp
import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from .config import (
    GPU_CSR_OVERHEAD_FACTOR,
    GPU_SIM_TEMP_OVERHEAD_FACTOR,
    INDEX_DTYPE_NP,
    GPUApproximateConfig,
)


def yield_batches_dynamic(
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


def compute_dynamic_max_peaks(
    config: GPUApproximateConfig,
    avg_peaks_per_spectrum: float,
) -> int:
    """
    Estimate maximum peaks per batch based on GPU memory and config.

    Why: GPU memory usage is dominated by:
    1. CSR matrices (left and right batches)
    2. Expanded right matrix (after tolerance window expansion)
    3. Similarity matrix (N_left x N_right dense/semi-dense)
    4. Temporary arrays during thresholding

    We solve a quadratic equation to find optimal batch size N that fits in
    target_mem * safety_factor.

    Memory model:
    target_mem * safety_factor = N^2 * (sim_bytes * temp_overhead) + N * (bytes_expanded + bytes_csr)

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

    # Solve quadratic: a*N^2 + b*N - c = 0
    # Why quadratic: similarity matrix is N x M (quadratic in batch size)
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
