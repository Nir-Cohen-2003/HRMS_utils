"""
Configuration constants and dataclasses for GPU approximate similarity.

This module centralizes all configuration used across the GPU similarity pipeline:
- Data type constants for CPU/GPU arrays
- Memory estimation factors
- GPUApproximateConfig dataclass with all tunable parameters
- AggregatedKernelTimings for profiling

Why separate module:
- Avoids circular imports between binning, gpu_operations, batching
- Single source of truth for constants
- Clean dataclass definitions without surrounding implementation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import polars as pl


# =============================================================================
# Data Type Constants
# =============================================================================

# Index types (spectrum IDs, row/col indices)
INDEX_DTYPE_NP = np.int32
INDEX_DTYPE_PL = pl.Int32

# Intensity and CSR data types
APPROX_INTENSITY_DTYPE_NP = np.float32
CSR_DATA_DTYPE_CPU = np.float32

# GPU-specific types
GPU_CSR_INDEX_DTYPE_NP = np.int32
GPU_CSR_INDPTR_DTYPE_NP = np.int32
GPU_SIM_DTYPE_NP = np.float32


# =============================================================================
# Memory Estimation Factors
# =============================================================================

# CSR overhead: accounts for fragmentation, metadata, cupy internals
GPU_CSR_OVERHEAD_FACTOR = 1.25

# Similarity matrix temporary overhead: sparse matmul intermediates
GPU_SIM_TEMP_OVERHEAD_FACTOR = 1.20

# Minimum m/z for ppm tolerance calculation (below this, use absolute tolerance)
MASS_TOLERANCE_CUTOFF = 200.0


# =============================================================================
# Profiling Dataclass
# =============================================================================


@dataclass
class AggregatedKernelTimings:
    """Aggregated timings for GPU kernel operations across all batches."""

    transfer_to_gpu_ms: float = 0.0
    normalize_left_ms: float = 0.0
    normalize_right_ms: float = 0.0
    expand_ms: float = 0.0
    spmm_ms: float = 0.0
    threshold_and_extract_ms: float = 0.0
    transfer_to_cpu_ms: float = 0.0
    total_ms: float = 0.0

    # Wall time (end-to-end including Python overhead, measured via perf_counter)
    wall_time_ms: float = 0.0

    # Percentages (computed by compute_percentages)
    transfer_to_gpu_pct: float = 0.0
    normalize_left_pct: float = 0.0
    normalize_right_pct: float = 0.0
    expand_pct: float = 0.0
    expand_left_pct: float = 0.0
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
            threshold_and_extract_ms=self.threshold_and_extract_ms
            + other.threshold_and_extract_ms,
            transfer_to_cpu_ms=self.transfer_to_cpu_ms + other.transfer_to_cpu_ms,
            total_ms=self.total_ms + other.total_ms,
            wall_time_ms=self.wall_time_ms + other.wall_time_ms,
        )

    def compute_percentages(self) -> None:
        """Compute percentage of total time for each operation."""
        if self.total_ms <= 0:
            return

        self.transfer_to_gpu_pct = (self.transfer_to_gpu_ms / self.total_ms) * 100.0
        self.normalize_left_pct = (self.normalize_left_ms / self.total_ms) * 100.0
        self.normalize_right_pct = (self.normalize_right_ms / self.total_ms) * 100.0
        self.expand_pct = (self.expand_ms / self.total_ms) * 100.0
        self.expand_left_pct = self.expand_pct
        self.spmm_pct = (self.spmm_ms / self.total_ms) * 100.0
        self.threshold_and_extract_pct = (
            self.threshold_and_extract_ms / self.total_ms
        ) * 100.0
        self.transfer_to_cpu_pct = (self.transfer_to_cpu_ms / self.total_ms) * 100.0


# =============================================================================
# Main Configuration Dataclass
# =============================================================================


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

        # Async writer memory management
        writer_max_queue_memory_bytes: Max memory for writer queue (None = auto 80% available)
        writer_memory_safety_ratio: Safety ratio for auto memory limit (default: 0.80)

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

    # Async writer memory management
    writer_max_queue_memory_bytes: Optional[int] = None
    writer_memory_safety_ratio: float = 0.80

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
        assert 0.0 < self.writer_memory_safety_ratio <= 1.0, (
            f"writer_memory_safety_ratio must be in (0, 1], got {self.writer_memory_safety_ratio}"
        )
        if self.writer_max_queue_memory_bytes is not None:
            assert self.writer_max_queue_memory_bytes > 0, (
                f"writer_max_queue_memory_bytes must be positive if provided, "
                f"got {self.writer_max_queue_memory_bytes}"
            )

        # Compute number of bins
        self.nbins = int(np.floor(self.upper_mass_bound / float(self.bin_size))) + 1
        assert self.nbins > 0, (
            f"Computed nbins={self.nbins} must be positive. "
            f"Check upper_mass_bound={self.upper_mass_bound} and bin_size={self.bin_size}."
        )
