#!/usr/bin/env python
"""
GPU Kernel Optimization Script for fast_cosine_sim.

This script provides comprehensive profiling, benchmarking, and optimization
analysis for the GPU-accelerated approximate similarity computation kernel.

Key Features:
- Synthetic spectrum generation with realistic diversity (1 in 1000 matching rate)
- Detailed kernel operation profiling (SpMM, normalization, expansion, etc.)
- Batch size optimization with throughput analysis
- Data type comparison (float32 vs float64)
- Visual analysis with matplotlib plots
- Actionable optimization recommendations

Why this script exists:
- Isolate GPU kernel performance from data distribution effects
- Identify bottlenecks in the computation pipeline
- Find optimal batch size for different GPU memory configurations
- Guide kernel optimization decisions with empirical data
- Test under realistic sparsity conditions (most spectra don't match)

Usage:
    python scripts/optimize_gpu_kernel.py \\
        --n-spectra 50000 \\
        --n-peaks-per-spectrum 100 \\
        --match-rate 0.001 \\
        --output-dir ./optimization_results

Requirements:
    - CuPy with CUDA toolkit
    - Polars
    - NumPy
    - Matplotlib/Seaborn (for plots)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Optional

import cupy as cp
import cupyx.scipy.sparse as cps
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from numpy.typing import NDArray

from fast_cosine_sim import (
    GPUApproximateConfig,
    batched_approximate_similarity_gpu,
)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10


# =============================================================================
# Data Classes for Profiling Results
# =============================================================================


@dataclass
class OperationProfile:
    """Profile data for a single GPU operation."""

    name: str
    duration_ms: float
    memory_before_bytes: int
    memory_after_bytes: int
    memory_delta_bytes: int
    percentage_of_total: float = 0.0


@dataclass
class InternalKernelTimings:
    """
    Detailed internal timings for kernel operations.

    Why: Identifies which specific operation (SpMM, expansion, thresholding, etc.)
    is the bottleneck within the GPU kernel.
    """

    transfer_to_gpu_ms: float
    normalize_left_ms: float
    normalize_right_ms: float
    expand_right_ms: float
    spmm_ms: float
    threshold_and_extract_ms: float
    transfer_to_cpu_ms: float
    total_batch_ms: float


@dataclass
class SpMMMetrics:
    """
    Sparse matrix multiply performance metrics.

    Why: SpMM (sparse matrix-matrix multiply) should dominate compute time.
    If it doesn't, other operations are bottlenecks that need optimization.
    """

    duration_ms: float
    left_shape: tuple[int, int]
    right_shape: tuple[int, int]
    left_nnz: int
    right_nnz: int
    output_nnz: int
    estimated_flops: int
    achieved_gflops: float
    memory_bandwidth_gb_s: float
    sparsity_factor: float
    efficiency_vs_peak_pct: float


@dataclass
class BatchBenchmarkResult:
    """Results from a single batch size configuration test."""

    dataset_size: int
    batch_size: int
    total_time_s: float
    throughput_pairs_per_sec: float  # Pairs compared per second (not matching pairs)
    peak_memory_gb: float
    avg_gpu_util_pct: float
    time_per_batch_ms: float
    pairs_compared: int  # Total number of spectrum pairs compared
    pairs_found: int  # Number of matching pairs above threshold


@dataclass
class DTypeComparison:
    """Comparison between float32 and float64 performance."""

    float32_time_s: float
    float32_memory_gb: float
    float32_avg_similarity: float
    float64_time_s: float
    float64_memory_gb: float
    float64_avg_similarity: float
    speedup_factor: float
    memory_reduction_pct: float
    accuracy_difference: float


@dataclass
class OptimizationSession:
    """Aggregates all optimization data for a complete run."""

    # Session metadata
    start_time: float = field(default_factory=perf_counter)
    gpu_name: str = ""
    compute_capability: str = ""
    total_gpu_memory_gb: float = 0.0
    peak_gpu_gflops: float = 0.0

    # Configuration
    n_spectra: int = 0
    n_peaks_per_spectrum: int = 0
    approx_threshold: float = 0.65
    ms2_tolerance_ppm: float = 10.0
    match_rate: float = 0.001

    # Profiling results
    operation_profiles: dict[str, OperationProfile] = field(default_factory=dict)
    internal_kernel_timings: Optional[InternalKernelTimings] = None
    spmm_metrics: Optional[SpMMMetrics] = None
    batch_benchmark_results: list[BatchBenchmarkResult] = field(default_factory=list)
    dtype_comparison: Optional[DTypeComparison] = None

    # Analysis results
    bottlenecks: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    optimal_batch_size: Optional[int] = None
    optimal_throughput: Optional[float] = None

    def finalize(self) -> None:
        """Compute final statistics."""
        self.total_time = perf_counter() - self.start_time


# =============================================================================
# GPU Info and Hardware Metrics
# =============================================================================


def get_gpu_info() -> dict[str, Any]:
    """
    Gather GPU hardware information using nvidia-smi.

    Returns:
        Dictionary with gpu_name, memory_gb, compute_capability, peak_gflops
    """
    info = {
        "gpu_name": "Unknown",
        "memory_gb": 0.0,
        "compute_capability": "0.0",
        "peak_gflops": 0.0,
    }

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,compute_cap",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            if len(parts) >= 3:
                info["gpu_name"] = parts[0]
                mem_str = parts[1].replace(" MiB", "")
                info["memory_gb"] = float(mem_str) / 1024.0
                info["compute_capability"] = parts[2]

                # Estimate peak GFLOPS based on known GPUs
                # Why: Used to compute SpMM efficiency (achieved vs peak)
                gpu_name = info["gpu_name"].lower()
                if "a100" in gpu_name:
                    info["peak_gflops"] = 19500.0  # 19.5 TFLOPS FP32
                elif "h100" in gpu_name:
                    info["peak_gflops"] = 51000.0  # 51 TFLOPS FP32
                elif "4090" in gpu_name or "rtx 4090" in gpu_name:
                    info["peak_gflops"] = 82600.0  # 82.6 TFLOPS FP32
                elif "3090" in gpu_name or "rtx 3090" in gpu_name:
                    info["peak_gflops"] = 35600.0  # 35.6 TFLOPS FP32
                elif "v100" in gpu_name:
                    info["peak_gflops"] = 15700.0  # 15.7 TFLOPS FP32
                else:
                    # Fallback estimate based on compute capability
                    cc_major = int(float(info["compute_capability"]))
                    if cc_major >= 8:  # Ampere or newer
                        info["peak_gflops"] = 20000.0
                    elif cc_major >= 7:  # Turing/Volta
                        info["peak_gflops"] = 15000.0
                    else:
                        info["peak_gflops"] = 10000.0

    except Exception:
        pass

    # Fallback to CuPy if nvidia-smi fails
    if info["gpu_name"] == "Unknown":
        device = cp.cuda.Device(0)
        _, total_mem = device.mem_info
        info["memory_gb"] = total_mem / (1024**3)
        info["gpu_name"] = f"GPU Device {device.id}"

    return info


# =============================================================================
# Synthetic Spectrum Generation
# =============================================================================


def generate_synthetic_spectrum(
    n_peaks: int = 100,
    mz_range: tuple[float, float] = (100.0, 1000.0),
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.float32]]:
    """
    Generate a single template spectrum with realistic characteristics.

    Why: Provides a controlled baseline for generating batch variations.
    Uses log-normal intensity distribution typical of MS data.

    Args:
        n_peaks: Number of peaks in spectrum (default: 100)
        mz_range: (min_mz, max_mz) in Da (default: 100-1000)
        seed: Random seed for reproducibility

    Returns:
        (mz_array, intensity_array) - template spectrum
            mz_array: shape (n_peaks,) in Da
            intensity_array: shape (n_peaks,) normalized to [0, 1]
    """
    rng = np.random.RandomState(seed)

    # Generate m/z values with small random jitter
    # Why: Evenly distributed but not perfectly uniform (more realistic)
    mz_min, mz_max = mz_range
    base_mz = np.linspace(mz_min, mz_max, n_peaks)
    jitter = rng.uniform(-1.0, 1.0, n_peaks)
    mz_array = base_mz + jitter
    mz_array = np.sort(mz_array).astype(np.float64)

    # Generate intensities with log-normal distribution
    # Why: MS data intensities are typically log-normally distributed
    log_intensities = rng.normal(loc=5.0, scale=1.5, size=n_peaks)
    intensities = np.exp(log_intensities)

    # Normalize to [0, 1] range
    intensities = intensities / np.max(intensities)
    intensity_array = intensities.astype(np.float32)

    return mz_array, intensity_array


def generate_batch_with_mass_shifts(
    template_mz: NDArray[np.float64],
    template_intensity: NDArray[np.float32],
    n_spectra: int,
    mass_shift_range_da: float = 0.01,
    intensity_noise_pct: float = 0.05,
    seed: Optional[int] = None,
    match_rate: float = 0.001,
) -> pl.DataFrame:
    """
    Create batch of spectra with realistic diversity (mostly non-matching).

    Why: Realistic profiling requires that most spectra don't match each other,
    simulating real-world scenarios where only a small fraction of spectrum pairs
    are similar. This ensures the GPU kernel is tested under realistic sparsity
    conditions.

    Strategy:
    - Matching spectra: Small variations of template (will score high similarity)
    - Non-matching spectra: Different fragmentation patterns within same m/z range
      (some overlapping peaks by chance, but different overall patterns)

    Args:
        template_mz: Template m/z values, shape (n_peaks,)
        template_intensity: Template intensities, shape (n_peaks,)
        n_spectra: Number of spectra to generate
        mass_shift_range_da: ± shift range in Da for matching spectra (default: ±0.01 Da)
        intensity_noise_pct: Intensity perturbation % for matching spectra (default: 5%)
        seed: Random seed for reproducibility
        match_rate: Desired fraction of PAIRS that should match (default: 0.001 = 0.1% of pairs)
            Note: This is converted to spectral match rate using sqrt(match_rate).
            Example: match_rate=0.01 → ~10% of spectra match → ~1% of pairs match

    Returns:
        DataFrame with columns: [idx, mz, intensity]
            - idx: Int32, spectrum index (0 to n_spectra-1)
            - mz: List[Float64], m/z values
            - intensity: List[Float32], intensity values
    """
    rng = np.random.RandomState(seed)

    n_peaks = len(template_mz)
    mz_min, mz_max = template_mz.min(), template_mz.max()
    mz_range = (mz_min, mz_max)

    idx_list = []
    mz_list = []
    intensity_list = []

    # Determine which spectra should match the template
    # Why: match_rate is the desired fraction of PAIRS that match, not spectra.
    # For self-comparison: total_pairs = n_spectra * (n_spectra - 1) / 2
    # Matching pairs from n_matching spectra = n_matching * (n_matching - 1) / 2
    # Solve: n_matching * (n_matching - 1) / 2 = match_rate * n_spectra * (n_spectra - 1) / 2
    # Simplifies to: n_matching * (n_matching - 1) ≈ match_rate * n_spectra * (n_spectra - 1)
    # For small match_rate: n_matching ≈ sqrt(match_rate) * n_spectra
    n_matching = max(1, int(np.sqrt(match_rate) * n_spectra))
    
    # Cap at n_spectra and ensure at least 2 for meaningful matching
    n_matching = min(n_matching, n_spectra)
    n_matching = max(2, n_matching)
    
    matching_indices = set(rng.choice(n_spectra, size=n_matching, replace=False))
    
    # Verify expected match count
    expected_match_pairs = n_matching * (n_matching - 1) // 2
    total_pairs = n_spectra * (n_spectra - 1) // 2
    actual_match_rate = expected_match_pairs / total_pairs if total_pairs > 0 else 0

    # Create diverse "cluster centers" for non-matching spectra
    # Why: Real datasets have diversity but aren't completely random
    n_clusters = max(10, n_spectra // 100)
    cluster_templates = []
    for _ in range(n_clusters):
        # Each cluster has peaks distributed across the m/z range
        # but with different positions and patterns
        cluster_mz = np.sort(rng.uniform(mz_min, mz_max, size=n_peaks))
        cluster_intensity = np.exp(rng.normal(loc=5.0, scale=1.5, size=n_peaks))
        cluster_intensity = cluster_intensity / np.max(cluster_intensity)
        cluster_templates.append((cluster_mz, cluster_intensity.astype(np.float32)))

    for i in range(n_spectra):
        if i in matching_indices:
            # Generate a matching spectrum (small variations of template)
            # Why: These should score high similarity with each other
            mass_shifts = rng.uniform(
                -mass_shift_range_da, mass_shift_range_da, size=n_peaks
            )
            shifted_mz = template_mz + mass_shifts

            noise_factors = 1.0 + rng.uniform(
                -intensity_noise_pct, intensity_noise_pct, size=n_peaks
            )
            noisy_intensity = template_intensity * noise_factors

            # Ensure positive intensities
            noisy_intensity = np.maximum(noisy_intensity, 1e-6)
            noisy_intensity = noisy_intensity / np.max(noisy_intensity)

        else:
            # Generate a non-matching spectrum from a random cluster
            # Why: Creates diversity while maintaining realistic structure
            # Spectra from different clusters will have low similarity

            cluster_idx = rng.randint(0, n_clusters)
            cluster_mz, cluster_intensity = cluster_templates[cluster_idx]

            # Apply larger mass shifts (simulates different compounds/fragments)
            # Why: Enough to prevent matching but not so large that we lose all overlap
            mass_shifts = rng.uniform(-5.0, 5.0, size=n_peaks)
            shifted_mz = cluster_mz + mass_shifts

            # Apply significant intensity variation (different fragmentation pattern)
            # Why: Same m/z region but very different fragmentation
            # Randomly drop 30-50% of peaks to 0 (not all peaks present)
            dropout_mask = rng.random(size=n_peaks) > 0.4
            noisy_intensity = cluster_intensity.copy()
            noisy_intensity[~dropout_mask] = 0.0

            # For remaining peaks, add substantial intensity noise
            remaining_peaks = dropout_mask.sum()
            if remaining_peaks > 0:
                noise_factors = rng.uniform(0.2, 2.0, size=n_peaks)
                noisy_intensity = noisy_intensity * noise_factors
                noisy_intensity = noisy_intensity / (np.max(noisy_intensity) + 1e-10)
            else:
                # Fallback: ensure at least some peaks
                noisy_intensity = rng.uniform(0.1, 1.0, size=n_peaks)
                noisy_intensity = noisy_intensity / np.max(noisy_intensity)

        idx_list.append(i)
        mz_list.append(shifted_mz.tolist())
        intensity_list.append(noisy_intensity.tolist())

    df = pl.DataFrame(
        {
            "idx": pl.Series(idx_list, dtype=pl.Int32),
            "mz": pl.Series(mz_list, dtype=pl.List(pl.Float64)),
            "intensity": pl.Series(intensity_list, dtype=pl.List(pl.Float32)),
        }
    )

    return df


# =============================================================================
# GPU Monitoring Thread
# =============================================================================


class GPUMonitor:
    """
    Background thread for monitoring GPU utilization via nvidia-smi.

    Why: Standard profiling tools don't capture aggregate utilization patterns.
    This samples nvidia-smi metrics during computation to detect idle time
    and low utilization periods.
    """

    def __init__(self, sample_interval_s: float = 0.1):
        self.sample_interval = sample_interval_s
        self.samples: list[dict[str, float]] = []
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start background monitoring thread."""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Stop monitoring and join thread."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)

    def _monitor_loop(self) -> None:
        """Background loop that samples GPU metrics."""
        while self.running:
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,utilization.memory,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=1.0,
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(", ")
                    if len(parts) >= 3:
                        self.samples.append(
                            {
                                "gpu_util_pct": float(parts[0]),
                                "mem_util_pct": float(parts[1]),
                                "mem_used_mb": float(parts[2]),
                            }
                        )
            except Exception:
                pass
            time.sleep(self.sample_interval)

    def get_avg_utilization(self) -> float:
        """Get average GPU utilization percentage."""
        if not self.samples:
            return 0.0
        return np.mean([s["gpu_util_pct"] for s in self.samples])

    def clear(self) -> None:
        """Clear accumulated samples."""
        self.samples.clear()


# =============================================================================
# Kernel Operation Profiling
# =============================================================================


def profile_single_operation(
    operation_name: str,
    operation_func: callable,
    memory_pool: cp.cuda.MemoryPool,
) -> OperationProfile:
    """
    Profile a single GPU operation with precise timing and memory tracking.

    Why: CUDA events provide microsecond-precision timing that's more
    accurate than CPU-side perf_counter for GPU operations.

    Args:
        operation_name: Human-readable operation name
        operation_func: Callable that performs the operation (no args)
        memory_pool: CuPy memory pool for tracking allocations

    Returns:
        OperationProfile with timing and memory data
    """
    # Capture memory before
    mem_before = memory_pool.used_bytes()

    # Create CUDA events for precise timing
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()

    # Record start
    start_event.record()

    # Execute operation
    operation_func()

    # Record end and synchronize
    end_event.record()
    end_event.synchronize()

    # Capture memory after
    mem_after = memory_pool.used_bytes()

    # Compute elapsed time
    duration_ms = cp.cuda.get_elapsed_time(start_event, end_event)

    return OperationProfile(
        name=operation_name,
        duration_ms=duration_ms,
        memory_before_bytes=mem_before,
        memory_after_bytes=mem_after,
        memory_delta_bytes=mem_after - mem_before,
    )


def profile_internal_kernel_operations(
    df: pl.DataFrame, config: GPUApproximateConfig, use_fused: bool = False
) -> InternalKernelTimings:
    """
    Profile internal kernel operations with detailed timing.

    Why: To identify whether SpMM dominates or if other operations
    (conversion, expansion, thresholding) are bottlenecks.

    This function manually executes one batch iteration and times each step.

    Args:
        df: Input DataFrame with spectra
        config: Configuration for similarity computation
        use_fused: If True, use fused normalize-expand kernel

    Returns:
        InternalKernelTimings with detailed operation timings
    """
    if use_fused:
        return _profile_single_batch_fused(df, config)
    else:
        return _profile_single_batch(df, config)


def _profile_single_batch(
    df: pl.DataFrame, config: GPUApproximateConfig
) -> InternalKernelTimings:
    """
    Profile internal kernel operations with detailed timing (OLD implementation).

    Why: To identify whether SpMM dominates or if other operations
    (conversion, expansion, thresholding) are bottlenecks.

    This function manually executes one batch iteration and times each step.

    Args:
        df: Input DataFrame with spectra
        config: Configuration for similarity computation

    Returns:
        InternalKernelTimings with detailed operation timings
    """
    from fast_cosine_sim.gpu_approximate_similarity import (
        APPROX_INTENSITY_DTYPE_NP,
        INDEX_DTYPE_NP,
        _expand_csr_horizontal_adaptive_gpu,
        _normalize_and_expand_csr_gpu,
        _normalize_csr_rows_inplace_gpu,
        _sparse_bin_spectra_df_to_csr,
    )

    # Add indices if needed
    if config.spectrum_id_col not in df.columns:
        df = df.with_row_index(config.spectrum_id_col).with_columns(
            pl.col(config.spectrum_id_col).cast(pl.Int32)
        )
    else:
        df = df.with_columns(pl.col(config.spectrum_id_col).cast(pl.Int32))

    # Convert to CSR (CPU operation, not timed here)
    csr_matrix = _sparse_bin_spectra_df_to_csr(
        df,
        config.mz_col,
        config.intensity_col,
        upper_bound=config.upper_mass_bound,
        intensity_power=config.intensity_power,
        bin_size=config.bin_size,
        apply_centroiding=config.centroiding_enabled,
        tolerance_ppm=config.ms2_tolerance_ppm,
        mass_tolerance_cutoff_mz=config.mass_tolerance_cutoff_mz,
    )

    # Take a representative batch (use first N rows for single batch test)
    batch_size = min(1000, csr_matrix.shape[0])
    left_csr = csr_matrix[:batch_size]
    right_csr = csr_matrix[:batch_size]

    # Create timing events
    events = {}
    for name in [
        "start",
        "left_xfer",
        "norm_left",
        "right_xfer",
        "norm_right",
        "expand",
        "spmm_start",
        "spmm_end",
        "thresh_start",
        "thresh_end",
        "cpu_xfer",
        "end",
    ]:
        events[name] = cp.cuda.Event()

    # Start profiling
    events["start"].record()

    # 1. Transfer LEFT to GPU
    l_data_gpu = cp.asarray(
        np.asarray(left_csr.data).astype(APPROX_INTENSITY_DTYPE_NP, copy=False)
    )
    l_indices_gpu = cp.asarray(np.asarray(left_csr.indices))
    l_indptr_gpu = cp.asarray(np.asarray(left_csr.indptr))
    L_gpu = cps.csr_matrix(
        (l_data_gpu, l_indices_gpu, l_indptr_gpu), shape=left_csr.shape
    )
    del l_data_gpu, l_indices_gpu, l_indptr_gpu
    events["left_xfer"].record()

    # 2. Normalize LEFT
    _ = _normalize_csr_rows_inplace_gpu(L_gpu)
    events["norm_left"].record()

    # 3. Transfer RIGHT to GPU
    r_data_gpu = cp.asarray(
        np.asarray(right_csr.data).astype(APPROX_INTENSITY_DTYPE_NP, copy=False)
    )
    r_indices_gpu = cp.asarray(np.asarray(right_csr.indices))
    r_indptr_gpu = cp.asarray(np.asarray(right_csr.indptr))
    R_gpu = cps.csr_matrix(
        (r_data_gpu, r_indices_gpu, r_indptr_gpu), shape=right_csr.shape
    )
    del r_data_gpu, r_indices_gpu, r_indptr_gpu
    events["right_xfer"].record()

    # 4. Normalize RIGHT
    _ = _normalize_csr_rows_inplace_gpu(R_gpu)
    events["norm_right"].record()

    # 5. Expand RIGHT (if tolerance configured)
    if config.ms2_tolerance_ppm > 0:
        nbins = int(config.upper_mass_bound / config.bin_size)
        R_gpu = _expand_csr_horizontal_adaptive_gpu(
            R_gpu,
            config.bin_size,
            config.ms2_tolerance_ppm,
            nbins,
        )
    events["expand"].record()

    # 6. SpMM: L @ R.T
    events["spmm_start"].record()
    sim = L_gpu.dot(R_gpu.T)
    events["spmm_end"].record()

    # 7. Thresholding & extraction
    events["thresh_start"].record()
    mask = sim.data >= config.approx_threshold

    out_rows = None
    out_cols = None
    out_data = None
    if int(mask.sum()) > 0:
        out_data = sim.data[mask]
        out_cols = sim.indices[mask]
        indices_in_data = cp.nonzero(mask)[0]
        out_rows = cp.searchsorted(sim.indptr, indices_in_data, side="right") - 1
    events["thresh_end"].record()

    # 8. Transfer results to CPU
    if out_rows is not None and out_cols is not None and out_data is not None:
        _ = cp.asnumpy(out_rows)
        _ = cp.asnumpy(out_cols)
        _ = cp.asnumpy(out_data)
    events["cpu_xfer"].record()

    events["end"].record()

    # Synchronize all events
    events["end"].synchronize()

    # Calculate timings
    transfer_to_gpu_ms = cp.cuda.get_elapsed_time(
        events["start"], events["left_xfer"]
    ) + cp.cuda.get_elapsed_time(events["norm_left"], events["right_xfer"])

    normalize_left_ms = cp.cuda.get_elapsed_time(
        events["left_xfer"], events["norm_left"]
    )
    normalize_right_ms = cp.cuda.get_elapsed_time(
        events["right_xfer"], events["norm_right"]
    )

    expand_right_ms = cp.cuda.get_elapsed_time(events["norm_right"], events["expand"])

    spmm_ms = cp.cuda.get_elapsed_time(events["spmm_start"], events["spmm_end"])

    threshold_and_extract_ms = cp.cuda.get_elapsed_time(
        events["thresh_start"], events["thresh_end"]
    )

    transfer_to_cpu_ms = cp.cuda.get_elapsed_time(
        events["thresh_end"], events["cpu_xfer"]
    )

    total_batch_ms = cp.cuda.get_elapsed_time(events["start"], events["end"])

    return InternalKernelTimings(
        transfer_to_gpu_ms=transfer_to_gpu_ms,
        normalize_left_ms=normalize_left_ms,
        normalize_right_ms=normalize_right_ms,
        expand_right_ms=expand_right_ms,
        spmm_ms=spmm_ms,
        threshold_and_extract_ms=threshold_and_extract_ms,
        transfer_to_cpu_ms=transfer_to_cpu_ms,
        total_batch_ms=total_batch_ms,
    )


def _profile_single_batch_fused(
    df: pl.DataFrame, config: GPUApproximateConfig
) -> InternalKernelTimings:
    """
    Profile internal kernel operations using FUSED normalize-expand kernel.

    Why: Test performance of new fused Numba CUDA kernel that combines
    normalization + expansion to eliminate intermediate allocations and
    reduce memory traffic. Expected speedup: 3-5x on expansion phase.

    This function is identical to _profile_single_batch but uses
    _normalize_and_expand_csr_gpu instead of separate normalize + expand.

    Args:
        df: Input DataFrame with spectra
        config: Configuration for similarity computation

    Returns:
        InternalKernelTimings with detailed operation timings
    """
    from fast_cosine_sim.gpu_approximate_similarity import (
        APPROX_INTENSITY_DTYPE_NP,
        INDEX_DTYPE_NP,
        _normalize_and_expand_csr_gpu,
        _normalize_csr_rows_inplace_gpu,
        _sparse_bin_spectra_df_to_csr,
    )

    # Add indices if needed
    if config.spectrum_id_col not in df.columns:
        df = df.with_row_index(config.spectrum_id_col).with_columns(
            pl.col(config.spectrum_id_col).cast(pl.Int32)
        )
    else:
        df = df.with_columns(pl.col(config.spectrum_id_col).cast(pl.Int32))

    # Convert to CSR (CPU operation, not timed here)
    csr_matrix = _sparse_bin_spectra_df_to_csr(
        df,
        config.mz_col,
        config.intensity_col,
        upper_bound=config.upper_mass_bound,
        intensity_power=config.intensity_power,
        bin_size=config.bin_size,
        apply_centroiding=config.centroiding_enabled,
        tolerance_ppm=config.ms2_tolerance_ppm,
        mass_tolerance_cutoff_mz=config.mass_tolerance_cutoff_mz,
    )

    # Take a representative batch (use first N rows for single batch test)
    batch_size = min(1000, csr_matrix.shape[0])
    left_csr = csr_matrix[:batch_size]
    right_csr = csr_matrix[:batch_size]

    # Create timing events (fused version combines norm_right + expand)
    events = {}
    for name in [
        "start",
        "left_xfer",
        "norm_left",
        "right_xfer",
        "fused_norm_expand",
        "spmm_start",
        "spmm_end",
        "thresh_start",
        "thresh_end",
        "cpu_xfer",
        "end",
    ]:
        events[name] = cp.cuda.Event()

    # Start profiling
    events["start"].record()

    # 1. Transfer LEFT to GPU
    l_data_gpu = cp.asarray(
        np.asarray(left_csr.data).astype(APPROX_INTENSITY_DTYPE_NP, copy=False)
    )
    l_indices_gpu = cp.asarray(np.asarray(left_csr.indices))
    l_indptr_gpu = cp.asarray(np.asarray(left_csr.indptr))
    L_gpu = cps.csr_matrix(
        (l_data_gpu, l_indices_gpu, l_indptr_gpu), shape=left_csr.shape
    )
    del l_data_gpu, l_indices_gpu, l_indptr_gpu
    events["left_xfer"].record()

    # 2. Normalize LEFT
    _ = _normalize_csr_rows_inplace_gpu(L_gpu)
    events["norm_left"].record()

    # 3. Transfer RIGHT to GPU (unnormalized)
    r_data_gpu = cp.asarray(
        np.asarray(right_csr.data).astype(APPROX_INTENSITY_DTYPE_NP, copy=False)
    )
    r_indices_gpu = cp.asarray(np.asarray(right_csr.indices))
    r_indptr_gpu = cp.asarray(np.asarray(right_csr.indptr))
    R_gpu = cps.csr_matrix(
        (r_data_gpu, r_indices_gpu, r_indptr_gpu), shape=right_csr.shape
    )
    del r_data_gpu, r_indices_gpu, r_indptr_gpu
    events["right_xfer"].record()

    # 4. FUSED Normalize + Expand RIGHT (if tolerance configured)
    if config.ms2_tolerance_ppm > 0:
        nbins = int(config.upper_mass_bound / config.bin_size)
        R_gpu = _normalize_and_expand_csr_gpu(
            R_gpu,
            config.bin_size,
            config.ms2_tolerance_ppm,
            nbins,
        )
    else:
        # No expansion, just normalize
        _ = _normalize_csr_rows_inplace_gpu(R_gpu)
    events["fused_norm_expand"].record()

    # 5. SpMM: L @ R.T
    events["spmm_start"].record()
    sim = L_gpu.dot(R_gpu.T)
    events["spmm_end"].record()

    # 6. Thresholding & extraction
    events["thresh_start"].record()
    mask = sim.data >= config.approx_threshold

    out_rows = None
    out_cols = None
    out_data = None
    if int(mask.sum()) > 0:
        out_data = sim.data[mask]
        out_cols = sim.indices[mask]
        indices_in_data = cp.nonzero(mask)[0]
        out_rows = cp.searchsorted(sim.indptr, indices_in_data, side="right") - 1
    events["thresh_end"].record()

    # 7. Transfer results to CPU
    if out_rows is not None and out_cols is not None and out_data is not None:
        _ = cp.asnumpy(out_rows)
        _ = cp.asnumpy(out_cols)
        _ = cp.asnumpy(out_data)
    events["cpu_xfer"].record()

    events["end"].record()

    # Synchronize all events
    events["end"].synchronize()

    # Calculate timings
    transfer_to_gpu_ms = cp.cuda.get_elapsed_time(
        events["start"], events["left_xfer"]
    ) + cp.cuda.get_elapsed_time(events["norm_left"], events["right_xfer"])

    normalize_left_ms = cp.cuda.get_elapsed_time(
        events["left_xfer"], events["norm_left"]
    )

    # Fused operation time (replaces separate normalize_right + expand_right)
    fused_norm_expand_ms = cp.cuda.get_elapsed_time(
        events["right_xfer"], events["fused_norm_expand"]
    )

    spmm_ms = cp.cuda.get_elapsed_time(events["spmm_start"], events["spmm_end"])

    threshold_and_extract_ms = cp.cuda.get_elapsed_time(
        events["thresh_start"], events["thresh_end"]
    )

    transfer_to_cpu_ms = cp.cuda.get_elapsed_time(
        events["thresh_end"], events["cpu_xfer"]
    )

    total_batch_ms = cp.cuda.get_elapsed_time(events["start"], events["end"])

    # Return with fused time split across normalize_right and expand_right for comparison
    # (for reporting purposes, we'll attribute it all to expand_right and set normalize_right to 0)
    return InternalKernelTimings(
        transfer_to_gpu_ms=transfer_to_gpu_ms,
        normalize_left_ms=normalize_left_ms,
        normalize_right_ms=0.0,  # Fused with expand
        expand_right_ms=fused_norm_expand_ms,  # Combined time
        spmm_ms=spmm_ms,
        threshold_and_extract_ms=threshold_and_extract_ms,
        transfer_to_cpu_ms=transfer_to_cpu_ms,
        total_batch_ms=total_batch_ms,
    )


def run_profiled_similarity_detailed(
    df: pl.DataFrame, config: GPUApproximateConfig, gpu_monitor: GPUMonitor
) -> tuple[pl.DataFrame, dict[str, OperationProfile]]:
    """
    Run similarity computation with detailed operation profiling.

    Why: Standard batched_approximate_similarity_gpu hides internal operation
    timing. This instruments the key operations to identify bottlenecks.

    Note: This is a simplified profiling wrapper. For production, we'd need
    to instrument the actual library code or use CuPy's profiler context.

    Args:
        df: Input DataFrame with spectra
        config: Configuration for similarity computation
        gpu_monitor: GPU monitoring thread

    Returns:
        (result_df, operation_profiles)
            result_df: Similarity pairs DataFrame
            operation_profiles: Dict mapping operation name to profile
    """
    memory_pool = cp.get_default_memory_pool()
    profiles = {}

    # Start GPU monitoring
    gpu_monitor.start()

    # Profile the full operation
    # Note: This is a high-level profile. For detailed operation breakdown,
    # we'd need to modify the library code or use cupyx.profiler
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()

    mem_before = memory_pool.used_bytes()
    start_event.record()

    # Run computation
    result = batched_approximate_similarity_gpu(
        left_df=df,
        config=config,
        right_df=None,
        output_path=None,
        logger=None,
    )

    # Synchronize
    end_event.record()
    end_event.synchronize()

    mem_after = memory_pool.used_bytes()
    duration_ms = cp.cuda.get_elapsed_time(start_event, end_event)

    # Stop monitoring
    gpu_monitor.stop()

    # Create profile for full computation
    profiles["full_computation"] = OperationProfile(
        name="full_computation",
        duration_ms=duration_ms,
        memory_before_bytes=mem_before,
        memory_after_bytes=mem_after,
        memory_delta_bytes=mem_after - mem_before,
    )

    # Collect result
    if isinstance(result, pl.LazyFrame):
        result = result.collect()

    return result, profiles


# =============================================================================
# SpMM Analysis
# =============================================================================


def estimate_spmm_metrics(
    operation_profile: OperationProfile,
    n_spectra: int,
    avg_peaks: int,
    expansion_factor: float,
    peak_gpu_gflops: float,
) -> SpMMMetrics:
    """
    Estimate SpMM (sparse matrix multiply) performance metrics.

    Why: SpMM is the dominant operation in approximate similarity. Understanding
    its efficiency helps determine if we're compute-bound or memory-bound.

    Args:
        operation_profile: Profile of the full computation
        n_spectra: Number of spectra processed
        avg_peaks: Average peaks per spectrum
        expansion_factor: Expansion from tolerance window (e.g., 2.0 = 2x density)
        peak_gpu_gflops: Peak GPU GFLOPS from hardware specs

    Returns:
        SpMMMetrics with performance analysis
    """
    # Estimate matrix dimensions
    left_shape = (n_spectra, 10_000_000)  # Approximate bin count
    right_shape = left_shape

    # Estimate non-zero counts
    left_nnz = n_spectra * avg_peaks
    right_nnz_expanded = int(left_nnz * expansion_factor)

    # Estimate FLOPs for sparse dot product
    # For each left row, we do dot product with each right row
    # FLOP count ≈ 2 * (avg_nnz_left * avg_nnz_right_per_row)
    avg_nnz_per_row = avg_peaks
    avg_nnz_per_row_expanded = int(avg_peaks * expansion_factor)
    flops_per_dot = 2 * avg_nnz_per_row * avg_nnz_per_row_expanded
    total_flops = flops_per_dot * n_spectra * n_spectra

    # Compute achieved GFLOPS
    # Note: This is a rough estimate since we're profiling the full pipeline
    duration_s = operation_profile.duration_ms / 1000.0
    achieved_gflops = (total_flops / duration_s) / 1e9

    # Estimate memory bandwidth
    # Data transferred: matrix elements (4 bytes each) * nnz
    data_volume_bytes = (left_nnz + right_nnz_expanded) * 4  # float32
    bandwidth_gb_s = (data_volume_bytes / duration_s) / 1e9

    # Compute efficiency
    efficiency_pct = (achieved_gflops / peak_gpu_gflops) * 100.0

    # Sparsity factor
    total_elements = left_shape[0] * left_shape[1]
    sparsity_factor = 1.0 - (left_nnz / total_elements) if total_elements > 0 else 1.0

    return SpMMMetrics(
        duration_ms=operation_profile.duration_ms,
        left_shape=left_shape,
        right_shape=right_shape,
        left_nnz=left_nnz,
        right_nnz=right_nnz_expanded,
        output_nnz=0,  # Unknown without running
        estimated_flops=total_flops,
        achieved_gflops=achieved_gflops,
        memory_bandwidth_gb_s=bandwidth_gb_s,
        sparsity_factor=sparsity_factor,
        efficiency_vs_peak_pct=efficiency_pct,
    )


# =============================================================================
# Batch Size Benchmarking
# =============================================================================


def benchmark_batch_sizes(
    template_spectrum: tuple[NDArray[np.float64], NDArray[np.float32]],
    dataset_size: int,
    batch_size_configs: list[int],
    config_base: GPUApproximateConfig,
    gpu_monitor: GPUMonitor,
    match_rate: float = 0.001,
) -> list[BatchBenchmarkResult]:
    """
    Benchmark different batch size configurations.

    Why: Find optimal batch size that maximizes throughput while staying
    within memory constraints.

    Args:
        template_spectrum: (mz, intensity) template for generation
        dataset_size: Number of spectra to generate
        batch_size_configs: List of batch sizes to test (in spectra)
        config_base: Base configuration (will modify max_peaks_per_batch)
        gpu_monitor: GPU monitoring thread
        match_rate: Fraction of spectra that should match (default: 0.001)

    Returns:
        List of BatchBenchmarkResult for each configuration
    """
    results = []
    template_mz, template_intensity = template_spectrum

    print(f"\n{'=' * 80}")
    print(f"Benchmarking Batch Sizes: {len(batch_size_configs)} configurations")
    print(f"Dataset Size: {dataset_size:,} spectra")
    print(f"{'=' * 80}\n")

    for i, batch_size in enumerate(batch_size_configs, 1):
        print(
            f"[{i}/{len(batch_size_configs)}] Testing batch_size={batch_size:,} spectra..."
        )

        # Generate synthetic dataset
        df = generate_batch_with_mass_shifts(
            template_mz,
            template_intensity,
            n_spectra=dataset_size,
            seed=42 + i,  # Different seed for each config
            match_rate=match_rate,
        )

        # Configure batch size via max_peaks_per_batch
        avg_peaks = len(template_mz)
        max_peaks = batch_size * avg_peaks

        config_test = GPUApproximateConfig(
            upper_mass_bound=config_base.upper_mass_bound,
            bin_size=config_base.bin_size,
            approx_threshold=config_base.approx_threshold,
            ms2_tolerance_ppm=config_base.ms2_tolerance_ppm,
            intensity_power=config_base.intensity_power,
            target_gpu_mem_ratio=config_base.target_gpu_mem_ratio,
            max_peaks_per_batch=max_peaks,
            comparison_mode="self",
            spectrum_id_col="idx",
            mz_col="mz",
            intensity_col="intensity",
            use_fused_kernel=config_base.use_fused_kernel,
        )

        # Clear GPU memory before test
        cp.get_default_memory_pool().free_all_blocks()
        gpu_monitor.clear()

        # Run benchmark
        t_start = perf_counter()
        mem_pool = cp.get_default_memory_pool()
        mem_before = mem_pool.used_bytes()

        result, _ = run_profiled_similarity_detailed(df, config_test, gpu_monitor)

        t_end = perf_counter()
        mem_after = mem_pool.used_bytes()
        peak_mem = mem_pool.total_bytes()

        total_time = t_end - t_start
        pairs_found = result.height

        # Calculate total pairs compared
        # Why: For self-comparison mode, we compare n*(n-1)/2 unique pairs
        pairs_compared = (dataset_size * (dataset_size - 1)) // 2
        throughput = pairs_compared / total_time if total_time > 0 else 0.0

        peak_memory_gb = peak_mem / (1024**3)
        avg_gpu_util = gpu_monitor.get_avg_utilization()

        # Estimate time per batch
        # Rough estimate: total_time / number_of_batches
        n_batches_estimate = max(1, dataset_size // batch_size)
        time_per_batch_ms = (total_time * 1000.0) / n_batches_estimate

        result_obj = BatchBenchmarkResult(
            dataset_size=dataset_size,
            batch_size=batch_size,
            total_time_s=total_time,
            throughput_pairs_per_sec=throughput,
            peak_memory_gb=peak_memory_gb,
            avg_gpu_util_pct=avg_gpu_util,
            time_per_batch_ms=time_per_batch_ms,
            pairs_compared=pairs_compared,
            pairs_found=pairs_found,
        )

        results.append(result_obj)

        print(
            f"  → Time: {total_time:.2f}s, Throughput: {throughput:,.0f} comparisons/s, "
            f"Memory: {peak_memory_gb:.2f} GB, Matches: {pairs_found:,}/{pairs_compared:,}, "
            f"GPU Util: {avg_gpu_util:.1f}%\n"
        )

        # Clean up
        del df, result
        cp.get_default_memory_pool().free_all_blocks()

    return results


# =============================================================================
# Data Type Comparison
# =============================================================================


def benchmark_dtypes(
    template_spectrum: tuple[NDArray[np.float64], NDArray[np.float32]],
    n_spectra: int,
    config_base: GPUApproximateConfig,
    gpu_monitor: GPUMonitor,
    match_rate: float = 0.001,
) -> DTypeComparison:
    """
    Compare float32 vs float64 performance.

    Why: Quantify speed/memory tradeoff for precision. float32 should be
    faster and use less memory, but we need empirical data.

    Args:
        template_spectrum: (mz, intensity) template
        n_spectra: Number of spectra to test
        config_base: Base configuration
        gpu_monitor: GPU monitoring thread
        match_rate: Fraction of spectra that should match (default: 0.001)

    Returns:
        DTypeComparison with metrics for both dtypes
    """
    template_mz, template_intensity = template_spectrum

    print(f"\n{'=' * 80}")
    print(f"Benchmarking Data Types: float32 vs float64")
    print(f"Dataset Size: {n_spectra:,} spectra")
    print(f"{'=' * 80}\n")

    results = {}

    for dtype_name, np_dtype in [("float32", np.float32), ("float64", np.float64)]:
        print(f"Testing {dtype_name}...")

        # Generate dataset
        df = generate_batch_with_mass_shifts(
            template_mz,
            template_intensity,
            n_spectra=n_spectra,
            seed=42,
            match_rate=match_rate,
        )

        # Configure with specific dtype
        config_test = GPUApproximateConfig(
            upper_mass_bound=config_base.upper_mass_bound,
            bin_size=config_base.bin_size,
            approx_threshold=config_base.approx_threshold,
            ms2_tolerance_ppm=config_base.ms2_tolerance_ppm,
            intensity_power=config_base.intensity_power,
            target_gpu_mem_ratio=config_base.target_gpu_mem_ratio,
            comparison_mode="self",
            spectrum_id_col="idx",
            mz_col="mz",
            intensity_col="intensity",
            csr_data_dtype=np.dtype(np_dtype),
            similarity_dtype=np.dtype(np_dtype),
            use_fused_kernel=config_base.use_fused_kernel,
        )

        # Clear memory
        cp.get_default_memory_pool().free_all_blocks()
        gpu_monitor.clear()

        # Run benchmark
        t_start = perf_counter()
        mem_pool = cp.get_default_memory_pool()

        result, _ = run_profiled_similarity_detailed(df, config_test, gpu_monitor)

        t_end = perf_counter()
        peak_mem = mem_pool.total_bytes()

        total_time = t_end - t_start
        peak_memory_gb = peak_mem / (1024**3)

        # Compute average similarity
        avg_similarity = 0.0
        if result.height > 0:
            avg_similarity = result["similarity"].mean()

        results[dtype_name] = {
            "time_s": total_time,
            "memory_gb": peak_memory_gb,
            "avg_similarity": avg_similarity,
        }

        print(
            f"  → Time: {total_time:.2f}s, Memory: {peak_memory_gb:.2f} GB, "
            f"Avg Similarity: {avg_similarity:.4f}\n"
        )

        # Clean up
        del df, result
        cp.get_default_memory_pool().free_all_blocks()

    # Compute comparison metrics
    speedup = results["float64"]["time_s"] / results["float32"]["time_s"]
    memory_reduction = (
        1.0 - results["float32"]["memory_gb"] / results["float64"]["memory_gb"]
    ) * 100.0
    accuracy_diff = abs(
        results["float32"]["avg_similarity"] - results["float64"]["avg_similarity"]
    )

    return DTypeComparison(
        float32_time_s=results["float32"]["time_s"],
        float32_memory_gb=results["float32"]["memory_gb"],
        float32_avg_similarity=results["float32"]["avg_similarity"],
        float64_time_s=results["float64"]["time_s"],
        float64_memory_gb=results["float64"]["memory_gb"],
        float64_avg_similarity=results["float64"]["avg_similarity"],
        speedup_factor=speedup,
        memory_reduction_pct=memory_reduction,
        accuracy_difference=accuracy_diff,
    )


# =============================================================================
# Visualization Functions
# =============================================================================


def plot_batch_optimization(
    results: list[BatchBenchmarkResult], output_path: Path
) -> None:
    """
    Plot batch size vs throughput and memory.

    Why: Visual identification of optimal batch size configuration.
    """
    batch_sizes = [r.batch_size for r in results]
    throughputs = [r.throughput_pairs_per_sec for r in results]
    memories = [r.peak_memory_gb for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Throughput plot
    ax1.plot(batch_sizes, throughputs, marker="o", linewidth=2, markersize=8)
    ax1.set_xlabel("Batch Size (spectra)", fontsize=12)
    ax1.set_ylabel("Throughput (comparisons/sec)", fontsize=12)
    ax1.set_title("Batch Size vs Throughput", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log")

    # Find and highlight optimal
    max_throughput_idx = np.argmax(throughputs)
    optimal_batch = batch_sizes[max_throughput_idx]
    optimal_throughput = throughputs[max_throughput_idx]
    ax1.axvline(optimal_batch, color="red", linestyle="--", alpha=0.7, label="Optimal")
    ax1.scatter(
        [optimal_batch], [optimal_throughput], color="red", s=200, zorder=5, marker="*"
    )
    ax1.legend()

    # Memory plot
    ax2.plot(
        batch_sizes, memories, marker="s", linewidth=2, markersize=8, color="green"
    )
    ax2.set_xlabel("Batch Size (spectra)", fontsize=12)
    ax2.set_ylabel("Peak Memory (GB)", fontsize=12)
    ax2.set_title("Batch Size vs Memory Usage", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale("log")
    ax2.axvline(optimal_batch, color="red", linestyle="--", alpha=0.7, label="Optimal")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_dtype_comparison(comparison: DTypeComparison, output_path: Path) -> None:
    """Plot float32 vs float64 comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Time comparison
    dtypes = ["float32", "float64"]
    times = [comparison.float32_time_s, comparison.float64_time_s]
    colors = ["#2ecc71", "#e74c3c"]

    ax1.bar(dtypes, times, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
    ax1.set_ylabel("Execution Time (seconds)", fontsize=12)
    ax1.set_title("Execution Time Comparison", fontsize=14, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    # Add speedup annotation
    speedup_text = f"Speedup: {comparison.speedup_factor:.2f}x"
    ax1.text(
        0.5,
        max(times) * 0.9,
        speedup_text,
        ha="center",
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.3),
    )

    # Memory comparison
    memories = [comparison.float32_memory_gb, comparison.float64_memory_gb]

    ax2.bar(dtypes, memories, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
    ax2.set_ylabel("Peak Memory (GB)", fontsize=12)
    ax2.set_title("Memory Usage Comparison", fontsize=14, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    # Add memory reduction annotation
    reduction_text = f"Reduction: {comparison.memory_reduction_pct:.1f}%"
    ax2.text(
        0.5,
        max(memories) * 0.9,
        reduction_text,
        ha="center",
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_internal_kernel_timings(
    timings: InternalKernelTimings, output_path: Path
) -> None:
    """
    Plot breakdown of internal kernel operation timings.

    Why: Visual identification of which operation is the bottleneck.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Operation names and timings
    operations = [
        ("GPU Transfer", timings.transfer_to_gpu_ms),
        ("Normalize Left", timings.normalize_left_ms),
        ("Normalize Right", timings.normalize_right_ms),
        ("Expand Right", timings.expand_right_ms),
        ("SpMM (L @ R.T)", timings.spmm_ms),
        ("Threshold & Extract", timings.threshold_and_extract_ms),
        ("CPU Transfer", timings.transfer_to_cpu_ms),
    ]

    op_names = [op[0] for op in operations]
    op_times = [op[1] for op in operations]

    # Calculate percentages
    op_percentages = [100.0 * t / timings.total_batch_ms for t in op_times]

    # Bar chart
    from matplotlib.colors import to_rgba

    colors_bar = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
    ]
    bars = ax1.barh(
        op_names,
        op_times,
        color=colors_bar[: len(op_names)],
        alpha=0.8,
        edgecolor="black",
        linewidth=1.2,
    )
    ax1.set_xlabel("Time (ms)", fontsize=12)
    ax1.set_title("Internal Kernel Operation Timings", fontsize=14, fontweight="bold")
    ax1.grid(axis="x", alpha=0.3)

    # Add value labels
    for bar, time_val in zip(bars, op_times):
        width = bar.get_width()
        ax1.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f" {time_val:.2f} ms",
            ha="left",
            va="center",
            fontsize=10,
        )

    # Pie chart with percentages
    colors_pie = colors_bar[: len(op_names)]
    wedges, texts, autotexts = ax2.pie(
        op_times,
        labels=op_names,
        autopct="%1.1f%%",
        colors=colors_pie,
        startangle=90,
        textprops={"fontsize": 10},
    )
    ax2.set_title("Time Distribution", fontsize=14, fontweight="bold")

    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# Report Generation
# =============================================================================


def generate_optimization_report(
    session: OptimizationSession, output_dir: Path
) -> None:
    """
    Generate comprehensive optimization report with recommendations.

    Why: Consolidate all profiling data into actionable insights.
    """
    session.finalize()

    report_path = output_dir / "optimization_report.md"
    json_path = output_dir / "profiling_data.json"

    # Build report sections
    lines = []

    def add_section(title: str, level: int = 1) -> None:
        prefix = "#" * level
        lines.append(f"\n{prefix} {title}\n")

    def add_line(text: str = "") -> None:
        lines.append(text)

    # Header
    add_line("# GPU Kernel Optimization Report")
    add_line(f"**Generated**: {datetime.now(timezone.utc).isoformat()}")
    add_line(f"**GPU**: {session.gpu_name}")
    add_line(f"**Memory**: {session.total_gpu_memory_gb:.2f} GB")
    add_line(f"**Compute Capability**: {session.compute_capability}")
    add_line(f"**Peak GFLOPS**: {session.peak_gpu_gflops:,.0f}")
    add_line()

    # Configuration
    add_section("Configuration", 2)
    add_line(f"- **Spectra**: {session.n_spectra:,}")
    add_line(f"- **Peaks per Spectrum**: {session.n_peaks_per_spectrum}")
    add_line(f"- **Approx Threshold**: {session.approx_threshold}")
    add_line(f"- **MS2 Tolerance**: {session.ms2_tolerance_ppm} ppm")
    add_line(
        f"- **Match Rate (Pairs)**: {session.match_rate:.4f} ({100*session.match_rate:.2f}% of pairs match)"
    )
    add_line()
    add_line(
        "**Note**: Match rate controls the fraction of spectrum PAIRS that match "
        "(e.g., 0.01 = 1% of pairs). This is achieved by making ~sqrt(match_rate) "
        "fraction of spectra similar to each other, creating sparse similarity matrices "
        "typical of production workloads."
    )
    add_line()

    # Batch Size Optimization Results
    if session.batch_benchmark_results:
        add_section("Batch Size Optimization", 2)

        # Find optimal
        optimal_result = max(
            session.batch_benchmark_results, key=lambda r: r.throughput_pairs_per_sec
        )
        session.optimal_batch_size = optimal_result.batch_size
        session.optimal_throughput = optimal_result.throughput_pairs_per_sec

        add_line(f"**Optimal Batch Size**: {optimal_result.batch_size:,} spectra")
        add_line(
            f"**Optimal Throughput**: {optimal_result.throughput_pairs_per_sec:,.0f} comparisons/sec"
        )
        add_line(f"**Pairs Compared**: {optimal_result.pairs_compared:,}")
        add_line(
            f"**Matches Found**: {optimal_result.pairs_found:,} ({100 * optimal_result.pairs_found / optimal_result.pairs_compared:.3f}%)"
        )
        add_line(f"**Memory at Optimal**: {optimal_result.peak_memory_gb:.2f} GB")
        add_line(f"**GPU Utilization**: {optimal_result.avg_gpu_util_pct:.1f}%")
        add_line()

        add_line("### All Batch Size Results")
        add_line()
        add_line(
            "| Batch Size | Throughput (cmp/s) | Matches Found | Memory (GB) | GPU Util (%) |"
        )
        add_line(
            "|------------|-------------------|---------------|-------------|--------------|"
        )
        for r in session.batch_benchmark_results:
            marker = " ⭐" if r.batch_size == optimal_result.batch_size else ""
            match_pct = (
                100 * r.pairs_found / r.pairs_compared if r.pairs_compared > 0 else 0
            )
            add_line(
                f"| {r.batch_size:,}{marker} | {r.throughput_pairs_per_sec:,.0f} | "
                f"{r.pairs_found:,} ({match_pct:.2f}%) | "
                f"{r.peak_memory_gb:.2f} | {r.avg_gpu_util_pct:.1f} |"
            )
        add_line()

    # Data Type Comparison
    if session.dtype_comparison:
        add_section("Data Type Comparison: float32 vs float64", 2)
        comp = session.dtype_comparison

        add_line("### float32")
        add_line(f"- **Time**: {comp.float32_time_s:.2f}s")
        add_line(f"- **Memory**: {comp.float32_memory_gb:.2f} GB")
        add_line(f"- **Avg Similarity**: {comp.float32_avg_similarity:.6f}")
        add_line()

        add_line("### float64")
        add_line(f"- **Time**: {comp.float64_time_s:.2f}s")
        add_line(f"- **Memory**: {comp.float64_memory_gb:.2f} GB")
        add_line(f"- **Avg Similarity**: {comp.float64_avg_similarity:.6f}")
        add_line()

        add_line("### Performance Gain with float32")
        add_line(f"- **Speedup**: {comp.speedup_factor:.2f}x faster")
        add_line(f"- **Memory Reduction**: {comp.memory_reduction_pct:.1f}%")
        add_line(f"- **Accuracy Difference**: {comp.accuracy_difference:.2e}")
        add_line()

    # Internal Kernel Timings
    if session.internal_kernel_timings:
        add_section("Internal Kernel Operation Breakdown", 2)
        timings = session.internal_kernel_timings

        add_line("**Single Batch Operation Timings (1000 spectra sample):**")
        add_line()
        add_line(f"- **GPU Transfer**: {timings.transfer_to_gpu_ms:.3f} ms")
        add_line(f"- **Normalize Left**: {timings.normalize_left_ms:.3f} ms")
        add_line(f"- **Normalize Right**: {timings.normalize_right_ms:.3f} ms")
        add_line(f"- **Expand Right**: {timings.expand_right_ms:.3f} ms")
        add_line(f"- **SpMM (L @ R.T)**: {timings.spmm_ms:.3f} ms ⭐")
        add_line(
            f"- **Threshold & Extract**: {timings.threshold_and_extract_ms:.3f} ms"
        )
        add_line(f"- **CPU Transfer**: {timings.transfer_to_cpu_ms:.3f} ms")
        add_line(f"- **Total Batch Time**: {timings.total_batch_ms:.3f} ms")
        add_line()

        # Calculate percentages
        total = timings.total_batch_ms
        if total > 0:
            add_line("**Percentage Breakdown:**")
            add_line()
            add_line(f"- GPU Transfer: {100 * timings.transfer_to_gpu_ms / total:.1f}%")
            add_line(
                f"- Normalize Left: {100 * timings.normalize_left_ms / total:.1f}%"
            )
            add_line(
                f"- Normalize Right: {100 * timings.normalize_right_ms / total:.1f}%"
            )
            add_line(f"- Expand Right: {100 * timings.expand_right_ms / total:.1f}%")
            add_line(f"- **SpMM (L @ R.T): {100 * timings.spmm_ms / total:.1f}%** ⭐")
            add_line(
                f"- Threshold & Extract: {100 * timings.threshold_and_extract_ms / total:.1f}%"
            )
            add_line(f"- CPU Transfer: {100 * timings.transfer_to_cpu_ms / total:.1f}%")
            add_line()

            # Identify bottleneck
            ops = [
                ("GPU Transfer", timings.transfer_to_gpu_ms),
                (
                    "Normalization",
                    timings.normalize_left_ms + timings.normalize_right_ms,
                ),
                ("Expansion", timings.expand_right_ms),
                ("SpMM", timings.spmm_ms),
                ("Threshold & Extract", timings.threshold_and_extract_ms),
                ("CPU Transfer", timings.transfer_to_cpu_ms),
            ]
            bottleneck = max(ops, key=lambda x: x[1])
            add_line(
                f"**Primary Bottleneck**: {bottleneck[0]} ({100 * bottleneck[1] / total:.1f}% of time)"
            )
            add_line()

    # SpMM Analysis
    if session.spmm_metrics:
        add_section("SpMM Kernel Analysis", 2)
        spmm = session.spmm_metrics

        add_line(f"**Duration**: {spmm.duration_ms:.2f} ms")
        add_line(f"**Estimated FLOPS**: {spmm.estimated_flops:,.0f}")
        add_line(f"**Achieved Performance**: {spmm.achieved_gflops:.2f} GFLOPS")
        add_line(
            f"**Efficiency vs Peak**: {spmm.efficiency_vs_peak_pct:.1f}% "
            f"(Peak: {session.peak_gpu_gflops:,.0f} GFLOPS)"
        )
        add_line(f"**Memory Bandwidth**: {spmm.memory_bandwidth_gb_s:.2f} GB/s")
        add_line(f"**Sparsity Factor**: {spmm.sparsity_factor:.6f}")
        add_line()

    # Recommendations
    add_section("Optimization Recommendations", 2)

    recommendations = []

    # Batch size recommendation
    if session.optimal_batch_size:
        recommendations.append(
            f"**Use Optimal Batch Size**: Configure `max_peaks_per_batch` to achieve "
            f"~{session.optimal_batch_size:,} spectra per batch for maximum throughput "
            f"({session.optimal_throughput:,.0f} comparisons/sec)."
        )

    # Dtype recommendation
    if session.dtype_comparison:
        if session.dtype_comparison.speedup_factor > 1.2:
            recommendations.append(
                f"**Prefer float32**: Provides {session.dtype_comparison.speedup_factor:.2f}x speedup "
                f"with {session.dtype_comparison.memory_reduction_pct:.1f}% less memory and negligible "
                f"accuracy loss ({session.dtype_comparison.accuracy_difference:.2e})."
            )

    # SpMM efficiency recommendation
    if session.spmm_metrics:
        if session.spmm_metrics.efficiency_vs_peak_pct < 20:
            recommendations.append(
                f"**Low SpMM Efficiency**: Only {session.spmm_metrics.efficiency_vs_peak_pct:.1f}% "
                f"of peak performance. This is typical for sparse operations and indicates "
                f"memory-bandwidth bound computation. Consider: (1) Increasing batch size to "
                f"amortize memory transfer costs, (2) Reducing expansion factor if possible."
            )
        elif session.spmm_metrics.efficiency_vs_peak_pct > 50:
            recommendations.append(
                f"**Excellent SpMM Efficiency**: Achieving {session.spmm_metrics.efficiency_vs_peak_pct:.1f}% "
                f"of peak performance. This is exceptional for sparse operations. Limited optimization headroom."
            )

    # GPU utilization recommendation
    if session.batch_benchmark_results:
        avg_util = np.mean(
            [r.avg_gpu_util_pct for r in session.batch_benchmark_results]
        )
        if avg_util < 50:
            recommendations.append(
                f"**Low GPU Utilization**: Average {avg_util:.1f}% utilization suggests GPU is "
                f"underutilized. Increase batch size or reduce CPU-GPU synchronization overhead."
            )

    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            add_line(f"{i}. {rec}")
            add_line()
    else:
        add_line(
            "No specific recommendations. Performance appears optimal for current configuration."
        )
        add_line()

    # Write report
    report_path.write_text("\n".join(lines))

    # Write JSON data
    json_data = {
        "gpu_info": {
            "name": session.gpu_name,
            "memory_gb": session.total_gpu_memory_gb,
            "compute_capability": session.compute_capability,
            "peak_gflops": session.peak_gpu_gflops,
        },
        "configuration": {
            "n_spectra": session.n_spectra,
            "n_peaks_per_spectrum": session.n_peaks_per_spectrum,
            "approx_threshold": session.approx_threshold,
            "ms2_tolerance_ppm": session.ms2_tolerance_ppm,
            "match_rate": session.match_rate,
        },
        "internal_kernel_timings": asdict(session.internal_kernel_timings)
        if session.internal_kernel_timings
        else None,
        "batch_benchmark_results": [asdict(r) for r in session.batch_benchmark_results],
        "dtype_comparison": asdict(session.dtype_comparison)
        if session.dtype_comparison
        else None,
        "spmm_metrics": asdict(session.spmm_metrics) if session.spmm_metrics else None,
        "recommendations": recommendations,
    }

    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"Optimization report written to: {report_path}")
    print(f"Raw data written to: {json_path}")
    print(f"{'=' * 80}\n")


# =============================================================================
# Main CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPU kernel optimization tool for fast_cosine_sim"
    )

    parser.add_argument(
        "--n-spectra",
        type=int,
        default=50_000,
        help="Number of synthetic spectra to generate (default: 50,000)",
    )

    parser.add_argument(
        "--n-peaks-per-spectrum",
        type=int,
        default=100,
        help="Peaks per spectrum in template (default: 100)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="optimization_results",
        help="Output directory for reports and plots (default: optimization_results)",
    )

    parser.add_argument(
        "--approx-threshold",
        type=float,
        default=0.65,
        help="Approximate similarity threshold (default: 0.65)",
    )

    parser.add_argument(
        "--ms2-tolerance-ppm",
        type=float,
        default=10.0,
        help="MS2 tolerance in ppm (default: 10.0)",
    )

    parser.add_argument(
        "--batch-configs",
        type=str,
        default=None,
        help="Comma-separated batch sizes to test (e.g., '100,500,1000,5000'). "
        "If not provided, uses automatic sweep.",
    )

    parser.add_argument(
        "--skip-dtype-comparison",
        action="store_true",
        help="Skip float32 vs float64 comparison (saves time)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    parser.add_argument(
        "--match-rate",
        type=float,
        default=0.001,
        help="Fraction of spectrum PAIRS that should match (default: 0.001 = 0.1%% of pairs). "
        "Example: 0.01 means 1%% of pairs match (~10%% of spectra similar to each other).",
    )

    parser.add_argument(
        "--use-fused-kernel",
        action="store_true",
        help="Use fused normalize-expand Numba CUDA kernel (experimental, for testing)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("GPU KERNEL OPTIMIZATION TOOL - fast_cosine_sim")
    print("=" * 80)
    print()

    if args.use_fused_kernel:
        print("⚡ USING FUSED NORMALIZE-EXPAND KERNEL (EXPERIMENTAL)")
        print("   Expected: 3-5× speedup on expansion, 1.5-2× overall improvement")
        print()

    # Initialize session
    session = OptimizationSession(
        n_spectra=args.n_spectra,
        n_peaks_per_spectrum=args.n_peaks_per_spectrum,
        approx_threshold=args.approx_threshold,
        ms2_tolerance_ppm=args.ms2_tolerance_ppm,
        match_rate=args.match_rate,
    )

    # Get GPU info
    gpu_info = get_gpu_info()
    session.gpu_name = gpu_info["gpu_name"]
    session.total_gpu_memory_gb = gpu_info["memory_gb"]
    session.compute_capability = gpu_info["compute_capability"]
    session.peak_gpu_gflops = gpu_info["peak_gflops"]

    print(f"GPU: {session.gpu_name}")
    print(f"Memory: {session.total_gpu_memory_gb:.2f} GB")
    print(f"Compute Capability: {session.compute_capability}")
    print(f"Peak GFLOPS: {session.peak_gpu_gflops:,.0f}")
    print(
        f"Match Rate: {args.match_rate:.4f} ({100*args.match_rate:.2f}% of pairs, "
        f"~{100*np.sqrt(args.match_rate):.1f}% of spectra match each other)"
    )
    print()

    # Generate template spectrum
    print(f"Generating template spectrum ({args.n_peaks_per_spectrum} peaks)...")
    template_spectrum = generate_synthetic_spectrum(
        n_peaks=args.n_peaks_per_spectrum, seed=args.seed
    )
    print(f"  ✓ Template spectrum generated\n")

    # Base configuration
    config_base = GPUApproximateConfig(
        upper_mass_bound=1000.0,
        bin_size=0.0001,
        approx_threshold=args.approx_threshold,
        ms2_tolerance_ppm=args.ms2_tolerance_ppm,
        intensity_power=0.5,
        target_gpu_mem_ratio=0.6,
        comparison_mode="self",
        spectrum_id_col="idx",
        mz_col="mz",
        intensity_col="intensity",
        use_fused_kernel=args.use_fused_kernel,
    )

    # GPU monitor
    gpu_monitor = GPUMonitor(sample_interval_s=0.1)

    # Determine batch size configurations to test
    if args.batch_configs:
        batch_configs = [int(x.strip()) for x in args.batch_configs.split(",")]
    else:
        # Automatic sweep: 10-15 configurations targeting 16-24 GB GPUs
        # Scale based on available memory
        mem_gb = session.total_gpu_memory_gb
        if mem_gb >= 20:
            # Large GPU: test larger batches
            batch_configs = [
                100,
                250,
                500,
                1_000,
                2_500,
                5_000,
                10_000,
                20_000,
                30_000,
                50_000,
            ]
        elif mem_gb >= 12:
            # Medium GPU: moderate batches
            batch_configs = [100, 250, 500, 1_000, 2_500, 5_000, 10_000, 15_000, 20_000]
        else:
            # Small GPU: conservative batches
            batch_configs = [100, 250, 500, 1_000, 2_500, 5_000, 7_500]

    # Profile internal kernel operations first (single batch for detailed timing)
    print(f"Step 1/4: Internal Kernel Operation Profiling")
    print(f"  Profiling single batch (1000 spectra) to identify bottlenecks...")

    template_mz, template_intensity = template_spectrum
    profile_df = generate_batch_with_mass_shifts(
        template_mz,
        template_intensity,
        n_spectra=min(5000, args.n_spectra),  # Use smaller dataset for profiling
        seed=args.seed,
        match_rate=args.match_rate,
    )

    try:
        session.internal_kernel_timings = profile_internal_kernel_operations(
            profile_df, config_base, use_fused=args.use_fused_kernel
        )

        kernel_type = "FUSED" if args.use_fused_kernel else "STANDARD"
        print(f"  ✓ Internal profiling complete ({kernel_type} kernel)")
        print(
            f"    - SpMM: {session.internal_kernel_timings.spmm_ms:.2f} ms "
            f"({100 * session.internal_kernel_timings.spmm_ms / session.internal_kernel_timings.total_batch_ms:.1f}%)"
        )
        print(
            f"    - Expansion: {session.internal_kernel_timings.expand_right_ms:.2f} ms "
            f"({100 * session.internal_kernel_timings.expand_right_ms / session.internal_kernel_timings.total_batch_ms:.1f}%)"
        )
        if not args.use_fused_kernel:
            print(
                f"    - Normalize Right: {session.internal_kernel_timings.normalize_right_ms:.2f} ms "
                f"({100 * session.internal_kernel_timings.normalize_right_ms / session.internal_kernel_timings.total_batch_ms:.1f}%)"
            )
        print(
            f"    - Thresholding: {session.internal_kernel_timings.threshold_and_extract_ms:.2f} ms "
            f"({100 * session.internal_kernel_timings.threshold_and_extract_ms / session.internal_kernel_timings.total_batch_ms:.1f}%)"
        )
        print(f"    - Total: {session.internal_kernel_timings.total_batch_ms:.2f} ms\n")

        # Plot internal timings
        plot_internal_kernel_timings(
            session.internal_kernel_timings, plots_dir / "internal_kernel_timings.png"
        )
        print(f"  ✓ Plot saved: {plots_dir / 'internal_kernel_timings.png'}\n")
    except Exception as e:
        print(f"  ⚠ Internal profiling failed: {e}")
        print(f"    Continuing with other benchmarks...\n")

    # Clean up
    del profile_df
    cp.get_default_memory_pool().free_all_blocks()

    # Run batch size benchmarking
    print(f"Step 2/4: Batch Size Optimization")
    session.batch_benchmark_results = benchmark_batch_sizes(
        template_spectrum=template_spectrum,
        dataset_size=args.n_spectra,
        batch_size_configs=batch_configs,
        config_base=config_base,
        gpu_monitor=gpu_monitor,
        match_rate=args.match_rate,
    )

    # Plot batch optimization results
    plot_batch_optimization(
        session.batch_benchmark_results, plots_dir / "batch_optimization.png"
    )
    print(f"  ✓ Plot saved: {plots_dir / 'batch_optimization.png'}\n")

    # Run dtype comparison
    if not args.skip_dtype_comparison:
        print(f"Step 3/4: Data Type Comparison (float32 vs float64)")
        session.dtype_comparison = benchmark_dtypes(
            template_spectrum=template_spectrum,
            n_spectra=min(args.n_spectra, 25_000),  # Use smaller dataset for speed
            config_base=config_base,
            gpu_monitor=gpu_monitor,
            match_rate=args.match_rate,
        )

        # Plot dtype comparison
        plot_dtype_comparison(
            session.dtype_comparison, plots_dir / "dtype_comparison.png"
        )
        print(f"  ✓ Plot saved: {plots_dir / 'dtype_comparison.png'}\n")
    else:
        print(f"Step 3/4: Data Type Comparison - SKIPPED\n")

    # Estimate SpMM metrics
    print(f"Step 4/4: SpMM Kernel Analysis")
    if session.batch_benchmark_results:
        # Use optimal result for SpMM analysis
        optimal_result = max(
            session.batch_benchmark_results, key=lambda r: r.throughput_pairs_per_sec
        )

        # Create a pseudo operation profile for SpMM analysis
        operation_profile = OperationProfile(
            name="spmm_estimate",
            duration_ms=optimal_result.total_time_s * 1000.0,
            memory_before_bytes=0,
            memory_after_bytes=int(optimal_result.peak_memory_gb * (1024**3)),
            memory_delta_bytes=int(optimal_result.peak_memory_gb * (1024**3)),
        )

        # Estimate expansion factor from tolerance
        expansion_factor = 1.0 + (args.ms2_tolerance_ppm / 1e6) * 10.0

        session.spmm_metrics = estimate_spmm_metrics(
            operation_profile=operation_profile,
            n_spectra=args.n_spectra,
            avg_peaks=args.n_peaks_per_spectrum,
            expansion_factor=expansion_factor,
            peak_gpu_gflops=session.peak_gpu_gflops,
        )

        print(
            f"  SpMM Efficiency: {session.spmm_metrics.efficiency_vs_peak_pct:.1f}% of peak"
        )
        print(f"  Achieved: {session.spmm_metrics.achieved_gflops:.2f} GFLOPS\n")

    # Generate comprehensive report
    print("Generating optimization report...")
    generate_optimization_report(session, output_dir)

    # Print summary
    print("\n" + "=" * 80)
    print("OPTIMIZATION SUMMARY")
    print("=" * 80)

    if session.optimal_batch_size:
        print(f"Optimal Batch Size: {session.optimal_batch_size:,} spectra")
        print(f"Optimal Throughput: {session.optimal_throughput:,.0f} comparisons/sec")

    if session.dtype_comparison:
        print(f"\nData Type Recommendation: float32")
        print(f"  Speedup: {session.dtype_comparison.speedup_factor:.2f}x")
        print(f"  Memory Savings: {session.dtype_comparison.memory_reduction_pct:.1f}%")

    if session.spmm_metrics:
        print(
            f"\nSpMM Efficiency: {session.spmm_metrics.efficiency_vs_peak_pct:.1f}% of peak"
        )

    print(f"\nTotal Runtime: {session.total_time:.1f}s")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
