#!/usr/bin/env python
"""
Throughput Benchmark Script for fast_cosine_sim.

This script measures the throughput of the SpMM expansion method for cosine similarity
with configurable batch sizes. It reports detailed breakdown of time spent in each
kernel operation (expansion, SpMM, normalization, etc.).

Usage:
    python benchmarks/benchmark_throughput.py --batch-size 10000 --n-spectra 2000000 --n-queries 10000
"""

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

import cupy as cp
import cupyx.scipy.sparse as cps
import numpy as np
import polars as pl
from numpy.typing import NDArray

from fast_cosine_sim import GPUApproximateConfig
from fast_cosine_sim.gpu_approximate_similarity import (
    APPROX_INTENSITY_DTYPE_NP,
    _normalize_csr_rows_inplace_gpu,
    _sparse_bin_spectra_df_to_csr,
    construct_expansion_matrix_gpu,
)

# =============================================================================
# Data Generation (Copied from scripts/optimize_gpu_kernel.py)
# =============================================================================

def generate_synthetic_spectrum(
    n_peaks: int = 100,
    mz_range: tuple[float, float] = (100.0, 1000.0),
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.float32]]:
    rng = np.random.RandomState(seed)
    mz_min, mz_max = mz_range
    base_mz = np.linspace(mz_min, mz_max, n_peaks)
    jitter = rng.uniform(-1.0, 1.0, n_peaks)
    mz_array = base_mz + jitter
    mz_array = np.sort(mz_array).astype(np.float64)
    log_intensities = rng.normal(loc=5.0, scale=1.5, size=n_peaks)
    intensities = np.exp(log_intensities)
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
    rng = np.random.RandomState(seed)
    n_peaks = len(template_mz)
    mz_min, mz_max = template_mz.min(), template_mz.max()
    
    idx_list = []
    mz_list = []
    intensity_list = []

    n_matching = max(1, int(np.sqrt(match_rate) * n_spectra))
    n_matching = min(n_matching, n_spectra)
    n_matching = max(2, n_matching)
    
    matching_indices = set(rng.choice(n_spectra, size=n_matching, replace=False))
    
    n_clusters = max(10, n_spectra // 100)
    cluster_templates = []
    for _ in range(n_clusters):
        cluster_mz = np.sort(rng.uniform(mz_min, mz_max, size=n_peaks))
        cluster_intensity = np.exp(rng.normal(loc=5.0, scale=1.5, size=n_peaks))
        cluster_intensity = cluster_intensity / np.max(cluster_intensity)
        cluster_templates.append((cluster_mz, cluster_intensity.astype(np.float32)))

    for i in range(n_spectra):
        if i in matching_indices:
            mass_shifts = rng.uniform(-mass_shift_range_da, mass_shift_range_da, size=n_peaks)
            shifted_mz = template_mz + mass_shifts
            noise_factors = 1.0 + rng.uniform(-intensity_noise_pct, intensity_noise_pct, size=n_peaks)
            noisy_intensity = template_intensity * noise_factors
            noisy_intensity = np.maximum(noisy_intensity, 1e-6)
            noisy_intensity = noisy_intensity / np.max(noisy_intensity)
        else:
            cluster_idx = rng.randint(0, n_clusters)
            cluster_mz, cluster_intensity = cluster_templates[cluster_idx]
            mass_shifts = rng.uniform(-5.0, 5.0, size=n_peaks)
            shifted_mz = cluster_mz + mass_shifts
            dropout_mask = rng.random(size=n_peaks) > 0.4
            noisy_intensity = cluster_intensity.copy()
            noisy_intensity[~dropout_mask] = 0.0
            remaining_peaks = dropout_mask.sum()
            if remaining_peaks > 0:
                noise_factors = rng.uniform(0.2, 2.0, size=n_peaks)
                noisy_intensity = noisy_intensity * noise_factors
                noisy_intensity = noisy_intensity / (np.max(noisy_intensity) + 1e-10)
            else:
                noisy_intensity = rng.uniform(0.1, 1.0, size=n_peaks)
                noisy_intensity = noisy_intensity / np.max(noisy_intensity)

        idx_list.append(i)
        mz_list.append(shifted_mz.tolist())
        intensity_list.append(noisy_intensity.tolist())

    return pl.DataFrame({
        "idx": pl.Series(idx_list, dtype=pl.Int32),
        "mz": pl.Series(mz_list, dtype=pl.List(pl.Float64)),
        "intensity": pl.Series(intensity_list, dtype=pl.List(pl.Float32)),
    })

# =============================================================================
# Profiling Logic
# =============================================================================

@dataclass
class AggregatedKernelTimings:
    transfer_to_gpu_ms: float = 0.0
    normalize_left_ms: float = 0.0
    normalize_right_ms: float = 0.0
    expand_right_ms: float = 0.0
    spmm_ms: float = 0.0
    threshold_and_extract_ms: float = 0.0
    transfer_to_cpu_ms: float = 0.0
    total_ms: float = 0.0
    
    num_batches: int = 0

class MockLogger:
    def info(self, msg): pass
    def warning(self, msg): print(f"[WARNING] {msg}")
    def error(self, msg): print(f"[ERROR] {msg}")

def prepare_data(df: pl.DataFrame, config: GPUApproximateConfig):
    print(f"Preprocessing data (converting to CSR)...")
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
    
    nbins = int(config.upper_mass_bound / config.bin_size) + 1
    
    print(f"Constructing expansion matrix...")
    expansion_matrix = construct_expansion_matrix_gpu(
        config.bin_size, config.ms2_tolerance_ppm, nbins, config.upper_mass_bound, logger=MockLogger()
    )
    if expansion_matrix is None:
        raise RuntimeError("Failed to construct expansion matrix. SpMM expansion method cannot be used.")
        
    return csr_matrix, expansion_matrix

def run_benchmark(
    csr_matrix,
    expansion_matrix,
    config: GPUApproximateConfig,
    batch_size: int,
    n_queries: int
) -> tuple[AggregatedKernelTimings, float]:
    
    timings = AggregatedKernelTimings()
    n_spectra = csr_matrix.shape[0]
    
    # Ensure n_queries is valid
    n_queries = min(n_queries, n_spectra)
    
    print(f"Starting batched processing...")
    print(f"  Queries:      {n_queries}")
    print(f"  Database:     {n_spectra}")
    print(f"  Batch Size:   {batch_size}")
    
    # Warmup
    cp.cuda.Stream.null.synchronize()
    
    # Events
    events = {name: cp.cuda.Event() for name in [
        "start_left", "end_left", "start_norm_left", "end_norm_left",
        "start_right", "end_right", "start_norm_right", "end_norm_right",
        "start_expand", "end_expand",
        "start_spmm", "end_spmm",
        "start_thresh", "end_thresh",
        "start_cpu", "end_cpu"
    ]}

    loop_start_time = time.perf_counter()

    # Outer Loop: Queries (Left side)
    for i in range(0, n_queries, batch_size):
        left_end = min(i + batch_size, n_queries)
        left_csr_cpu = csr_matrix[i:left_end]
        if left_csr_cpu.shape[0] == 0:
            continue
            
        # 1. Transfer Left (Once per query batch)
        events["start_left"].record()
        left_gpu = cps.csr_matrix(left_csr_cpu, dtype=np.float32)
        events["end_left"].record()

        # 2. Normalize Left (Once per query batch)
        events["start_norm_left"].record()
        _normalize_csr_rows_inplace_gpu(left_gpu)
        events["end_norm_left"].record()
        
        # Accumulate Left-side timings immediately (will happen multiple times if multiple query batches)
        # Note: We can't synchronize yet for max performance, but we record the events.
        
        # Inner Loop: Database (Right side)
        for j in range(0, n_spectra, batch_size):
            right_end = min(j + batch_size, n_spectra)
            right_csr_cpu = csr_matrix[j:right_end]
            if right_csr_cpu.shape[0] == 0:
                continue

            # 3. Transfer Right
            events["start_right"].record()
            right_gpu = cps.csr_matrix(right_csr_cpu, dtype=np.float32)
            events["end_right"].record()

            # 4. Normalize Right
            events["start_norm_right"].record()
            _normalize_csr_rows_inplace_gpu(right_gpu)
            events["end_norm_right"].record()

            # 5. Expand Right
            events["start_expand"].record()
            right_expanded = right_gpu.dot(expansion_matrix)
            events["end_expand"].record()

            # 6. SpMM
            events["start_spmm"].record()
            sim = left_gpu.dot(right_expanded.T)
            events["end_spmm"].record()

            # 7. Threshold
            events["start_thresh"].record()
            mask = sim.data >= config.approx_threshold
            if int(mask.sum()) > 0:
                _ = sim.data[mask]
            events["end_thresh"].record()

            # 8. Transfer back
            events["start_cpu"].record()
            # In real usage we would transfer indices/scores
            events["end_cpu"].record()
            
            # Synchronize periodically or at end of inner loop iteration to get timings
            # For accurate timing breakdown we need to sync, but this hurts pipeline overlap.
            # Benchmarking script usually prioritizes measurement.
            events["end_cpu"].synchronize()
            
            # Accumulate Inner Loop Timings
            timings.transfer_to_gpu_ms += cp.cuda.get_elapsed_time(events["start_right"], events["end_right"])
            timings.normalize_right_ms += cp.cuda.get_elapsed_time(events["start_norm_right"], events["end_norm_right"])
            timings.expand_right_ms += cp.cuda.get_elapsed_time(events["start_expand"], events["end_expand"])
            timings.spmm_ms += cp.cuda.get_elapsed_time(events["start_spmm"], events["end_spmm"])
            timings.threshold_and_extract_ms += cp.cuda.get_elapsed_time(events["start_thresh"], events["end_thresh"])
            timings.transfer_to_cpu_ms += cp.cuda.get_elapsed_time(events["start_cpu"], events["end_cpu"])
            
            # Add Left side overhead to this batch? 
            # Or just accumulate Left side once per outer loop?
            # The 'timings' object structure implies total time spent in kernels.
            # We should add the Left side setup to 'transfer_to_gpu' and 'normalize_left'.
            # Since we only do it once per outer loop, we can just add it here?
            # No, if we add it every inner loop it's wrong.
            # We should add it once per outer loop.

            timings.num_batches += 1
            
        # Accumulate Left Side Timings (Once per outer loop)
        events["end_norm_left"].synchronize()
        timings.transfer_to_gpu_ms += cp.cuda.get_elapsed_time(events["start_left"], events["end_left"])
        timings.normalize_left_ms += cp.cuda.get_elapsed_time(events["start_norm_left"], events["end_norm_left"])
        
    loop_wall_time = time.perf_counter() - loop_start_time
    
    # Calculate total GPU time
    timings.total_ms = (
        timings.transfer_to_gpu_ms + 
        timings.normalize_left_ms + 
        timings.normalize_right_ms + 
        timings.expand_right_ms + 
        timings.spmm_ms + 
        timings.threshold_and_extract_ms + 
        timings.transfer_to_cpu_ms
    )
    
    return timings, loop_wall_time

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Throughput benchmark for SpMM expansion")
    parser.add_argument("--batch-size", type=int, default=10000, help="Batch size (spectra)")
    parser.add_argument("--n-spectra", type=int, default=50000, help="Total number of spectra in database")
    parser.add_argument("--n-queries", type=int, default=None, help="Number of query spectra (default: batch-size)")
    parser.add_argument("--n-peaks", type=int, default=100, help="Peaks per spectrum")
    parser.add_argument("--match-rate", type=float, default=0.001, help="Match rate")
    args = parser.parse_args()

    # Default n_queries to n_spectra if not specified (All vs All)
    if args.n_queries is None:
        args.n_queries = args.n_spectra
        print(f"No --n-queries specified. Defaulting to All-vs-All comparison ({args.n_spectra} vs {args.n_spectra}).")
        print("Note: This may take a long time. Use --n-queries <N> for a partial benchmark.")
    else:
        print(f"Running partial benchmark: {args.n_queries} queries vs {args.n_spectra} database.")


    print(f"Generating {args.n_spectra} synthetic spectra...")
    template_mz, template_intensity = generate_synthetic_spectrum(n_peaks=args.n_peaks)
    df = generate_batch_with_mass_shifts(
        template_mz, template_intensity, 
        n_spectra=args.n_spectra, 
        match_rate=args.match_rate
    )

    config = GPUApproximateConfig(
        upper_mass_bound=1000.0,
        bin_size=0.0001,
        approx_threshold=0.65,
        ms2_tolerance_ppm=10.0,
        intensity_power=0.5,
    )
    
    # Setup
    setup_start = time.perf_counter()
    csr_matrix, expansion_matrix = prepare_data(df, config)
    setup_time = time.perf_counter() - setup_start
    print(f"Setup Time (CSR + Expansion Matrix): {setup_time:.2f} s")

    # Run Benchmark
    print("Running benchmark loop...")
    timings, loop_wall_time = run_benchmark(csr_matrix, expansion_matrix, config, args.batch_size, args.n_queries)

    # Clean up
    del expansion_matrix

    # Report
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    
    n_queries_actual = min(args.n_queries, args.n_spectra)
    total_pairs = n_queries_actual * args.n_spectra
    
    print(f"Configuration:")
    print(f"  Batch Size:    {args.batch_size}")
    print(f"  Database Size: {args.n_spectra}")
    print(f"  Query Size:    {n_queries_actual}")
    print(f"  Total Pairs:   {total_pairs:,}")
    print("-" * 60)
    
    print(f"Timings:")
    print(f"  Setup Time:      {setup_time*1000:.2f} ms")
    print(f"  Loop Wall Time:  {loop_wall_time*1000:.2f} ms")
    print(f"  Total GPU Time:  {timings.total_ms:.2f} ms")
    print("-" * 60)
    
    # Calculate throughputs
    # GPU Throughput (Theoretical Kernel Max)
    gpu_throughput = total_pairs / (timings.total_ms / 1000.0) if timings.total_ms > 0 else 0
    
    # System Throughput (End-to-End excluding one-time setup)
    system_throughput = total_pairs / loop_wall_time if loop_wall_time > 0 else 0
    
    print(f"Throughput:")
    print(f"  Kernel Throughput (GPU only):   {gpu_throughput:,.0f} pairs/sec")
    print(f"  System Throughput (Wall Time):  {system_throughput:,.0f} pairs/sec")
    print("-" * 60)
    print("TIMING BREAKDOWN (Accumulated GPU Time)")
    print(f"{'Operation':<25} {'Time (ms)':>12} {'Pct (%)':>10}")
    print("-" * 60)
    
    ops = [
        ("GPU Transfer", timings.transfer_to_gpu_ms),
        ("Normalize Left", timings.normalize_left_ms),
        ("Normalize Right", timings.normalize_right_ms),
        ("Expand Right (SpMM)", timings.expand_right_ms),
        ("SpMM (L @ R.T)", timings.spmm_ms),
        ("Threshold & Extract", timings.threshold_and_extract_ms),
        ("CPU Transfer", timings.transfer_to_cpu_ms),
    ]
    
    for name, t in ops:
        pct = (t / timings.total_ms) * 100 if timings.total_ms > 0 else 0
        print(f"{name:<25} {t:12.2f} {pct:10.1f}")
        
    print("-" * 60)
    print("="*60)

if __name__ == "__main__":
    main()
