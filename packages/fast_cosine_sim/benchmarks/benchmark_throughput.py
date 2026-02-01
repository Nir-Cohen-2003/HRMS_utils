#!/usr/bin/env python
"""
Throughput Benchmark Script for fast_cosine_sim.

This script measures the throughput of the GPU approximate similarity computation
by using the library directly. It reports detailed breakdown of time spent in each
kernel operation (expansion, SpMM, normalization, etc.).

Usage:
    pixi run -e default python benchmarks/benchmark_throughput.py --n-spectra 50000
    pixi run -e default python benchmarks/benchmark_throughput.py --n-left 10000 --n-right 200000
"""

import argparse
from typing import Optional

import numpy as np
import polars as pl
from numpy.typing import NDArray

from fast_cosine_sim import GPUApproximateConfig, batched_approximate_similarity_gpu


# =============================================================================
# Data Generation
# =============================================================================


def generate_synthetic_spectrum(
    n_peaks: int = 100,
    mz_range: tuple[float, float] = (100.0, 1000.0),
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.float32]]:
    """Generate a single synthetic spectrum with realistic peak distribution."""
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
    """
    Generate a batch of synthetic spectra with controlled similarity.

    A fraction of spectra (based on match_rate) are similar to the template,
    while the rest are randomly generated from cluster templates.
    """
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
            mass_shifts = rng.uniform(
                -mass_shift_range_da, mass_shift_range_da, size=n_peaks
            )
            shifted_mz = template_mz + mass_shifts
            noise_factors = 1.0 + rng.uniform(
                -intensity_noise_pct, intensity_noise_pct, size=n_peaks
            )
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

    return pl.DataFrame(
        {
            "idx": pl.Series(idx_list, dtype=pl.Int32),
            "mz": pl.Series(mz_list, dtype=pl.List(pl.Float64)),
            "intensity": pl.Series(intensity_list, dtype=pl.List(pl.Float32)),
        }
    )


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Throughput benchmark for GPU approximate similarity"
    )
    parser.add_argument(
        "--n-spectra",
        type=int,
        default=None,
        help="Total spectra for self-comparison (mutually exclusive with --n-left/--n-right)",
    )
    parser.add_argument(
        "--n-left",
        type=int,
        default=None,
        help="Number of left (query) spectra for cross-comparison",
    )
    parser.add_argument(
        "--n-right",
        type=int,
        default=None,
        help="Number of right (database) spectra for cross-comparison",
    )
    parser.add_argument("--n-peaks", type=int, default=100, help="Peaks per spectrum")
    parser.add_argument("--match-rate", type=float, default=0.001, help="Match rate")
    parser.add_argument(
        "--threshold", type=float, default=0.65, help="Similarity threshold"
    )
    args = parser.parse_args()

    # Determine mode based on arguments
    if args.n_spectra is not None:
        if args.n_left is not None or args.n_right is not None:
            parser.error(
                "--n-spectra is mutually exclusive with --n-left/--n-right. "
                "Use --n-spectra for self-comparison or --n-left/--n-right for cross-comparison."
            )
        mode = "self"
        n_left = args.n_spectra
        n_right = args.n_spectra
    elif args.n_left is not None and args.n_right is not None:
        mode = "cross"
        n_left = args.n_left
        n_right = args.n_right
    else:
        # Default: cross-comparison with reasonable defaults
        mode = "cross"
        n_left = 10000
        n_right = 50000
        print(
            f"No size specified. Defaulting to cross-comparison: {n_left} queries vs {n_right} database."
        )

    print(f"Mode: {mode}")
    print(f"Left (queries): {n_left:,}")
    print(f"Right (database): {n_right:,}")
    print(f"Total pairs: {n_left * n_right:,}")
    print()

    # Generate synthetic data
    print(f"Generating synthetic spectra...")
    template_mz, template_intensity = generate_synthetic_spectrum(n_peaks=args.n_peaks)

    left_df = generate_batch_with_mass_shifts(
        template_mz,
        template_intensity,
        n_spectra=n_left,
        match_rate=args.match_rate,
        seed=42,
    )

    if mode == "cross":
        right_df = generate_batch_with_mass_shifts(
            template_mz,
            template_intensity,
            n_spectra=n_right,
            match_rate=args.match_rate,
            seed=123,
        )
    else:
        right_df = None

    print(f"  Left DataFrame: {len(left_df):,} spectra")
    if right_df is not None:
        print(f"  Right DataFrame: {len(right_df):,} spectra")
    print()

    # Configure the library
    config = GPUApproximateConfig(
        upper_mass_bound=1000.0,
        bin_size=0.0001,
        approx_threshold=args.threshold,
        ms2_tolerance_ppm=10.0,
        intensity_power=0.5,
        comparison_mode=mode,
    )

    # Run the benchmark using the library directly
    print("Running benchmark...")
    result, timings = batched_approximate_similarity_gpu(
        left_df,
        config,
        right_df=right_df,
        log_timings=True,
    )

    # Compute percentages
    timings.compute_percentages()

    # Collect result to get count
    if hasattr(result, "collect"):
        result_df = result.collect()
    else:
        result_df = result
    n_pairs_found = len(result_df)

    # Calculate throughput
    total_pairs = n_left * n_right
    if mode == "self":
        # Upper triangular: N*(N+1)/2, but library reports full pairs
        total_pairs = n_left * n_right

    gpu_throughput = (
        total_pairs / (timings.total_ms / 1000.0) if timings.total_ms > 0 else 0
    )
    wall_throughput = (
        total_pairs / (timings.wall_time_ms / 1000.0) if timings.wall_time_ms > 0 else 0
    )

    # Report results
    print()
    print("=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    print(f"Configuration:")
    print(f"  Mode:          {mode}")
    print(f"  Left Size:     {n_left:,}")
    print(f"  Right Size:    {n_right:,}")
    print(f"  Total Pairs:   {total_pairs:,}")
    print(f"  Pairs Found:   {n_pairs_found:,}")
    print("-" * 60)

    print(f"Timings:")
    print(f"  Wall Time:     {timings.wall_time_ms:.2f} ms")
    print(f"  Total GPU:     {timings.total_ms:.2f} ms")
    print("-" * 60)

    print(f"Throughput:")
    print(f"  GPU Kernel:    {gpu_throughput:,.0f} pairs/sec")
    print(f"  Wall Time:     {wall_throughput:,.0f} pairs/sec")
    print("-" * 60)

    print("TIMING BREAKDOWN (GPU Kernel Time)")
    print(f"{'Operation':<28} {'Time (ms)':>12} {'Pct (%)':>10}")
    print("-" * 60)

    operations = [
        ("GPU Transfer", timings.transfer_to_gpu_ms, timings.transfer_to_gpu_pct),
        ("Normalize Left", timings.normalize_left_ms, timings.normalize_left_pct),
        ("Normalize Right", timings.normalize_right_ms, timings.normalize_right_pct),
        ("Expand Left (SpMM)", timings.expand_ms, timings.expand_pct),
        ("SpMM (L @ R.T)", timings.spmm_ms, timings.spmm_pct),
        (
            "Threshold & Extract",
            timings.threshold_and_extract_ms,
            timings.threshold_and_extract_pct,
        ),
        ("CPU Transfer", timings.transfer_to_cpu_ms, timings.transfer_to_cpu_pct),
    ]

    for name, time_ms, pct in operations:
        print(f"{name:<28} {time_ms:12.2f} {pct:10.1f}")

    print("-" * 60)
    print("=" * 60)


if __name__ == "__main__":
    main()
