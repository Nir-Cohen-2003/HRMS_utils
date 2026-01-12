#!/usr/bin/env python
"""
Example script demonstrating batched approximate + exact cosine similarity pipeline.

This script shows how to use the new batched exact cosine functionality with both
CPU and GPU exact computation modes.
"""

from __future__ import annotations

import time

import polars as pl
from approximate_similarity import SimilarityConfig
from batched_exact_cosine import run_approximate_and_exact_similarity


def compare_cpu_vs_gpu_exact(
    df: pl.DataFrame,
    threshold: float = 0.7,
    ms2_tolerance_ppm: float = 10.0,
    verbose: bool = True,
) -> None:
    """
    Compare CPU vs GPU exact cosine computation on the same approximate candidates.

    Why: This demonstrates the performance difference and validates that both
    implementations produce consistent results.
    """
    # Configure for CPU exact
    config_cpu = SimilarityConfig(
        upper_mass_bound=1000.0,
        bin_size=0.0001,
        ms2_tolerance_ppm=ms2_tolerance_ppm,
        intensity_power=0.5,
        threshold=threshold,
        use_gpu_exact_cosine=False,
    )

    # Configure for GPU exact
    config_gpu = SimilarityConfig(
        upper_mass_bound=1000.0,
        bin_size=0.0001,
        ms2_tolerance_ppm=ms2_tolerance_ppm,
        intensity_power=0.5,
        threshold=threshold,
        use_gpu_exact_cosine=True,
    )

    print("=" * 80)
    print("Running with CPU exact cosine...")
    print("=" * 80)
    t_cpu0 = time.perf_counter()
    results_cpu = run_approximate_and_exact_similarity(
        df,
        config_cpu,
        batch_size=1000,
        target_gpu_mem_ratio=0.6,
        max_peaks_per_exact_batch=10_000_000,
        verbose=verbose,
    )
    t_cpu1 = time.perf_counter()
    print(f"\nCPU total time: {t_cpu1 - t_cpu0:.3f}s")
    print(f"CPU results: {len(results_cpu)} pairs above threshold")

    print("\n" + "=" * 80)
    print("Running with GPU exact cosine (dynamic batching)...")
    print("=" * 80)
    t_gpu0 = time.perf_counter()
    results_gpu = run_approximate_and_exact_similarity(
        df,
        config_gpu,
        batch_size=1000,
        target_gpu_mem_ratio=0.6,
        max_peaks_per_exact_batch=10_000_000,
        verbose=verbose,
    )
    t_gpu1 = time.perf_counter()
    print(f"\nGPU total time: {t_gpu1 - t_gpu0:.3f}s")
    print(f"GPU results: {len(results_gpu)} pairs above threshold")

    # Compare results
    if len(results_cpu) > 0 and len(results_gpu) > 0:
        # Sort both by idx, idx_right for comparison
        cpu_sorted = results_cpu.sort(["idx", "idx_right"])
        gpu_sorted = results_gpu.sort(["idx", "idx_right"])

        assert len(cpu_sorted) == len(gpu_sorted), (
            f"Different number of pairs: CPU={len(cpu_sorted)}, GPU={len(gpu_sorted)}"
        )

        # Compare exact similarity scores
        cpu_scores = cpu_sorted["exact_similarity"].to_numpy()
        gpu_scores = gpu_sorted["exact_similarity"].to_numpy()

        import numpy as np

        diff = np.abs(cpu_scores - gpu_scores)
        print("\n" + "=" * 80)
        print("Comparison CPU vs GPU:")
        print(f"  Mean absolute difference: {diff.mean():.6f}")
        print(f"  Max absolute difference: {diff.max():.6f}")
        print(
            f"  Pairs with diff > 1e-3: {(diff > 1e-3).sum()} ({(diff > 1e-3).mean() * 100:.2f}%)"
        )
        print(
            f"  Pairs with diff > 1e-2: {(diff > 1e-2).sum()} ({(diff > 1e-2).mean() * 100:.2f}%)"
        )
        print("=" * 80)

        speedup = (t_cpu1 - t_cpu0) / (t_gpu1 - t_gpu0)
        print(f"\nSpeedup (GPU vs CPU): {speedup:.2f}x")


def run_single_library_example(
    parquet_path: str, n_spectra: int = 5000, threshold: float = 0.7
) -> None:
    """
    Run batched exact cosine on a single library.

    Why: This is the typical use case for all-vs-all similarity search within
    a spectral library.
    """
    print(f"Loading {n_spectra} spectra from {parquet_path}...")
    spectra = pl.scan_parquet(parquet_path).head(n_spectra).collect()

    print(f"Loaded {len(spectra)} spectra")
    print(f"Columns: {spectra.columns}")

    compare_cpu_vs_gpu_exact(spectra, threshold=threshold, verbose=True)


def run_gpu_only_example(
    parquet_path: str, n_spectra: int = 10000, threshold: float = 0.7
) -> None:
    """
    Run batched exact cosine using only GPU (no CPU comparison).

    Why: For production use, you typically just want the fastest path.
    """
    print(f"Loading {n_spectra} spectra from {parquet_path}...")
    spectra = pl.scan_parquet(parquet_path).head(n_spectra).collect()

    config = SimilarityConfig(
        upper_mass_bound=1000.0,
        bin_size=0.0001,
        ms2_tolerance_ppm=10.0,
        intensity_power=0.5,
        threshold=threshold,
        use_gpu_exact_cosine=True,
    )

    print("=" * 80)
    print(f"Running GPU-only pipeline with threshold={threshold}")
    print("=" * 80)

    t0 = time.perf_counter()
    results = run_approximate_and_exact_similarity(
        spectra,
        config,
        batch_size=1000,
        target_gpu_mem_ratio=0.6,
        max_peaks_per_exact_batch=10_000_000,
        verbose=True,
    )
    t1 = time.perf_counter()

    print(f"\nTotal time: {t1 - t0:.3f}s")
    print(f"Found {len(results)} pairs above threshold={threshold}")

    if len(results) > 0:
        print("\nSample of top 10 pairs by exact similarity:")
        print(results.sort("exact_similarity", descending=True).head(10))


if __name__ == "__main__":
    # Example usage - adjust path to your data
    parquet_path = (
        "/home/analytit_admin/Data/spectral_libs/info_score/combined_library.parquet"
    )

    print("Example 1: Compare CPU vs GPU exact cosine (5000 spectra)")
    print("-" * 80)
    run_single_library_example(parquet_path, n_spectra=5000, threshold=0.7)

    print("\n\n")
    print("Example 2: GPU-only production mode (10000 spectra)")
    print("-" * 80)
    run_gpu_only_example(parquet_path, n_spectra=10000, threshold=0.7)
