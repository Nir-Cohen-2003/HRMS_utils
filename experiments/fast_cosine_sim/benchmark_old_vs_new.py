"""
Benchmark script to compare old vs new GPU-accelerated cosine similarity implementations.

Old: experiments/fast_cosine_sim/batched_gpu.py (full pipeline: approx + exact)
New: packages/fast_cosine_sim/ (approximate stage only, cleaner modular package)

This script compares ONLY the approximate stage timings for a fair comparison.

Usage:
    # Quick test with 100k spectra, 3 runs (default)
    python experiments/fast_cosine_sim/benchmark_old_vs_new.py

    # Full test with all datasets (100k, 300k, all), 5 runs
    python experiments/fast_cosine_sim/benchmark_old_vs_new.py --full-test --num-runs 5

    # Single run for faster iteration
    python experiments/fast_cosine_sim/benchmark_old_vs_new.py --num-runs 1
"""

from __future__ import annotations

import argparse
import gc
import logging
import re
import shutil
import sys
import tempfile
from io import StringIO
from pathlib import Path
from time import perf_counter
from typing import NamedTuple

import numpy as np
import polars as pl

# Add experiments directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Old implementation imports
from approximate_similarity import SimilarityConfig
from batched_gpu import build_and_write_pairs_parquet_gpu_batched
from batched_utils import BatchedGPUConfig

# Add packages to path for new implementation
sys.path.insert(
    0, str(Path(__file__).parent.parent.parent / "packages" / "fast_cosine_sim" / "src")
)

# New implementation imports
from fast_cosine_sim import (
    ApproximateGpuBatchedSimilarityConfig,
    ApproximateGpuDtypesConfig,
    BatchSizingConfig,
    IntensityTransformConfig,
    OutputParquetConfig,
    compute_gpu_batched_approximate_similarity_pairs,
)


class BenchmarkResult(NamedTuple):
    """Results from a single benchmark run."""

    dataset_name: str
    implementation: str
    run_number: int
    time_seconds: float
    approx_pairs: int
    dataset_size: int
    # Detailed timing breakdown (only for new implementation)
    t_flatten: float | None = None
    t_centroid: float | None = None
    t_binning: float | None = None
    t_gpu_compute: float | None = None
    t_write: float | None = None


class DatasetConfig(NamedTuple):
    """Configuration for a test dataset."""

    name: str
    path: Path
    num_spectra: int | None  # None = all spectra


# Dataset configurations
DATASETS = {
    "fraghub_100k": DatasetConfig(
        name="fraghub_100k",
        path=Path(
            "/home/analytit_admin/Data/spectral_libs/fast_similarity/fraghub_P_100k.parquet"
        ),
        num_spectra=None,  # Use all in file (already 100k)
    ),
    "fraghub_300k": DatasetConfig(
        name="fraghub_300k",
        path=Path(
            "/home/analytit_admin/Data/spectral_libs/fast_similarity/fraghub_P_300k.parquet"
        ),
        num_spectra=None,
    ),
    "fraghub_all": DatasetConfig(
        name="fraghub_all",
        path=Path(
            "/home/analytit_admin/Data/spectral_libs/fast_similarity/fraghub_P.parquet"
        ),
        num_spectra=None,
    ),
}

# Shared similarity parameters (must match for fair comparison)
# Note: The old implementation has auto-reduction logic where if you set threshold=X,
# it automatically sets approx_threshold = max(0.0, X - 0.15). To ensure both
# implementations use the EXACT same threshold, we explicitly set approx_threshold here.
SIMILARITY_PARAMS = {
    "upper_mass_bound": 1000.0,
    "bin_size": 0.0001,
    "ms2_tolerance_ppm": 10.0,
    "intensity_power": 0.5,
    "approx_threshold": 0.65,  # Actual threshold used for approximate stage
    "exact_threshold": 0.8,  # Would be used for exact stage (not relevant here)
    "target_gpu_mem_ratio": 0.3,
}


def cleanup_gpu_memory() -> None:
    """Clear GPU memory between tests."""
    try:
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass
    gc.collect()


def parse_old_log_for_approx_time(log_path: Path) -> float:
    """
    Extract approximate stage compute time from old implementation log.

    Why: The old implementation logs detailed profiling info including
    `approx_compute_seconds` which is the time we want to compare.

    Returns: approximate compute time in seconds
    """
    assert log_path.exists(), f"Log file does not exist: {log_path}"

    log_text = log_path.read_text()

    # Look for the approximate profiling summary
    # Format: "approx_compute_seconds=X.Xs"
    pattern = r"approx_compute_seconds=(\d+\.\d+)s"
    match = re.search(pattern, log_text)

    assert match is not None, (
        f"Could not find approx_compute_seconds in log file {log_path}. "
        f"Log content:\n{log_text[:1000]}"
    )

    return float(match.group(1))


def run_old_implementation(
    dataset: DatasetConfig,
    output_dir: Path,
    log_path: Path,
) -> tuple[float, int]:
    """
    Run old GPU batched similarity implementation (approximate stage only timing).

    Returns: (approx_compute_time_seconds, num_pairs)
    """
    # Old implementation requires these specific column names
    # Assuming the data has: cleaned_normalized_mz, cleaned_normalized_intensity, etc.

    # Build configuration matching old implementation
    # Why: Old implementation has auto-reduction: approx_threshold = threshold - 0.15
    # We explicitly set both to override the auto-reduction and ensure exact match with new impl
    approx_cfg = SimilarityConfig(
        upper_mass_bound=float(SIMILARITY_PARAMS["upper_mass_bound"]),
        bin_size=float(SIMILARITY_PARAMS["bin_size"]),
        ms2_tolerance_ppm=float(SIMILARITY_PARAMS["ms2_tolerance_ppm"]),
        intensity_power=float(SIMILARITY_PARAMS["intensity_power"]),
        threshold=float(
            SIMILARITY_PARAMS["exact_threshold"]
        ),  # For exact stage (not used here)
        approx_threshold=float(
            SIMILARITY_PARAMS["approx_threshold"]
        ),  # Explicitly set to match new impl
        use_gpu_exact_cosine=False,  # We only care about approximate stage
    )

    # Verify the threshold is set correctly (not auto-reduced)
    assert approx_cfg.approx_threshold == SIMILARITY_PARAMS["approx_threshold"], (
        f"Old implementation approx_threshold mismatch: "
        f"expected {SIMILARITY_PARAMS['approx_threshold']}, got {approx_cfg.approx_threshold}"
    )

    batched_cfg = BatchedGPUConfig(
        batch_size=10000,
        gpu_batch_write_interval=100,
        approx_config=approx_cfg,
        target_gpu_mem_ratio=float(SIMILARITY_PARAMS["target_gpu_mem_ratio"]),
    )

    # Run the old implementation
    build_and_write_pairs_parquet_gpu_batched(
        parquet_paths=[dataset.path],
        output_path=output_dir / "pairs.parquet",
        batched_config=batched_cfg,
        num_spectra=dataset.num_spectra,
    )

    # Extract timing from log
    approx_time = parse_old_log_for_approx_time(log_path)

    # Count pairs
    pairs_df = pl.scan_parquet(output_dir / "pairs.parquet")
    num_pairs = pairs_df.select(pl.len()).collect().item()

    return approx_time, int(num_pairs)


def parse_new_timing_log(log_text: str) -> dict[str, float] | None:
    """
    Extract detailed timing breakdown from new implementation log.

    Why: The new implementation logs detailed timing breakdown:
    "Timing breakdown: total=X.XXs | flatten=X.XXs | centroid=X.XXs | binning=X.XXs | gpu_compute=X.XXs | write=X.XXs"

    Returns: dict with timing components, or None if not found
    """
    # Look for timing breakdown line
    pattern = r"Timing breakdown: total=(\d+\.\d+)s \| flatten=(\d+\.\d+)s \| centroid=(\d+\.\d+)s \| binning=(\d+\.\d+)s \| gpu_compute=(\d+\.\d+)s(?:\s*\|\s*write=(\d+\.\d+)s)?"
    match = re.search(pattern, log_text)

    if match is None:
        return None

    return {
        "total": float(match.group(1)),
        "flatten": float(match.group(2)),
        "centroid": float(match.group(3)),
        "binning": float(match.group(4)),
        "gpu_compute": float(match.group(5)),
        "write": float(match.group(6)) if match.group(6) else 0.0,
    }


def run_new_implementation(
    dataset: DatasetConfig,
    output_dir: Path,
) -> tuple[float, int, dict[str, float] | None]:
    """
    Run new GPU batched similarity implementation (approximate stage).

    Returns: (compute_time_seconds, num_pairs, timing_breakdown)
    """
    # New implementation expects: mz, intensity columns
    # We need to load the data and map column names appropriately

    # Load data
    df = pl.read_parquet(dataset.path)

    # Map old column names to new expected names if needed
    if "cleaned_normalized_mz" in df.columns:
        df = df.rename(
            {
                "cleaned_normalized_mz": "mz",
                "cleaned_normalized_intensity": "intensity",
            }
        )

    # Ensure idx column exists
    if "idx" not in df.columns:
        df = df.with_row_index("idx")

    # Build configuration for new implementation
    config = ApproximateGpuBatchedSimilarityConfig(
        upper_mass_bound=float(SIMILARITY_PARAMS["upper_mass_bound"]),
        bin_size=float(SIMILARITY_PARAMS["bin_size"]),
        approx_threshold=float(SIMILARITY_PARAMS["approx_threshold"]),
        ms2_tolerance_ppm=float(SIMILARITY_PARAMS["ms2_tolerance_ppm"]),
        dtypes=ApproximateGpuDtypesConfig(index_dtype=np.dtype(np.int32)),
        intensity=IntensityTransformConfig(
            power=float(SIMILARITY_PARAMS["intensity_power"])
        ),
        batching=BatchSizingConfig(
            target_gpu_memory_usage_ratio=float(
                SIMILARITY_PARAMS["target_gpu_mem_ratio"]
            ),
            min_spectra_per_batch=10000,
            flush_to_parquet_every_n_batches=100,
        ),
        output_parquet=OutputParquetConfig(path=output_dir),
        comparison_mode="self",
        spectrum_id_column="idx",
        mz_column="mz",
        intensity_column="intensity",
    )

    # Set up logger to capture timing logs
    logger = logging.getLogger("fast_cosine_sim_benchmark")
    logger.setLevel(logging.INFO)
    log_stream = StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    # Time the computation
    t0 = perf_counter()
    result = compute_gpu_batched_approximate_similarity_pairs(
        left=df,
        config=config,
        right=None,
        logger=logger,
    )
    compute_time = perf_counter() - t0

    # Count pairs (result is a LazyFrame scan in write mode)
    if isinstance(result, pl.LazyFrame):
        num_pairs = result.select(pl.len()).collect().item()
    else:
        num_pairs = result.height

    # Parse timing breakdown from log
    log_text = log_stream.getvalue()
    timing_breakdown = parse_new_timing_log(log_text)

    # Clean up logger
    logger.removeHandler(handler)
    handler.close()

    return compute_time, int(num_pairs), timing_breakdown


def run_benchmark(
    dataset: DatasetConfig,
    run_number: int,
    implementation: str,
) -> BenchmarkResult:
    """
    Run a single benchmark for one implementation on one dataset.

    Args:
        dataset: Dataset configuration
        run_number: Run number (for multiple runs)
        implementation: "old" or "new"

    Returns: BenchmarkResult with timing and pair count
    """
    temp_dir = Path(tempfile.mkdtemp(prefix=f"benchmark_{implementation}_"))

    try:
        print(
            f"  Run {run_number}: {implementation} implementation...",
            end=" ",
            flush=True,
        )

        timing_breakdown = None

        if implementation == "old":
            output_dir = temp_dir / "output"
            output_dir.mkdir()
            log_path = output_dir / "pairs.log"

            time_seconds, num_pairs = run_old_implementation(
                dataset=dataset,
                output_dir=output_dir,
                log_path=log_path,
            )
        elif implementation == "new":
            output_dir = temp_dir / "output"

            time_seconds, num_pairs, timing_breakdown = run_new_implementation(
                dataset=dataset,
                output_dir=output_dir,
            )

            # Print timing breakdown if available
            if timing_breakdown:
                breakdown_str = " | ".join(
                    f"{k}={v:.2f}s" for k, v in timing_breakdown.items() if k != "total"
                )
                print(
                    f"{time_seconds:.3f}s ({num_pairs:,} pairs) [{breakdown_str}]",
                    flush=True,
                )
            else:
                print(f"{time_seconds:.3f}s ({num_pairs:,} pairs)", flush=True)
        else:
            raise ValueError(f"Unknown implementation: {implementation}")

        # Load dataset to get actual size
        df = pl.read_parquet(dataset.path)
        dataset_size = len(df)

        if implementation == "old":
            print(f"{time_seconds:.3f}s ({num_pairs:,} pairs)")

        return BenchmarkResult(
            dataset_name=dataset.name,
            implementation=implementation,
            run_number=run_number,
            time_seconds=time_seconds,
            approx_pairs=num_pairs,
            dataset_size=dataset_size,
            t_flatten=timing_breakdown.get("flatten") if timing_breakdown else None,
            t_centroid=timing_breakdown.get("centroid") if timing_breakdown else None,
            t_binning=timing_breakdown.get("binning") if timing_breakdown else None,
            t_gpu_compute=timing_breakdown.get("gpu_compute")
            if timing_breakdown
            else None,
            t_write=timing_breakdown.get("write") if timing_breakdown else None,
        )

    finally:
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        cleanup_gpu_memory()


def format_results_table(results: list[BenchmarkResult]) -> str:
    """Format benchmark results as a readable table."""

    # Group by dataset and implementation, compute min time
    from collections import defaultdict

    grouped = defaultdict(list)
    for r in results:
        grouped[(r.dataset_name, r.implementation)].append(r)

    # Compute summary statistics
    summary = []
    for (dataset_name, impl), runs in sorted(grouped.items()):
        min_time = min(r.time_seconds for r in runs)
        avg_time = sum(r.time_seconds for r in runs) / len(runs)
        num_pairs = runs[0].approx_pairs  # Should be same for all runs
        dataset_size = runs[0].dataset_size

        # For new implementation, get timing breakdown from min-time run
        timing_breakdown = None
        if impl == "new":
            min_run = min(runs, key=lambda r: r.time_seconds)
            if min_run.t_flatten is not None:
                timing_breakdown = {
                    "flatten": min_run.t_flatten,
                    "centroid": min_run.t_centroid,
                    "binning": min_run.t_binning,
                    "gpu_compute": min_run.t_gpu_compute,
                    "write": min_run.t_write,
                }

        summary.append(
            {
                "dataset": dataset_name,
                "implementation": impl,
                "min_time": min_time,
                "avg_time": avg_time,
                "num_pairs": num_pairs,
                "dataset_size": dataset_size,
                "timing_breakdown": timing_breakdown,
            }
        )

    # Build table
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("BENCHMARK RESULTS (Approximate Stage Only)")
    lines.append("=" * 100)
    lines.append(f"Parameters: {SIMILARITY_PARAMS}")
    lines.append("")

    # Group by dataset
    datasets = sorted(set(s["dataset"] for s in summary))

    for dataset in datasets:
        dataset_summary = [s for s in summary if s["dataset"] == dataset]

        old_data = next(
            (s for s in dataset_summary if s["implementation"] == "old"), None
        )
        new_data = next(
            (s for s in dataset_summary if s["implementation"] == "new"), None
        )

        lines.append(f"\nDataset: {dataset}")
        lines.append(f"  Size: {dataset_summary[0]['dataset_size']:,} spectra")
        lines.append(f"  Pairs found: {dataset_summary[0]['num_pairs']:,}")
        lines.append("")

        if old_data and new_data:
            speedup = old_data["min_time"] / new_data["min_time"]
            lines.append(
                f"  OLD implementation (min): {old_data['min_time']:>8.3f}s  (avg: {old_data['avg_time']:.3f}s)"
            )
            lines.append(
                f"  NEW implementation (min): {new_data['min_time']:>8.3f}s  (avg: {new_data['avg_time']:.3f}s)"
            )
            lines.append(
                f"  Speedup: {speedup:.2f}x {'FASTER' if speedup > 1 else 'SLOWER'}"
            )

            # Show timing breakdown for new implementation
            if new_data["timing_breakdown"]:
                tb = new_data["timing_breakdown"]
                lines.append("")
                lines.append("  NEW implementation breakdown (min-time run):")
                lines.append(
                    f"    flatten:     {tb['flatten']:>6.3f}s ({tb['flatten'] / new_data['min_time'] * 100:>5.1f}%)"
                )
                lines.append(
                    f"    centroid:    {tb['centroid']:>6.3f}s ({tb['centroid'] / new_data['min_time'] * 100:>5.1f}%)"
                )
                lines.append(
                    f"    binning:     {tb['binning']:>6.3f}s ({tb['binning'] / new_data['min_time'] * 100:>5.1f}%)"
                )
                lines.append(
                    f"    gpu_compute: {tb['gpu_compute']:>6.3f}s ({tb['gpu_compute'] / new_data['min_time'] * 100:>5.1f}%)"
                )
                if tb["write"] > 0:
                    lines.append(
                        f"    write:       {tb['write']:>6.3f}s ({tb['write'] / new_data['min_time'] * 100:>5.1f}%)"
                    )
        else:
            for s in dataset_summary:
                lines.append(
                    f"  {s['implementation'].upper()} (min): {s['min_time']:>8.3f}s  (avg: {s['avg_time']:.3f}s)"
                )

                # Show breakdown if available
                if s.get("timing_breakdown"):
                    tb = s["timing_breakdown"]
                    lines.append("    Breakdown:")
                    lines.append(
                        f"      flatten={tb['flatten']:.3f}s, centroid={tb['centroid']:.3f}s, binning={tb['binning']:.3f}s, gpu_compute={tb['gpu_compute']:.3f}s, write={tb['write']:.3f}s"
                    )

    lines.append("\n" + "=" * 100)

    return "\n".join(lines)


def save_results_csv(results: list[BenchmarkResult], output_path: Path) -> None:
    """Save detailed results to CSV."""
    df = pl.DataFrame(
        [
            {
                "dataset_name": r.dataset_name,
                "dataset_size": r.dataset_size,
                "implementation": r.implementation,
                "run_number": r.run_number,
                "time_seconds": r.time_seconds,
                "approx_pairs": r.approx_pairs,
                "t_flatten": r.t_flatten,
                "t_centroid": r.t_centroid,
                "t_binning": r.t_binning,
                "t_gpu_compute": r.t_gpu_compute,
                "t_write": r.t_write,
            }
            for r in results
        ]
    )

    df.write_csv(output_path)
    print(f"\nDetailed results saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark old vs new GPU-accelerated cosine similarity implementations."
    )

    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of runs per configuration (default: 3). Min time is reported.",
    )

    parser.add_argument(
        "--full-test",
        action="store_true",
        help="Test all datasets (100k, 300k, all). Default: only 100k.",
    )

    parser.add_argument(
        "--output-csv",
        type=str,
        default="benchmark_results.csv",
        help="Output CSV file path (default: benchmark_results.csv).",
    )

    return parser.parse_args()


def main() -> None:
    """Main benchmark execution."""
    args = parse_args()

    # Select datasets
    if args.full_test:
        datasets_to_test = ["fraghub_100k", "fraghub_300k", "fraghub_all"]
    else:
        datasets_to_test = ["fraghub_100k"]

    print("=" * 100)
    print("GPU-ACCELERATED COSINE SIMILARITY BENCHMARK")
    print("=" * 100)
    print(f"Comparing: OLD (experiments) vs NEW (packages)")
    print(f"Datasets: {', '.join(datasets_to_test)}")
    print(f"Runs per config: {args.num_runs}")
    print(f"Parameters: {SIMILARITY_PARAMS}")
    print(
        f"\nIMPORTANT: Both implementations use approx_threshold={SIMILARITY_PARAMS['approx_threshold']}"
    )
    print(
        f"           (Old impl auto-reduction disabled by explicit approx_threshold setting)"
    )
    print("=" * 100)

    # Verify all dataset paths exist
    for dataset_name in datasets_to_test:
        dataset = DATASETS[dataset_name]
        assert dataset.path.exists(), f"Dataset not found: {dataset.path}"

    # Run benchmarks
    all_results = []

    for dataset_name in datasets_to_test:
        dataset = DATASETS[dataset_name]
        print(f"\n{'=' * 100}")
        print(f"Testing: {dataset_name}")
        print(f"{'=' * 100}")

        for run_num in range(1, args.num_runs + 1):
            # Test OLD implementation
            result_old = run_benchmark(
                dataset=dataset,
                run_number=run_num,
                implementation="old",
            )
            all_results.append(result_old)

            # Test NEW implementation
            result_new = run_benchmark(
                dataset=dataset,
                run_number=run_num,
                implementation="new",
            )
            all_results.append(result_new)

    # Print results
    print(format_results_table(all_results))

    # Save to CSV
    output_path = Path(args.output_csv)
    save_results_csv(all_results, output_path)

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
