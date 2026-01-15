"""
Benchmark/runner for GPU-batched approximate cosine similarity candidate generation.

This script is intentionally scoped to the *approximate* GPU-batched stage only:
- it bins and normalizes spectra into sparse vectors
- computes sparse dot-products on GPU (CuPy)
- thresholds to emit candidate pairs
- optionally writes partitions to parquet
- logs timing + basic stats

It does NOT run any "exact similarity" refinement stage.

Usage example:

  python HRMS_utils/packages/fast_cosine_sim/scripts/run_gpu_batched_approx_similarity.py \
    --left-parquet /home/analytit_admin/Data/spectral_libs/fast_similarity/fraghub_P_100k.parqeut \
    --right-parquet /home/analytit_admin/Data/spectral_libs/fast_similarity/fraghub_P_100k.parqeut \
    --comparison-mode self \
    --output-parquet-dir ./output_approx_pairs \
    --log-path ./approx_benchmark.log

Notes:
- The example parquet path above is taken from `experiments/fast_cosine_sim/batched_gpu.py`.
- This script expects input parquet(s) to contain at least:
    - mz column (default: "mz") as list/array
    - intensity column (default: "intensity") as list/array
  and optionally an id column (default: "idx"). If missing, it will be added.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Optional

import polars as pl

from fast_cosine_sim import (
    ApproximateGpuBatchedSimilarityConfig,
    ApproximateGpuDtypesConfig,
    BatchSizingConfig,
    IntensityTransformConfig,
    LoggingConfig,
    OutputParquetConfig,
    compute_gpu_batched_approximate_similarity_pairs,
)


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log_line(message: str, *, log_path: Optional[Path]) -> None:
    line = f"[{_utc_iso()}] {message}"
    print(line)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            log_path.read_text() + line + "\n" if log_path.exists() else line + "\n"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GPU-batched approximate cosine similarity (candidate generation) and log timings."
    )

    # Paths (lifted from the experiment's example)
    parser.add_argument(
        "--left-parquet",
        type=str,
        default="/home/analytit_admin/Data/spectral_libs/fast_similarity/fraghub_P_100k.parqeut",
        help="Left parquet path (default taken from experiments/fast_cosine_sim/batched_gpu.py).",
    )
    parser.add_argument(
        "--right-parquet",
        type=str,
        default=None,
        help="Right parquet path (optional). If set, you likely want --comparison-mode cross.",
    )

    parser.add_argument(
        "--comparison-mode",
        type=str,
        choices=("self", "cross"),
        default="self",
        help="self: upper triangle i<j within one set; cross: full left x right.",
    )

    parser.add_argument("--mz-column", type=str, default="mz")
    parser.add_argument("--intensity-column", type=str, default="intensity")
    parser.add_argument("--id-column", type=str, default="idx")

    # Similarity/approx knobs
    parser.add_argument("--upper-mass-bound", type=float, default=1000.0)
    parser.add_argument("--bin-size", type=float, default=0.0001)
    parser.add_argument("--approx-threshold", type=float, default=0.8)
    parser.add_argument("--intensity-power", type=float, default=0.5)

    # Batching knobs
    parser.add_argument(
        "--target-gpu-mem-ratio",
        type=float,
        default=0.3,
        help="Fraction of free GPU memory to target for batching (approximate stage).",
    )
    parser.add_argument("--min-spectra-per-batch", type=int, default=256)
    parser.add_argument(
        "--max-peaks-per-batch",
        type=int,
        default=None,
        help="Optional clamp on peak-count-based dynamic batching.",
    )
    parser.add_argument(
        "--flush-to-parquet-every-n-batches",
        type=int,
        default=100,
        help="If set, enables parquet write mode and flushes every N inner batch-products.",
    )

    # Output options
    parser.add_argument(
        "--output-parquet-dir",
        type=str,
        default="output_approximate_pairs",
        help="Directory to write parquet partitions to (must not already exist).",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default="approx_gpu_batched_benchmark.log",
        help="Log file path for timings + run metadata.",
    )

    # IO / columns
    parser.add_argument(
        "--select-columns",
        type=str,
        default=None,
        help="Comma-separated list of columns to read from parquet. "
        "If not set, reads all columns. Recommended: idx,mz,intensity",
    )

    return parser.parse_args()


def _build_config(
    args: argparse.Namespace, *, output_dir: Path, log_path: Path
) -> ApproximateGpuBatchedSimilarityConfig:
    intensity_cfg = IntensityTransformConfig(power=float(args.intensity_power))
    dtypes_cfg = ApproximateGpuDtypesConfig()  # leave defaults (int64 ids, int32 CSR, float32 data)

    batching_cfg = BatchSizingConfig(
        target_gpu_memory_usage_ratio=float(args.target_gpu_mem_ratio),
        min_spectra_per_batch=int(args.min_spectra_per_batch),
        max_peaks_per_batch=None
        if args.max_peaks_per_batch is None
        else int(args.max_peaks_per_batch),
        flush_to_parquet_every_n_batches=int(args.flush_to_parquet_every_n_batches)
        if args.flush_to_parquet_every_n_batches is not None
        else None,
    )

    output_cfg = OutputParquetConfig(path=output_dir)
    logging_cfg = LoggingConfig(log_path=log_path)

    config = ApproximateGpuBatchedSimilarityConfig(
        upper_mass_bound=float(args.upper_mass_bound),
        bin_size=float(args.bin_size),
        approx_threshold=float(args.approx_threshold),
        dtypes=dtypes_cfg,
        intensity=intensity_cfg,
        batching=batching_cfg,
        output_parquet=output_cfg,
        logging=logging_cfg,
        comparison_mode=str(args.comparison_mode),  # Literal enforced in __post_init__
        spectrum_id_column=str(args.id_column),
        mz_column=str(args.mz_column),
        intensity_column=str(args.intensity_column),
    )
    return config


def _read_parquet(path: Path, *, select_columns: Optional[list[str]]) -> pl.DataFrame:
    assert path.exists(), f"Input parquet does not exist: {path}"
    if select_columns is None:
        return pl.read_parquet(path)
    missing = [c for c in select_columns if not c]
    assert not missing, f"select_columns contained empty column names: {select_columns}"
    return pl.read_parquet(path, columns=select_columns)


def main() -> None:
    args = _parse_args()

    left_path = Path(args.left_parquet)
    right_path = None if args.right_parquet is None else Path(args.right_parquet)

    output_dir = Path(args.output_parquet_dir)
    log_path = Path(args.log_path)

    # Fail fast: writer requires directory not to exist (mirrors library contract).
    if args.flush_to_parquet_every_n_batches is not None:
        assert not output_dir.exists(), (
            f"Output parquet dir already exists: {output_dir}. Remove it or choose a new one."
        )

    select_columns = None
    if args.select_columns is not None:
        select_columns = [c.strip() for c in args.select_columns.split(",") if c.strip()]

    _log_line("GPU-batched approximate similarity benchmark: start", log_path=log_path)
    _log_line(f"left_parquet={left_path}", log_path=log_path)
    _log_line(f"right_parquet={right_path}", log_path=log_path)
    _log_line(f"comparison_mode={args.comparison_mode}", log_path=log_path)
    _log_line(f"output_parquet_dir={output_dir}", log_path=log_path)
    _log_line("", log_path=log_path)

    config = _build_config(args, output_dir=output_dir, log_path=log_path)

    # Log config in a stable-ish way (dataclasses nested)
    config_payload = {
        "upper_mass_bound": config.upper_mass_bound,
        "bin_size": config.bin_size,
        "approx_threshold": config.approx_threshold,
        "comparison_mode": config.comparison_mode,
        "spectrum_id_column": config.spectrum_id_column,
        "mz_column": config.mz_column,
        "intensity_column": config.intensity_column,
        "dtypes": asdict(config.dtypes),
        "intensity": asdict(config.intensity),
        "batching": asdict(config.batching),
        "output_parquet": {
            "path": str(config.output_parquet.path) if config.output_parquet.path else None
        },
        "logging": {"log_path": str(config.logging.log_path) if config.logging.log_path else None},
    }
    _log_line("config=" + json.dumps(config_payload, indent=2, sort_keys=True), log_path=log_path)

    t0 = perf_counter()
    t_load0 = perf_counter()
    left_df = _read_parquet(left_path, select_columns=select_columns)
    right_df = (
        left_df if right_path is None else _read_parquet(right_path, select_columns=select_columns)
    )
    load_time = perf_counter() - t_load0

    _log_line(
        f"loaded: left_rows={left_df.height}, right_rows={right_df.height}, load_time_s={load_time:.3f}",
        log_path=log_path,
    )

    # Run approximate stage only.
    t_compute0 = perf_counter()
    result = compute_gpu_batched_approximate_similarity_pairs(
        left=left_df,
        right=None if right_path is None else right_df,
        config=config,
        logger=None,  # file logging is handled here; library logging is optional
    )
    compute_time = perf_counter() - t_compute0

    # Summarize result without doing any exact refinement.
    if isinstance(result, pl.LazyFrame):
        # In write-mode, this is a scan over partitions. Fetch only a count.
        t_count0 = perf_counter()
        approx_pairs = result.select(pl.len()).collect().item()
        count_time = perf_counter() - t_count0
        _log_line(
            f"result: type=LazyFrame(scan), approx_pairs={approx_pairs}, count_time_s={count_time:.3f}",
            log_path=log_path,
        )
        _log_line(f"parquet_partitions_glob={output_dir / '*.parquet'}", log_path=log_path)
    else:
        approx_pairs = result.height
        _log_line(
            f"result: type=DataFrame(in-memory), approx_pairs={approx_pairs}", log_path=log_path
        )

    total_time = perf_counter() - t0
    _log_line(
        f"timing: total_s={total_time:.3f} (load_s={load_time:.3f}, approx_compute_s={compute_time:.3f})",
        log_path=log_path,
    )
    _log_line("GPU-batched approximate similarity benchmark: done", log_path=log_path)


if __name__ == "__main__":
    main()
