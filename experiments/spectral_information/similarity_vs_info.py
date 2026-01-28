import datetime
import logging
import math
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import polars as pl

# Add fast_cosine_sim to path to import batched_gpu and approximate_similarity
# Assumes this script is in experiments/spectral_information/
# and fast_cosine_sim is in experiments/fast_cosine_sim/
sys.path.append(str(Path(__file__).parents[1] / "fast_cosine_sim"))

from approximate_similarity import SimilarityConfig
from batched_gpu import build_and_write_pairs_parquet_gpu_batched
from batched_utils import BatchedGPUConfig
from utils import compute_and_save_tanimoto_scores

logging.basicConfig(level=logging.INFO)
os.environ["RUST_BACKTRACE"] = "full"


if __name__ == "__main__":
    LIBRARY_PATH = Path(
        "/home/analytit_admin/Data/spectral_libs/info_score/combined_library.parquet"
    )
    PAIRS_PATH = Path(
        "/home/analytit_admin/Data/spectral_libs/info_score/combined_library_pairs_260104.parquet"
    )
    LEFT_LIBRARY_SNAPSHOT = PAIRS_PATH.with_suffix(".left_library.parquet")
    PAIRS_WITH_TANIMOTO_PATH = Path(
        "/home/analytit_admin/Data/spectral_libs/info_score/combined_library_pairs_with_tanimoto_260104.parquet"
    )

    # approx_cfg = SimilarityConfig(
    #     upper_mass_bound=1000.0,
    #     bin_size=0.0001,
    #     ms2_tolerance_ppm=10.0,
    #     intensity_power=0.5,
    #     threshold=0.5,
    # )

    logging.info(
        f"Starting similarity calculation. Approx threshold: {APPROX_THRESHOLD}, Exact threshold: {EXACT_THRESHOLD}"
    )

    # =========================================================================
    # Step 1: Generate approximate similarity candidate pairs
    # =========================================================================
    # Why: GPU-batched approximate similarity uses binned sparse matrices to quickly
    # identify candidate pairs above a lower threshold. This is much faster than exact
    # computation but less accurate due to binning artifacts.
    #
    # Output: `idx_left`, `idx_right`, `similarity` columns written to parquet

    logging.info(f"Loading library from {LIBRARY_PATH}")
    library_lf = pl.scan_parquet(str(LIBRARY_PATH))

    # Configure approximate similarity computation
    approx_cfg = GPUApproximateConfig(
        # Binning parameters
        upper_mass_bound=1000.0,
        bin_size=0.001,
        ms2_tolerance_ppm=5.0,
        intensity_power=0.5,
        approx_threshold=APPROX_THRESHOLD,
        # Comparison mode
        comparison_mode="self",  # Self-comparison with upper triangular optimization
        # Memory management
        target_gpu_mem_ratio=0.1,  # Use 10% of free GPU memory (conservative)
        safety_factor=0.5,  # Additional safety margin for memory estimation
        write_buffer_batches=1000,  # Flush to parquet every 100 GPU batches
        # Column names (match the library schema)
        spectrum_id_col="msp_index",
        mz_col="cleaned_normalized_mz",
        intensity_col="cleaned_normalized_intensity",
    )

    logging.info("Running approximate similarity computation on GPU...")
    t_approx_start = perf_counter()

    # Run approximate similarity and write directly to parquet
    # Why: Writing to file allows us to stream the results through the exact computation
    # stage without holding everything in memory.
    result_lf = batched_approximate_similarity_gpu(
        library_lf,
        config=approx_cfg,
        output_path=APPROX_PAIRS_PATH,
        logger=logging.getLogger(__name__),
    )

    t_approx_end = perf_counter()
    logging.info(
        f"Approximate similarity complete in {t_approx_end - t_approx_start:.3f}s"
    )

    # =========================================================================
    # Step 2: Compute exact similarity on candidate pairs (CPU streaming)
    # =========================================================================
    # Why: Exact similarity uses unbinned m/z arrays and precise peak matching with
    # tolerance windows. This is slower but more accurate. We stream from the approximate
    # results file to avoid loading everything into memory.
    #
    # Output: `idx_left`, `idx_right`, `approx_similarity`, `dotprod_similarity`

    logging.info("Computing exact similarity on candidates (CPU streaming)...")
    t_exact_start = perf_counter()

    # Load approximate pairs from file (lazy)
    # Note: The package outputs columns as `idx_left`, `idx_right`, `similarity`
    approx_pairs_lf = pl.scan_parquet(str(APPROX_PAIRS_PATH))

    # Rename `similarity` to `approx_similarity` for clarity in final output
    approx_pairs_lf = approx_pairs_lf.rename({"similarity": "approx_similarity"})

    logging.info(f"Loaded approximate pairs from {APPROX_PAIRS_PATH}")

    # Join left and right spectra
    # Note: 'idx_left' and 'idx_right' in pairs correspond to 'msp_index' in library
    pairs_with_spectra = (
        approx_pairs_lf.join(
            library_lf,
            left_on="idx_left",
            right_on="msp_index",
        )
        .rename(
            {
                "cleaned_normalized_mz": "mz1",
                "cleaned_normalized_intensity": "intensities1",
                "precursor_mz": "precursor_mz1",
            }
        )
        .join(
            library_lf,
            left_on="idx_right",
            right_on="msp_index",
            suffix="_right",
        )
        .rename(
            {
                "cleaned_normalized_mz": "mz2",
                "cleaned_normalized_intensity": "intensities2",
                "precursor_mz": "precursor_mz2",
            }
        )
    )

    # Compute exact dotprod and filter
    # Why: Use Polars spectral_similarity plugin for CPU-based exact cosine similarity.
    # This is slower than GPU but simpler and doesn't require complex batching logic.
    pairs_exact = (
        pairs_with_spectra.with_columns(
            pl.struct(
                mz1=pl.col("mz1"),
                intensities1=pl.col("intensities1"),
                mz2=pl.col("mz2"),
                intensities2=pl.col("intensities2"),
                precursor_mz1=pl.col("precursor_mz1"),
                precursor_mz2=pl.col("precursor_mz2"),
            )
            .spectral_similarity.dotprod_similarity(
                ms2_tolerance_in_ppm=approx_cfg.ms2_tolerance_ppm,
                clean_spectra_first=False,
                ignore_precursor=True,
            )
            .alias("dotprod_similarity")
        )
        .filter(pl.col("dotprod_similarity") >= EXACT_THRESHOLD)
        .select(
            "idx_left",
            "idx_right",
            "approx_similarity",
            "dotprod_similarity",
        )
    )

    # Collect and write final results
    # Why: Streaming sink for memory efficiency
    logging.info("Streaming exact results to final output...")
    pairs_exact.sink_parquet(PAIRS_PATH, maintain_order=False)

    t_exact_end = perf_counter()
    logging.info(f"Exact similarity complete in {t_exact_end - t_exact_start:.3f}s")

    # Read final count (lightweight, just metadata)
    final_count = pl.scan_parquet(PAIRS_PATH).select(pl.len()).collect().item()
    logging.info(f"Final results: {final_count} pairs written to {PAIRS_PATH}")

    total_time = t_exact_end - t_approx_start
    logging.info(
        f"Total time: {total_time:.3f}s "
        f"(approx: {t_approx_end - t_approx_start:.3f}s, exact: {t_exact_end - t_exact_start:.3f}s)"
    )
