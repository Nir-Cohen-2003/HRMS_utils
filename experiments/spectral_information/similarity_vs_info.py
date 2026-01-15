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

# Use the packaged fast_cosine_sim implementation (packages/fast_cosine_sim) instead of
# experiments-local modules.
#
# This experiment should be runnable without installing the wheel/conda package, so we
# add the local packages source tree (`packages/fast_cosine_sim/src`) to sys.path.
# _FAST_COSINE_SIM_SRC = (
#     Path(__file__).parents[2] / "packages" / "fast_cosine_sim" / "src"
# )
# assert _FAST_COSINE_SIM_SRC.exists(), (
#     f"Expected fast_cosine_sim source tree at {_FAST_COSINE_SIM_SRC}. "
#     "If you moved the package, update this path."
# )
# sys.path.insert(0, str(_FAST_COSINE_SIM_SRC))
from fast_cosine_sim import (  # noqa: E402
    ApproximateGpuBatchedSimilarityConfig,
    BatchSizingConfig,
    IntensityTransformConfig,
    OutputParquetConfig,
    compute_gpu_batched_approximate_similarity_pairs,
)
from utils import compute_and_save_tanimoto_scores

logging.basicConfig(level=logging.INFO)
os.environ["RUST_BACKTRACE"] = "full"


if __name__ == "__main__":
    LIBRARY_PATH = Path(
        "/home/analytit_admin/Data/spectral_libs/info_score/combined_library.parquet"
    )
    PAIRS_PATH = Path(
        "/home/analytit_admin/Data/spectral_libs/info_score/combined_library_pairs_260115.parquet"
    )
    LEFT_LIBRARY_SNAPSHOT = PAIRS_PATH.with_suffix(".left_library.parquet")
    PAIRS_WITH_TANIMOTO_PATH = Path(
        "/home/analytit_admin/Data/spectral_libs/info_score/combined_library_pairs_with_tanimoto_260115.parquet"
    )

    # Generate approximate similarity candidate pairs using the *packaged* implementation
    # (packages/fast_cosine_sim) instead of experiments-local modules.
    #
    # Note: this stage produces `idx_left`, `idx_right`, `approx_similarity` only.
    # Tanimoto scoring is computed later via `compute_and_save_tanimoto_scores(...)`.
    #
    approx_cfg = ApproximateGpuBatchedSimilarityConfig(
        upper_mass_bound=1000.0,
        bin_size=0.0001,
        approx_threshold=0.5,
        intensity=IntensityTransformConfig(power=0.5),
        batching=BatchSizingConfig(
            target_gpu_memory_usage_ratio=0.1,
            # min_spectra_per_batch=10_000,
            flush_to_parquet_every_n_batches=100,
        ),
        output_parquet=OutputParquetConfig(path=PAIRS_PATH.with_suffix(".parts")),
        comparison_mode="self",
        # combined_library.parquet schema:
        # - spectrum id: `msp_index`
        # - peaks: `cleaned_normalized_mz`, `cleaned_normalized_intensity`
        spectrum_id_column="msp_index",
        mz_column="cleaned_normalized_mz",
        intensity_column="cleaned_normalized_intensity",
    )

    pairs_lazy_or_df = compute_gpu_batched_approximate_similarity_pairs(
        pl.scan_parquet(str(LIBRARY_PATH)),
        config=approx_cfg,
        logger=logging.getLogger(__name__),
    )

    # If write-mode is enabled, `pairs_lazy_or_df` is a LazyFrame scanning the written partitions.
    pairs_df = (
        pairs_lazy_or_df.collect()
        if isinstance(pairs_lazy_or_df, pl.LazyFrame)
        else pairs_lazy_or_df
    )
    pairs_df.write_parquet(PAIRS_PATH)

    # compute_and_save_tanimoto_scores(
    #     input_parquet_path=PAIRS_PATH,
    #     output_path=PAIRS_WITH_TANIMOTO_PATH,
    #     left_library_parquet_path=LEFT_LIBRARY_SNAPSHOT,
    #     right_library_parquet_path=None,
    #     batch_size=30_000_000,
    # )
