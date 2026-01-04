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
from batched_gpu import BatchedGPUConfig, build_and_write_pairs_parquet_gpu_batched
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

    approx_cfg = SimilarityConfig(
        upper_mass_bound=1000.0,
        bin_size=0.0001,
        ms2_tolerance_ppm=10.0,
        intensity_power=0.5,
    )

    batched_cfg = BatchedGPUConfig(
        batch_size=10000,
        gpu_batch_write_interval=100,
        threshold=0.5,
        approx_config=approx_cfg,
        target_gpu_mem_ratio=0.1,
    )

    build_and_write_pairs_parquet_gpu_batched(
        parquet_paths=[LIBRARY_PATH],
        output_path=PAIRS_PATH,
        batched_config=batched_cfg,
    )

    compute_and_save_tanimoto_scores(
        input_parquet_path=PAIRS_PATH,
        output_path=PAIRS_WITH_TANIMOTO_PATH,
        left_library_parquet_path=LEFT_LIBRARY_SNAPSHOT,
        right_library_parquet_path=None,
        batch_size=10_000_000,
    )
