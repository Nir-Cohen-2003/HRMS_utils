import logging
import math
import os
from pathlib import Path
from time import perf_counter
from tkinter import TclError
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import polars as pl
from proximate_simialrity import build_and_write_pairs_parquet
from utils import compute_and_save_tanimoto_scores

logging.basicConfig(level=logging.INFO)
os.environ["RUST_BACKTRACE"] = "full"


if __name__ == "__main__":
    LIBRARY_PATH = Path(
        "/home/analytit_admin/Data/spectral_libs/msp_for_Yonathan/combined_spectral_lib.parquet"
    )
    PAIRS_PATH = Path(
        "/home/analytit_admin/Data/spectral_libs/info_score/combined_library_pairs_full_test.parquet"
    )
    PAIRS_WITH_TANIMOTO_PATH = Path(
        "/home/analytit_admin/Data/spectral_libs/info_score/combined_library_pairs_with_tanimoto_test.parquet"
    )

    build_and_write_pairs_parquet(
        output_path=PAIRS_PATH,
        parquet_paths=[LIBRARY_PATH],
        threshold=0.8,
        num_spectra=None,
        mass_range=None,
        batch_size=32_000,
        num_workers=16,
    )

    # compute_and_save_tanimoto_scores(
    #     input_parquet_path=PAIRS_PATH,
    #     output_path=PAIRS_WITH_TANIMOTO_PATH,
    #     batch_size=None,
    # )
