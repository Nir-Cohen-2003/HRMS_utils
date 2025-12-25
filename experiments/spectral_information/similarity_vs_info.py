import logging
import math
import os
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import polars as pl
from nvmolkit.fingerprints import MorganFingerprintGenerator
from nvmolkit.similarity import crossTanimotoSimilarityMemoryConstrained
from rdkit import Chem
from utils import build_and_write_pairs_parquet, compute_and_save_tanimoto_scores

import hrms_utils
from hrms_utils.rdkit import sanitize_smiles

logging.basicConfig(level=logging.DEBUG)
os.environ["RUST_BACKTRACE"] = "full"


if __name__ == "__main__":
    LIBRARY_PATH = Path(
        "/home/analytit_admin/Data/spectral_libs/msp_for_Yonathan/combined_spectral_lib.parquet"
    )
    PAIRS_PATH = Path(
        "/home/analytit_admin/Data/spectral_libs/info_score/combined_library_pairs_200_300.parquet"
    )
    PAIRS_WITH_TANIMOTO_PATH = Path(
        "/home/analytit_admin/Data/spectral_libs/info_score/combined_library_pairs_with_tanimoto.parquet"
    )

    # build_and_write_pairs_parquet(
    #     output_path=PAIRS_PATH,
    #     parquet_paths=[LIBRARY_PATH],
    #     threshold=0.8,
    #     num_spectra=None,
    #     mass_range=(200.0, 300.0),
    #     batch_size=2000,
    #     use_pyarrow_batching=True,
    # )

    compute_and_save_tanimoto_scores(
        input_parquet_path=PAIRS_PATH,
        output_path=PAIRS_WITH_TANIMOTO_PATH,
        batch_size=None,
    )
