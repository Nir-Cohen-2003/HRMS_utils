from time import perf_counter

import marimo

__generated_with = "0.18.3"
app = marimo.App(width="full")


@app.cell
def _():
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

    os.environ["RUST_BACKTRACE"] = "full"
    return (
        Chem,
        List,
        MorganFingerprintGenerator,
        Path,
        Union,
        crossTanimotoSimilarityMemoryConstrained,
        np,
        pl,
        sanitize_smiles,
    )


@app.cell
def _(Path):
    LIBRARY_PATH = Path(
        "/home/analytit_admin/Data/spectral_libs/msp_for_Yonathan/combined_spectral_lib.parquet"
    )
    PAIRS_PATH = Path(
        "/home/analytit_admin/Data/spectral_libs/info_score/combined_library_pairs.parquet"
    )
    PAIRS_WITH_TANIMOTO_PATH = Path(
        "/home/analytit_admin/Data/spectral_libs/info_score/combined_library_pairs_with_tanimoto.parquet"
    )
    return LIBRARY_PATH, PAIRS_PATH, PAIRS_WITH_TANIMOTO_PATH


@app.cell
def _(LIBRARY_PATH, PAIRS_PATH, build_and_write_pairs_parquet):
    build_and_write_pairs_parquet(
        output_path=PAIRS_PATH,
        parquet_paths=[LIBRARY_PATH],
        threshold=0.8,
        num_mols_to_sample=10_000,
    )
    return


@app.cell
def _(PAIRS_PATH, PAIRS_WITH_TANIMOTO_PATH, compute_and_save_tanimoto_scores):
    compute_and_save_tanimoto_scores(
        input_parquet_path=PAIRS_PATH,
        output_path=PAIRS_WITH_TANIMOTO_PATH,
        batch_size=100_000,
    )
    return


if __name__ == "__main__":
    app.run()
