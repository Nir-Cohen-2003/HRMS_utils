from functools import partial
from pathlib import Path
from time import perf_counter
from typing import List, Union

import numpy as np
import polars as pl
from nvmolkit.fingerprints import MorganFingerprintGenerator
from nvmolkit.similarity import crossTanimotoSimilarityMemoryConstrained
from rdkit import Chem

from hrms_utils.rdkit import sanitize_smiles


def build_and_write_pairs_parquet(
    parquet_paths: List[Path],
    output_path: Union[str, Path],
    threshold: float = 0.8,
    num_mols_to_sample: int | None = None,
    ms2_tolerance_ppm: float = 10.0,
) -> None:
    """
    Build unioned library LF, compute pairwise dot-product similarities (ignoring precursor),
    and write pairs with high similarity to parquet.

    Args:
      - parquet_paths: list of Path objects pointing at library parquet files
      - output_path: where to write the pairs with similarities (required)
      - threshold: float (default 0.8). Only pairs with dotprod_similarity >= threshold are saved.
      - num_mols_to_sample: Optional[int]. If provided, limit the number of molecules read from the
        unioned input using a lazy .limit(num_mols_to_sample) to avoid collecting the full dataset.

    Returns:
      - None (writes parquet to output_path)
    """
    assert len(parquet_paths) > 0, "parquet_paths must contain at least one path"
    # Load and union into a single lazyframe
    lf_list = []
    for PARQUET_PATH in parquet_paths:
        assert Path(PARQUET_PATH).exists(), (
            f"Requested parquet does not exist: {PARQUET_PATH}"
        )
        lf_list.append(pl.scan_parquet(PARQUET_PATH))

    # Keep only precursors that passed cleaning
    lf = pl.union(lf_list).filter(pl.col("clean_precursor"))
    # Use an internal default tolerance for MS2 matching since tolerance is not exposed here.
    ms2_tolerance_ppm = 10.0

    # Optionally limit the number of molecules sampled from the unioned libraries.
    if num_mols_to_sample is not None:
        assert isinstance(num_mols_to_sample, int) and num_mols_to_sample > 0, (
            "num_mols_to_sample must be a positive integer or None"
        )
        lf = lf.limit(num_mols_to_sample)

    # Keep only necessary columns; add idx and nominal_mass to join on
    lf = (
        lf.collect()
        .lazy()
        .select(
            [
                "precursor_type",
                "precursor_mz",
                "precursor_formula_array",
                "ion_mode",
                "base_inchikey",
                "spectral_information_score",
                "cleaned_normalized_mz",
                "cleaned_normalized_intensity",
                "smiles",
            ]
        )
        .filter(pl.col("smiles").is_not_null())
        .with_row_index("idx")
        .with_columns(mol_idx=pl.col("idx").min().over(["base_inchikey", "ion_mode"]))
        #     .with_columns(
        #         nominal_mass=pl.col("precursor_mz").round(0),
        #         spectral_entropy=(
        #             pl.col("cleaned_normalized_intensity")
        #             / pl.col("cleaned_normalized_intensity").list.sum()
        #         )
        #         .list.eval(pl.element().log(base=math.e).mul(pl.element()))
        #         .list.sum()
        #         .neg(),
        #         num_clean_peaks=pl.col("cleaned_normalized_mz").list.len(),
        #         normalized_spectral_information_score=(
        #             # here we normalize the SIS per molecule+Ion mode, so its a fraction of the max possible SIS for that molecule
        #             pl.col("spectral_information_score").truediv(
        #                 pl.col("spectral_information_score")
        #                 .mean()
        #                 .over(["base_inchikey", "ion_mode"])
        #             )
        #         ),
        #     )
        #     .with_columns(
        #         most_informative=pl.col("normalized_spectral_information_score").eq(
        #             1.0
        #         ),
        #         normalized_spectral_entropy=pl.col("spectral_entropy").truediv(
        #             pl.col("spectral_entropy")
        #             .mean()
        #             .over(["base_inchikey", "ion_mode"])
        #         ),
        #         normalized_num_clean_peaks=pl.col("num_clean_peaks").truediv(
        #             pl.col("num_clean_peaks").mean().over(["base_inchikey", "ion_mode"])
        #         ),
        #     )
        #     .collect()
        #     .lazy()
    )
    start = perf_counter()
    # Join on nominal mass and ion mode; require different base_inchikey (different molecules).
    pairs_filtered = (
        lf.join(other=lf, on=["ion_mode"], suffix="_right")
        .filter(
            pl.col("base_inchikey") != pl.col("base_inchikey_right"),
        )
        .with_columns(
            spectra=pl.struct(
                mz1=pl.col("cleaned_normalized_mz").alias("mz1"),
                intensities1=pl.col("cleaned_normalized_intensity").alias(
                    "intensities1"
                ),
                mz2=pl.col("cleaned_normalized_mz_right").alias("mz2"),
                intensities2=pl.col("cleaned_normalized_intensity_right").alias(
                    "intensities2"
                ),
                precursor_mz1=pl.col("precursor_mz").alias("precursor_mz1"),
                precursor_mz2=pl.col("precursor_mz_right").alias("precursor_mz2"),
            )
        )
        .drop(
            [
                "cleaned_normalized_mz",
                "cleaned_normalized_intensity",
                "cleaned_normalized_mz_right",
                "cleaned_normalized_intensity_right",
                "precursor_mz",
                "precursor_mz_right",
            ]
        )
        .with_columns(
            # Only compute dot-product similarity (ignore precursor).
            dotprod_similarity=pl.col("spectra").spectral_similarity.dotprod_similarity(
                ms2_tolerance_in_ppm=ms2_tolerance_ppm,
                clean_spectra_first=False,
                ignore_precursor=True,
            )
        )
        .drop("spectra")
        .filter(
            pl.col("dotprod_similarity").is_not_null(),
            pl.col("dotprod_similarity").ge(threshold),
        )
        .with_columns(
            max_similairty_per_input=pl.col("dotprod_similarity")
            .max()
            .over(["idx", "idx_right"]),
        )
        .filter(pl.col("dotprod_similarity") == pl.col("max_similairty_per_input"))
        .drop("max_similairty_per_input")
        .select(
            "idx",
            "idx_right",
            "mol_idx",
            "mol_idx_right",
            "base_inchikey",
            "ion_mode",
            "base_inchikey_right",
            "smiles",
            "smiles_right",
            "dotprod_similarity",
            "spectral_information_score",
            "spectral_information_score_right",
        )
    )

    # Sink to parquet for downstream analysis and return the lazyframe
    pairs_filtered.sink_parquet(str(output_path), engine="streaming")
    end = perf_counter()
    print(
        f"wrote resutls of library search to file {str(output_path)} in time {end - start} "
    )
    return None


def process_batch(
    df: pl.DataFrame,
    fp_radius: int = 2,
    fp_size: int = 2048,
) -> pl.DataFrame:
    """Process one batch (a polars DataFrame) of pairs and compute Tanimoto similarity.

    The function expects the DataFrame to contain columns `smiles` and `smiles_right`.

    Args:
      - df: polars DataFrame containing pair rows
      - fp_radius: radius parameter for Morgan fingerprints
      - fp_size: fingerprint size (#bits)

    Returns:
      - A DataFrame equal to the input with an added column `tanimoto_similarity` (Float32)
    """
    # Check for required columns
    if "smiles" not in df.columns or "smiles_right" not in df.columns:
        raise ValueError(
            "Input dataframe must have 'smiles' and 'smiles_right' columns"
        )

    assert isinstance(fp_radius, int) and fp_radius > 0, (
        "fp_radius must be a positive integer"
    )
    assert isinstance(fp_size, int) and fp_size > 0, (
        "fp_size must be a positive integer"
    )

    s1 = df.get_column("smiles")
    s2 = df.get_column("smiles_right")

    # Identify unique SMILES to sanitize (combine both columns)
    unique_smiles = set(s1.drop_nulls().to_list()) | set(s2.drop_nulls().to_list())
    unique_list = list(unique_smiles)

    # Sanitize (returns list of canonical strings or empty strings)
    sanitized_list = sanitize_smiles(unique_list)

    # Map original -> RDKit Mol
    mol_map = {}
    for orig, clean in zip(unique_list, sanitized_list):
        if clean:
            m = Chem.MolFromSmiles(clean)
            if m:
                mol_map[orig] = m

    # Prepare generator
    fpgen = MorganFingerprintGenerator(radius=fp_radius, fpSize=fp_size)

    # Process in sub-chunks to manage memory/GPU
    sub_chunk_size = 4096
    n_rows = len(df)
    scores = np.full(n_rows, np.nan, dtype=np.float32)

    s1_list = s1.to_list()
    s2_list = s2.to_list()

    for start in range(0, n_rows, sub_chunk_size):
        end = min(start + sub_chunk_size, n_rows)

        batch_s1 = s1_list[start:end]
        batch_s2 = s2_list[start:end]

        # Convert to mols
        mols1 = [mol_map.get(s) for s in batch_s1]
        mols2 = [mol_map.get(s) for s in batch_s2]

        # Identify valid pairs
        valid_indices = [
            i
            for i, (m1, m2) in enumerate(zip(mols1, mols2))
            if m1 is not None and m2 is not None
        ]

        if not valid_indices:
            continue

        mols1_valid = [mols1[i] for i in valid_indices]
        mols2_valid = [mols2[i] for i in valid_indices]

        # Generate FPs
        fps1 = fpgen.GetFingerprints(mols1_valid)
        fps2 = fpgen.GetFingerprints(mols2_valid)

        # Calculate Similarity (Diagonal of cross-matrix)
        sim_mat = crossTanimotoSimilarityMemoryConstrained(fps1, fps2)
        if hasattr(sim_mat, "get"):
            sim_mat = sim_mat.get()

        # Extract diagonal
        diagonals = np.diag(sim_mat)

        # Assign back to scores
        for i, score in zip(valid_indices, diagonals):
            scores[start + i] = score

    return df.with_columns(pl.Series("tanimoto_similarity", scores, dtype=pl.Float32))


def compute_and_save_tanimoto_scores(
    input_parquet_path: Union[str, Path],
    output_path: Union[str, Path],
    batch_size: int = 100_000,
    fp_radius: int = 2,
    fp_size: int = 2048,
) -> None:
    """
    Computes Tanimoto similarity for pairs in the input parquet using nvmolkit.
    Reads 'smiles' and 'smiles_right' columns, computes similarity, and appends 'tanimoto_similarity'.
    """
    input_path = Path(input_parquet_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Robust input path handling
    parquet_paths = []
    path_str = str(input_parquet_path)
    if input_path.is_dir():
        parquet_paths = sorted(input_path.rglob("*.parquet"))
    elif any(ch in path_str for ch in ("*", "?", "[")):
        from glob import glob as _glob

        parquet_paths = [Path(p) for p in sorted(_glob(path_str, recursive=True))]
    else:
        parquet_paths = [input_path]

    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found for {input_path}")
    start = perf_counter()
    # Scan and stream
    lf = pl.scan_parquet(parquet_paths)

    # Stream batches and compute tanimoto per batch (process_batch adds the new column)

    lf.map_batches(
        partial(process_batch, fp_radius=fp_radius, fp_size=fp_size),
        streamable=True,
    ).sink_parquet(output_path)
    end = perf_counter()
    print(f"wrote the pairs with tanimoto to file {output_path} in time {end - start}")
