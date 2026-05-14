import shutil
from pathlib import Path
from time import time
from typing import BinaryIO, Optional, cast

import numpy as np
import polars as pl
import requests
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt  # type: ignore[unresolved-import]

from parallel_rdkit import sanitize_smiles_parallel


def _sanitize_smiles_polars(smiles: pl.Series) -> pl.Series:
    """Polars batch wrapper around parallel_rdkit.sanitize_smiles_parallel."""
    return pl.Series(sanitize_smiles_parallel(smiles), dtype=pl.String)


def reduce_pubchem_data(
    pubchem: pl.DataFrame | pl.LazyFrame,
) -> pl.DataFrame | pl.LazyFrame:
    pubchem = pubchem.with_columns(
        pl.col("InChIKey").str.split(by="-").list.get(0).alias("base_InChIKey"),
    )
    pubchem = pubchem.unique("base_InChIKey")
    return pubchem


def extract_pubchem_inchi(input_path: str | Path, output_path: str | Path):
    """
    Reads the CID-InChI-Key file using streaming, processes it,
    and writes the output to a Parquet file.
    Assumes the input file is tab-separated with columns: CID, InChI, InChIKey.

    Args:
        input_path: Path to the input CID-InChI-Key file.
        output_path: Path to save the output Parquet file.
    """
    print(f"Starting extraction from {input_path}")
    start_time = time()

    # Define the schema for the input file
    schema = {"CID": pl.Int64, "InChI": pl.String, "InChIKey": pl.String}

    try:
        # Scan the CSV file using streaming
        # Assuming tab separator based on common PubChem formats
        lf = pl.scan_csv(
            input_path,
            separator="\t",
            has_header=False,
            schema=schema,
        )

        # Collect the data using streaming
        data = lf.collect(engine="streaming")

        # Write the result to Parquet
        data.write_parquet(output_path)

        end_time = time()
        print(f"Successfully wrote {data.height} rows to {output_path}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")

    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        # Consider adding fallback logic here if needed (e.g., try different separators)


def extract_pubchem_smiles(input_path: str | Path, output_path: str | Path):
    """
    Reads the CID-SMILES file using streaming, canonicalizes SMILES,
    and writes the output to a Parquet file.
    Assumes the input file is tab-separated with columns: CID, SMILES.

    Args:
        input_path: Path to the input CID-SMILES file.
        output_path: Path to save the output Parquet file.
    """
    print(f"Starting extraction from {input_path}")
    start_time = time()

    # Define the schema for the input file
    schema = {
        "CID": pl.Int64,
        "SMILES_raw": pl.String,  # Read raw SMILES first
    }

    try:
        # Scan the CSV file using streaming
        # Assuming tab separator based on common PubChem formats
        lf = pl.scan_csv(
            input_path,
            separator="\t",
            has_header=False,
            schema=schema,
        )

        # Collect the data using streaming
        data = lf.collect(engine="streaming")

        # Apply canonicalization after collecting
        # Note: map_elements can be slower on very large data compared to pure Polars expressions
        print("Canonicalizing SMILES...")
        data = data.with_columns(
            pl.col("SMILES_raw")
            .map_batches(function=_sanitize_smiles_polars, return_dtype=pl.String)
            .alias("SMILES")
        ).drop("SMILES_raw")  # Drop the original raw SMILES column

        # Write the result to Parquet
        data.write_parquet(output_path)

        end_time = time()
        print(f"Successfully wrote {data.height} rows to {output_path}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")

    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        # Consider adding fallback logic here if needed


def extract_pubchem_mass(input_path: str | Path, output_path: str | Path):
    """
    Reads the CID-Mass file using streaming and writes the output to a Parquet file.

    Args:
        input_path: Path to the input CID-Mass file.
        output_path: Path to save the output Parquet file.
    """
    print(f"Starting extraction from {input_path}")
    start_time = time()
    # Define the schema for the input file
    schema = {
        "CID": pl.Int64,
        "Formula": pl.String,
        "monoisotopic_mass": pl.Float64,
        "exact_mass": pl.Float64,
    }

    try:
        # Scan the CSV file using streaming
        lf = pl.scan_csv(
            input_path,
            separator="\t",
            has_header=False,
            schema=schema,
        )

        # Collect the data using streaming
        data = lf.collect(engine="streaming")

        # Write the result to Parquet
        data.write_parquet(output_path)

        end_time = time()
        print(f"Successfully wrote {data.height} rows to {output_path}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")

    except Exception as e:
        print(f"Error processing {input_path}: {e}")


def download_pubchem_data(input_dir: Path, download_if_missing: bool = False):
    pubchem_smiles_url = (
        "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz"
    )
    pubchem_inchi_url = (
        "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-InChI-Key.gz"
    )

    input_dir = Path(input_dir)
    smiles_gz_path = input_dir / "CID-SMILES.gz"
    inchi_gz_path = input_dir / "CID-InChI-Key.gz"

    if not download_if_missing:
        if smiles_gz_path.exists() and inchi_gz_path.exists():
            print(f"PubChem files already exist in {input_dir}, skipping download.")
            return

    start = time()

    if download_if_missing or not smiles_gz_path.exists():
        r = requests.get(pubchem_smiles_url, stream=True)
        r.raise_for_status()
        with open(smiles_gz_path, "wb") as f:
            shutil.copyfileobj(cast(BinaryIO, r.raw), f)
        r.close()
        print(f"Downloaded {smiles_gz_path}")

    if download_if_missing or not inchi_gz_path.exists():
        r = requests.get(pubchem_inchi_url, stream=True)
        r.raise_for_status()
        with open(inchi_gz_path, "wb") as f:
            shutil.copyfileobj(cast(BinaryIO, r.raw), f)
        r.close()
        print(f"Downloaded {inchi_gz_path}")

    print(f"Download time: {time() - start:.2f} seconds")


def process_pubchem_data(
    input_dir: Path | str,
    output_path: Path | str,
    download_if_missing: bool = False,
) -> None:
    """
    Creates a merged PubChem parquet file for spectral library enrichment.

    Args:
        input_dir: Directory containing or where to store downloaded PubChem gz files.
        output_path: Path for the output merged Parquet file.
        download_if_missing: If True, download missing PubChem files from FTP.
    """
    input_dir = Path(input_dir)
    output_path = Path(output_path)

    download_pubchem_data(input_dir, download_if_missing=download_if_missing)

    smiles_gz_path = input_dir / "CID-SMILES.gz"
    inchi_gz_path = input_dir / "CID-InChI-Key.gz"

    if not smiles_gz_path.exists() or not inchi_gz_path.exists():
        raise FileNotFoundError(
            f"Required PubChem files not found in {input_dir}. "
            "Set download_if_missing=True to download them."
        )

    print("Processing PubChem data...")
    start_time = time()

    schema_smiles = {"CID": pl.Int64, "SMILES": pl.String}
    lf_smiles = pl.scan_csv(
        smiles_gz_path,
        separator="\t",
        has_header=False,
        schema=schema_smiles,
    )
    df_smiles = lf_smiles.collect(engine="streaming")
    lf_smiles = df_smiles.lazy()

    schema_inchi = {"CID": pl.Int64, "InChI": pl.String, "InChIKey": pl.String}
    lf_inchi = pl.scan_csv(
        inchi_gz_path,
        separator="\t",
        has_header=False,
        schema=schema_inchi,
    )

    lf_merged = lf_smiles.join(lf_inchi, on="CID", how="left")

    lf_merged = lf_merged.with_columns(
        pl.col("InChIKey").str.split(by="-").list.get(0).alias("base_inchikey")
    ).unique(subset="base_inchikey")

    lf_merged = lf_merged.select(
        [
            "CID",
            "InChIKey",
            "SMILES",
            "InChI",
        ]
    )

    lf_merged.sink_parquet(output_path, engine="streaming")

    print(f"Successfully wrote to {output_path}")
    print(f"Total time: {time() - start_time:.2f} seconds")


def get_mass_from_smiles(smiles: str):
    if smiles is None:
        return -1.0
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return ExactMolWt(mol)


get_mass_from_smiles_batch = np.vectorize(get_mass_from_smiles)


def get_pubchem_isomers_by_mass(
    pubchem: pl.LazyFrame,
    mass: float,
    tolerance_in_ppm: float | int = 3,
    num_isomers: Optional[int] = None,
    compound_base_inchikey: Optional[str] = None,
    column="monoisotopic_mass",
) -> pl.LazyFrame:
    """
    Get isomers from PubChem database that match a specific mass within a given tolerance.

    Args:
        pubchem: A polars LazyFrame containing PubChem data
        mass: The mass to search for in the database
        tolerance_in_ppm: Tolerance in parts per million for mass matching
        num_isomers: Maximum number of isomers to return (None for all)
        compound_base_inchikey: Base InChIKey to exclude from results
        column: Column name to use for mass comparison (default: 'monoisotopic_mass')

    Returns:
        A LazyFrame containing matched isomers

    Note:
        Best used for batch processing multiple masses, followed by collect_all()
        The exact_mass column is not the precursorMZ, there is the difference of the adduct
    """
    pubchem_isomers = pubchem.filter(
        pl.col(column).le(mass * (1 + tolerance_in_ppm / 1e6))
        & pl.col(column).ge(mass * (1 - tolerance_in_ppm / 1e6))
    )
    if "base_inchikey" not in pubchem_isomers.collect_schema().names():
        pubchem_isomers = pubchem_isomers.with_columns(
            base_inchikey=pl.col("InChIKey").str.split("-").list.get(index=0)
        )

    pubchem_isomers = pubchem_isomers.unique("base_inchikey")
    if compound_base_inchikey is not None:
        pubchem_isomers = pubchem_isomers.filter(
            pl.col("base_inchikey") != compound_base_inchikey
        )
    if num_isomers is not None:
        pubchem_isomers = pubchem_isomers.head(num_isomers)
    return pubchem_isomers


def get_pubchem_isomers_by_formula(
    pubchem_path: str | Path,
    formula: str,
    num_isomers: int | None = None,
    compound_base_inchikey: str | None = None,
) -> pl.DataFrame:
    """don't use this, the more correct way is to use the masses and the tolerance"""
    pubchem = pl.scan_parquet(
        source=f"{pubchem_path}/Formula={formula}",
        hive_partitioning=True,
        hive_schema={"Formula": pl.String},
    )
    pubchem_isomers = pubchem.filter(pl.col("Formula") == formula)
    pubchem_isomers = pubchem_isomers.with_columns(
        base_inchikey=pl.col("InChIKey").str.split("-").list.get(index=0)
    )
    pubchem_isomers = pubchem_isomers.unique("base_inchikey").drop("base_inchikey")
    if compound_base_inchikey is not None:
        pubchem_isomers = pubchem_isomers.filter(
            pl.col("base_inchikey") != compound_base_inchikey
        )
    if num_isomers is not None:
        pubchem_isomers = pubchem_isomers.head(num_isomers)
    return pubchem_isomers.collect(engine="streaming")


if __name__ == "__main__":
    start = time()
    print("Total script time : ", time() - start)
