"""
NIST MSP/MSPEC format parser.

This module provides parsing functions for NIST MSP and MSPEC format files.
It uses the common spectral processing pipeline for consistency across formats.
"""

import re
from pathlib import Path
from typing import TypeVar, cast

import polars as pl

from ..formula_annotation.element_table import NUM_ELEMENTS
from ..formula_annotation.utils import format_formula_string_to_array
from ..hrms_core import *
from .spectra_pipeline import (
    filter_metadata,
    extract_collision_energy_values,
    annotate_spectra,
    add_precursor_type_indicators,
    add_molecular_ion_info,
    add_spectral_information_score,
)

polarsFrame = TypeVar("polarsFrame", pl.DataFrame, pl.LazyFrame)


def create_nist_dataframe(
    named_file_list: list[tuple[str | Path, str]],
) -> pl.DataFrame:
    """
    Read multiple MSP/MSPEC files and combine into a single DataFrame.
    
    Takes a list of tuples where each tuple contains:
    - File path (str or Path)
    - Database name (str) to add as "DB_Name" column
    
    Args:
        named_file_list: List of (file_path, db_name) tuples
        
    Returns:
        Combined DataFrame from all files with DB_Name column
    """
    for file_path, db_name in named_file_list:
        if not isinstance(file_path, Path):
            file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist.")
        if not file_path.is_file():
            raise ValueError(f"Path {file_path} is not a file.")
        # make sure the file is a MSPEC, mspec, MSP or msp file
        if file_path.suffix.lower() not in [".mspec", ".msp"]:
            raise ValueError(f"File {file_path} is not a MSPEC or MSP file.")
    
    dataframes = []
    for file_path, db_name in named_file_list:
        df = read_MSPEC_file(file_path)
        df = df.with_columns(pl.lit(db_name).alias("DB_Name"))
        dataframes.append(df)
    
    combined_df = pl.concat(dataframes, how="vertical")
    return combined_df


def read_MSPEC_file(
    path: Path | str,
    raw_fragment_tolerance_ppm: float = 10.0,
    normalized_fragment_tolerance_ppm: float = 5.0,
    molecular_ion_tolerance_ppm: float = 5.0,
    lazy: bool = False,
) -> pl.DataFrame | pl.LazyFrame:
    """
    Read an MSP or MSPEC file and apply the standard processing pipeline.
    
    Args:
        path: Path to MSP or MSPEC file
        raw_fragment_tolerance_ppm: Tolerance for initial fragment annotation (ppm)
        normalized_fragment_tolerance_ppm: Tolerance after normalization (ppm)
        molecular_ion_tolerance_ppm: Tolerance for molecular ion matching (ppm)
        lazy: If True, return a LazyFrame instead of collecting
        
    Returns:
        Processed DataFrame or LazyFrame with standardized columns
    """
    with open(path, "r") as file:
        file_contents = file.read()

    # Parse raw entries
    data = _parse_mspec_entries(file_contents)
    
    # Apply common processing pipeline
    data = filter_metadata(data)
    data = extract_collision_energy_values(data)
    data = annotate_spectra(
        data,
        raw_fragment_tolerance_ppm=raw_fragment_tolerance_ppm,
        normalized_fragment_tolerance_ppm=normalized_fragment_tolerance_ppm,
    )
    data = add_precursor_type_indicators(data)
    data = add_molecular_ion_info(data, molecular_ion_tolerance_ppm)
    data = add_spectral_information_score(data)
    
    # Select final columns in standard order
    data = data.select(
        [
            "name",
            "nist_id",
            "db_id",
            "instrument_type",
            "instrument",
            "ionization",
            "ion_mode",
            "mslevel",
            "collision_energy_NCE",
            "collision_energy_ev",
            "collision_energy_list",
            "multiple_collision_energies",
            "collision_energy_mean",
            "cas",
            "inchikey",
            "base_inchikey",
            "smiles",
            "inchi",
            "is_orbitrap",
            "is_TOF",
            "is_ESI",
            "precursor_type",
            "precursor_mz",
            "molecular_formula",
            "molecular_formula_array",
            "precursor_formula_array",
            "clean_precursor",
            "exact_mass",
            "raw_spectrum_mz",
            "raw_spectrum_intensity",
            "cleaned_normalized_mz",
            "cleaned_normalized_intensity",
            "cleaned_fragment_formulas",
            "cleaned_fragment_formulas_str",
            "cleaned_fragment_errors_ppm",
            "explained_intensity",
            "molecular_ion_intensity",
            "spectral_information_score",
            "spectral_information_score_with_hydrogens",
        ]
    )

    if not lazy:
        return data.collect(engine="streaming")
    else:
        return data


def _parse_mspec_entries(file_contents: str) -> pl.LazyFrame:
    """
    Parse MSP/MSPEC file contents into a LazyFrame with raw data.
    
    This is an internal function that extracts all metadata and spectrum data
    from MSP/MSPEC format entries.
    
    Args:
        file_contents: Raw file contents as string
        
    Returns:
        LazyFrame with parsed but unprocessed data
    """
    mz_intensity_pattern = r"(\d+\.\d+)\s(\d+(\.\d+)?)"

    entries = _split_entries(file_contents)
    data = pl.DataFrame(entries, schema={"raw": pl.String}).lazy()
    data = (
        data.with_columns(
            pl.col("raw")
            .str.extract(pattern=r"(?i)Name: (.+)", group_index=1)
            .alias("name"),
            pl.col("raw")
            .str.extract(pattern=r"(?i)NIST#: (\d+)", group_index=1)
            .alias("nist_id"),
            pl.col("raw")
            .str.extract(pattern=r"(?i)DB#: (\d+)", group_index=1)
            .alias("db_id"),
            pl.col("raw")
            .str.extract(pattern=r"(?i)Instrument_?type: (.+)", group_index=1)
            .alias("instrument_type"),
            pl.col("raw")
            .str.extract(pattern=r"(?i)Instrument: (.+)", group_index=1)
            .alias("instrument"),
            pl.col("raw")
            .str.extract(
                pattern=r"(?i)(?:Spectrum_type|MSLEVEL): (?:MS)?(\d+)", group_index=1
            )
            .str.to_integer()
            .alias("mslevel"),  # extract the numeric MS level
            pl.col("raw")
            .str.extract(pattern=r"(?i)Collision_gas: (.+)", group_index=1)
            .alias("collision_gas"),
            pl.col("raw")
            .str.extract(pattern=r"(?i)Collision_?energy: (.+)", group_index=1)
            .alias("collision_energy_raw"),
            pl.col("raw")
            .str.extract(pattern=r"(?i)Ionization: (.+)", group_index=1)
            .alias("ionization"),
            pl.col("raw")
            .str.extract(pattern=r"(?i)Ion_?mode: (p|n)", group_index=1)
            .alias(
                "ion_mode"
            ),  # works for P,N, and negative/positive in any capitalization
            pl.col("raw")
            .str.extract(pattern=r"(?i)Precursor_?type: (.+)", group_index=1)
            .alias("precursor_type"),
            pl.col("raw")
            .str.extract(pattern=r"(?i)PrecursorMZ: (\d+\.?\d*)", group_index=1)
            .alias("precursor_mz"),
            pl.col("raw")
            .str.extract(pattern=r"(?i)MW: (\d+)", group_index=1)
            .alias("mw"),
            pl.col("raw")
            .str.extract(pattern=r"(?i)Formula: (.+)", group_index=1)
            .alias("molecular_formula"),
            pl.col("raw")
            .str.extract(pattern=r"(?i)Num Peaks: (\d+)", group_index=1)
            .alias("num_peaks"),
            pl.col("raw")
            .str.extract(pattern=r"(?i)\nCAS#: ([0-9,-]+)", group_index=1)
            .alias("cas"),
            pl.col("raw")
            .str.extract(pattern=r"(?i)\nRelated_CAS#: ([0-9,-]+)", group_index=1)
            .alias("related_cas"),
            pl.col("raw")
            .str.extract(pattern=r"(?i)\nInChIKey: (.+)", group_index=1)
            .alias("inchikey"),
            pl.col("raw")
            .str.extract(pattern=r"(?i)\nExactMass: (\d+\.\d+)", group_index=1)
            .alias("exact_mass"),
            pl.col("raw")
            .str.extract(pattern=r"(?i)[Mm]z_diff=(-?\d+\.\d+)", group_index=1)
            .alias("mz_diff"),
            pl.col("raw")
            .str.extract_all(pattern=r"(?i)Synon: (.+)")
            .list.eval(
                pl.element().str.extract(pattern=r"(?i)Synon: (.+)", group_index=1)
            )
            .alias("synonyms"),
            pl.col("raw")
            .str.extract(pattern=r"(?i)Peptide_sequence: (.+)")
            .alias("peptide_sequence"),
            pl.col("raw")
            .str.extract(pattern=r"(?i)Peptide_mods: (.+)")
            .alias("peptide_mods"),
            pl.col("raw").str.extract(pattern=r"(?i)InChI: (.+)").alias("inchi"),
            pl.col("raw").str.extract(pattern=r"(?i)SMILES: (.+)").alias("smiles"),
            pl.col("raw")
            .str.extract_all(pattern=mz_intensity_pattern)
            .alias("mz_intensity"),
        )
        .drop("raw")
        .with_columns(
            pl.col("inchikey").str.extract(r"(.+?)-").alias("base_inchikey"),
            pl.col("nist_id").str.to_integer(),
            pl.col("db_id").str.to_integer(),
            pl.col("mw").str.to_integer(),
            pl.col("ion_mode").str.to_uppercase(),
            pl.col("num_peaks").str.to_integer(),
            pl.col("precursor_mz").cast(pl.Float64),
            pl.col("exact_mass").cast(pl.Float64, strict=False),
            pl.col("mz_diff").cast(pl.Float64),
            pl.col("molecular_formula")
            .map_elements(
                format_formula_string_to_array, return_dtype=pl.List(pl.Int32)
            )
            .list.to_array(width=NUM_ELEMENTS)
            .alias("molecular_formula_array"),
            pl.col("mz_intensity")
            .list.eval(
                pl.element().str.split(by=" ").list.get(index=0).cast(pl.Float64)
            )
            .alias("raw_spectrum_mz"),
            pl.col("mz_intensity")
            .list.eval(
                pl.element().str.split(by=" ").list.get(index=1).cast(pl.Float64)
            )
            .alias("raw_spectrum_intensity"),
            pl.when(  # when we know the precursor type, we can get the precursor formula directly
                pl.col("precursor_type").is_not_null()
            )
            .then(pl.col("precursor_type").str.replace(r"\[(M.*)\][+\\-]?\\d*", r"$1"))
            .otherwise(  # we assume [M+H]+ or [M-H]- based on ionization mode
                pl.when(pl.col("ion_mode").str.to_uppercase().eq("P"))
                .then(pl.lit("M+H"))
                .otherwise(pl.lit("M-H"))
            )
            .str.replace("M", pl.col("molecular_formula"))
            .alias("precursor_formula"),
        )
        .with_columns(
            pl.col("precursor_formula")
            .map_elements(
                format_formula_string_to_array, return_dtype=pl.List(pl.Int32)
            )
            .list.to_array(width=NUM_ELEMENTS)
            .alias("precursor_formula_array"),
        )
    )
    return data


def _split_entries(file_contents: str) -> list:
    """Split file contents into individual MSP/MSPEC entries."""
    entries = re.split(r"\n\s*\n", file_contents)
    if entries[len(entries) - 1] == "":
        entries.pop()
    return entries


if __name__ == "__main__":
    # Example usage
    from time import perf_counter

    start_time = perf_counter()
    
    # Example: Read a single file
    # df = read_MSPEC_file("/path/to/file.msp")
    
    # Example: Create combined dataframe from multiple files
    # file_list = [
    #     ("/path/to/file1.msp", "database_1"),
    #     ("/path/to/file2.msp", "database_2"),
    # ]
    # df = create_nist_dataframe(file_list)
    
    print("Module loaded successfully")
    print(f"Time taken: {perf_counter() - start_time:.4f} seconds")
