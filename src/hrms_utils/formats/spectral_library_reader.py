"""
Universal spectral library reader.

This module provides a unified interface for reading spectral library files
in various formats (MSP, MGF, MSPEC). It automatically detects the format
based on file extension and applies the common processing pipeline.

For MSn spectra (MSLEVEL > 2), the reader handles precursor annotation and
returns separate dataframes for valid, no-options, and ambiguous cases.
"""

from pathlib import Path
from typing import Tuple

import polars as pl

from .spectra_schema import SpectralLibrarySchema, validate_spectral_library
from .spectra_pipeline import (
    filter_metadata,
    extract_collision_energy_values,
    annotate_spectra,
    add_precursor_type_indicators,
    add_molecular_ion_info,
    add_spectral_information_score,
    annotate_msn_precursors,
)


def read_spectral_library(
    path: Path | str,
    raw_fragment_tolerance_ppm: float = 10.0,
    normalized_fragment_tolerance_ppm: float = 5.0,
    molecular_ion_tolerance_ppm: float = 10.0,
    msn_precursor_tolerance_ppm: float = 10.0,
    lazy: bool = False,
) -> Tuple[pl.DataFrame | pl.LazyFrame, pl.DataFrame | pl.LazyFrame, pl.DataFrame | pl.LazyFrame]:
    """
    Read a spectral library file and apply standard processing pipeline.
    
    Automatically detects file format from extension (.msp, .mspec, .mgf) and
    routes to the appropriate parser. Applies the common processing pipeline
    including metadata filtering, collision energy extraction, spectra annotation,
    and spectral information score calculation.
    
    For MSn spectra (MSLEVEL > 2), returns three dataframes:
    1. Valid spectra: Exactly 1 formula option for all MSn precursors
    2. No formula options: At least one precursor with 0 formula options
    3. Ambiguous spectra: At least one precursor with 2+ formula options
    
    Args:
        path: Path to spectral library file (.msp, .mspec, or .mgf)
        raw_fragment_tolerance_ppm: Tolerance for initial fragment annotation (ppm)
        normalized_fragment_tolerance_ppm: Tolerance after normalization (ppm)
        molecular_ion_tolerance_ppm: Tolerance for molecular ion matching (ppm)
        msn_precursor_tolerance_ppm: Tolerance for MSn precursor annotation (ppm)
        lazy: If True, return LazyFrames instead of DataFrames
        
    Returns:
        Tuple of (valid_df, no_options_df, ambiguous_df):
        - valid_df: Valid spectra with validated schema
        - no_options_df: Spectra with no formula options for at least one precursor
        - ambiguous_df: Spectra with 2+ formula options for at least one precursor
        
        For MS2-only data (no MSLEVEL > 2), no_options_df and ambiguous_df
        will be empty DataFrames.
        
    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file format is not supported
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"File {path} does not exist.")
    
    if not path.is_file():
        raise ValueError(f"Path {path} is not a file.")
    
    # Parse file based on extension
    extension = path.suffix.lower()
    
    if extension in [".msp", ".mspec"]:
        from .nist_mspec import _parse_mspec_entries
        raw_df = _parse_mspec_entries(path)
    elif extension == ".mgf":
        from .mgf import _parse_mgf_entries
        raw_df = _parse_mgf_entries(path)
    else:
        raise ValueError(
            f"Unsupported file format: {extension}. "
            f"Supported formats: .msp, .mspec, .mgf"
        )
    
    # Apply common processing pipeline
    processed_df = _apply_pipeline(
        raw_df,
        raw_fragment_tolerance_ppm=raw_fragment_tolerance_ppm,
        normalized_fragment_tolerance_ppm=normalized_fragment_tolerance_ppm,
        molecular_ion_tolerance_ppm=molecular_ion_tolerance_ppm,
    )
    
    # Handle MSn precursor annotation and separation
    valid_df, no_options_df, ambiguous_df = annotate_msn_precursors(
        processed_df,
        tolerance_ppm=msn_precursor_tolerance_ppm,
    )
    
    # Validate against schema
    validate_spectral_library(valid_df)
    
    if not lazy:
        valid_df = valid_df.collect() if isinstance(valid_df, pl.LazyFrame) else valid_df
        no_options_df = no_options_df.collect() if isinstance(no_options_df, pl.LazyFrame) else no_options_df
        ambiguous_df = ambiguous_df.collect() if isinstance(ambiguous_df, pl.LazyFrame) else ambiguous_df
    
    return valid_df, no_options_df, ambiguous_df


def read_spectral_libraries(
    named_file_list: list[Tuple[Path | str, str]],
    raw_fragment_tolerance_ppm: float = 10.0,
    normalized_fragment_tolerance_ppm: float = 5.0,
    molecular_ion_tolerance_ppm: float = 10.0,
    msn_precursor_tolerance_ppm: float = 10.0,
    lazy: bool = False,
) -> Tuple[pl.DataFrame | pl.LazyFrame, pl.DataFrame | pl.LazyFrame, pl.DataFrame | pl.LazyFrame]:
    """
    Read multiple spectral library files with associated database names.
    
    Takes a list of tuples where each tuple contains:
    - File path (Path or str)
    - Database name (str) to add as 'db_name' column
    
    Args:
        named_file_list: List of (file_path, db_name) tuples
        raw_fragment_tolerance_ppm: Tolerance for initial fragment annotation
        normalized_fragment_tolerance_ppm: Tolerance after normalization
        molecular_ion_tolerance_ppm: Tolerance for molecular ion matching
        msn_precursor_tolerance_ppm: Tolerance for MSn precursor annotation
        lazy: If True, return LazyFrames
        
    Returns:
        Tuple of (valid_df, no_options_df, ambiguous_df) concatenated from all files
    """
    valid_dfs = []
    no_options_dfs = []
    ambiguous_dfs = []
    
    for file_path, db_name in named_file_list:
        valid, no_options, ambiguous = read_spectral_library(
            file_path,
            raw_fragment_tolerance_ppm=raw_fragment_tolerance_ppm,
            normalized_fragment_tolerance_ppm=normalized_fragment_tolerance_ppm,
            molecular_ion_tolerance_ppm=molecular_ion_tolerance_ppm,
            msn_precursor_tolerance_ppm=msn_precursor_tolerance_ppm,
            lazy=True,  # Keep lazy for concatenation
        )
        
        # Add db_name column
        valid = valid.with_columns(pl.lit(db_name).alias("db_name"))
        if no_options.height > 0 if isinstance(no_options, pl.DataFrame) else True:
            no_options = no_options.with_columns(pl.lit(db_name).alias("db_name"))
        if ambiguous.height > 0 if isinstance(ambiguous, pl.DataFrame) else True:
            ambiguous = ambiguous.with_columns(pl.lit(db_name).alias("db_name"))
        
        valid_dfs.append(valid)
        no_options_dfs.append(no_options)
        ambiguous_dfs.append(ambiguous)
    
    # Concatenate all dataframes
    valid_combined = pl.concat(valid_dfs, how="vertical")
    
    # Only concatenate non-empty dataframes
    no_options_combined = pl.concat(
        [df for df in no_options_dfs if (df.height > 0 if isinstance(df, pl.DataFrame) else True)],
        how="vertical"
    ) if any(df.height > 0 if isinstance(df, pl.DataFrame) else True for df in no_options_dfs) else \
        (pl.DataFrame().lazy() if lazy else pl.DataFrame())
    
    ambiguous_combined = pl.concat(
        [df for df in ambiguous_dfs if (df.height > 0 if isinstance(df, pl.DataFrame) else True)],
        how="vertical"
    ) if any(df.height > 0 if isinstance(df, pl.DataFrame) else True for df in ambiguous_dfs) else \
        (pl.DataFrame().lazy() if lazy else pl.DataFrame())
    
    if not lazy:
        valid_combined = valid_combined.collect() if isinstance(valid_combined, pl.LazyFrame) else valid_combined
        no_options_combined = no_options_combined.collect() if isinstance(no_options_combined, pl.LazyFrame) else no_options_combined
        ambiguous_combined = ambiguous_combined.collect() if isinstance(ambiguous_combined, pl.LazyFrame) else ambiguous_combined
    
    return valid_combined, no_options_combined, ambiguous_combined


def _apply_pipeline(
    data: pl.DataFrame | pl.LazyFrame,
    raw_fragment_tolerance_ppm: float,
    normalized_fragment_tolerance_ppm: float,
    molecular_ion_tolerance_ppm: float,
) -> pl.DataFrame | pl.LazyFrame:
    """
    Apply the common processing pipeline to raw spectral data.
    
    Pipeline steps:
    1. Filter metadata (remove QQ instruments, add instrument flags)
    2. Extract collision energy values
    3. Add precursor type indicators
    4. Annotate spectra (clean/normalize with formula annotation)
    5. Add molecular ion info
    6. Add spectral information score
    
    Args:
        data: Raw parsed spectral data
        raw_fragment_tolerance_ppm: Fragment annotation tolerance
        normalized_fragment_tolerance_ppm: Post-normalization tolerance
        molecular_ion_tolerance_ppm: Molecular ion matching tolerance
        
    Returns:
        Processed dataframe with standardized columns
    """
    # Apply pipeline steps
    data = filter_metadata(data)
    data = extract_collision_energy_values(data)
    data = add_precursor_type_indicators(data)
    data = annotate_spectra(
        data,
        raw_fragment_tolerance_ppm=raw_fragment_tolerance_ppm,
        normalized_fragment_tolerance_ppm=normalized_fragment_tolerance_ppm,
    )
    data = add_molecular_ion_info(data, tolerance_ppm=molecular_ion_tolerance_ppm)
    data = add_spectral_information_score(data)
    
    return data
