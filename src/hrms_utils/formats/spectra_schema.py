"""
Dataframely schema for standardized spectral library data.

This module defines the standard schema for spectral library dataframes,
enforcing consistent column names and types across all supported formats (MSP, MGF).
"""

from typing import Optional
import polars as pl
import dataframely as dy

from ..formula_annotation.element_table import NUM_ELEMENTS


class SpectralLibrarySchema(dy.Schema):
    """
    Standard schema for spectral library dataframes.
    
    This schema enforces the common column structure across all supported
    spectral library formats (MSP/MGF). Columns are based on MSP naming conventions.
    
    Note on MSn data: For MSn spectra (MSLEVEL > 2), additional optional columns
    may be present for precursor tracking at each fragmentation level.
    """
    
    # Core identification
    name: dy.String()
    nist_id: dy.Int64(nullable=True)
    db_id: dy.Int64(nullable=True)
    
    # Instrument metadata
    instrument_type: dy.String(nullable=True)
    instrument: dy.String(nullable=True)
    ionization: dy.String(nullable=True)
    ion_mode: dy.Enum(["P", "N"], nullable=True)  # P=positive, N=negative
    mslevel: dy.Int64()
    
    # Collision energy information
    collision_energy_NCE: dy.Float64(nullable=True)
    collision_energy_ev: dy.Float64(nullable=True)
    collision_energy_list: dy.List(dy.Float64(), nullable=True)
    multiple_collision_energies: dy.Bool()
    collision_energy_mean: dy.Float64(nullable=True)
    
    # Compound identifiers
    cas: dy.String(nullable=True)
    inchikey: dy.String(nullable=True)
    base_inchikey: dy.String(nullable=True)
    smiles: dy.String(nullable=True)
    inchi: dy.String(nullable=True)
    
    # Instrument flags
    is_orbitrap: dy.Bool()
    is_TOF: dy.Bool()
    is_ESI: dy.Bool()
    
    # Precursor information
    precursor_type: dy.String(nullable=True)
    precursor_mz: dy.Float64()
    molecular_formula: dy.String(nullable=True)
    molecular_formula_array: dy.Array(dy.Int32(), NUM_ELEMENTS, nullable=True)
    precursor_formula_array: dy.Array(dy.Int32(), NUM_ELEMENTS, nullable=True)
    clean_precursor: dy.Bool()
    exact_mass: dy.Float64(nullable=True)
    
    # Raw spectrum data (nested)
    raw_spectrum_mz: dy.List(dy.Float64())
    raw_spectrum_intensity: dy.List(dy.Float64())
    
    # Cleaned/normalized spectrum data (nested)
    cleaned_normalized_mz: dy.List(dy.Float64())
    cleaned_normalized_intensity: dy.List(dy.Float64())
    cleaned_fragment_formulas: dy.List(dy.Array(dy.Int32(), NUM_ELEMENTS), nullable=True)
    cleaned_fragment_formulas_str: dy.List(dy.String(), nullable=True)
    cleaned_fragment_errors_ppm: dy.List(dy.Float64(), nullable=True)
    
    # Quality metrics
    explained_intensity: dy.Float64(nullable=True)
    molecular_ion_intensity: dy.Float64(nullable=True)
    spectral_information_score: dy.Float64(nullable=True)
    spectral_information_score_with_hydrogens: dy.Float64(nullable=True)
    
    # MSn-specific fields (optional, for MSLEVEL > 2)
    # These track the fragmentation cascade
    MSn_precursor_mzs: dy.List(dy.Float64(), nullable=True)  # Precursor m/z at each level
    MSn_collision_energies: dy.List(dy.Float64(), nullable=True)  # CE at each level
    MSn_fragmentation_methods: dy.List(dy.String(), nullable=True)  # Method at each level


def validate_spectral_library(df: pl.DataFrame | pl.LazyFrame) -> bool:
    """
    Validate a dataframe against the spectral library schema.
    
    Only validates that required columns exist with correct types.
    Extra columns are allowed.
    
    Args:
        df: DataFrame or LazyFrame to validate
        
    Returns:
        True if valid, raises SchemaError otherwise
    """
    return SpectralLibrarySchema.validate(df, strict=False)
