"""
MGF format parser - internal module.

This module provides internal parsing functions for MGF files.
The public API is available through spectral_library_reader.py.
"""

import polars as pl
from pathlib import Path
import re
from typing import List
from ..formula_annotation.utils import format_formula_string_to_array
from ..formula_annotation.element_table import NUM_ELEMENTS


def _parse_mgf_entries(
    mgf_path: str | Path,
    includes_MSn: bool = False
) -> pl.DataFrame:
    """
    Parse MGF file entries into a DataFrame with MSP-standard column names.
    
    This is an internal function used by the spectral library reader.
    Maps MGF field names to MSP-standard column names for compatibility.
    
    MGF to MSP column name mapping:
    - NAME -> name
    - PEPMASS -> precursor_mz
    - CHARGE -> (parsed for ion_mode)
    - MSLEVEL -> mslevel
    - FORMULA -> molecular_formula
    - INCHIAUX -> inchikey
    - INCHI -> inchi
    - SMILES -> smiles
    - EXACTMASS -> exact_mass
    - COLLISION_ENERGY -> collision_energy_raw
    - INSTRUMENT_TYPE -> instrument_type
    - INSTRUMENT -> instrument
    - IONMODE -> ion_mode (P/N)
    - ION_SOURCE -> ionization
    
    Args:
        mgf_path: Path to MGF file
        includes_MSn: Whether to parse MSn-specific fields
        
    Returns:
        DataFrame with parsed entries using MSP-standard column names
    """
    with open(mgf_path, 'r') as f:
        mgf_text = f.read()
        entries = re.findall(r'BEGIN IONS(.*?)END IONS', mgf_text, re.DOTALL)
    
    lf = pl.DataFrame({'entry': entries})
    del entries

    # MGF field names to MSP-standard column names
    field_mapping = {
        "NAME": "name",
        "DESCRIPTION": "description",
        "EXACTMASS": "exact_mass",
        "FORMULA": "molecular_formula",
        "INCHI": "inchi",
        "INCHIAUX": "inchikey",
        "SMILES": "smiles",
        "FEATURE_ID": "feature_id",
        "MSLEVEL": "mslevel",
        "RTINSECONDS": "rt_in_seconds",
        "ADDUCT": "adduct",
        "PEPMASS": "precursor_mz",
        "CHARGE": "charge_raw",
        "FEATURE_MS1_HEIGHT": "feature_ms1_height",
        "SPECTYPE": "spectype",
        "COLLISION_ENERGY": "collision_energy_raw",
        "FRAGMENTATION_METHOD": "fragmentation_method",
        "ISOLATION_WINDOW": "isolation_window",
        "ACQUISITION": "acquisition",
        "INSTRUMENT_TYPE": "instrument_type",
        "INSTRUMENT": "instrument",
        "SOURCE_INSTRUMENT": "source_instrument",
        "IMS_TYPE": "ims_type",
        "ION_SOURCE": "ionization",
        "IONMODE": "ion_mode_raw",
        "PI": "pi",
        "DATACOLLECTOR": "data_collector",
        "DATASET_ID": "dataset_id",
        "USI": "usi",
        "SCANS": "scans",
        "PRECURSOR_PURITY": "precursor_purity",
        "QUALITY_CHIMERIC": "quality_chimeric",
        "QUALITY_EXPLAINED_INTENSITY": "quality_explained_intensity",
        "QUALITY_EXPLAINED_SIGNALS": "quality_explained_signals",
        "Num peaks": "num_peaks",
    }
    
    exprs = []
    for mgf_key, msp_col in field_mapping.items():
        exprs.append(
            pl.col("entry").str.extract(rf"(?m)^{mgf_key}=(.+)$", 1).alias(msp_col)
        )
    
    # Extract spectrum data (mz and intensity pairs)
    exprs.append(
        pl.col("entry")
        .str.extract_all(r"(?m)^(\d+\.\d+)\s+(\d+\.\d+(?:[eE][+-]?\d+)?)$")
        .alias("mz_int_pairs")
    )
    
    lf = lf.with_columns(exprs).drop(["entry"]).lazy()
    
    # Convert ion_mode from "positive"/"negative" to "P"/"N"
    # and fill null spectype
    lf = lf.with_columns(
        pl.col("spectype").fill_null(value="SINGLE_BEST_SCAN"),
        pl.when(pl.col("ion_mode_raw").str.to_lowercase().eq("positive"))
        .then(pl.lit("P"))
        .when(pl.col("ion_mode_raw").str.to_lowercase().eq("negative"))
        .then(pl.lit("N"))
        .otherwise(pl.col("ion_mode_raw"))
        .alias("ion_mode")
    )
    
    # Cast columns to appropriate types
    lf = lf.cast(
        {
            "exact_mass": pl.Float64,
            "rt_in_seconds": pl.Float64,
            "precursor_mz": pl.Float64,
            "feature_ms1_height": pl.Float64,
            "mslevel": pl.Int64,
            "isolation_window": pl.Float64,
            "num_peaks": pl.Int64,
            "precursor_purity": pl.Float64,
            "quality_explained_intensity": pl.Float64,
            "quality_explained_signals": pl.Float64,
        }
    )
    
    # Extract spectrum mz and intensity from pairs
    lf = lf.with_columns(
        pl.col("mz_int_pairs")
        .list.eval(pl.element().str.split(by=" ").list.get(0).cast(pl.Float64))
        .alias("raw_spectrum_mz"),
        pl.col("mz_int_pairs")
        .list.eval(pl.element().str.split(by=" ").list.get(1).cast(pl.Float64))
        .alias("raw_spectrum_intensity"),
    ).drop(["mz_int_pairs"])

    # Parse molecular formula to array
    lf = lf.with_columns(
        pl.col("molecular_formula")
        .map_elements(format_formula_string_to_array, return_dtype=pl.List(pl.Int32))
        .list.to_array(width=NUM_ELEMENTS)
        .alias("molecular_formula_array")
    )
    
    # For precursor formula, we need to apply adduct to molecular formula
    # Default adducts: M+H for positive, M-H for negative
    lf = lf.with_columns(
        pl.when(pl.col("adduct").is_not_null())
        .then(pl.col("adduct"))
        .when(pl.col("ion_mode").eq("P"))
        .then(pl.lit("M+H"))
        .otherwise(pl.lit("M-H"))
        .str.replace("M", pl.col("molecular_formula"))
        .alias("precursor_formula")
    ).with_columns(
        pl.col("precursor_formula")
        .map_elements(format_formula_string_to_array, return_dtype=pl.List(pl.Int32))
        .list.to_array(width=NUM_ELEMENTS)
        .alias("precursor_formula_array")
    )
    
    # Parse collision energy - extract list if present
    lf = lf.with_columns(
        pl.col("collision_energy_raw")
        .str.strip_chars("[]")
        .str.split(by=",")
        .list.eval(pl.element().str.strip_chars(" "))
        .cast(pl.List(pl.Float64))
        .alias("collision_energy_list")
    ).with_columns(
        pl.when(pl.col("collision_energy_list").list.len() > 1)
        .then(pl.lit(True))
        .otherwise(pl.lit(False))
        .alias("multiple_collision_energies"),
        pl.col("collision_energy_list").list.mean().alias("collision_energy_mean")
    )
    
    # Add placeholder columns for instrument type flags (will be computed in pipeline)
    lf = lf.with_columns(
        pl.lit(None).alias("is_orbitrap"),
        pl.lit(None).alias("is_TOF"),
        pl.lit(None).alias("is_ESI"),
        pl.lit(None).alias("clean_precursor"),
    )
    
    # Extract base inchikey (first 14 characters)
    lf = lf.with_columns(
        pl.col("inchikey").str.extract(r"(.+?)-").alias("base_inchikey")
    )
    
    # Parse precursor_type from adduct or generate from ion_mode
    lf = lf.with_columns(
        pl.when(pl.col("adduct").is_not_null())
        .then(
            pl.when(pl.col("ion_mode").eq("P"))
            .then(pl.col("adduct").str.replace(r"^(.*)$", r"[$1]+"))
            .otherwise(pl.col("adduct").str.replace(r"^(.*)$", r"[$1]-"))
        )
        .when(pl.col("ion_mode").eq("P"))
        .then(pl.lit("[M+H]+"))
        .otherwise(pl.lit("[M-H]-"))
        .alias("precursor_type")
    )
    
    # --- MSn fields ---
    if includes_MSn:
        msn_expr = [
            pl.col("entry").str.extract(rf"(?m)^{mgf_key}=(.+)$", 1).alias(msp_col)
            for mgf_key, msp_col in [
                ("MSn_collision_energies", "MSn_collision_energies"),
                ("MSn_precursor_mzs", "MSn_precursor_mzs"),
                ("MSn_fragmentation_methods", "MSn_fragmentation_methods"),
                ("MSn_isolation_windows", "MSn_isolation_windows"),
            ]
        ]
        
        lf = lf.with_columns(msn_expr)
        
        lf = lf.with_columns(
            pl.when(pl.col("MSn_precursor_mzs").is_not_null())
            .then(
                pl.col("MSn_precursor_mzs")
                .str.strip_chars("[]")
                .str.split(",")
                .list.eval(pl.element().str.strip_chars(" ").cast(pl.Float64))
            )
            .alias("MSn_precursor_mzs"),
            
            pl.when(pl.col("MSn_fragmentation_methods").is_not_null())
            .then(
                pl.col("MSn_fragmentation_methods")
                .str.strip_chars("[]")
                .str.split(",")
                .list.eval(pl.element().str.strip_chars(" "))
            )
            .alias("MSn_fragmentation_methods"),
            
            pl.when(pl.col("MSn_isolation_windows").is_not_null())
            .then(
                pl.col("MSn_isolation_windows")
                .str.strip_chars("[]")
                .str.split(",")
                .list.eval(pl.element().str.strip_chars(" ").cast(pl.Float64))
            )
            .alias("MSn_isolation_windows"),
        )
    
    return lf.collect()


# Keep for backward compatibility during transition
# TODO: Remove these after all code is migrated to use spectral_library_reader
def _deprecated_read_mgf_to_dataframe(
        mgf_path: str | Path,
        includes_MSn: bool = False
        ) -> pl.DataFrame:
    """Deprecated: Use spectral_library_reader.read_spectral_library() instead."""
    import warnings
    warnings.warn(
        "read_mgf_to_dataframe is deprecated. Use spectral_library_reader.read_spectral_library() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _parse_mgf_entries(mgf_path, includes_MSn)


if __name__ == "__main__":
    # Example usage for testing
    from time import perf_counter
    start_time = perf_counter()
    mgf_file = Path("/home/analytit_admin/Data/MSnLib/20241003_enammol_pos_msn.mgf")
    df = _parse_mgf_entries(mgf_path=mgf_file)
    end_time = perf_counter()
    print(df)
    print(df.schema)
    print(df["mslevel"].value_counts(sort=True))
    print(df["spectype"].value_counts(sort=True))
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Number of entries: {df.height}")
    print(f'time per entry: {(end_time - start_time) / df.height:.8f} seconds')
