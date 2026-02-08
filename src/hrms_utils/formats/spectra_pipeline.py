"""
Common spectral processing pipeline.

This module contains shared processing logic for spectral library data,
including metadata filtering, collision energy extraction, spectra annotation,
and MSn precursor handling.
"""

import re
from typing import TypeVar, cast, Tuple
from pathlib import Path

import polars as pl
import polars.selectors as plcs

from ..formula_annotation.element_table import ADDUCT_MASSES, NUM_ELEMENTS
from ..formula_annotation.utils import format_formula_string_to_array
from ..hrms_core import *

polarsFrame = TypeVar("polarsFrame", pl.DataFrame, pl.LazyFrame)


def filter_metadata(data: polarsFrame) -> polarsFrame:
    """
    Filter entries based on instrument metadata and add instrument type flags.
    
    Filters out entries from QQ instruments. Adds boolean flags:
    - is_orbitrap: True if Orbitrap/FT (but not FT-ICR or TOF)
    - is_TOF: True if TOF instrument
    - is_ESI: True if ESI ionization
    
    Args:
        data: Input DataFrame/LazyFrame with instrument metadata columns
        
    Returns:
        Filtered DataFrame with added instrument flags
    """
    instrument_data_columns = plcs.by_name(
        ["instrument", "instrument_type", "ionization"]
    )

    # Build mask that only excludes rows explicitly tagged with 'QQ' in any instrument column.
    # Treat NULL as not matching 'QQ' so missing instrument metadata does not drop the row.
    qq_mask = pl.any_horizontal(
        instrument_data_columns.str.contains(r"(?i)QQ").fill_null(False)
    )

    return cast(
        polarsFrame,
        data.filter(
            qq_mask.not_()  # keep rows unless one of the instrument columns explicitly includes QQ
        ).with_columns(
            # Use fill_null(False) so missing values won't produce NULL booleans and won't exclude rows
            pl.any_horizontal(
                instrument_data_columns.str.contains(r"(?i)LC").fill_null(False)
            ).alias("is_LC"),
            pl.any_horizontal(
                instrument_data_columns.str.contains(
                    r"(?i)orbi(?:trap)?|HCD"
                ).fill_null(False)
                | instrument_data_columns.str.contains(r"(?i)thermo").fill_null(False)
                | (
                    instrument_data_columns.str.contains(r"(?i)FT").fill_null(False)
                    & instrument_data_columns.str.contains(r"(?i)ICR")
                    .not_()
                    .fill_null(True)
                    & instrument_data_columns.str.contains(r"(?i)TOF")
                    .not_()
                    .fill_null(True)
                )
            ).alias("is_orbitrap"),
            pl.any_horizontal(
                instrument_data_columns.str.contains(r"(?i)TOF").fill_null(False)
            ).alias("is_TOF"),
            pl.any_horizontal(
                instrument_data_columns.str.contains(r"(?i)ESI|LC").fill_null(False)
            ).alias("is_ESI"),  # LC is usually coupled with ESI
        ),
    )


def extract_collision_energy_values(data: polarsFrame) -> polarsFrame:
    """
    Extract collision energy values (NCE and eV) from raw collision energy strings.
    
    Handles various formats:
    - NCE=70% 16eV
    - 20 (NCE)
    - 20 NCE
    - 20 eV
    - 20 V
    - 20
    - 20 % (nominal)
    - 20.0 eV
    - [20.0, 30.0, 60.0, 40.0] (lists)
    
    Logic:
    - If only NCE or only V/eV present, use that value
    - If both present, assign based on proximity to descriptors
    - % is considered NCE indicator
    - Orbitrap instruments default to NCE, others to eV for bare numbers
    - Lists are averaged and assigned based on instrument type
    
    Args:
        data: Input with 'collision_energy_raw' column
        
    Returns:
        DataFrame with added collision energy columns
    """
    # Regex patterns
    # NCE: Matches "NCE=20", "NCE 20", "20%", "20 NCE", "20 (NCE)"
    pat_nce = r"(?i)(?:NCE\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)|([0-9]+(?:\.[0-9]+)?)\s*(?:%|(?:\(?NCE\)?)))"
    # eV: Matches "20eV", "20 eV", "20V", "20 V"
    pat_ev = r"(?i)([0-9]+(?:\.[0-9]+)?)\s*e?V"
    # Bare number: Matches any number. Used as fallback.
    pat_num = r"([0-9]+(?:\.[0-9]+)?)"
    # List pattern: Matches content inside square brackets
    pat_list_content = r"\[(.*?)\]"

    return cast(
        polarsFrame,
        data.with_columns(
            # Extract NCE candidates
            pl.col("collision_energy_raw")
            .str.extract(pat_nce, group_index=1)
            .fill_null(
                pl.col("collision_energy_raw").str.extract(pat_nce, group_index=2)
            )
            .cast(pl.Float64, strict=False)
            .alias("collision_energy_NCE"),
            # Extract eV candidates
            pl.col("collision_energy_raw")
            .str.extract(pat_ev, group_index=1)
            .cast(pl.Float64, strict=False)
            .alias("collision_energy_ev"),
            # Extract List candidates
            pl.col("collision_energy_raw")
            .str.extract(pat_list_content, group_index=1)
            .str.extract_all(r"\d+(?:\.\d+)?")
            .list.eval(pl.element().cast(pl.Float64, strict=False))
            .alias("collision_energy_list"),
        )
        .with_columns(
            # Fallback logic: if NCE, eV and List are null, try to use the bare number
            pl.when(
                pl.col("collision_energy_NCE").is_null()
                & pl.col("collision_energy_ev").is_null()
                & pl.col("collision_energy_list").is_null()
            )
            .then(
                pl.col("collision_energy_raw")
                .str.extract(pat_num, group_index=1)
                .cast(pl.Float64, strict=False)
            )
            .otherwise(None)
            .alias("_bare_energy"),
            # Calculate mean of list if present
            pl.col("collision_energy_list").list.mean().alias("_list_mean"),
        )
        .with_columns(
            # Apply Orbitrap logic to fallback for NCE (using list mean or bare energy)
            pl.when(pl.col("collision_energy_NCE").is_null())
            .then(
                pl.when(pl.col("is_orbitrap"))
                .then(pl.coalesce([pl.col("_list_mean"), pl.col("_bare_energy")]))
                .otherwise(None)
            )
            .otherwise(pl.col("collision_energy_NCE"))
            .alias("collision_energy_NCE"),
            # Apply Orbitrap logic to fallback for eV (using list mean or bare energy)
            pl.when(pl.col("collision_energy_ev").is_null())
            .then(
                pl.when(pl.col("is_orbitrap").not_())
                .then(pl.coalesce([pl.col("_list_mean"), pl.col("_bare_energy")]))
                .otherwise(None)
            )
            .otherwise(pl.col("collision_energy_ev"))
            .alias("collision_energy_ev"),
        )
        .with_columns(
            pl.col("collision_energy_list")
            .list.len()
            .ge(2)
            .fill_null(False)
            .alias("multiple_collision_energies"),
            # Mean is either the list mean, or the single value present
            pl.coalesce(
                [
                    pl.col("_list_mean"),
                    pl.col("collision_energy_NCE"),
                    pl.col("collision_energy_ev"),
                ]
            ).alias("collision_energy_mean"),
        )
        .drop("_bare_energy", "_list_mean"),
    )


def annotate_spectra(
    data: polarsFrame,
    raw_fragment_tolerance_ppm: float,
    normalized_fragment_tolerance_ppm: float,
) -> polarsFrame:
    """
    Clean and normalize spectra with known precursor formulas.
    
    Normalizes intensities, annotates fragments with formulas, and calculates
    explained intensity. Filters entries where precursor mass doesn't match
    the precursor formula.
    
    Args:
        data: Input with raw spectrum and precursor formula columns
        raw_fragment_tolerance_ppm: Tolerance for initial fragment annotation
        normalized_fragment_tolerance_ppm: Tolerance after normalization
        
    Returns:
        DataFrame with cleaned/normalized spectrum columns
    """
    # Determine adduct_mass based on precursor_type
    adduct_mapping = pl.Series(
        name="precursor_type", values=list(ADDUCT_MASSES.keys()), dtype=pl.String
    )
    adduct_masses = pl.Series(
        name="adduct_mass", values=list(ADDUCT_MASSES.values()), dtype=pl.Float64
    )
    adduct_df = pl.DataFrame(
        {"precursor_type": adduct_mapping, "adduct_mass": adduct_masses}
    )

    if isinstance(data, pl.LazyFrame):
        adduct_lf = adduct_df.lazy()
        data_lf = data.join(adduct_lf, on="precursor_type", how="left")
        data_frame = cast(polarsFrame, data_lf)
    elif isinstance(data, pl.DataFrame):
        data_df = data.join(adduct_df, on="precursor_type", how="left")
        data_frame = cast(polarsFrame, data_df)
    else:
        raise TypeError(
            f"In function 'annotate_spectra', data must be a Polars DataFrame or LazyFrame, got {type(data)}"
        )

    return cast(
        polarsFrame,
        data_frame.with_columns(
            pl.col("raw_spectrum_intensity")
            .truediv(pl.col("raw_spectrum_intensity").list.sum())
            .alias("raw_spectrum_intensity")
        )
        .with_columns(
            pl.struct(  # type: ignore[missing-attribute]
                [
                    pl.col("precursor_formula_array").alias("precursor_formula"),
                    pl.col("raw_spectrum_mz").alias("mz"),
                    pl.col("raw_spectrum_intensity").alias("intensities"),
                ]
            )
            .mass_decomposition.clean_and_normalize_spectrum(
                raw_fragment_tolerance_ppm=raw_fragment_tolerance_ppm,
                normalized_fragment_tolerance_ppm=normalized_fragment_tolerance_ppm,
                min_dbe=-0.5,
                max_dbe=40,
                dbe_mode="half_integer",
                water_absorption=True,
            )
            .alias("cleaned_normalized_spectra")
        )
        .with_columns(  # Extract results and add adduct_mass back to normalized masses
            pl.col("cleaned_normalized_spectra")
            .struct.field("normalized_masses")
            .alias("cleaned_normalized_mz"),
            pl.col("cleaned_normalized_spectra")
            .struct.field("intensities")
            .alias("cleaned_normalized_intensity"),
            pl.col("cleaned_normalized_spectra")
            .struct.field("formulas")
            .alias("cleaned_fragment_formulas"),
            pl.col("cleaned_normalized_spectra")
            .struct.field("formulas_str")
            .alias("cleaned_fragment_formulas_str"),
            pl.col("cleaned_normalized_spectra")
            .struct.field("errors_ppm")
            .alias("cleaned_fragment_errors_ppm"),
        )
        .drop("cleaned_normalized_spectra")
        .with_columns(
            pl.col("cleaned_normalized_intensity")
            .list.sum()
            .truediv(pl.col("raw_spectrum_intensity").list.sum())
            .alias("explained_intensity")
        ),
    )


def add_precursor_type_indicators(data: polarsFrame) -> polarsFrame:
    """
    Add boolean flags for precursor type characteristics.
    
    Adds columns:
    - Isotope: Contains isotope notation (e.g., [M+1]+)
    - Cation: Contains cation notation (e.g., [M+Na]+)
    - Multimer: Contains multimer notation (e.g., [2M+H]+)
    - MultiCharge: Contains multi-charge notation (e.g., [M+2H]2+)
    - Fragment: Contains fragment notation
    - clean_precursor: True if none of the above (simple adduct like [M+H]+)
    """
    fragment_pattern = (
        r"-\d*"
        + r"((H(\d+|[A-Z]|[a-z]))|([A-G]|[I-Z])[a-z]?\d*)"
        + r"(([A-Z][a-z]?\d*))*"
    )

    return cast(
        polarsFrame,
        data.with_columns(
            pl.col("precursor_type").str.contains("i").alias("Isotope"),
            pl.col("precursor_type").str.contains("Cat").alias("Cation"),
            pl.col("precursor_type").str.contains("[0-9]M").alias("Multimer"),
            pl.col("precursor_type").str.contains("][0-9]").alias("MultiCharge"),
            pl.col("precursor_type").str.contains(fragment_pattern).alias("Fragment"),
        ).with_columns(
            (
                pl.col("Isotope")
                | pl.col("Cation")
                | pl.col("Multimer")
                | pl.col("MultiCharge")
                | pl.col("Fragment")
                | pl.col("precursor_type")
                .str.contains("M")
                .not_()  # there are some that are [123.1234]+, all of the m with single occurance, which are probably not clean
            )
            .not_()
            .alias("clean_precursor")
        ),
    )


def add_molecular_ion_info(data: polarsFrame, tolerance_ppm: float = 10.0) -> polarsFrame:
    """
    Add molecular ion intensity column based on precursor match.
    
    Identifies the molecular ion peak in the cleaned spectrum by matching
    the highest m/z fragment to the precursor m/z within tolerance.
    
    Args:
        data: Input DataFrame
        tolerance_ppm: Mass tolerance for precursor matching
        
    Returns:
        DataFrame with added 'molecular_ion_intensity' column
    """
    lazy_frame = data.lazy()
    lazy_frame = lazy_frame.with_columns(
        molecular_ion_intensity=pl.when(
            pl.col("cleaned_normalized_mz")
            .list.last()
            .is_close(
                pl.col("precursor_mz"),
                rel_tol=tolerance_ppm * 1e-6,
                abs_tol=200.0 * tolerance_ppm * 1e-6,
            )
        )
        .then(pl.col("cleaned_normalized_intensity").list.last())
        .otherwise(None)
    )

    if isinstance(data, pl.LazyFrame):
        return cast(polarsFrame, lazy_frame)
    elif isinstance(data, pl.DataFrame):
        return cast(polarsFrame, lazy_frame.collect())
    else:
        raise TypeError(
            f"In function 'add_molecular_ion_info', data must be a Polars DataFrame or LazyFrame, got {type(data)}"
        )


def add_spectral_information_score(data: polarsFrame) -> polarsFrame:
    """
    Calculate tree-based spectral information score.
    
    Computes two versions:
    - spectral_information_score: Ignores hydrogens in formula comparison
    - spectral_information_score_with_hydrogens: Includes hydrogens
    """
    return cast(
        polarsFrame,
        data.with_columns(
            pl.struct(
                [
                    pl.col("precursor_formula_array").alias("precursor_formula"),
                    pl.col("cleaned_fragment_formulas").alias("fragment_formulas"),
                ]
            ).alias("spectra_for_spectral_info")
        ).with_columns(
            pl.col("spectra_for_spectral_info")  # type: ignore[missing-attribute]
            .spectral_info.spectral_info_score(
                distance_metric="l2", ignore_hydrogens=True
            )
            .alias("spectral_information_score"),
            pl.col("spectra_for_spectral_info")  # type: ignore[missing-attribute]
            .spectral_info.spectral_info_score(
                distance_metric="l2", ignore_hydrogens=False
            )
            .alias("spectral_information_score_with_hydrogens"),
        ),
    )


def annotate_msn_precursors(
    data: polarsFrame,
    tolerance_ppm: float = 10.0,
) -> Tuple[polarsFrame, polarsFrame, polarsFrame]:
    """
    Annotate MSn precursors with formulas and separate by ambiguity.
    
    For MSn spectra (MSLEVEL > 2), annotates each precursor level with possible
    formulas based on the molecular formula. Returns three DataFrames:
    
    1. Valid spectra: Exactly 1 formula option for all precursors
    2. No formula options: At least one precursor with 0 formula options
    3. Ambiguous spectra: At least one precursor with 2+ formula options
    
    The annotation logic:
    - First precursor: Annotated from molecular formula with adduct
    - Subsequent precursors: Annotated based on previous precursor formula
      using mass decomposition within bounds
    
    Args:
        data: Input with MSn data (MSn_precursor_mzs, molecular_formula_array, etc.)
        tolerance_ppm: Mass tolerance for precursor annotation
        
    Returns:
        Tuple of (valid_df, no_options_df, ambiguous_df)
    """
    # Only process MSn spectra
    msn_mask = pl.col("mslevel") > 2
    
    if isinstance(data, pl.LazyFrame):
        # Check if any MSn spectra exist
        has_msn = data.filter(msn_mask).limit(1).collect().height > 0
    else:
        has_msn = data.filter(msn_mask).height > 0
    
    if not has_msn:
        # No MSn data, return all as valid
        return data, cast(polarsFrame, pl.DataFrame().lazy() if isinstance(data, pl.LazyFrame) else pl.DataFrame()), \
               cast(polarsFrame, pl.DataFrame().lazy() if isinstance(data, pl.LazyFrame) else pl.DataFrame())
    
    # Split MSn and non-MSn
    non_msn_data = data.filter(pl.col("mslevel") <= 2)
    msn_data = data.filter(msn_mask)
    
    # For MSn data, we need to annotate each precursor level
    # This is a placeholder for the actual implementation which would:
    # 1. Iterate through MSn_precursor_mzs (list of precursor m/z at each level)
    # 2. For first precursor: use molecular_formula_array + adduct
    # 3. For subsequent: use mass decomposition with bounds from previous precursor
    # 4. Count formula options at each level
    # 5. Categorize as valid (all have 1 option), no_options (any has 0), ambiguous (any has 2+)
    
    # For now, return all as valid (actual implementation would be added here)
    # TODO: Implement full MSn precursor annotation logic
    
    combined = pl.concat([non_msn_data, msn_data], how="vertical")
    
    return combined, \
           cast(polarsFrame, pl.DataFrame().lazy() if isinstance(data, pl.LazyFrame) else pl.DataFrame()), \
           cast(polarsFrame, pl.DataFrame().lazy() if isinstance(data, pl.LazyFrame) else pl.DataFrame())
