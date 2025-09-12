import polars as pl
from typing import List, Tuple, Dict
import numpy as np
from numba import njit, jit
from .isotopic_pattern import deduce_isotopic_pattern
from .mass_decomposition import decompose_mass_per_bounds, clean_and_normalize_spectra_known_precursor, NUM_ELEMENTS
from .element_table import ELEMENT_INDEX, ELEMENT_MASSES

PROTON_MASS = ELEMENT_MASSES[ELEMENT_INDEX['H']]

def annotate_chromatogram_with_formulas(
    chromatogram: pl.DataFrame,
    addcut_mass: float = PROTON_MASS,
    max_bounds: dict|None = None,
    precursor_mass_accuracy_ppm: float = 3.0,
    fragment_mass_accuracy_ppm: float = 5.0,
    normalized_fragment_mass_accuracy_ppm: float = 4.0,
    isotopic_mass_accuracy_ppm: float = 2.0,
    isotopic_minimum_intensity: float = 5e4,
    isotopic_intensity_absolute_tolerance: float = 5e5,
    isotopic_intensity_relative_tolerance: float = 0.05,
) -> pl.DataFrame:
    # Isotopic pattern deduction
    chromatogram = chromatogram.with_columns(
        pl.struct(
            ["Precursor_mz_MSDIAL", "ms1_isotopes_m/z", "ms1_isotopes_intensity"]
        ).map_batches(
            lambda batch: deduce_isotopic_pattern(
                batch.struct.field("Precursor_mz_MSDIAL"),
                batch.struct.field("ms1_isotopes_m/z"),
                batch.struct.field("ms1_isotopes_intensity"),
                ms1_mass_tolerance_ppm=precursor_mass_accuracy_ppm,
                isotopic_mass_tolerance_ppm=isotopic_mass_accuracy_ppm,
                minimum_intensity=isotopic_minimum_intensity,
                intensity_absolute_tolerance=isotopic_intensity_absolute_tolerance,
                intensity_relative_tolerance=isotopic_intensity_relative_tolerance,
                max_bounds=max_bounds,
            ),
            return_dtype=pl.Array(inner=pl.Int32, shape=(2*NUM_ELEMENTS,))
        ).alias("bounds")
    ).with_columns(
        pl.col("bounds").arr.slice(0, length=NUM_ELEMENTS).list.to_array(width=NUM_ELEMENTS).alias("min_bounds"),
        pl.col("bounds").arr.slice(NUM_ELEMENTS, length=NUM_ELEMENTS).list.to_array(width=NUM_ELEMENTS).alias("max_bounds")
    )

    # Mass decomposition
    chromatogram = chromatogram.with_columns(
        non_ionized_mass = pl.col("Precursor_mz_MSDIAL") - addcut_mass
    ).with_columns(
        pl.struct(
            ["non_ionized_mass", "min_bounds", "max_bounds"]
        ).map_batches(
            lambda batch: decompose_mass_per_bounds(
                batch.struct.field("non_ionized_mass"),
                batch.struct.field("min_bounds"),
                batch.struct.field("max_bounds"),
                tolerance_ppm=precursor_mass_accuracy_ppm,
            ),
            return_dtype=pl.List(pl.Array(inner=pl.Int32, shape=(NUM_ELEMENTS,)))
        ).alias("decomposed_formulas")
    )
    
    chromatogram = chromatogram.with_columns(pl.col("msms_m/z").sub(addcut_mass).alias("non_ionized_msms_m/z"))
    chromatogram = chromatogram.explode("decomposed_formulas")
    
    # Cleaning + normalization (updated API requires observed precursor mass series)
    chromatogram = chromatogram.with_columns(
        pl.struct(["decomposed_formulas", "non_ionized_mass", "non_ionized_msms_m/z", "msms_intensity"]).map_batches(
            lambda batch: clean_and_normalize_spectra_known_precursor(
                precursor_formula_series=batch.struct.field("decomposed_formulas"),
                precursor_masses_series=batch.struct.field("non_ionized_mass"),
                fragment_masses_series=batch.struct.field("non_ionized_msms_m/z"),
                fragment_intensities_series=batch.struct.field("msms_intensity"),
                tolerance_ppm=fragment_mass_accuracy_ppm,
                max_allowed_normalized_mass_error_ppm=normalized_fragment_mass_accuracy_ppm,
            ),
            return_dtype=pl.Struct({
                "masses_normalized": pl.List(pl.Float64),
                "intensities": pl.List(pl.Float64),
                "fragment_formulas": pl.List(pl.Array(inner=pl.Int32, shape=(NUM_ELEMENTS,))),
                "fragment_errors_ppm": pl.List(pl.Float64),
            }),
        ).alias("cleaned_spectra")
    ).with_columns(
        pl.col("cleaned_spectra").struct.unnest()
    ).rename(
        {
            "masses_normalized": "cleaned_msms_mz",
            "intensities": "cleaned_msms_intensity",
            "fragment_formulas": "cleaned_spectrum_formulas",
            "fragment_errors_ppm": "cleaned_fragment_errors_ppm",
        }
    ).drop(
        "cleaned_spectra"
    )
    return chromatogram