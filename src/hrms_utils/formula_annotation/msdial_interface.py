import polars as pl
from typing import List, Tuple, Dict
import numpy as np
from numba import njit, jit
from .isotopic_pattern import deduce_isotopic_pattern
from .mass_decomposition import decompose_mass_per_bounds, decompose_spectra_known_precursor
from .element_table import ELEMENT_INDEX, ELEMENT_MASSES

PROTON_MASS = ELEMENT_MASSES[ELEMENT_INDEX['H']]

def annotate_chromatogram_with_formulas(
    chromatogram: pl.DataFrame,
    addcut_mass: float = PROTON_MASS,
    max_bounds: dict|None = None,
    precursor_mass_accuracy_ppm: float = 3.0,
    fragment_mass_accuracy_ppm: float = 5.0,
    isotopic_mass_accuracy_ppm: float = 2.0,
    isotopic_minimum_intensity: float = 5e4,
    isotopic_intensity_absolute_tolerance: float = 5e5,
    isotopic_intensity_relative_tolerance: float = 0.05,
) -> pl.DataFrame:
    """
    Annotate chromatogram with possible formulas using isotopic pattern deduction and mass decomposition. If annotate_spectrum is True, also annotate the fragments, and then you get each possible formula as a new line.

    Args:
        chromatogram (pl.DataFrame): Input chromatogram.
        base_bounds (dict): Minimum element bounds, e.g. {'C': 1}.
        max_bounds (dict): Maximum element bounds, e.g. {'C': 50, 'H': 100, ...}.
        precursor_mass_accuracy_ppm (float): Precursor mass accuracy in ppm—the mass accuracy for the MS1.
        fragment_mass_accuracy_ppm (float): Fragment mass accuracy in ppm—the mass accuracy for the MS2.
        isotopic_mass_accuracy_ppm (float): Isotope mass accuracy in ppm—the mass accuracy of the difference between the monoisotopic peak and the isotopic peak. Note it can be much smaller than the MS1 mass accuracy, since we get a cancellation of errors.
        annotate_spectrum (bool): Whether to annotate the fragments, based on their precursor.

    Returns:
        pl.DataFrame: Chromatogram with 'decomposed_formulas' column.
    """
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
            return_dtype=pl.Array(inner=pl.Int32, shape=(30,))
        ).alias("bounds")
    ).with_columns(
        pl.col("bounds").arr.slice(0, length=15).list.to_array(width=15).alias("min_bounds"),
        pl.col("bounds").arr.slice(15, length=15).list.to_array(width=15).alias("max_bounds")
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
            return_dtype=pl.List(pl.Array(inner=pl.Int32, shape=(15,)))
        ).alias("decomposed_formulas")
    )
    
    chromatogram = chromatogram.with_columns(pl.col("msms_m/z").sub(addcut_mass).alias("non_ionized_msms_m/z"))
    chromatogram = chromatogram.explode("decomposed_formulas")
    
    chromatogram = chromatogram.with_columns(
        pl.struct(["non_ionized_msms_m/z", "decomposed_formulas"]).map_batches(
            lambda row: decompose_spectra_known_precursor(
                precursor_formula_series=row.struct.field("decomposed_formulas"),
                fragment_masses_series=row.struct.field("non_ionized_msms_m/z"),
                tolerance_ppm=fragment_mass_accuracy_ppm,
            ), return_dtype=pl.List(pl.List(pl.Array(inner=pl.Int32, shape=(15,))))
        ).list.eval(pl.element().list.first()).alias("decomposed_spectra_formulas") # the type of this is now pl.List(pl.Array(inner=pl.Int32, shape=(15,)))
    ).with_columns(
        pl.struct(["decomposed_spectra_formulas", "msms_m/z", "msms_intensity"]).map_batches(
            function=lambda batch: _clean_spectrum(
                decomposed_spectra_formulas=batch.struct.field("decomposed_spectra_formulas"),
                msms_mz=batch.struct.field("msms_m/z"),
                msms_intensity=batch.struct.field("msms_intensity")
            ),
            return_dtype=pl.Struct({
                "cleaned_spectrum_formulas": pl.List(pl.Array(inner=pl.Int32, shape=(15,))),
                "cleaned_msms_mz": pl.List(pl.Float32),
                "cleaned_msms_intensity": pl.List(pl.Float32)
            }),is_elementwise=True
        ).alias("cleaned_spectra")
    ).with_columns(
        pl.col("cleaned_spectra").struct.unnest()
    ).drop(
        "cleaned_spectra"
    )
    return chromatogram



def _clean_spectrum(
        decomposed_spectra_formulas: pl.Series,
        msms_mz: pl.Series,
        msms_intensity: pl.Series
        ):
    '''cleans the raw spectrum based on the spectrum decomposition results, removing fragments that were not assigned any formula'''
    decomposed_spectra_formulas = decomposed_spectra_formulas.to_list()
    # print(type(decomposed_spectra_formulas[0])) # list
    # print(type(decomposed_spectra_formulas[2][0])) # NoneType | list/np.ndarray?
    msms_mz = msms_mz.to_list()
    msms_intensity = msms_intensity.to_list()
    cleaned_spectra = []


    for i in range(len(decomposed_spectra_formulas)):
        cleaned_spectrum_formulas, cleaned_msms_mz, cleaned_msms_intensity = _clean_spectrum_single(
            decomposed_spectra_formulas[i],
            msms_mz[i],
            msms_intensity[i]
        )
        cleaned_spectra.append({
            "cleaned_spectrum_formulas": cleaned_spectrum_formulas,
            "cleaned_msms_mz": cleaned_msms_mz,
            "cleaned_msms_intensity": cleaned_msms_intensity
        })
    # print(cleaned_spectra[0])
    cleaned_spectra = pl.Series(
        values=cleaned_spectra,
        dtype=pl.Struct({
            "cleaned_spectrum_formulas": pl.List(pl.Array(inner=pl.Int32, shape=(15,))),
            "cleaned_msms_mz": pl.List(pl.Float32),
            "cleaned_msms_intensity": pl.List(pl.Float32)
        })
    )
    # print(cleaned_spectra[0])
    return cleaned_spectra

# @njit(nogil=True)
# @jit()
def _clean_spectrum_single(
        decomposed_spectra_formulas: List[np.typing.NDArray],
        msms_mz: List[float],
        msms_intensity: List[float]
) -> Tuple[List[np.typing.NDArray], List[float], List[float]]:
    '''cleans the raw spectrum based on the spectrum decomposition results, removing fragments that were not assigned any formula'''
    num_raw_fragments = len(decomposed_spectra_formulas)
    assert num_raw_fragments == len(msms_mz) == len(msms_intensity)

    clean_spectrum_formulas = []
    clean_spectrum_mz = []
    clean_spectrum_intensity = []
    for i in range(num_raw_fragments):
        if decomposed_spectra_formulas[i] is not None:
            clean_spectrum_formulas.append(decomposed_spectra_formulas[i])
            clean_spectrum_mz.append(msms_mz[i])
            clean_spectrum_intensity.append(msms_intensity[i])


    return clean_spectrum_formulas, clean_spectrum_mz, clean_spectrum_intensity