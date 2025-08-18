import polars as pl
from .isotopic_pattern import deduce_isotopic_pattern
from .mass_decomposition import decompose_mass_per_bounds, decompose_spectra_known_precursor
from .element_table import ELEMENT_INDEX, ELEMENT_MASSES

PROTON_MASS = ELEMENT_MASSES[ELEMENT_INDEX['H']]

def annotate_chromatogram_with_formulas(
    chromatogram: pl.DataFrame,
    max_bounds: dict|None = None,
    precursor_mass_accuracy_ppm: float = 3.0,
    fragment_mass_accuracy_ppm: float = 5.0,
    isotopic_mass_accuracy_ppm: float = 2.0,
    annotate_spectrum: bool = False
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
                intensity_relative_tolerance=0.05,
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
        non_protonated_mass = pl.col("Precursor_mz_MSDIAL") - PROTON_MASS
    ).with_columns(
        pl.struct(
            ["non_protonated_mass", "min_bounds", "max_bounds"]
        ).map_batches(
            lambda batch: decompose_mass_per_bounds(
                batch.struct.field("non_protonated_mass"),
                batch.struct.field("min_bounds"),
                batch.struct.field("max_bounds"),
                tolerance_ppm=precursor_mass_accuracy_ppm,
            ),
            return_dtype=pl.List(pl.Array(inner=pl.Int32, shape=(15,)))
        ).alias("decomposed_formulas")
    )
    

    if annotate_spectrum:
        chromatogram = chromatogram.with_columns(pl.col("msms_m/z").sub(PROTON_MASS).alias("non_protonated_msms_m/z"))
        chromatogram = chromatogram.explode("decomposed_formulas")
        
        chromatogram = chromatogram.with_columns(
            pl.struct(["non_protonated_msms_m/z", "decomposed_formulas"]).map_batches(
                lambda row: decompose_spectra_known_precursor(
                    precursor_formula_series=row.struct.field("decomposed_formulas"),
                    fragment_masses_series=row.struct.field("non_protonated_msms_m/z"),
                    tolerance_ppm=fragment_mass_accuracy_ppm,
                ), return_dtype=pl.List(pl.List(pl.Array(inner=pl.Int32, shape=(15,))))
            ).alias("decomposed_spectra")
        )
    return chromatogram

if __name__ == "__main__":
    from ..interfaces import get_chromatogram

    annotated_chromatogram = annotate_chromatogram_with_formulas(
        get_chromatogram(),
        mass_accuracy_ppm=3.0,
        annotate_spectrum=True
    )
    
    print(annotated_chromatogram)