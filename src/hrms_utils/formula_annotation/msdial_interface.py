import polars as pl
from .isotopic_pattern import deduce_isotopic_pattern
from .mass_decomposition import decompose_mass_per_bounds, decompose_spectra_known_precursor

def annotate_chromatogram_with_formulas(
    chromatogram: pl.DataFrame,
    base_bounds: dict,
    max_bounds: dict,
    mass_accuracy_ppm: float = 3.0,
    annotate_spectrum: bool = False
) -> pl.DataFrame:
    """
    Annotate chromatogram with possible formulas using isotopic pattern deduction and mass decomposition. if annotate_spectrum is True, also annotate the fragments, and then you get the each possible formula as a new line.

    Args:
        chromatogram (pl.DataFrame): Input chromatogram.
        base_bounds (dict): Minimum element bounds, e.g. {'C': 1}.
        max_bounds (dict): Maximum element bounds, e.g. {'C': 50, 'H': 100, ...}.
        mass_accuracy_ppm (float): Mass accuracy in ppm.
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
                mass_tolerance_ppm=mass_accuracy_ppm,
                intensity_relative_tolerance=0.05,
                min_bounds=base_bounds,
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
        non_protonated_mass = pl.col("Precursor_mz_MSDIAL") - 1.007825
    ).with_columns(
        pl.struct(
            ["non_protonated_mass", "min_bounds", "max_bounds"]
        ).map_batches(
            lambda batch: decompose_mass_per_bounds(
                batch.struct.field("non_protonated_mass"),
                batch.struct.field("min_bounds"),
                batch.struct.field("max_bounds"),
                tolerance_ppm=mass_accuracy_ppm,
            ),
            return_dtype=pl.List(pl.Array(inner=pl.Int32, shape=(15,)))
        ).alias("decomposed_formulas")
    )

    if annotate_spectrum:
        chromatogram = chromatogram.with_columns(
            pl.struct(["msms_m/z", "decomposed_formula"]).map_batches(
                lambda row: decompose_spectra_known_precursor(
                    precursor_formula_series=row.struct.field("decomposed_formula"),
                    fragment_masses_series=row.struct.field("msms_m/z"),
                tolerance_ppm=5.0,
            )).alias("decomposed_spectra")
        )
    return chromatogram