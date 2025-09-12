from hrms_utils.interfaces import get_chromatogram
from hrms_utils.formula_annotation import annotate_chromatogram_with_formulas
import polars as pl
from pathlib import Path

if __name__ == "__main__":
        
    # Resolve the chromatogram file relative to this script to fail fast if missing.
    chromatogram_path = Path(__file__).parent / "250515_006.txt"
    assert chromatogram_path.exists(), f"Required chromatogram file not found: {chromatogram_path}"

    chromatogram_df = get_chromatogram(str(chromatogram_path)).filter(
        pl.col("Height") > 2e6,
        pl.col("ms1_isotopes_m/z").is_not_null(),
        pl.col("msms_m/z").is_not_null(),
    )

    annotated_chromatogram = annotate_chromatogram_with_formulas(
        chromatogram_df,
        max_bounds={
            "C": 50,
            "H": 100,
            "O": 10,
            "N": 10,
            "S": 2,
            "P": 2,
        },
        precursor_mass_accuracy_ppm=3.0,
        fragment_mass_accuracy_ppm=5.0,
        isotopic_mass_accuracy_ppm=2.0,
        isotopic_intensity_relative_tolerance=0.05,
        isotopic_intensity_absolute_tolerance=1e6,
    )

    print(f"Number of annotated formulas: {annotated_chromatogram.filter(
        pl.col("decomposed_formulas").is_not_null()
    ).height}")
    print(f"number of peaks with any annotation: {annotated_chromatogram.filter(
        pl.col("decomposed_formulas").is_not_null()
    ).unique(subset=pl.col("Peak ID")).height}")
    print(annotated_chromatogram.select([
        "Peak ID", "Precursor_mz_MSDIAL", "decomposed_formulas","cleaned_spectrum_formulas"
    ]))
    print(annotated_chromatogram.head(1).select([
        "Peak ID", "Precursor_mz_MSDIAL", "decomposed_formulas","cleaned_spectrum_formulas", "cleaned_fragment_errors_ppm"
    ]).to_init_repr())
    print(annotated_chromatogram.schema)