from hrms_utils.interfaces import get_chromatogram
from hrms_utils.formula_annotation import annotate_chromatogram_with_formulas
import polars as pl

if __name__ == "__main__":
        
    annotated_chromatogram = annotate_chromatogram_with_formulas(
        get_chromatogram(r"/home/analytit_admin/Data/raw_data/meth/250120_04amph.txt").filter(
        pl.col("Height") > 2e6,
        pl.col("ms1_isotopes_m/z").is_not_null(),
        pl.col("msms_m/z").is_not_null(),
    ),
    max_bounds={
        "C": 50,
        "H": 100,
        "O": 10,
        "N": 10,
        "S": 2,
        "P":2
    },
    precursor_mass_accuracy_ppm=3.0,
    fragment_mass_accuracy_ppm=5.0,
    isotopic_mass_accuracy_ppm=2.0,
    )

    print(f"Number of annotated formulas: {annotated_chromatogram.filter(
        pl.col("decomposed_formulas").is_not_null()
    ).height}")
    print(f"number of peaks with any annotation: {annotated_chromatogram.filter(
        pl.col("decomposed_formulas").is_not_null()
    ).unique(subset=pl.col("Peak ID")).height}")
    print(annotated_chromatogram.filter(
        pl.col("Precursor_mz_MSDIAL").is_close(other=pl.lit(150.1277),rel_tol=7e-6)
    ).select(["decomposed_formulas","decomposed_spectra_formulas","msms_m/z","msms_intensity","Height"]).to_init_repr())
    print(annotated_chromatogram.filter(
        pl.col("Precursor_mz_MSDIAL").is_close(other=pl.lit(150.1277),rel_tol=7e-6)
    ).select(["cleaned_spectrum_formulas","cleaned_msms_mz","cleaned_msms_intensity"]).to_init_repr())