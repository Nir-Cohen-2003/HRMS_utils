from hrms_utils.interfaces import get_chromatogram
from hrms_utils.formula_annotation import annotate_chromatogram_with_formulas
import polars as pl

if __name__ == "__main__":
        
    annotated_chromatogram = annotate_chromatogram_with_formulas(
        get_chromatogram(r"D:\Nir\MSDIAL_scratch\250120_04amph.txt").filter(
        pl.col("Height") > 2e6,
        pl.col("ms1_isotopes_m/z").is_not_null(),
        pl.col("msms_m/z").is_not_null(),

    )
    ,
        mass_accuracy_ppm=3.0,
        annotate_spectrum=True
    )

    print(annotated_chromatogram)
    print(annotated_chromatogram.filter(
        pl.col("Precursor_mz_MSDIAL").is_close(other=pl.lit(150.1277),rel_tol=7e-6)
    ).select(["decomposed_formulas","decomposed_spectra","msms_m/z","msms_intensity","Height"]).to_init_repr())