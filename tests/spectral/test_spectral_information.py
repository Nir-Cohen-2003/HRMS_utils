import spectral_information
from hrms_utils.formats import get_chromatogram, annotate_chromatogram_with_formulas
import polars as pl
from pathlib import Path
import numpy as np

# Why: calculate spectral information scores using tree-based algorithm with formulas of precursor and fragments


if __name__ == "__main__":
    # get chromatogram data
    chromatogram_path = Path(__file__).parent.parent / "data" / "250515_006.txt"
    assert chromatogram_path.exists(), f"Required chromatogram file not found: {chromatogram_path}"

    chromatogram_df = get_chromatogram(str(chromatogram_path)).filter(
        pl.col("Height") > 2e6,
        pl.col("ms1_isotopes_m/z").is_not_null(),
        pl.col("msms_m/z").is_not_null(),
    )
    chromatogram_df = annotate_chromatogram_with_formulas(
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
            fragment_mass_accuracy_ppm=10.0,
            normalized_fragment_mass_accuracy_ppm=5.0,
            isotopic_mass_accuracy_ppm=2.0,
            isotopic_intensity_relative_tolerance=0.05,
            isotopic_intensity_absolute_tolerance=1e6,
        ).with_columns(
            explained_intensity=pl.col("cleaned_msms_intensity").list.sum().truediv(pl.col("msms_intensity").list.sum())
        ).with_columns(
            maximal_explained_intensity=pl.col("explained_intensity").max().over("Peak ID")
        )
    print(chromatogram_df.select([
        pl.col("explained_intensity").mean().alias("avg_explained_intensity"),
        pl.col("explained_intensity").std().alias("stddev_explained_intensity"),
        pl.col("explained_intensity").median().alias("median_explained_intensity"),
        pl.col("maximal_explained_intensity").mean().alias("avg_maximal_explained_intensity"),
        pl.col("maximal_explained_intensity").std().alias("stddev_maximal_explained_intensity"),
        pl.col("maximal_explained_intensity").median().alias("median_maximal_explained_intensity"),
    ]))
    chromatogram_df = chromatogram_df.filter(
            pl.col("explained_intensity") > 0.9,
            # Why: select only the decomposition with maximal explained intensity per Peak ID group
            pl.col("explained_intensity") == pl.col("maximal_explained_intensity")
        )
    

    # compute the spectral information scores
    chromatogram_df = chromatogram_df.with_columns(
        pl.struct(
            pl.col("precursor_formula"), 
            pl.col("cleaned_spectrum_formulas").alias("fragment_formulas")
        ).spectral_info.spectral_info_score(ignore_hydrogens=True).alias("spectral_info_score")
    )
    
    # Why: validate that score computation succeeded without producing invalid values
    assert not chromatogram_df.select(pl.col("spectral_info_score").is_nan().any()).item(), \
        "NaN scores produced by spectral_info_score_polars"
    
    # print summary statistics for the scores
    print(f"Spectral Information Score Summary Statistics for {chromatogram_df.height} features:")
    print(chromatogram_df.select([
        pl.col("spectral_info_score").min().alias("min_score"),
        pl.col("spectral_info_score").max().alias("max_score"),
        pl.col("spectral_info_score").mean().alias("avg_score"),
        pl.col("spectral_info_score").median().alias("median_score"),
        pl.col("spectral_info_score").std().alias("stddev_score"),
        pl.col("spectral_info_score").filter(pl.col("spectral_info_score") > 1).count().alias("number_over_1"),
    ]))

