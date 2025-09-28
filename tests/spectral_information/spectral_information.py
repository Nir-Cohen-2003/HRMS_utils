from hrms_utils.spectral_information import spectral_info_polars, tree_spectral_info_score_polars
from hrms_utils.formats import get_chromatogram, annotate_chromatogram_with_formulas
import polars as pl
from pathlib import Path
import numpy as np

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
            chromatogram_df,  # clone to ensure each run sees the same input state
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
    
    # calculate spectral information scores, we need the formuals of the precursor and the fragments
    def calculate_scores(s: pl.Series) -> pl.Series:
        precursors = s.struct.field("decomposed_formulas")
        fragments = s.struct.field("cleaned_spectrum_formulas")
        
        score1 = spectral_info_polars(precursors, fragments)
        score2 = tree_spectral_info_score_polars(precursors, fragments, distance_metric="l2")
        
        return pl.struct([score1.alias("spectral_info_score"), score2.alias("tree_spectral_info_score")])

    chromatogram_df = chromatogram_df.with_columns(
        pl.struct(["decomposed_formulas", "cleaned_spectrum_formulas"])
          .map_batches(function=calculate_scores,return_dtype=pl.Struct([
              pl.Field("spectral_info_score", pl.Float64),
              pl.Field("tree_spectral_info_score", pl.Float64),
          ]))
          .alias("scores"),
        explained_intensity=pl.col("cleaned_msms_intensity").list.sum()
    ).unnest("scores")

    # Analysis for spectral_info_score
    in_group_correlation = chromatogram_df.filter(
        pl.col("explained_intensity").is_not_null(),
        pl.col("explained_intensity") > 0,
        pl.col("spectral_info_score").is_not_null(),
    ).group_by("Peak ID").agg(
        pl.corr("explained_intensity", "spectral_info_score").alias("in_group_corr")
    ).select(pl.col("in_group_corr")).filter(
        pl.col("in_group_corr").is_not_null(),
        pl.col("in_group_corr").is_not_nan(),
    ).select(pl.col("in_group_corr").mean()).item()
    print(f"Correlation between explained intensity and spectral info score: {in_group_correlation:.4f}")
    print(f"Average spectral info score: {chromatogram_df.select(pl.col('spectral_info_score').mean()).item():.4f}")
    print(f"Average score (highest explained intensity): {chromatogram_df.sort(by='explained_intensity', descending=True).group_by(by='Peak ID').first().select(pl.col('spectral_info_score').mean()).item():.4f}")
    print("-" * 30)

    # Analysis for tree_spectral_info_score
    tree_in_group_correlation = chromatogram_df.filter(
        pl.col("explained_intensity").is_not_null(),
        pl.col("explained_intensity") > 0,
        pl.col("tree_spectral_info_score").is_not_null(),
        pl.col("tree_spectral_info_score") > 0, # Scores can be 0, which breaks correlation
    ).group_by("Peak ID").agg(
        pl.corr("explained_intensity", "tree_spectral_info_score").alias("in_group_corr")
    ).select(pl.col("in_group_corr")).filter(
        pl.col("in_group_corr").is_not_null(),
        pl.col("in_group_corr").is_not_nan(),
    ).select(pl.col("in_group_corr").mean()).item()
    print(f"Correlation between explained intensity and tree spectral info score: {tree_in_group_correlation:.4f}")
    print(f"Average tree spectral info score: {chromatogram_df.select(pl.col('tree_spectral_info_score').mean()).item():.4f}")
    print(f"Average tree score (highest explained intensity): {chromatogram_df.sort(by='explained_intensity', descending=True).group_by(by='Peak ID').first().select(pl.col('tree_spectral_info_score').mean()).item():.4f}")