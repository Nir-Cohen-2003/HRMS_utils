from hrms_utils.spectral_information import spectral_info_polars, tree_spectral_info_score_polars
from hrms_utils.formats import get_chromatogram, annotate_chromatogram_with_formulas
import polars as pl
from pathlib import Path
import numpy as np

# calculate spectral information scores, we need the formuals of the precursor and the fragments


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
        ).with_columns(
            explained_intensity=pl.col("cleaned_msms_intensity").list.sum()
        ).filter(
            pl.col("explained_intensity") > 0.0,
            pl.col("explained_intensity") == pl.col("explained_intensity").max().over("Peak ID") # select only the decomposition with the maximnal explained intensity, over the Peak ID groups
        )
    

    # compute the spectral information scores
    chromatogram_df = chromatogram_df.with_columns(
        pl.struct(["decomposed_formulas", "cleaned_spectrum_formulas"])
          .map_batches(
              function=lambda s: spectral_info_polars(s.struct.field("decomposed_formulas"), s.struct.field("cleaned_spectrum_formulas")),
              return_dtype=pl.Float64).alias("spectral_space_info_score"))
    # check if any nan scores were produced
    assert not chromatogram_df.select(pl.col("spectral_space_info_score").is_nan().any()).item(), "NaN scores produced by spectral_info_polars"
    chromatogram_df = chromatogram_df.with_columns(
        pl.struct(["decomposed_formulas", "cleaned_spectrum_formulas"])
          .map_batches(
              function=lambda s: tree_spectral_info_score_polars(s.struct.field("decomposed_formulas"), s.struct.field("cleaned_spectrum_formulas")),
              return_dtype=pl.Float64).alias("spectral_tree_info_score"),
          )
    # check if any nan scores were produced
    assert not chromatogram_df.select(pl.col("spectral_tree_info_score").is_nan().any()).item(), "NaN scores produced by tree_spectral_info_score_polars"
    # pritn summary on the scores: min/max, avg, stddev
    print(chromatogram_df.select([
        pl.col("spectral_space_info_score").min().alias("min_space_score"),
        pl.col("spectral_space_info_score").max().alias("max_space_score"),
        pl.col("spectral_space_info_score").mean().alias("avg_space_score"),
        pl.col("spectral_space_info_score").std().alias("stddev_space_score"),
        pl.col("spectral_tree_info_score").min().alias("min_tree_score"),
        pl.col("spectral_tree_info_score").max().alias("max_tree_score"),
        pl.col("spectral_tree_info_score").mean().alias("avg_tree_score"),
        pl.col("spectral_tree_info_score").std().alias("stddev_tree_score"),
    ]))
    # find the correlation between the two scores
    print(chromatogram_df.select([
        pl.col("spectral_space_info_score"),
        pl.col("spectral_tree_info_score"),
    ]).corr())

    