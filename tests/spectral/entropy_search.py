import polars as pl
from pathlib import Path
import numpy as np
import timeit
import sys

from hrms_utils.formats.msdial import get_chromatogram
from hrms_utils.spectral_search.entropy_search import entropy_score_polars_external
from spectral_similarity import calculate_similarity


def test_entropy_search_parity(
        ms1_ppm_tolerance: float = 5.0,
        ms2_ppm_tolerance: float = 5.0,
        noise_threshold: float = 0.001,
    ):
    # Resolve the chromatogram file relative to this script to fail fast if missing.
    chromatogram_path = Path(__file__).parent.parent / "data" / "250120_04amph.txt"
    assert chromatogram_path.exists(), f"Required chromatogram file not found: {chromatogram_path}"

    chromatogram_df = get_chromatogram(chromatogram_path).filter(
        pl.col("msms_m/z").is_not_null(),
        pl.col("msms_intensity").list.len() > 3,
        # pl.col("Precursor_mz_MSDIAL") > 300
    )
    print(f"Loaded chromatogram with {chromatogram_df.height:,} spectra")

    df_sorted = chromatogram_df.sort("Precursor_mz_MSDIAL").select(
        ["Precursor_mz_MSDIAL", "msms_m/z", "msms_intensity"]
    )
    
    # join_asof for tolerance join
    joined_df = df_sorted.join_where(
        df_sorted,
        pl.col("Precursor_mz_MSDIAL") > pl.col("Precursor_mz_MSDIAL_right") * (1 - ms1_ppm_tolerance / 1e6),
        pl.col("Precursor_mz_MSDIAL") < pl.col("Precursor_mz_MSDIAL_right") * (1 + ms1_ppm_tolerance / 1e6),
        suffix="_right",
    )
    
    if len(sys.argv) > 1:
        try:
            n_repeats = int(sys.argv[1])
            print(f"Repeating dataframe {n_repeats} times.")
            joined_df = pl.concat([joined_df] * n_repeats, rechunk=True)
        except ValueError:
            print("Invalid number of repeats, using original dataframe.")

    print(f"Running entropy search parity test on {joined_df.height:,} spectrum pairs")
    
    # Ensure there are no nulls in key columns before creating the struct
    joined_df = joined_df.filter(
        pl.col("msms_m/z").is_not_null() &
        pl.col("msms_intensity").is_not_null() &
        pl.col("msms_m/z_right").is_not_null() &
        pl.col("msms_intensity_right").is_not_null()
    )
    
    joined_df = joined_df.with_columns(
        struct_col = pl.struct([
            pl.col("msms_m/z").cast(pl.List(inner=pl.Float32)).alias("mz1"),
            pl.col("msms_intensity").cast(pl.List(inner=pl.Float32)).alias("intensities1"),
            pl.col("msms_m/z_right").cast(pl.List(inner=pl.Float32)).alias("mz2"),
            pl.col("msms_intensity_right").cast(pl.List(inner=pl.Float32)).alias("intensities2"),
        ],
    ))
    # print(joined_df)

    # Call the external python entropy score function
    start_time = timeit.default_timer()
    results_df = joined_df.with_columns(
        pl.col("struct_col").map_batches(
            lambda batch: entropy_score_polars_external(
                spec1_mz=batch.struct.field("mz1"),
                spec1_intensity=batch.struct.field("intensities1"),
                spec2_mz=batch.struct.field("mz2"),
                spec2_intensity=batch.struct.field("intensities2"),
                ms2_mass_tolerance=ms2_ppm_tolerance,
                noise_threshold=noise_threshold,
            ),
            return_dtype=pl.Float64,
    ).alias("external_results"))

    external_time = timeit.default_timer() - start_time
    print(f"External entropy search time: {external_time:.4f} s")

    # Call the rust-based similarity function
    start_time = timeit.default_timer()
    
    results_df = results_df.with_columns(
        rust_similarity=calculate_similarity(
            pl.col("struct_col"),
            ms2_tolerance_in_ppm=ms2_ppm_tolerance,
            noise_threshold=noise_threshold,
            clean_spectra_first=True
        )
    )
   
    rust_time = timeit.default_timer() - start_time
    print(f"Rust-based similarity time: {rust_time:.4f} s")
    results_df = results_df.with_columns(
        abs_score_difference=(pl.col("external_results") - pl.col("rust_similarity")).abs()
    ).with_columns(
        scores_equal=pl.col("abs_score_difference") < 1e-3,
        scores_close = pl.col("abs_score_difference") < 1e-2,
        scores_similar = pl.col("abs_score_difference") < 5e-2,
    )
    # Compare the results
    assert results_df.height > 0, "No results to compare"
    assert results_df.select([pl.col("external_results"), pl.col("rust_similarity")]).null_count().equals(pl.DataFrame({"external_results": [0], "rust_similarity": [0]})), "Null values found in results"
    score_corr = results_df.select(pl.col(["external_results", "rust_similarity"])).corr().item(0,1)
    assert score_corr > 0.99, f"Low correlation between implementations: {score_corr}"
    print(f"The correlation between the two implementations is: {score_corr}, which is above the threshold of 0.99.")
    num_unequal = results_df.filter(~pl.col("scores_equal")).height
    unequal_corr = results_df.filter(~pl.col("scores_equal")).select(["external_results", "rust_similarity"]).corr().item(0,1)
    num_non_close = results_df.filter(~pl.col("scores_close")).height
    non_close_corr = results_df.filter(~pl.col("scores_close")).select(["external_results", "rust_similarity"]).corr().item(0,1)
    num_non_similar = results_df.filter(~pl.col("scores_similar")).height
    non_similar_corr = results_df.filter(~pl.col("scores_similar")).select(["external_results", "rust_similarity"]).corr().item(0,1)
    print(f"Number of unequal scores (>1e-3 difference): {num_unequal}")
    print(f"Correlation of unequal scores: {unequal_corr}")
    print(f"Number of non-close scores (>1e-2 difference): {num_non_close}")
    print(f"Correlation of non-close scores: {non_close_corr}")
    print(f"Number of non-similar scores (>5e-2 difference): {num_non_similar}")
    print(f"Correlation of non-similar scores: {non_similar_corr}")
    # assert num_unequal == 0, f"Found {num_unequal} unequal scores between implementations, max difference: {results_df.select(pl.col('score_difference').max()).item()}"

    # assert results_df.select(pl.all("scores_equal")).item(), "Not all scores are equal"
    # # print("All entropy similarity scores match between implementations")

if __name__ == "__main__":
    print("Running entropy search parity test...")
    test_entropy_search_parity()
    print("Test completed successfully.")
