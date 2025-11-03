
import polars as pl
from pathlib import Path
import numpy as np
import pytest
import spectral_similarity # This is needed to register the plugin

# --- User-provided function ---
def _read_SRCRESLT_type_file(file_path : str | Path ) -> pl.DataFrame:
    if isinstance(file_path,str):
        file_path = Path(file_path)
    
    # Handle file not found
    if not file_path.exists():
        # We will return an empty dataframe with the expected schema if the file doesn't exist,
        # and the test will be skipped.
        print(f"Warning: SRCRESLT file not found at {file_path}. Skipping test.")
        return pl.DataFrame({
            'Peak ID': pl.Series([], dtype=pl.Int64),
            'DotProd': pl.Series([], dtype=pl.Int64),
            'DB_ID': pl.Series([], dtype=pl.Int64),
        })

    with open(file_path, mode='r',encoding='ANSI',errors="ignore") as srcrsults:
        search_results = srcrsults.read()
    
    search_results = search_results.split('\nUnknown: ')
    if search_results:
        search_results[0] = search_results[0].strip("Unknown: ")

    if not search_results or (len(search_results) == 1 and not search_results[0]):
        return pl.DataFrame({
            'Peak ID': pl.Series([], dtype=pl.Int64),
            'DotProd': pl.Series([], dtype=pl.Int64),
            'DB_ID': pl.Series([], dtype=pl.Int64),
        })

    NIST_results = pl.DataFrame(search_results,schema={'raw':pl.String})
    NIST_results = NIST_results.with_columns(
        pl.col('raw').str.extract(pattern=r"^(\d+)",group_index=1).alias('Peak ID'))
    
    NIST_results = NIST_results.with_columns(
        pl.col('raw').str.split(by='\n').alias('raw')
    )
    NIST_results = NIST_results.explode('raw')

    NIST_results = NIST_results.with_columns(
        pl.col('raw').str.extract(pattern=r"; MF:(\s+)(\d+);",group_index=2).cast(pl.Int64)
            .alias('Score'),
        pl.col('raw').str.extract(pattern=r"; RMF:(\s+)(\d+);",group_index=2).cast(pl.Int64)
            .alias('DotProd'),
        pl.col('raw').str.extract(pattern=r"Id:(\s+)(\d+)\.",group_index=2).cast(pl.Int64)
            .alias('DB_ID'),
    )
    
    # The user asked not to filter the results, so the call to filter_NIST_results is removed.
    return NIST_results.select(['Peak ID', 'DotProd', 'DB_ID']).drop_nulls()

# --- Test Implementation ---

# --- !! IMPORTANT !! ---
# Please replace this placeholder with the actual path to your SRCRESLT file.
# The test will be skipped if this file does not exist.
NIST_SRCRESLT_FILE = "path/to/your/nist_results.txt" 

NIST_DB_PATH = "/home/analytit_admin/Data/NIST_hr_msms/NIST23.parquet"

@pytest.mark.skipif(not Path(NIST_SRCRESLT_FILE).exists() or not Path(NIST_DB_PATH).exists(), reason=f"NIST data files not found. Searched for {NIST_SRCRESLT_FILE} and {NIST_DB_PATH}")
def test_dotprod_similarity_against_nist():
    """
    Compares the dotprod_similarity function with NIST's own results.
    """
    # 1. Read NIST search results
    nist_results_df = _read_SRCRESLT_type_file(NIST_SRCRESLT_FILE)

    if nist_results_df.is_empty():
        pytest.skip(f"No data found in {NIST_SRCRESLT_FILE}")

    # 2. Extract unique NIST IDs to fetch from the main database
    query_ids = nist_results_df["Peak ID"].unique().to_list()
    db_ids = nist_results_df["DB_ID"].unique().to_list()
    all_nist_ids = list(set(query_ids + db_ids))

    # 3. Load NIST spectra database and filter for required spectra
    nist_spectra_db = pl.read_parquet(NIST_DB_PATH).filter(pl.col("NIST_ID").is_in(all_nist_ids))

    # 4. Prepare the DataFrame for comparison by joining spectra for query and library
    spectra_for_join = nist_spectra_db.select(
        pl.col("NIST_ID"),
        pl.col("raw_spectrum_mz"),
        pl.col("raw_spectrum_intensity")
    )

    # Join for query spectra
    df_with_query_spectra = nist_results_df.join(
        spectra_for_join.rename({"NIST_ID": "Peak ID", "raw_spectrum_mz": "mz1", "raw_spectrum_intensity": "intensities1"}),
        on="Peak ID",
        how="inner",
    )

    # Join for matched spectra
    df_with_all_spectra = df_with_query_spectra.join(
        spectra_for_join.rename({"NIST_ID": "DB_ID", "raw_spectrum_mz": "mz2", "raw_spectrum_intensity": "intensities2"}),
        on="DB_ID",
        how="inner",
    )

    # 5. Calculate dot-product similarity using the spectral-similarity plugin
    # Create the struct column required by the plugin
    comparison_df = df_with_all_spectra.with_columns(
        pl.struct([
            pl.col("mz1"),
            pl.col("intensities1"),
            pl.col("mz2"),
            pl.col("intensities2"),
        ]).alias("spectra_pair_struct")
    )

    # Apply the dotprod_similarity function
    # Using a tolerance of 20ppm as a reasonable default for HRMS data
    result_df = comparison_df.with_columns(
        pl.col("spectra_pair_struct").spectral.dotprod_similarity(
            ms2_tolerance_in_ppm=20.0,
            clean_spectra_first=True, # NIST likely uses cleaned spectra
            noise_threshold=0.01, # Common noise threshold
        ).alias("my_dotprod_score")
    )

    # 6. Calculate the difference between our score and NIST's score
    final_results = result_df.with_columns(
        (pl.col("DotProd") / 999.0).alias("nist_dotprod_score") # NIST score is out of 999, not 1000
    ).with_columns(
        (pl.col("my_dotprod_score") - pl.col("nist_dotprod_score")).abs().alias("difference")
    )

    # 7. Calculate statistics and assert
    max_diff = final_results["difference"].max()
    mean_diff = final_results["difference"].mean()
    median_diff = final_results["difference"].median()

    print(f"Max difference: {max_diff}")
    print(f"Mean difference: {mean_diff}")
    print(f"Median difference: {median_diff}")

    # Assertions: These thresholds are examples and may need to be adjusted
    # based on the specific dataset and cleaning parameters.
    assert max_diff < 0.1, "Maximum difference is too high"
    assert mean_diff < 0.05, "Mean difference is too high"
    assert median_diff < 0.05, "Median difference is too high"

