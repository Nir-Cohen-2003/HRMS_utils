import polars as pl
from pathlib import Path
import numpy as np
import pytest
import spectral_similarity # Why: needed to register the plugin

def _read_SRCRESLT_type_file(file_path: str | Path) -> pl.DataFrame:
    """
    Parse NIST SRCRESLT file format into a Polars DataFrame.
    
    Why: SRCRESLT files contain NIST search results with query IDs, match scores, 
    and library references that need to be extracted via regex patterns.
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
    
    # Why: fail fast if required resource is missing
    assert file_path.exists(), f"SRCRESLT file not found at {file_path}. Cannot proceed with test."

    with open(file_path, mode='r', errors="ignore") as srcrsults:
        search_results = srcrsults.read()
    
    search_results = search_results.split('\nUnknown: ')
    if search_results:
        search_results[0] = search_results[0].strip("Unknown: ")

    assert search_results and not (len(search_results) == 1 and not search_results[0]), \
        f"No valid search results found in {file_path}"

    nist_results = pl.DataFrame(search_results, schema={'raw': pl.String})
    nist_results = nist_results.with_columns(
        pl.col('raw').str.extract(pattern=r"^(\d+)", group_index=1).cast(pl.Int64).alias('query_ID')
    )
    
    nist_results = nist_results.with_columns(
        pl.col('raw').str.split(by='\n').alias('raw')
    )
    nist_results = nist_results.explode('raw')

    nist_results = nist_results.with_columns(
        pl.col('raw').str.extract(pattern=r"; MF:(\s+)(\d+);", group_index=2).cast(pl.Int64).alias('Score'),
        pl.col('raw').str.extract(pattern=r"; RMF:(\s+)(\d+);", group_index=2).cast(pl.Int64).alias('DotProd'),
        pl.col('raw').str.extract(pattern=r"Id:(\s+)(\d+)\.", group_index=2).cast(pl.Int64).alias('DB_ID'),
        # Why: extract DB_Name to join with parquet DB_Name column for mapping DB_ID to NIST_ID
        pl.col('raw').str.extract(pattern=r"Lib:\s+<<(.+?)>>", group_index=1).alias('DB_Name'),
    )
    
    return nist_results.select(['query_ID', 'DotProd', 'DB_ID', 'DB_Name']).drop_nulls()


NIST_SRCRESLT_FILE = "/home/analytit_admin/Data/nist_vs_nist/NIST_results_1154442_1188674.txt"
NIST_DB_PATH = "/home/analytit_admin/Data/NIST_hr_msms/NIST23.parquet"

@pytest.mark.skipif(
    not Path(NIST_SRCRESLT_FILE).exists() or not Path(NIST_DB_PATH).exists(), 
    reason=f"NIST data files not found. Searched for {NIST_SRCRESLT_FILE} and {NIST_DB_PATH}"
)
def test_dotprod_similarity_against_nist() -> None:
    """
    Validate dotprod_similarity function against NIST reference implementation.
    
    Why: NIST is the gold standard for spectral matching. This test ensures our implementation
    produces results within acceptable tolerance of NIST's own dot-product scores.
    """
    # 1. Read NIST search results (contains query_ID, DB_ID, DB_Name, DotProd)
    nist_results_df: pl.DataFrame = _read_SRCRESLT_type_file(NIST_SRCRESLT_FILE)
    assert not nist_results_df.is_empty(), f"No data found in {NIST_SRCRESLT_FILE}"

    # 2. Load NIST spectra database and filter out QTOF instruments
    # Why: QTOF instruments have different peak shapes and mass accuracy characteristics
    # that may introduce systematic differences in dot-product calculations
    nist_spectra_db: pl.DataFrame = pl.read_parquet(NIST_DB_PATH)
    nist_spectra_db = nist_spectra_db.filter(
        ~pl.col("Instrument_type").str.contains(r"(?i)TOF")
    )
    
    assert not nist_spectra_db.is_empty(), \
        f"No spectra remaining after filtering QTOF instruments from {NIST_DB_PATH}"
    
    # 3. Get query spectra by joining on NIST_ID
    # Why: query_ID in search results corresponds to NIST_ID in the database
    query_spectra: pl.DataFrame = nist_results_df.select("query_ID").unique().join(
        nist_spectra_db.select(["NIST_ID", "raw_spectrum_mz", "raw_spectrum_intensity"]),
        left_on="query_ID",
        right_on="NIST_ID",
        how="inner"
    ).rename({
        "raw_spectrum_mz": "mz1",
        "raw_spectrum_intensity": "intensities1"
    })

    # 4. Join query spectra to results
    results_with_query_spectra: pl.DataFrame = nist_results_df.join(
        query_spectra,
        on="query_ID",
        how="inner"
    )

    # 5. Get library hit spectra using DB_ID and DB_Name
    # Why: DB_ID is library-specific identifier; must join on both DB_ID and DB_Name
    # to correctly map to NIST_ID in the parquet database
    results_with_all_spectra: pl.DataFrame = results_with_query_spectra.join(
        # include the library NIST_ID so we can report it for the hit with largest difference
        nist_spectra_db.select([
            "DB_ID",
            "DB_Name",
            "NIST_ID",  # <- include original NIST identifier for the library entry
            "raw_spectrum_mz",
            "raw_spectrum_intensity"
        ]).rename({
            "NIST_ID": "library_NIST_ID",               # avoid collision with query NIST_ID
            "raw_spectrum_mz": "mz2",
            "raw_spectrum_intensity": "intensities2"
        }),
        on=["DB_ID", "DB_Name"],
        how="inner"
    )

    assert not results_with_all_spectra.is_empty(), \
        "No matching spectra found after joining. Check DB_ID and DB_Name mapping or QTOF filtering may have removed all matches."

    # 6. Calculate dot-product similarity using spectral_similarity plugin
    comparison_df: pl.DataFrame = results_with_all_spectra.with_columns(
        pl.struct([
            pl.col("mz1"),
            pl.col("intensities1"),
            pl.col("mz2"),
            pl.col("intensities2"),
        ]).alias("spectra_pair_struct")
    )

    result_df: pl.DataFrame = comparison_df.with_columns(
        pl.col("spectra_pair_struct").spectral.dotprod_similarity(
            ms2_tolerance_in_ppm=12.0,
            clean_spectra_first=False,
            # noise_threshold=0.01,
        ).alias("my_dotprod_score")
    )

    # 7. Compare with NIST scores
    # Why: NIST reports dot-product scores as integers out of 999, not normalized to 1.0
    final_results: pl.DataFrame = result_df.with_columns(
        (pl.col("DotProd") / 999.0).alias("nist_dotprod_score")
    ).with_columns(
        (pl.col("my_dotprod_score") - pl.col("nist_dotprod_score")).abs().alias("difference")
    )

    # 8. Calculate statistics and validate
    max_diff: float = final_results["difference"].max()
    mean_diff: float = final_results["difference"].mean()
    median_diff: float = final_results["difference"].median()

    # Why: identify the specific spectrum pair with maximum difference for debugging
    max_diff_row: pl.DataFrame = final_results.filter(pl.col("difference") == max_diff)
    
    print(f"Max absolute difference: {max_diff}")
    print(f"Mean absolute difference: {mean_diff}")
    print(f"Median absolute difference: {median_diff}")
    print(f"\nRow with maximum difference:")
    print(f"Query NIST_ID: {max_diff_row['query_ID'][0]}")
    print(f"Library NIST_ID: {max_diff_row['library_NIST_ID'][0]} (via DB_ID={max_diff_row['DB_ID'][0]}, DB_Name={max_diff_row['DB_Name'][0]})")
    print(f"NIST DotProd score: {max_diff_row['nist_dotprod_score'][0]:.6f}")
    print(f"Our DotProd score: {max_diff_row['my_dotprod_score'][0]:.6f}")
    print(f"Difference: {max_diff_row['difference'][0]:.6f}")

    # Print spectra for the pair with the largest difference to aid debugging.
    # Why: inspect raw mz/intensity arrays to see what drove the discrepancy.
    query_mz = max_diff_row["mz1"][0]
    query_intensities = max_diff_row["intensities1"][0]
    library_mz = max_diff_row["mz2"][0]
    library_intensities = max_diff_row["intensities2"][0]

    # Print lengths and first N pairs to avoid enormous output while still being informative.
    N_PREVIEW = 20
    print(f"Query spectrum length: {len(query_mz)}")
    print("Query spectrum (mz, intensity) preview:",
          list(zip(query_mz, query_intensities))[:N_PREVIEW])
    print(f"Library spectrum length: {len(library_mz)}")
    print("Library spectrum (mz, intensity) preview:",
          list(zip(library_mz, library_intensities))[:N_PREVIEW])

    # Print error distribution using numpy histogram instead of asserting thresholds
    # Why: provide full error distribution for inspection rather than failing the test immediately
    differences: np.ndarray = final_results["difference"].to_numpy()  # shape: (n_matches,)
    counts, bin_edges = np.histogram(differences, bins=10)
    print("Error distribution histogram counts:", counts)
    print("Error distribution histogram bin edges:", bin_edges)

    # Provide a few summary percentiles for quick inspection, including 90th and 99th percentiles
    percentiles_values = np.percentile(differences, [0, 25, 50, 75, 90, 95, 99, 100])
    percentile_labels = ["0th", "25th", "50th", "75th", "90th", "95th", "99th", "100th"]
    print("Difference percentiles:")
    for label, val in zip(percentile_labels, percentiles_values):
        print(f"  {label}: {val:.6f}")

    # Why: these thresholds validate our implementation matches NIST within acceptable bounds
    # for HRMS spectral matching (accounting for minor differences in peak cleaning/alignment)
    assert max_diff < 0.1, f"Maximum difference {max_diff} exceeds threshold of 0.1"
    assert mean_diff < 0.05, f"Mean difference {mean_diff} exceeds threshold of 0.05"
    assert median_diff < 0.05, f"Median difference {median_diff} exceeds threshold of 0.05"

