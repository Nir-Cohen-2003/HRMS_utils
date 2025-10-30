import time
from pathlib import Path
import traceback
import polars as pl
from hrms_utils.formats.nist_mspec import read_MSPEC_file
import os

def run_mspec_reader_profile(msp_file_path: Path):
    """
    Reads a MSPEC file, profiles the execution time,
    and prints the number of spectra read.
    """
    print(f"\nProcessing file: {msp_file_path.name}")
    start_time = time.perf_counter()
    
    # Read the MSPEC file
    try:
        spectra_df = read_MSPEC_file(msp_file_path,raw_fragment_tolerance_ppm=10.0, normalized_fragment_tolerance_ppm=5.0, molecular_ion_tolerance_ppm=5.0)
        is_empty = spectra_df.is_empty()
    except Exception as e:
        print(f"An error occurred while reading the MSPEC file: {e}")
        traceback.print_exc()
        return

    end_time = time.perf_counter()

    # Calculate the duration
    duration = end_time - start_time
    
    # Get the number of spectra
    num_spectra = len(spectra_df)
    
    # Print the profiling information
    print(f"Read {num_spectra} spectra in {duration:.4f} seconds.")
    
    # Assert that the DataFrame is not empty
    assert not is_empty, "The resulting DataFrame should not be empty."
    print(spectra_df.schema)
    print(spectra_df.head())

    # Identify non-nested columns
    non_nested_cols = [col for col, dtype in spectra_df.schema.items() if not isinstance(dtype, (pl.List, pl.Struct))]

    # Count unique values for non_nested_cols
    unique_counts = {}
    for col in non_nested_cols:
        unique_counts[col] = spectra_df[col].n_unique()

    # Select columns with 20 or less unique values
    filtered_cols = {col: count for col, count in unique_counts.items() if count <= 20}

    # Print the filtered columns and their unique counts
    print("\nColumns with 20 or less unique values:")
    for col, count in filtered_cols.items():
        print(f"  - {col}: {count} unique values, values: {spectra_df[col].unique().to_list()}")
    
    # Always print values for specific columns (USER-CONFIGURABLE)
    always_print_cols = ["Instrument", "Instrument_type", "Ionization", "Ion_mode", "MSLEVEL", "Precursor_type"]
    print("\nAlways printed columns:")
    for col in always_print_cols:
        if col in spectra_df.columns:
            value_counts = spectra_df[col].value_counts()
            print(f"  - {col}:")
            for value, count in value_counts.iter_rows():
                print(f"    - {value}: {count} rows")
        else:
            print(f"  - {col}: Not found in DataFrame")

    # Check for "NOT FOUND" values in specific columns
    print("\n'NOT FOUND' value counts:")

    not_found_cols = ["Instrument", "Instrument_type", "Ionization"]
    for i in range(len(not_found_cols)):
        for j in range(i + 1, len(not_found_cols)):
            col1 = not_found_cols[i]
            col2 = not_found_cols[j]
            if col1 in spectra_df.columns and col2 in spectra_df.columns:
                count = spectra_df.filter((pl.col(col1) == "NOT FOUND") & (pl.col(col2) == "NOT FOUND")).height
                print(f"  - Rows with '{col1}' AND '{col2}' as 'NOT FOUND': {count}")

    if all(col in spectra_df.columns for col in not_found_cols):
        count = spectra_df.filter((pl.col("Instrument") == "NOT FOUND") & (pl.col("Instrument_type") == "NOT FOUND") & (pl.col("Ionization") == "NOT FOUND")).height
        print(f"  - Rows with 'Instrument', 'Instrument_type', AND 'Ionization' as 'NOT FOUND': {count}")
    # print how many are orbi, tof or other
    orbi_count = spectra_df.filter(pl.col("is_orbitrap")).height
    tof_count = spectra_df.filter(pl.col("is_TOF")).height
    other_count = spectra_df.filter((pl.col("is_orbitrap").not_()) & (pl.col("is_TOF").not_())).height
    print("\nInstrument type counts:")
    print(f"  - Orbitrap: {orbi_count}")
    print(f"  - TOF: {tof_count}")
    print(f"  - Other: {other_count}")
    # print how many of the orbi, TOF are ESI
    esi_orbi_count = spectra_df.filter(pl.col("is_ESI") & pl.col("is_orbitrap")).height
    esi_tof_count = spectra_df.filter(pl.col("is_ESI") & pl.col("is_TOF")).height
    esi_other_count = spectra_df.filter(pl.col("is_ESI") & (pl.col("is_orbitrap").not_()) & (pl.col("is_TOF").not_())).height
    print("\nESI counts by instrument type:")
    print(f"  - ESI Orbitrap: {esi_orbi_count}")
    print(f"  - ESI TOF: {esi_tof_count}")
    print(f"  - ESI Other: {esi_other_count}")

    # print teh distribution of "entropy_similarity" column if it exists
    if "entropy_similarity" in spectra_df.columns:
        print("\n'entropy_similarity' column statistics, for clean precursors only:")
        entropy_stats = spectra_df.filter(pl.col("clean_precursor")).select([
            pl.col("entropy_similarity").mean().alias("mean"),
            pl.col("entropy_similarity").median().alias("median"),
            pl.col("entropy_similarity").min().alias("min"),
            pl.col("entropy_similarity").max().alias("max"),
            pl.col("entropy_similarity").std().alias("std_dev"),
        ]).to_dict(as_series=False)
        for stat, value in entropy_stats.items():
            print(f"  - {stat}: {value[0]}")
    print(spectra_df.filter(pl.col("clean_precursor"),pl.col("Precursor_type").eq("[M+H]+")).sort(by="entropy_similarity").head().select(["entropy_similarity","NIST_ID","cleaned_fragment_formulas_str","cleaned_fragment_errors_ppm"]))

if __name__ == "__main__":
    # Make data_dir relative to this test file's location (resolve to absolute path)
    data_dir = (Path(__file__).resolve().parent.parent / "data").resolve()
    assert data_dir.exists(), f"data directory not found: {data_dir}"
    files_to_test = [f for f in os.listdir(data_dir) if f.lower().endswith(('.msp', '.mspec'))]
    
    for file_name in files_to_test:
        file_path = data_dir / file_name
        run_mspec_reader_profile(file_path)
