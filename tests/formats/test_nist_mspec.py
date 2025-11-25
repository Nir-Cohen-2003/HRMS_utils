import sys
import time
from pathlib import Path
import traceback
import polars as pl
from hrms_utils.formats.nist_mspec import read_MSPEC_file
import os
import numpy as np

def run_mspec_reader_profile(msp_file_path: Path):
    """
    Reads a MSPEC file, profiles the execution time,
    and prints the number of spectra read.
    """
    print(f"\nProcessing file: {msp_file_path.name}")
    start_time = time.perf_counter()
    
    # Read the MSPEC file
    try:
        spectra_df = read_MSPEC_file(
            msp_file_path,
            raw_fragment_tolerance_ppm=15.0,
            normalized_fragment_tolerance_ppm=10.0,
            molecular_ion_tolerance_ppm=10.0
        )
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
    always_print_cols = ["instrument", "instrument_type", "ionization", "ion_mode", "mslevel"]
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

    not_found_cols = ["instrument", "instrument_type", "ionization"]
    for i in range(len(not_found_cols)):
        for j in range(i + 1, len(not_found_cols)):
            col1 = not_found_cols[i]
            col2 = not_found_cols[j]
            if col1 in spectra_df.columns and col2 in spectra_df.columns:
                count = spectra_df.filter((pl.col(col1) == "NOT FOUND") & (pl.col(col2) == "NOT FOUND")).height
                print(f"  - Rows with '{col1}' AND '{col2}' as 'NOT FOUND': {count}")

    if all(col in spectra_df.columns for col in not_found_cols):
        count = spectra_df.filter((pl.col("instrument") == "NOT FOUND") & (pl.col("instrument_type") == "NOT FOUND") & (pl.col("ionization") == "NOT FOUND")).height
        print(f"  - Rows with 'instrument', 'instrument_type', AND 'ionization' as 'NOT FOUND': {count}")
    
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


    
    # print a histogram (in numpy) of explained_intensity for clean precursors
    # Why: numeric histograms help CI and debugging without introducing plot artifacts.
    for hist_col in ["explained_intensity", "spectral_information_score"]:
        if hist_col not in spectra_df.columns:
            print(f"\nColumn '{hist_col}' not found; skipping histogram.")
            continue

        # Filter to clean precursors and drop nulls to ensure meaningful histogram bins.
        filtered = spectra_df.filter(pl.col("clean_precursor") & pl.col(hist_col).is_not_null()).select(pl.col(hist_col))
        if filtered.height == 0:
            print(f"\nNo rows for clean precursors with non-null '{hist_col}'; skipping histogram.")
            continue

        # Convert to 1D numpy array for numeric histogram computation.
        # shape: (n_rows,)
        values: np.ndarray = filtered.to_numpy().ravel()

        # Choose adaptive bins; 'auto' is a reasonable default that adjusts to the distribution
        # to provide informative buckets without needing visualization.
        counts, edges = np.histogram(values, bins=20)
        percentages = (counts / counts.sum()) * 100.0

        print(f"\nNumeric histogram for '{hist_col}' (clean_precursor):")
        print(f"  - bins: {len(counts)}")
        print(f"  - counts: {counts.tolist()}")
        print(f"  - percentages: {[round(p, 2) for p in percentages.tolist()]}")
        print(f"  - bin_edges (len {len(edges)}): {edges.tolist()}")
        # Also print the basic distribution summary for this column
        summary = filtered.select([
            pl.col(hist_col).mean().alias("mean"),
            pl.col(hist_col).median().alias("median"),
            pl.col(hist_col).min().alias("min"),
            pl.col(hist_col).max().alias("max"),
            pl.col(hist_col).std().alias("std_dev"),
        ]).to_dict(as_series=False)
        print(f"  - summary: { {k:v[0] for k,v in summary.items()} }")

    print(spectra_df.filter(
        pl.col("clean_precursor"),
        pl.col("precursor_type").eq("[M+H]+")
    ).sort(by="explained_intensity").head(1).select([
        "precursor_formula_array",
        "explained_intensity",
        "nist_id",
        "raw_spectrum_mz",
        "raw_spectrum_intensity",
        "cleaned_normalized_mz",
        "cleaned_normalized_intensity",
        "cleaned_fragment_formulas_str",
        "cleaned_fragment_errors_ppm"
    ]).to_init_repr())


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # CLI mode: process provided file or directory path
        input_path = Path(sys.argv[1]).resolve()
        assert input_path.exists(), f"Input path does not exist: {input_path}"
        
        if input_path.is_file():
            assert input_path.suffix.lower() in ['.msp', '.mspec'], f"File must have .msp or .mspec extension, got: {input_path.suffix}"
            run_mspec_reader_profile(input_path)
        elif input_path.is_dir():
            files_to_test = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in ['.msp', '.mspec']]
            assert len(files_to_test) > 0, f"No MSPEC/MSP files found in directory: {input_path}"
            for file_path in sorted(files_to_test):
                run_mspec_reader_profile(file_path)
        else:
            raise ValueError(f"Path is neither a file nor a directory: {input_path}")
    else:
        # Default mode: process all files in tests/data directory
        data_dir = (Path(__file__).resolve().parent.parent / "data").resolve()
        assert data_dir.exists(), f"data directory not found: {data_dir}"
        files_to_test = [f for f in os.listdir(data_dir) if f.lower().endswith(('.msp', '.mspec'))]
        
        for file_name in files_to_test:
            file_path = data_dir / file_name
            run_mspec_reader_profile(file_path)
