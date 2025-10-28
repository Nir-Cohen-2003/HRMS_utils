import time
from pathlib import Path
import traceback
from hrms_utils.formats.nist_mspec import read_MSPEC_file

def run_mspec_reader_profile():
    """
    Reads a MSPEC file, profiles the execution time,
    and prints the number of spectra read.
    """
    msp_file_path = Path("/home/analytit_admin/dev/HRMS_utils/tests/data/NEG_LC.msp")

    start_time = time.perf_counter()
    
    # Read the MSPEC file
    try:
        spectra_df = read_MSPEC_file(msp_file_path)
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

if __name__ == "__main__":
    run_mspec_reader_profile()