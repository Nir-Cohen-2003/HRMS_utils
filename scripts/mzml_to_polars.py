import pymzml
import polars as pl
import numpy as np
import os
import time

def mzml_to_polars(mzml_path):
    if not os.path.exists(mzml_path):
        raise FileNotFoundError(f"File not found: {mzml_path}")

    data = []
    run = pymzml.run.Reader(mzml_path)
    
    print(f"Reading {mzml_path}...")
    
    for spectrum in run:
        scan_time = None
        if spectrum.scan_time:
            scan_time = spectrum.scan_time[0]
            
        mz_array = spectrum.mz
        intensity_array = spectrum.i
        
        spec_data = {
            "id": str(spectrum.ID),
            "ms_level": spectrum.ms_level,
            "scan_time": scan_time,
            "mz": mz_array,
            "intensity": intensity_array
        }
        data.append(spec_data)

    schema = {
        "id": pl.String,
        "ms_level": pl.Int64,
        "scan_time": pl.Float64,
        "mz": pl.List(pl.Float64),
        "intensity": pl.List(pl.Float64)
    }

    df = pl.DataFrame(data, schema=schema, orient="row")
    return df

if __name__ == "__main__":
    file_path = "tests/data/Metabolomics_2.mzML"
    try:
        start_time = time.time() # Start timing
        df = mzml_to_polars(file_path)
        end_time = time.time() # End timing
        
        print("Successfully converted mzML to Polars DataFrame:")
        print(df)
        print("\nSchema:")
        print(df.schema)
        print(f"\nShape: {df.shape}")
        print(f"\nTime taken: {end_time - start_time:.2f} seconds") # Print elapsed time
    except Exception as e:
        print(f"Error: {e}")
