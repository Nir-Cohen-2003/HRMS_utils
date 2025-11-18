"""
Convert MSP/MSPEC files to a single Parquet file.

Usage:
    python convert_msp.py <path>

Where <path> can be:
- A single .msp/.mspec file
- A directory containing .msp/.mspec files

The script will read all matching files, concatenate them into a single Polars DataFrame,
and write the result as a Parquet file adjacent to the input with the same base name.
"""

import sys
from pathlib import Path
import polars as pl
from hrms_utils.formats.nist_mspec import read_MSPEC_file
from time import perf_counter

def collect_mspec_files(path: Path) -> list[Path]:
    """
    Collect all MSPEC/MSP files from the given path.
    
    Args:
        path: Either a file or directory path
        
    Returns:
        List of Path objects for all matching files
    """
    valid_suffixes = {'.msp', '.mspec', '.MSP', '.MSPEC'}
    
    if path.is_file():
        assert path.suffix in valid_suffixes, f"File {path} does not have a valid MSPEC/MSP suffix: {path.suffix}"
        return [path]
    
    if path.is_dir():
        files = [f for f in path.iterdir() if f.is_file() and f.suffix in valid_suffixes]
        assert len(files) > 0, f"No MSPEC/MSP files found in directory: {path}"
        return sorted(files)
    
    raise ValueError(f"Path does not exist or is not a file/directory: {path}")


def main():
    assert len(sys.argv) == 2, "Usage: python convert_msp.py <path_to_file_or_directory>"
    
    input_path = Path(sys.argv[1]).resolve()
    assert input_path.exists(), f"Input path does not exist: {input_path}"
    
    # Collect all matching files
    mspec_files = collect_mspec_files(input_path)
    print(f"Found {len(mspec_files)} MSPEC/MSP file(s) to process")
    
    # Read and concatenate all dataframes
    start = perf_counter()
    lazyframes = []
    for file_path in mspec_files:
        print(f"Reading {file_path.name}...")
        df = read_MSPEC_file(
            file_path,
            raw_fragment_tolerance_ppm=10.0,
            normalized_fragment_tolerance_ppm=5.0,
            molecular_ion_tolerance_ppm=5.0,
            lazy=True
        )
        lazyframes.append(df)
    
    print(f"Concatenating {len(lazyframes)} lazyframes and collecting them")
    combined_df = pl.concat(lazyframes, how="vertical").collect(engine='streaming')
    end = perf_counter()
    print(f"Combined dataframe has {combined_df.height} spectra, and {combined_df.unique(subset="base_inchikey").height} unique 2d structures (by base_inchikey)")
    print(f"Completed reading and concatenation in {end - start:.2f} seconds")
    combined_df = combined_df.filter(
        pl.col("clean_precursor"),
        pl.col("explained_intensity") > 0.95,
        pl.col("is_ESI"),
        pl.col("is_orbitrap")
        )
    print(f"afetr filtering, we are left with {combined_df.height} spectra and {combined_df.unique(subset='base_inchikey').height} unique 2d structures (by base_inchikey)")
    
    # Determine output path: same location as input, with .parquet extension
    if input_path.is_file():
        output_path = input_path.with_suffix('.parquet')
    else:
        # For directories, use the directory name as the base file name
        output_path = input_path / f"{input_path.name}.parquet"
    
    print(f"Writing to {output_path}...")
    combined_df.write_parquet(output_path)
    print(f"Successfully wrote {len(combined_df)} spectra to {output_path}")


if __name__ == "__main__":
    main()