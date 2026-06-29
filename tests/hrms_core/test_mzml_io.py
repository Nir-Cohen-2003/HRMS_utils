import pytest
import polars as pl
import os
from hrms_utils.hrms_core import read_mzml_files

@pytest.fixture
def mzml_files():
    # Return list of absolute paths to all mzML files in tests/data
    current_dir = os.path.dirname(__file__)
    data_dir = os.path.join(current_dir, "../../tests/data")
    
    if not os.path.exists(data_dir):
        pytest.skip(f"Data directory not found: {data_dir}")
        
    files = [
        os.path.abspath(os.path.join(data_dir, f)) 
        for f in os.listdir(data_dir) 
        if f.lower().endswith('.mzml')
    ]
    
    if not files:
        pytest.skip(f"No mzML files found in {data_dir}")
        
    return files

def test_read_mzml_files(mzml_files):
    """Test reading all mzML files found in data directory."""
    # Test reading all files at once
    dfs = read_mzml_files(mzml_files)
    
    assert isinstance(dfs, list)
    assert len(dfs) == len(mzml_files)
    
    for df in dfs:
        assert isinstance(df, pl.DataFrame)
        
        # Check schema
        expected_columns = {
            "id", "ms_level", "scan_time", "polarity", "mz", "intensity",
            "precursor_mz", "isolation_window_lower_bound",
            "isolation_window_upper_bound", "collision_energy",
            "collision_energy_unit", "injection_time", "filter_string"
        }

        assert set(df.columns) == expected_columns

        # Check data types
        schema = df.schema
        assert schema["id"] == pl.String
        assert schema["ms_level"] == pl.UInt8
        assert schema["mz"] == pl.List(pl.Float64)
        assert schema["intensity"] == pl.List(pl.Float64)
        assert schema["injection_time"] == pl.Float64
        assert schema["filter_string"] == pl.String
        
        # Check content (basic checks)
        assert df.height > 0
        
        # Check MS2 specific fields if any MS2 spectra exist
        ms2_df = df.filter(pl.col("ms_level") == 2)
        if ms2_df.height > 0:
            assert ms2_df.null_count().item(column="precursor_mz", row=0) == 0
            assert ms2_df.null_count().item(column="collision_energy", row=0) == 0

def test_read_multiple_files(mzml_files):
    """Test reading multiple files (using the first file twice)."""
    if not mzml_files:
        pytest.skip("No mzML files to test")
        
    first_file = mzml_files[0]
    paths = [first_file, first_file]
    dfs = read_mzml_files(paths)
    
    assert len(dfs) == 2
    assert dfs[0].equals(dfs[1])

def test_missing_file():
    """Test behavior with missing file."""
    # Depending on implementation, this might raise an error or return empty/None
    # The Rust implementation uses map_err, so it should raise a PyRuntimeError
    with pytest.raises(RuntimeError):
        read_mzml_files(["/path/to/nonexistent/file.mzML"])
