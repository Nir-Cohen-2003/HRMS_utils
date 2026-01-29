import os

import polars as pl
import pytest

from hrms_utils.hrms_core import read_thermo_files


@pytest.fixture
def raw_file():
    # Path to the specific placeholder file in tests/data
    current_dir = os.path.dirname(__file__)
    data_dir = os.path.join(current_dir, "../../tests/data")
    file_path = os.path.join(data_dir, "S17582.raw")

    if not os.path.exists(file_path):
        pytest.skip(f"Test file not found: {file_path}")

    return file_path


def test_read_thermo_file(raw_file):
    """Test reading the placeholder Thermo RAW file."""
    dfs = read_thermo_files([raw_file])

    assert isinstance(dfs, list)
    assert len(dfs) == 1
    df = dfs[0]

    assert isinstance(df, pl.DataFrame)

    # Check schema
    expected_columns = {
        "id",
        "ms_level",
        "scan_time",
        "polarity",
        "mz",
        "intensity",
        "precursor_mz",
        "isolation_window_lower_bound",
        "isolation_window_upper_bound",
        "collision_energy",
        "injection_time",
        "filter_string",
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
    # Only assert height if we actually have data, which depends on the placeholder file
    if df.height > 0:
        pass


def test_missing_file():
    """Test behavior with missing file."""
    with pytest.raises(RuntimeError):
        read_thermo_files(["/path/to/nonexistent/file.raw"])
