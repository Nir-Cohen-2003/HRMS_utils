"""Tests for MS-DIAL CLI integration.

This module tests the MS-DIAL Console App integration including:
- Auto-detection of MS-DIAL executable
- Parameter file generation for positive and negative polarity
- Running MS-DIAL analysis on mzML files
- Reading and verifying the output chromatograms
"""

import tempfile
from pathlib import Path

import pytest

from hrms_utils.formats.msdial import (
    MSDialRunnerConfig,
    _find_msdial_executable,
    _generate_params_file,
    get_chromatogram,
    run_msdial_lcmsdda,
)


# Get paths to test data
TEST_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "mzmls"
POSITIVE_MZML_DIR = TEST_DATA_DIR / "positive"
NEGATIVE_MZML_DIR = TEST_DATA_DIR / "negative"


def test_find_msdial_executable():
    """Test that MS-DIAL executable can be found in packages/msdial directory."""
    executable = _find_msdial_executable()
    assert executable.exists(), f"MS-DIAL executable not found at {executable}"
    print(f"Found MS-DIAL executable at: {executable}")
    
    # Verify it's the expected binary in packages/msdial
    expected_path = Path(__file__).resolve().parent.parent.parent / "packages" / "msdial"
    assert expected_path in executable.parents, (
        f"MS-DIAL executable found at {executable} is not in expected location {expected_path}"
    )


def test_generate_params_file_positive():
    """Test parameter file generation for positive polarity."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        params_file = _generate_params_file(
            polarity="positive",
            output_path=output_dir,
            threads=4,
            minimum_peak_height=50000,
        )
        
        assert params_file.exists(), "Params file was not created"
        assert params_file.name == "Msdial-lcms-dda-positive-params.txt"
        
        content = params_file.read_text()
        assert "Ion mode: Positive" in content, "Positive ion mode not set correctly"
        assert "[M+H]+" in content, "Positive adducts not included"
        assert "Number of threads: 4" in content, "Thread count not set correctly"
        assert "Minimum peak height: 50000" in content, "Peak height not set correctly"


def test_generate_params_file_negative():
    """Test parameter file generation for negative polarity."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        params_file = _generate_params_file(
            polarity="negative",
            output_path=output_dir,
            threads=8,
            minimum_peak_height=100000,
        )
        
        assert params_file.exists(), "Params file was not created"
        assert params_file.name == "Msdial-lcms-dda-negative-params.txt"
        
        content = params_file.read_text()
        assert "Ion mode: Negative" in content, "Negative ion mode not set correctly"
        assert "[M-H]-" in content, "Negative adducts not included"
        assert "Number of threads: 8" in content, "Thread count not set correctly"
        assert "Minimum peak height: 100000" in content, "Peak height not set correctly"


@pytest.mark.skipif(
    not POSITIVE_MZML_DIR.exists() or not any(POSITIVE_MZML_DIR.glob("*.mzML")),
    reason="Positive mzML test data not available"
)
def test_run_msdial_positive_and_read_output():
    """Test running MS-DIAL on positive mode data and reading the output."""
    # First find and verify the binary being used
    msdial_binary = _find_msdial_executable()
    print(f"\nUsing MS-DIAL binary: {msdial_binary}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # Run MS-DIAL on positive mzML files
        result = run_msdial_lcmsdda(
            input_dir=POSITIVE_MZML_DIR,
            output_dir=output_dir,
            polarity="positive",
        )
        
        assert result.returncode == 0, f"MS-DIAL execution failed: {result.stderr}"
        
        # Check that output files were created
        # MS-DIAL outputs various files; check for peak/alignment files
        output_files = (
            list(output_dir.glob("*.txt")) + 
            list(output_dir.glob("*.mdpeak")) +
            list(output_dir.glob("*Peak*")) +
            list(output_dir.glob("*Alignment*"))
        )
        assert len(output_files) > 0, "No output files were created"
        
        # Try to read the chromatogram from the output
        chromatogram_found = False
        for output_file in output_files:
            try:
                chromatogram = get_chromatogram(output_file)
                assert not chromatogram.is_empty(), f"Chromatogram from {output_file.name} is empty"
                assert "Peak ID" in chromatogram.columns, "Missing Peak ID column"
                assert "RT (min)" in chromatogram.columns, "Missing RT column"
                assert "Precursor_mz_MSDIAL" in chromatogram.columns, "Missing Precursor_mz_MSDIAL column"
                print(f"Successfully read chromatogram from {output_file.name}: {len(chromatogram)} peaks")
                chromatogram_found = True
            except Exception as e:
                # Some output files might not be chromatograms, that's okay
                print(f"Could not read {output_file.name} as chromatogram: {e}")
                continue
        
        assert chromatogram_found, "Could not read any chromatogram from MS-DIAL output files"


@pytest.mark.skipif(
    not NEGATIVE_MZML_DIR.exists() or not any(NEGATIVE_MZML_DIR.glob("*.mzML")),
    reason="Negative mzML test data not available"
)
def test_run_msdial_negative_and_read_output():
    """Test running MS-DIAL on negative mode data and reading the output."""
    # First find and verify the binary being used
    msdial_binary = _find_msdial_executable()
    print(f"\nUsing MS-DIAL binary: {msdial_binary}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # Run MS-DIAL on negative mzML files
        result = run_msdial_lcmsdda(
            input_dir=NEGATIVE_MZML_DIR,
            output_dir=output_dir,
            polarity="negative",
        )
        
        assert result.returncode == 0, f"MS-DIAL execution failed: {result.stderr}"
        
        # Check that output files were created
        output_files = (
            list(output_dir.glob("*.txt")) + 
            list(output_dir.glob("*.mdpeak")) +
            list(output_dir.glob("*Peak*")) +
            list(output_dir.glob("*Alignment*"))
        )
        assert len(output_files) > 0, "No output files were created"
        
        # Try to read the chromatogram from the output
        chromatogram_found = False
        for output_file in output_files:
            try:
                chromatogram = get_chromatogram(output_file)
                assert not chromatogram.is_empty(), f"Chromatogram from {output_file.name} is empty"
                assert "Peak ID" in chromatogram.columns, "Missing Peak ID column"
                assert "RT (min)" in chromatogram.columns, "Missing RT column"
                assert "Precursor_mz_MSDIAL" in chromatogram.columns, "Missing Precursor_mz_MSDIAL column"
                print(f"Successfully read chromatogram from {output_file.name}: {len(chromatogram)} peaks")
                chromatogram_found = True
            except Exception as e:
                # Some output files might not be chromatograms, that's okay
                print(f"Could not read {output_file.name} as chromatogram: {e}")
                continue
        
        assert chromatogram_found, "Could not read any chromatogram from MS-DIAL output files"


def test_run_msdial_with_custom_config():
    """Test running MS-DIAL with custom configuration."""
    # First find the executable
    msdial_path = _find_msdial_executable()
    
    # Create custom config
    config = MSDialRunnerConfig(
        msdial_path=msdial_path,
        threads=2,
        minimum_peak_height=200000,
    )
    
    # Just verify the config is created properly
    assert config.msdial_path == msdial_path
    assert config.threads == 2
    assert config.minimum_peak_height == 200000


def test_invalid_polarity_raises_error():
    """Test that invalid polarity raises ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="polarity must be 'positive' or 'negative'"):
            run_msdial_lcmsdda(
                input_dir=POSITIVE_MZML_DIR,
                output_dir=Path(tmpdir),
                polarity="invalid",  # type: ignore
            )


def test_nonexistent_input_dir_raises_error():
    """Test that non-existent input directory raises FileNotFoundError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError, match="Input directory does not exist"):
            run_msdial_lcmsdda(
                input_dir="/nonexistent/path/to/data",
                output_dir=Path(tmpdir),
                polarity="positive",
            )


def test_nonexistent_params_file_raises_error():
    """Test that non-existent custom params file raises FileNotFoundError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError, match="Parameters file not found"):
            run_msdial_lcmsdda(
                input_dir=POSITIVE_MZML_DIR,
                output_dir=Path(tmpdir),
                polarity="positive",
                params_file="/nonexistent/params.txt",
            )


if __name__ == "__main__":
    # Run tests when called directly
    print("Testing MS-DIAL executable detection...")
    test_find_msdial_executable()
    print("\nTesting positive parameter generation...")
    test_generate_params_file_positive()
    print("\nTesting negative parameter generation...")
    test_generate_params_file_negative()
    print("\nTesting custom config...")
    test_run_msdial_with_custom_config()
    
    # Run integration tests if data is available
    if POSITIVE_MZML_DIR.exists() and any(POSITIVE_MZML_DIR.glob("*.mzML")):
        print("\nRunning positive mode integration test...")
        test_run_msdial_positive_and_read_output()
    else:
        print("\nSkipping positive mode integration test (no data)")
    
    if NEGATIVE_MZML_DIR.exists() and any(NEGATIVE_MZML_DIR.glob("*.mzML")):
        print("\nRunning negative mode integration test...")
        test_run_msdial_negative_and_read_output()
    else:
        print("\nSkipping negative mode integration test (no data)")
    
    print("\nAll tests completed!")
