"""Tests for MS-DIAL CLI integration.

This module tests the MS-DIAL Console App integration including:
- Auto-detection of MS-DIAL executable
- Parameter file generation for positive and negative polarity
- Running MS-DIAL analysis on mzML files
- Reading and verifying the output chromatograms
"""

import tempfile
import time
import threading
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

import psutil
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

# Persistent output directory for inspection
OUTPUT_BASE_DIR = Path(__file__).resolve().parent.parent / "output" / "msdial_cli"


@dataclass
class ResourceSnapshot:
    """Snapshot of system resources at a point in time."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    
    def __str__(self) -> str:
        return f"[{self.timestamp:.1f}s] CPU: {self.cpu_percent:.1f}%, Memory: {self.memory_percent:.1f}% ({self.memory_used_mb:.1f} MB)"


class ResourceMonitor:
    """Monitor CPU and memory usage during test execution."""
    
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.snapshots: List[ResourceSnapshot] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None
        self._process = psutil.Process()
    
    def start(self):
        """Start monitoring resources."""
        self._start_time = time.time()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop monitoring resources."""
        if self._thread:
            self._stop_event.set()
            self._thread.join(timeout=self.interval + 1)
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while not self._stop_event.is_set():
            if self._start_time is not None:
                elapsed = time.time() - self._start_time
                try:
                    # Get CPU usage (percent over interval)
                    cpu_percent = self._process.cpu_percent(interval=None)
                    # Get memory info
                    mem_info = self._process.memory_info()
                    mem_used_mb = mem_info.rss / (1024 * 1024)
                    mem_percent = self._process.memory_percent()
                    
                    snapshot = ResourceSnapshot(
                        timestamp=elapsed,
                        cpu_percent=cpu_percent,
                        memory_percent=mem_percent,
                        memory_used_mb=mem_used_mb
                    )
                    self.snapshots.append(snapshot)
                except psutil.NoSuchProcess:
                    break
            self._stop_event.wait(self.interval)
    
    def get_summary(self) -> str:
        """Generate a summary of resource usage."""
        if not self.snapshots:
            return "No resource data collected"
        
        cpu_values = [s.cpu_percent for s in self.snapshots]
        mem_values = [s.memory_percent for s in self.snapshots]
        mem_mb_values = [s.memory_used_mb for s in self.snapshots]
        
        summary = [
            "\n=== Resource Usage Summary ===",
            f"Monitoring duration: {self.snapshots[-1].timestamp:.1f}s",
            f"Samples collected: {len(self.snapshots)}",
            "",
            "CPU Usage:",
            f"  Average: {sum(cpu_values) / len(cpu_values):.1f}%",
            f"  Min: {min(cpu_values):.1f}%",
            f"  Max: {max(cpu_values):.1f}%",
            "",
            "Memory Usage:",
            f"  Average: {sum(mem_values) / len(mem_values):.1f}%",
            f"  Min: {min(mem_values):.1f}%",
            f"  Max: {max(mem_values):.1f}%",
            f"  Average (MB): {sum(mem_mb_values) / len(mem_mb_values):.1f} MB",
            f"  Max (MB): {max(mem_mb_values):.1f} MB",
            "",
            "Timeline (CPU%/Memory%):",
        ]
        
        for snapshot in self.snapshots:
            summary.append(f"  {snapshot}")
        
        return "\n".join(summary)
    
    def detect_low_activity_periods(self, cpu_threshold: float = 5.0, duration_threshold: float = 5.0) -> List[str]:
        """Detect periods of low CPU activity."""
        low_activity_periods = []
        current_period_start = None
        
        for i, snapshot in enumerate(self.snapshots):
            if snapshot.cpu_percent < cpu_threshold:
                if current_period_start is None:
                    current_period_start = snapshot.timestamp
            else:
                if current_period_start is not None:
                    duration = snapshot.timestamp - current_period_start
                    if duration >= duration_threshold:
                        low_activity_periods.append(
                            f"  Low activity from {current_period_start:.1f}s to {snapshot.timestamp:.1f}s (duration: {duration:.1f}s)"
                        )
                    current_period_start = None
        
        # Check if still in low activity at end
        if current_period_start is not None and self.snapshots:
            duration = self.snapshots[-1].timestamp - current_period_start
            if duration >= duration_threshold:
                low_activity_periods.append(
                    f"  Low activity from {current_period_start:.1f}s to end (duration: {duration:.1f}s)"
                )
        
        return low_activity_periods


def get_persistent_output_dir(test_name: str) -> Path:
    """Get a persistent output directory for test results.
    
    Creates a timestamped subdirectory for each test run.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_BASE_DIR / f"{test_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


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
    output_dir = get_persistent_output_dir("params_positive")
    
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
    
    print(f"Positive params file saved to: {params_file}")


def test_generate_params_file_negative():
    """Test parameter file generation for negative polarity."""
    output_dir = get_persistent_output_dir("params_negative")
    
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
    
    print(f"Negative params file saved to: {params_file}")


@pytest.mark.skipif(
    not POSITIVE_MZML_DIR.exists() or not any(POSITIVE_MZML_DIR.glob("*.mzML")),
    reason="Positive mzML test data not available"
)
def test_run_msdial_positive_and_read_output():
    """Test running MS-DIAL on positive mode data and reading the output."""
    # First find and verify the binary being used
    msdial_binary = _find_msdial_executable()
    print(f"\nUsing MS-DIAL binary: {msdial_binary}")
    
    # Use persistent output directory
    output_dir = get_persistent_output_dir("positive_mode_run")
    print(f"Output directory: {output_dir}")
    
    # Start resource monitoring
    monitor = ResourceMonitor(interval=1.0)
    
    # Run MS-DIAL with timing and monitoring
    start_time = time.time()
    monitor.start()
    
    try:
        result = run_msdial_lcmsdda(
            input_dir=POSITIVE_MZML_DIR,
            output_dir=output_dir,
            polarity="positive",
        )
    finally:
        monitor.stop()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\n=== Timing Results ===")
    print(f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(monitor.get_summary())
    
    # Check for low activity periods
    low_activity = monitor.detect_low_activity_periods(cpu_threshold=5.0, duration_threshold=5.0)
    if low_activity:
        print("\n=== Low Activity Periods Detected ===")
        for period in low_activity:
            print(period)
    else:
        print("\nNo significant low activity periods detected")
    
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
    
    print(f"\nOutput files created ({len(output_files)} total):")
    for f in sorted(output_files):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.2f} MB)")
    
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
    
    print(f"\nAll output files preserved in: {output_dir}")


@pytest.mark.skipif(
    not NEGATIVE_MZML_DIR.exists() or not any(NEGATIVE_MZML_DIR.glob("*.mzML")),
    reason="Negative mzML test data not available"
)
def test_run_msdial_negative_and_read_output():
    """Test running MS-DIAL on negative mode data and reading the output."""
    # First find and verify the binary being used
    msdial_binary = _find_msdial_executable()
    print(f"\nUsing MS-DIAL binary: {msdial_binary}")
    
    # Use persistent output directory
    output_dir = get_persistent_output_dir("negative_mode_run")
    print(f"Output directory: {output_dir}")
    
    # Start resource monitoring
    monitor = ResourceMonitor(interval=1.0)
    
    # Run MS-DIAL with timing and monitoring
    start_time = time.time()
    monitor.start()
    
    try:
        result = run_msdial_lcmsdda(
            input_dir=NEGATIVE_MZML_DIR,
            output_dir=output_dir,
            polarity="negative",
        )
    finally:
        monitor.stop()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\n=== Timing Results ===")
    print(f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(monitor.get_summary())
    
    # Check for low activity periods
    low_activity = monitor.detect_low_activity_periods(cpu_threshold=5.0, duration_threshold=5.0)
    if low_activity:
        print("\n=== Low Activity Periods Detected ===")
        for period in low_activity:
            print(period)
    else:
        print("\nNo significant low activity periods detected")
    
    assert result.returncode == 0, f"MS-DIAL execution failed: {result.stderr}"
    
    # Check that output files were created
    output_files = (
        list(output_dir.glob("*.txt")) + 
        list(output_dir.glob("*.mdpeak")) +
        list(output_dir.glob("*Peak*")) +
        list(output_dir.glob("*Alignment*"))
    )
    assert len(output_files) > 0, "No output files were created"
    
    print(f"\nOutput files created ({len(output_files)} total):")
    for f in sorted(output_files):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.2f} MB)")
    
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
    
    print(f"\nAll output files preserved in: {output_dir}")


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
    output_dir = get_persistent_output_dir("invalid_polarity")
    with pytest.raises(ValueError, match="polarity must be 'positive' or 'negative'"):
        run_msdial_lcmsdda(
            input_dir=POSITIVE_MZML_DIR,
            output_dir=output_dir,
            polarity="invalid",  # type: ignore
        )


def test_nonexistent_input_dir_raises_error():
    """Test that non-existent input directory raises FileNotFoundError."""
    output_dir = get_persistent_output_dir("nonexistent_input")
    with pytest.raises(FileNotFoundError, match="Input directory does not exist"):
        run_msdial_lcmsdda(
            input_dir="/nonexistent/path/to/data",
            output_dir=output_dir,
            polarity="positive",
        )


def test_nonexistent_params_file_raises_error():
    """Test that non-existent custom params file raises FileNotFoundError."""
    output_dir = get_persistent_output_dir("nonexistent_params")
    with pytest.raises(FileNotFoundError, match="Parameters file not found"):
        run_msdial_lcmsdda(
            input_dir=POSITIVE_MZML_DIR,
            output_dir=output_dir,
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
    print(f"\nAll output files are preserved in: {OUTPUT_BASE_DIR}")
