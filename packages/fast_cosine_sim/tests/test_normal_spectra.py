"""
Tests for normal (well-behaved) spectra.

This module validates that the GPU-accelerated binning-based cosine similarity
implementation produces results consistent with the reference implementation
for well-separated peaks with no edge cases.

These tests require exact agreement (±0.001) and will fail fast on the first
failing spectrum when a failure occurs.

Bin sizes tested: 1e-5, 1e-4, 1e-3, 1e-2 Da
MS2 tolerance: 20 ppm (kept identical between implementations)
Tolerance handling: For m/z >= 200 Da, uses m/z * ppm * 1e-6; below 200 Da, uses 200 * ppm * 1e-6
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from utils import (
    AgreementStatistics,
    ComparisonResult,
    SpectrumPair,
    compare_results,
    compute_agreement_statistics,
    run_fast_cosine_sim,
    run_reference_cosine_greedy,
    skip_if_no_gpu,
)


def generate_normal_spectra_pairs() -> list[SpectrumPair]:
    """
    Generate 'normal' spectrum pairs with well-separated peaks and no edge cases.

    These should produce exact agreement (±0.001) between implementations.

    Returns:
        list of SpectrumPair objects
    """
    pairs = []

    # 1. Identical spectra (should give similarity ~1.0)
    mz_identical = np.array([100.0, 200.0, 300.0, 400.0, 500.0], dtype=np.float64)
    intensity_identical = np.array([1.0, 0.8, 0.6, 0.4, 0.2], dtype=np.float32)
    pairs.append(
        SpectrumPair(
            idx_left=0,
            idx_right=0,
            mz_left=mz_identical.copy(),
            intensity_left=intensity_identical.copy(),
            mz_right=mz_identical.copy(),
            intensity_right=intensity_identical.copy(),
            description="identical_spectra",
        )
    )

    # 2. Well-separated peaks, same positions different intensities
    mz_sep = np.array([100.0, 200.0, 300.0, 400.0, 500.0], dtype=np.float64)
    intensity_left_sep = np.array([1.0, 0.5, 0.3, 0.2, 0.1], dtype=np.float32)
    intensity_right_sep = np.array([0.5, 1.0, 0.4, 0.3, 0.15], dtype=np.float32)
    pairs.append(
        SpectrumPair(
            idx_left=1,
            idx_right=1,
            mz_left=mz_sep.copy(),
            intensity_left=intensity_left_sep,
            mz_right=mz_sep.copy(),
            intensity_right=intensity_right_sep,
            description="same_peaks_different_intensities",
        )
    )

    # 3. Partial overlap (some shared peaks)
    mz_left_partial = np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float64)
    intensity_left_partial = np.array([1.0, 0.8, 0.6, 0.4], dtype=np.float32)
    mz_right_partial = np.array([250.0, 300.0, 400.0, 500.0], dtype=np.float64)
    intensity_right_partial = np.array([1.0, 0.7, 0.5, 0.3], dtype=np.float32)
    pairs.append(
        SpectrumPair(
            idx_left=2,
            idx_right=2,
            mz_left=mz_left_partial,
            intensity_left=intensity_left_partial,
            mz_right=mz_right_partial,
            intensity_right=intensity_right_partial,
            description="partial_overlap",
        )
    )

    # 4. No overlap (should give similarity ~0.0)
    mz_left_no_overlap = np.array([100.0, 150.0, 200.0], dtype=np.float64)
    intensity_left_no_overlap = np.array([1.0, 0.8, 0.6], dtype=np.float32)
    mz_right_no_overlap = np.array([600.0, 650.0, 700.0], dtype=np.float64)
    intensity_right_no_overlap = np.array([1.0, 0.8, 0.6], dtype=np.float32)
    pairs.append(
        SpectrumPair(
            idx_left=3,
            idx_right=3,
            mz_left=mz_left_no_overlap,
            intensity_left=intensity_left_no_overlap,
            mz_right=mz_right_no_overlap,
            intensity_right=intensity_right_no_overlap,
            description="no_overlap",
        )
    )

    # 5. Dense spectrum with many peaks
    mz_dense = np.arange(100.0, 500.0, 5.0, dtype=np.float64)  # peaks every 5 Da
    intensity_dense = (
        np.random.RandomState(42)
        .uniform(0.1, 1.0, size=len(mz_dense))
        .astype(np.float32)
    )
    pairs.append(
        SpectrumPair(
            idx_left=4,
            idx_right=4,
            mz_left=mz_dense.copy(),
            intensity_left=intensity_dense.copy(),
            mz_right=mz_dense.copy(),
            intensity_right=intensity_dense.copy(),
            description="dense_spectrum",
        )
    )

    return pairs


@pytest.fixture
def normal_spectra_pairs() -> list[SpectrumPair]:
    """Fixture providing normal (well-behaved) spectrum pairs."""
    return generate_normal_spectra_pairs()


@pytest.mark.runtime
@pytest.mark.parametrize("bin_size", [1e-5, 1e-4, 1e-3, 1e-2])
@pytest.mark.parametrize("tolerance_ppm", [20.0])
@pytest.mark.parametrize("comparison_mode", ["self", "cross"])
def test_normal_spectra_exact_agreement(
    normal_spectra_pairs: list[SpectrumPair],
    bin_size: float,
    tolerance_ppm: float,
    comparison_mode: str,
) -> None:
    """
    Test that normal spectra produce exact agreement (±0.001) between implementations.

    Why: Well-separated peaks with no edge cases should produce identical results
    regardless of binning strategy, as long as tolerance is sufficient.

    If a failure occurs, this test will print the first failing spectrum and halt.
    """
    skip_if_no_gpu()

    # Run both implementations
    reference_scores = run_reference_cosine_greedy(
        normal_spectra_pairs, tolerance_ppm=tolerance_ppm
    )
    fast_cosine_scores = run_fast_cosine_sim(
        normal_spectra_pairs,
        bin_size=bin_size,
        tolerance_ppm=tolerance_ppm,
        comparison_mode=comparison_mode,
    )

    # Compare results
    comparison_results = compare_results(reference_scores, fast_cosine_scores)
    stats = compute_agreement_statistics(comparison_results)

    # Print report
    report = stats.format_report(
        bin_size=bin_size,
        tolerance_ppm=tolerance_ppm,
        test_name=f"Normal Spectra ({comparison_mode} mode)",
    )
    print(report)

    # Check for failures and print first failing spectrum if found
    failures = [r for r in comparison_results if not r.is_exact_match(0.001)]
    
    if failures:
        first_failure = failures[0]
        
        # Create mapping from (idx_left, idx_right) to original SpectrumPair for detailed output
        # Why: compare_results generates descriptions like "pair_0_0", but we need the original
        # SpectrumPair objects to show m/z and intensity arrays in error messages
        pair_lookup = {
            (pair.idx_left, pair.idx_right): pair for pair in normal_spectra_pairs
        }
        
        # Extract indices from description like "pair_0_0"
        # Why: ComparisonResult.description uses format "pair_{idx_left}_{idx_right}"
        desc_parts = first_failure.description.split("_")
        assert len(desc_parts) == 3 and desc_parts[0] == "pair", (
            f"Expected description format 'pair_<idx_left>_<idx_right>', got {first_failure.description}"
        )
        failing_idx_left = int(desc_parts[1])
        failing_idx_right = int(desc_parts[2])
        failing_pair = pair_lookup.get((failing_idx_left, failing_idx_right))
        
        # Build concise error message with all debugging info
        if failing_pair is not None:
            error_msg = (
                f"\nFIRST FAILURE | bin_size={bin_size:.2e} Da, tolerance={tolerance_ppm} ppm, mode={comparison_mode}\n"
                f"Spectrum: {failing_pair.description}\n"
                f"Scores: reference={first_failure.reference_score:.8f}, fast_cosine={first_failure.fast_cosine_score:.8f}, diff={first_failure.absolute_difference:.8f}\n"
                f"Left (idx={failing_pair.idx_left}):  mz={failing_pair.mz_left}, intensity={failing_pair.intensity_left}\n"
                f"Right (idx={failing_pair.idx_right}): mz={failing_pair.mz_right}, intensity={failing_pair.intensity_right}\n"
                f"Failed: {len(failures)}/{stats.total_pairs} pairs"
            )
        else:
            error_msg = (
                f"\nFIRST FAILURE | bin_size={bin_size:.2e} Da, tolerance={tolerance_ppm} ppm, mode={comparison_mode}\n"
                f"Scores: reference={first_failure.reference_score:.8f}, fast_cosine={first_failure.fast_cosine_score:.8f}, diff={first_failure.absolute_difference:.8f}\n"
                f"Description: {first_failure.description}\n"
                f"Failed: {len(failures)}/{stats.total_pairs} pairs"
            )
        
        pytest.fail(error_msg)
    
    # Assert exact agreement for all normal spectra
    assert stats.exact_agreement_rate == 1.0, (
        f"Expected 100% exact agreement (±0.001) for normal spectra, "
        f"got {stats.exact_agreement_rate * 100:.1f}% with bin_size={bin_size:.2e} Da, "
        f"tolerance={tolerance_ppm} ppm, mode={comparison_mode}"
    )
