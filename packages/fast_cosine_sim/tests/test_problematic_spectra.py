"""
Tests for problematic (edge case) spectra.

This module tests binning artifacts, boundary conditions, and extreme values
that may cause small differences between the reference and fast implementations.

These tests report agreement at multiple thresholds but do not enforce strict
pass/fail criteria (except for soft warnings).

Bin sizes tested: 1e-5, 3e-5, 1e-4 Da
MS2 tolerance: 20 ppm (kept identical between implementations)
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


def generate_problematic_spectra_pairs() -> list[SpectrumPair]:
    """
    Generate 'problematic' spectrum pairs with edge cases.

    These test binning artifacts, boundary conditions, and extreme values.

    Returns:
        list of SpectrumPair objects
    """
    pairs = []

    # 1. Very close peaks (within 0.001 Da - potential binning artifacts)
    for i, offset in enumerate([0.0001, 0.0005, 0.001, 0.002]):
        mz_left = np.array([500.0000], dtype=np.float64)
        intensity_left = np.array([1.0], dtype=np.float32)
        mz_right = np.array([500.0000 + offset], dtype=np.float64)
        intensity_right = np.array([1.0], dtype=np.float32)
        pairs.append(
            SpectrumPair(
                idx_left=100 + i,
                idx_right=100 + i,
                mz_left=mz_left,
                intensity_left=intensity_left,
                mz_right=mz_right,
                intensity_right=intensity_right,
                description=f"very_close_peaks_offset_{offset:.4f}Da",
            )
        )

    # 2. Peaks at bin boundaries (for various bin sizes)
    # Test peaks that land exactly on multiples of common bin sizes
    for i, base_mz in enumerate([100.0, 500.0, 1000.0]):
        mz_boundary = np.array(
            [base_mz, base_mz + 0.001, base_mz + 0.01, base_mz + 0.1], dtype=np.float64
        )
        intensity_boundary = np.array([1.0, 0.8, 0.6, 0.4], dtype=np.float32)
        pairs.append(
            SpectrumPair(
                idx_left=200 + i,
                idx_right=200 + i,
                mz_left=mz_boundary.copy(),
                intensity_left=intensity_boundary.copy(),
                mz_right=mz_boundary.copy(),
                intensity_right=intensity_boundary.copy(),
                description=f"bin_boundary_mz_{base_mz:.0f}",
            )
        )

    # 3. High-mass peaks (near upper_mass_bound, default 2000 Da)
    mz_high_mass = np.array([1900.0, 1950.0, 1980.0, 1990.0], dtype=np.float64)
    intensity_high_mass = np.array([1.0, 0.8, 0.6, 0.4], dtype=np.float32)
    pairs.append(
        SpectrumPair(
            idx_left=300,
            idx_right=300,
            mz_left=mz_high_mass.copy(),
            intensity_left=intensity_high_mass.copy(),
            mz_right=mz_high_mass.copy(),
            intensity_right=intensity_high_mass.copy(),
            description="high_mass_peaks",
        )
    )

    # 4. Low-mass peaks (below mass_tolerance_cutoff, default 200 Da)
    mz_low_mass = np.array([50.0, 100.0, 150.0, 199.0], dtype=np.float64)
    intensity_low_mass = np.array([1.0, 0.8, 0.6, 0.4], dtype=np.float32)
    pairs.append(
        SpectrumPair(
            idx_left=400,
            idx_right=400,
            mz_left=mz_low_mass.copy(),
            intensity_left=intensity_low_mass.copy(),
            mz_right=mz_low_mass.copy(),
            intensity_right=intensity_low_mass.copy(),
            description="low_mass_peaks_below_cutoff",
        )
    )

    # 5. Mixed intensity scales (spanning many orders of magnitude)
    mz_mixed_int = np.array([100.0, 200.0, 300.0, 400.0, 500.0], dtype=np.float64)
    intensity_mixed_left = np.array([1e6, 1e3, 1e0, 1e-2, 1e-4], dtype=np.float32)
    intensity_mixed_right = np.array([1e5, 1e2, 1e1, 1e-1, 1e-3], dtype=np.float32)
    pairs.append(
        SpectrumPair(
            idx_left=500,
            idx_right=500,
            mz_left=mz_mixed_int,
            intensity_left=intensity_mixed_left,
            mz_right=mz_mixed_int,
            intensity_right=intensity_mixed_right,
            description="mixed_intensity_scales",
        )
    )

    # 6. Single peak spectra
    pairs.append(
        SpectrumPair(
            idx_left=600,
            idx_right=600,
            mz_left=np.array([500.0], dtype=np.float64),
            intensity_left=np.array([1.0], dtype=np.float32),
            mz_right=np.array([500.0], dtype=np.float64),
            intensity_right=np.array([1.0], dtype=np.float32),
            description="single_peak_identical",
        )
    )

    return pairs


@pytest.fixture
def problematic_spectra_pairs() -> list[SpectrumPair]:
    """Fixture providing problematic (edge case) spectrum pairs."""
    return generate_problematic_spectra_pairs()


@pytest.mark.runtime
@pytest.mark.parametrize("bin_size", [1e-5, 3e-5, 1e-4])
@pytest.mark.parametrize("tolerance_ppm", [20.0])
@pytest.mark.parametrize("comparison_mode", ["self", "cross"])
def test_problematic_spectra_agreement(
    problematic_spectra_pairs: list[SpectrumPair],
    bin_size: float,
    tolerance_ppm: float,
    comparison_mode: str,
) -> None:
    """
    Test problematic spectra and report agreement at multiple thresholds.

    Why: Edge cases (very close peaks, bin boundaries) may have small differences
    due to binning artifacts. We report statistics but don't enforce strict pass/fail.
    
    Only test on selected bin sizes (1e-5, 3e-5, 1e-4 Da) to reduce test time.
    """
    skip_if_no_gpu()

    # Run both implementations
    reference_scores = run_reference_cosine_greedy(
        problematic_spectra_pairs, tolerance_ppm=tolerance_ppm
    )
    fast_cosine_scores = run_fast_cosine_sim(
        problematic_spectra_pairs,
        bin_size=bin_size,
        tolerance_ppm=tolerance_ppm,
        comparison_mode=comparison_mode,
    )

    # Compare results
    comparison_results = compare_results(reference_scores, fast_cosine_scores)
    stats = compute_agreement_statistics(comparison_results)

    # Print detailed report
    report = stats.format_report(
        bin_size=bin_size,
        tolerance_ppm=tolerance_ppm,
        test_name=f"Problematic Spectra ({comparison_mode} mode)",
    )
    print(report)

    # Print details for pairs with large differences
    large_diff_threshold = 0.01
    large_diff_pairs = [
        r for r in comparison_results if r.absolute_difference > large_diff_threshold
    ]
    if large_diff_pairs:
        print(f"\nPairs with difference > {large_diff_threshold}:")
        for r in sorted(
            large_diff_pairs, key=lambda x: x.absolute_difference, reverse=True
        ):
            print(
                f"  {r.description}: "
                f"reference={r.reference_score:.6f}, "
                f"fast_cosine={r.fast_cosine_score:.6f}, "
                f"diff={r.absolute_difference:.6f}"
            )

    # Soft assertion: warn if exact agreement is too low, but don't fail
    # Why: Some binning artifacts are expected for problematic cases
    if stats.exact_agreement_rate < 0.90:
        print(
            f"\nWARNING: Exact agreement rate ({stats.exact_agreement_rate * 100:.1f}%) "
            f"is below 90% for problematic spectra with bin_size={bin_size:.2e}"
        )
