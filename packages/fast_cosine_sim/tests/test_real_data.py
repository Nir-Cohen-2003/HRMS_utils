"""
Tests for real HRMS data.

This module validates the implementation on actual mass spectrometry data
and reports agreement statistics without enforcing strict thresholds.

Real data may have characteristics not covered by synthetic tests, so we
report comprehensive statistics and only enforce soft assertions.

Bin sizes tested: 1e-5, 3e-5, 1e-4 Da
MS2 tolerance: 20 ppm (kept identical between implementations)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
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


@pytest.fixture
def real_spectra_path() -> Path:
    """Fixture providing path to real HRMS spectra data."""
    path = Path(
        "/home/analytit_admin/Data/spectral_libs/fast_similarity/fraghub_P_100k.parquet"
    )
    assert path.exists(), f"Real spectra file not found: {path}"
    return path


@pytest.fixture
def max_spectra(request) -> int:
    """Fixture providing the maximum number of spectra to test."""
    return int(request.config.getoption("--max-spectra"))


@pytest.mark.runtime
@pytest.mark.parametrize("bin_size", [1e-5, 3e-5, 1e-4])
@pytest.mark.parametrize("tolerance_ppm", [20.0])
def test_real_spectra_self_comparison(
    real_spectra_path: Path,
    max_spectra: int,
    bin_size: float,
    tolerance_ppm: float,
) -> None:
    """
    Test self-comparison on real HRMS data.

    Why: Self-comparison (spectrum vs itself) should give similarity ~1.0 and allows
    us to validate normalization. Only test on selected bin sizes to avoid slow
    reference computation.

    The number of spectra can be adjusted via --max-spectra CLI argument.
    """
    skip_if_no_gpu()

    # Load a subset of real spectra
    df = pl.read_parquet(real_spectra_path)

    # Select random subset if dataset is large
    if df.height > max_spectra:
        df = df.sample(n=max_spectra, seed=42)

    # Handle different column naming conventions
    # Why: Real HRMS data may use different column names
    mz_col = None
    intensity_col = None

    if "mz" in df.columns:
        mz_col = "mz"
        intensity_col = "intensity"
    elif "cleaned_normalized_mz" in df.columns:
        mz_col = "cleaned_normalized_mz"
        intensity_col = "cleaned_normalized_intensity"
    elif "raw_spectrum_mz" in df.columns:
        mz_col = "raw_spectrum_mz"
        intensity_col = "raw_spectrum_intensity"
    else:
        raise ValueError(
            f"Real data missing expected m/z columns. Available columns: {df.columns}"
        )

    assert mz_col in df.columns, f"Real data missing '{mz_col}' column: {df.columns}"
    assert intensity_col in df.columns, (
        f"Real data missing '{intensity_col}' column: {df.columns}"
    )

    # Add idx column if missing
    if "idx" not in df.columns:
        df = df.with_row_index("idx")

    # Convert to SpectrumPair objects - self-comparison only
    spectrum_pairs = []

    for i in range(min(max_spectra, df.height)):
        row = df.row(i, named=True)
        mz = np.array(row[mz_col], dtype=np.float64)
        intensity = np.array(row[intensity_col], dtype=np.float32)
        idx = int(row["idx"])

        # Self-comparison: each spectrum vs itself (should be similarity ~1.0)
        spectrum_pairs.append(
            SpectrumPair(
                idx_left=idx,
                idx_right=idx,
                mz_left=mz,
                intensity_left=intensity,
                mz_right=mz,
                intensity_right=intensity,
                description=f"real_spectrum_{idx}_self",
            )
        )

    assert len(spectrum_pairs) > 0, "No spectrum pairs generated from real data"

    # Run both implementations
    reference_scores = run_reference_cosine_greedy(
        spectrum_pairs, tolerance_ppm=tolerance_ppm
    )
    fast_cosine_scores = run_fast_cosine_sim(
        spectrum_pairs,
        bin_size=bin_size,
        tolerance_ppm=tolerance_ppm,
        comparison_mode="self",
    )

    # Compare results
    comparison_results = compare_results(reference_scores, fast_cosine_scores)
    stats = compute_agreement_statistics(comparison_results)

    # Print detailed report
    report = stats.format_report(
        bin_size=bin_size,
        tolerance_ppm=tolerance_ppm,
        test_name=f"Real HRMS Data (self-comparison, n={len(spectrum_pairs)} pairs)",
    )
    print(report)

    # Print distribution of differences
    differences = [r.absolute_difference for r in comparison_results]
    print(f"\nDifference distribution:")
    print(f"  Min: {np.min(differences):.6f}")
    print(f"  Median: {np.median(differences):.6f}")
    print(f"  95th percentile: {np.percentile(differences, 95):.6f}")
    print(f"  Max: {np.max(differences):.6f}")

    # Print worst cases
    print(f"\nTop 5 largest differences:")
    worst_cases = sorted(
        comparison_results, key=lambda r: r.absolute_difference, reverse=True
    )[:5]
    for r in worst_cases:
        print(
            f"  {r.description}: "
            f"reference={r.reference_score:.6f}, "
            f"fast_cosine={r.fast_cosine_score:.6f}, "
            f"diff={r.absolute_difference:.6f}"
        )

    # Relaxed assertion for real data self-comparison
    # Why: GPU implementation may have minor numerical differences, but should be close
    assert stats.close_agreement_rate > 0.90, (
        f"Self-comparison on real data should have >90% close agreement (±0.01), "
        f"got {stats.close_agreement_rate * 100:.1f}%"
    )
