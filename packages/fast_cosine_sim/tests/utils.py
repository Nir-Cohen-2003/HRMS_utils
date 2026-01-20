"""
Shared utilities for reference comparison tests.

This module contains common data structures, helper functions, and utilities
used across all reference comparison test files.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import polars as pl
import pytest
from numpy.typing import NDArray
from reference_cosine_greedy import cosine_greedy_ppm

from fast_cosine_sim import ApproximateGpuBatchedSimilarityConfig
from fast_cosine_sim.gpu_batched_approximate import (
    compute_gpu_batched_approximate_similarity_pairs,
)


def skip_if_no_gpu() -> None:
    """
    Skip if GPU/CuPy not available.

    Why: This package requires GPU; all tests need CUDA.
    """
    try:
        import cupy as cp
    except Exception as exc:
        pytest.skip(f"CuPy import failed: {exc!r}")

    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
        if device_count <= 0:
            pytest.skip(f"CUDA runtime reports 0 devices")
    except Exception as exc:
        pytest.skip(f"CUDA device probe failed: {exc!r}")


@dataclass(frozen=True, slots=True)
class SpectrumPair:
    """Container for a pair of spectra with metadata for testing."""

    idx_left: int
    idx_right: int
    mz_left: NDArray[np.float64]
    intensity_left: NDArray[np.float32]
    mz_right: NDArray[np.float64]
    intensity_right: NDArray[np.float32]
    description: str


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """Results from comparing reference and fast_cosine_sim on a test case."""

    bin_size: float
    tolerance_ppm: float
    description: str
    reference_score: float
    fast_cosine_score: float
    absolute_difference: float

    def is_exact_match(self, threshold: float = 0.001) -> bool:
        """Check if scores agree within threshold."""
        return abs(self.absolute_difference) <= threshold


@dataclass(frozen=True, slots=True)
class AgreementStatistics:
    """Summary statistics for a batch of comparisons."""

    total_pairs: int
    exact_agreement_count: int  # ±0.001
    close_agreement_count: int  # ±0.01
    loose_agreement_count: int  # ±0.1
    mean_absolute_difference: float
    max_absolute_difference: float

    @property
    def exact_agreement_rate(self) -> float:
        """Fraction of pairs with exact agreement (±0.001)."""
        return self.exact_agreement_count / max(self.total_pairs, 1)

    @property
    def close_agreement_rate(self) -> float:
        """Fraction of pairs with close agreement (±0.01)."""
        return self.close_agreement_count / max(self.total_pairs, 1)

    @property
    def loose_agreement_rate(self) -> float:
        """Fraction of pairs with loose agreement (±0.1)."""
        return self.loose_agreement_count / max(self.total_pairs, 1)

    def format_report(
        self, bin_size: float, tolerance_ppm: float, test_name: str
    ) -> str:
        """Generate a formatted report string."""
        lines = [
            f"\n{'=' * 80}",
            f"Test: {test_name}",
            f"Bin size: {bin_size:.2e} Da | MS2 tolerance: {tolerance_ppm} ppm",
            f"{'-' * 80}",
            f"Total pairs evaluated: {self.total_pairs}",
            f"  Exact agreement (±0.001): {self.exact_agreement_rate * 100:.1f}% ({self.exact_agreement_count}/{self.total_pairs})",
            f"  Close agreement (±0.01):  {self.close_agreement_rate * 100:.1f}% ({self.close_agreement_count}/{self.total_pairs})",
            f"  Loose agreement (±0.1):   {self.loose_agreement_rate * 100:.1f}% ({self.loose_agreement_count}/{self.total_pairs})",
            f"",
            f"Mean absolute difference: {self.mean_absolute_difference:.6f}",
            f"Max absolute difference:  {self.max_absolute_difference:.6f}",
            f"{'=' * 80}\n",
        ]
        return "\n".join(lines)


def run_reference_cosine_greedy(
    spectra_pairs: list[SpectrumPair],
    tolerance_ppm: float,
    intensity_power: float = 0.5,
    mz_power: float = 0.0,
    apply_centroiding: bool = True,
) -> dict[tuple[int, int], float]:
    """
    Run reference greedy cosine on spectrum pairs.

    Args:
        spectra_pairs: list of SpectrumPair objects
        tolerance_ppm: MS2 tolerance in ppm
        intensity_power: power to raise intensities to before similarity
        mz_power: power to raise m/z to in score calculation (default: 0.0)
        apply_centroiding: if True, centroid spectra before matching (default: True)

    Returns:
        dict mapping (idx_left, idx_right) -> similarity_score
        
    Why centroiding by default:
        Both reference and GPU implementations should see the same centroided data
        for fair comparison. Centroiding prevents one-to-many matching.
    """
    results = {}

    for pair in spectra_pairs:
        score, num_matches = cosine_greedy_ppm(
            mz1=pair.mz_left,
            intensity1=pair.intensity_left,
            mz2=pair.mz_right,
            intensity2=pair.intensity_right,
            tolerance_ppm=tolerance_ppm,
            intensity_power=intensity_power,
            mz_power=mz_power,
            apply_centroiding=apply_centroiding,
        )

        results[(pair.idx_left, pair.idx_right)] = float(score)

    return results


def run_fast_cosine_sim(
    spectra_pairs: list[SpectrumPair],
    bin_size: float,
    tolerance_ppm: float,
    intensity_power: float = 0.5,
    comparison_mode: str = "cross",
    upper_mass_bound: float = 2000.0,
) -> dict[tuple[int, int], float]:
    """
    Run fast_cosine_sim on spectrum pairs.

    Args:
        spectra_pairs: list of SpectrumPair objects
        bin_size: binning resolution in Da
        tolerance_ppm: MS2 tolerance in ppm (used for adaptive expansion)
        intensity_power: power to raise intensities to before similarity
        comparison_mode: 'self' or 'cross' (parameter kept for compatibility but always uses cross)
        upper_mass_bound: upper m/z bound for binning

    Returns:
        dict mapping (idx_left, idx_right) -> similarity_score
        
    Why cross mode:
        - Cross mode works for both self-matches (idx_left == idx_right) and different matches
        - Self mode excludes diagonal (spectrum vs itself), which fails tests expecting self-similarity=1.0
        - For true self-comparison mode (upper triangle without diagonal), tests should pass one
          set of spectra, not pairs.
    """
    skip_if_no_gpu()

    # Build Polars DataFrames from spectrum pairs
    # Why: Use row index as internal spectrum ID for the GPU computation, but keep
    # a mapping back to the original idx values for result lookup. This ensures:
    # 1. Unique IDs in the dataframe (row 0,1,2,... maps to pairs[0], pairs[1], ...)
    # 2. We can extract the correct diagonal pairs (i,i) from the cross-product
    # 3. We can map back to the original (idx_left, idx_right) for the test framework
    
    left_data = {
        "idx": list(range(len(spectra_pairs))),
        "mz": [pair.mz_left.tolist() for pair in spectra_pairs],
        "intensity": [pair.intensity_left.tolist() for pair in spectra_pairs],
    }

    right_data = {
        "idx": list(range(len(spectra_pairs))),
        "mz": [pair.mz_right.tolist() for pair in spectra_pairs],
        "intensity": [pair.intensity_right.tolist() for pair in spectra_pairs],
    }

    left_df = pl.DataFrame(left_data)
    right_df = pl.DataFrame(right_data)

    # Why: always use cross mode to support both self-matches and different matches.
    # Self mode excludes diagonal, which breaks tests expecting self-similarity scores.
    config = ApproximateGpuBatchedSimilarityConfig(
        upper_mass_bound=float(upper_mass_bound),
        bin_size=float(bin_size),
        approx_threshold=0.0,  # Get all pairs, filter later
        ms2_tolerance_ppm=float(tolerance_ppm),
        comparison_mode="cross",
        spectrum_id_column="idx",
        mz_column="mz",
        intensity_column="intensity",
    )

    # Override intensity power
    config = replace(
        config,
        intensity=replace(config.intensity, power=float(intensity_power)),
    )

    result_df = compute_gpu_batched_approximate_similarity_pairs(
        left=left_df,
        right=right_df,
        config=config,
    )

    # Convert to dict
    # Why: Extract only diagonal pairs (i, i) from the full cross-product.
    # The GPU implementation computes all pairs (i, j), but we only want to compare
    # the specific pairs defined in the test (left[i] vs right[i]).
    results = {}
    for row in result_df.iter_rows(named=True):
        idx_left = int(row["idx_left"])
        idx_right = int(row["idx_right"])
        
        # Only keep diagonal pairs (same row index in left and right)
        if idx_left == idx_right:
            # Map back to original test pair indices
            pair_idx = idx_left
            if pair_idx < len(spectra_pairs):
                pair = spectra_pairs[pair_idx]
                key = (int(pair.idx_left), int(pair.idx_right))
                results[key] = float(row["approx_similarity"])

    return results


def compare_results(
    reference_scores: dict[tuple[int, int], float],
    fast_cosine_scores: dict[tuple[int, int], float],
) -> list[ComparisonResult]:
    """
    Compare reference and fast_cosine_sim results.

    Args:
        reference_scores: dict from (idx_left, idx_right) -> similarity
        fast_cosine_scores: dict from (idx_left, idx_right) -> similarity

    Returns:
        list of ComparisonResult objects for each pair
    """
    # Get all unique keys from both dicts
    all_keys = set(reference_scores.keys()) | set(fast_cosine_scores.keys())

    results = []
    for key in sorted(all_keys):
        reference_score = reference_scores.get(key, 0.0)
        fast_cosine_score = fast_cosine_scores.get(key, 0.0)

        results.append(
            ComparisonResult(
                bin_size=0.0,  # Will be filled by caller
                tolerance_ppm=0.0,  # Will be filled by caller
                description=f"pair_{key[0]}_{key[1]}",
                reference_score=reference_score,
                fast_cosine_score=fast_cosine_score,
                absolute_difference=abs(reference_score - fast_cosine_score),
            )
        )

    return results


def compute_agreement_statistics(
    comparison_results: list[ComparisonResult],
) -> AgreementStatistics:
    """
    Compute agreement statistics from comparison results.

    Args:
        comparison_results: list of ComparisonResult objects

    Returns:
        AgreementStatistics object
    """
    if not comparison_results:
        return AgreementStatistics(
            total_pairs=0,
            exact_agreement_count=0,
            close_agreement_count=0,
            loose_agreement_count=0,
            mean_absolute_difference=0.0,
            max_absolute_difference=0.0,
        )

    total = len(comparison_results)
    exact_count = sum(1 for r in comparison_results if r.is_exact_match(0.001))
    close_count = sum(1 for r in comparison_results if r.is_exact_match(0.01))
    loose_count = sum(1 for r in comparison_results if r.is_exact_match(0.1))

    differences = [r.absolute_difference for r in comparison_results]
    mean_diff = float(np.mean(differences))
    max_diff = float(np.max(differences))

    return AgreementStatistics(
        total_pairs=total,
        exact_agreement_count=exact_count,
        close_agreement_count=close_count,
        loose_agreement_count=loose_count,
        mean_absolute_difference=mean_diff,
        max_absolute_difference=max_diff,
    )
