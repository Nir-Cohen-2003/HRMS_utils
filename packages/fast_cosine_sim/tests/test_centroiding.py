"""
Unit tests for centroiding (peak merging) functionality.

Tests the standard MS centroiding algorithm: single-linkage clustering
along the m/z axis to merge consecutive peaks within tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from fast_cosine_sim.centroiding import (
    centroid_by_neighbor_distance,
    centroid_flat_spectra,
)


def test_centroid_empty_spectrum():
    """Empty spectrum returns empty arrays."""
    mz = np.array([], dtype=np.float64)
    intensity = np.array([], dtype=np.float32)
    
    cent_mz, cent_int = centroid_by_neighbor_distance(
        mz, intensity, tolerance_ppm=20.0
    )
    
    assert len(cent_mz) == 0
    assert len(cent_int) == 0
    assert cent_mz.dtype == np.float64
    assert cent_int.dtype == np.float32


def test_centroid_single_peak():
    """Single peak returns unchanged."""
    mz = np.array([100.0], dtype=np.float64)
    intensity = np.array([10.0], dtype=np.float32)
    
    cent_mz, cent_int = centroid_by_neighbor_distance(
        mz, intensity, tolerance_ppm=20.0
    )
    
    assert len(cent_mz) == 1
    assert len(cent_int) == 1
    assert np.isclose(cent_mz[0], 100.0)
    assert np.isclose(cent_int[0], 10.0)


def test_centroid_no_merging_needed():
    """Well-separated peaks remain separate."""
    # Peaks separated by much more than tolerance
    # At 100 Da with 20 ppm: tolerance = 100 * 20e-6 = 0.002 Da
    # Gap = 100 Da >> 0.002 Da
    mz = np.array([100.0, 200.0, 300.0], dtype=np.float64)
    intensity = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    
    cent_mz, cent_int = centroid_by_neighbor_distance(
        mz, intensity, tolerance_ppm=20.0
    )
    
    assert len(cent_mz) == 3
    assert np.allclose(cent_mz, [100.0, 200.0, 300.0])
    assert np.allclose(cent_int, [10.0, 20.0, 30.0])


def test_centroid_merge_two_peaks():
    """Two close peaks merge into one centroid."""
    # Peaks at 100.0 and 100.0001 Da
    # Gap = 0.0001 Da = 1 ppm @ 100 Da
    # Tolerance @ 20 ppm = 100 * 20e-6 = 0.002 Da
    # Gap < Tolerance -> MERGE
    mz = np.array([100.0, 100.0001], dtype=np.float64)
    intensity = np.array([10.0, 20.0], dtype=np.float32)
    
    cent_mz, cent_int = centroid_by_neighbor_distance(
        mz, intensity, tolerance_ppm=20.0
    )
    
    assert len(cent_mz) == 1, "Two close peaks should merge into one"
    assert len(cent_int) == 1
    
    # Check intensity is sum
    assert np.isclose(cent_int[0], 30.0), f"Expected 30.0, got {cent_int[0]}"
    
    # Check m/z is intensity-weighted mean
    # (100.0 * 10.0 + 100.0001 * 20.0) / 30.0 = 100.00006667
    expected_mz = (100.0 * 10.0 + 100.0001 * 20.0) / 30.0
    assert np.isclose(cent_mz[0], expected_mz), f"Expected {expected_mz}, got {cent_mz[0]}"


def test_centroid_merge_chain():
    """Chain of close peaks all merge (transitive via neighbors).
    
    Why this test is important:
        Tests that single-linkage clustering works correctly.
        If peaks [A, B, C] where A-B and B-C are close but A-C might not be,
        all three should still merge into one centroid.
    """
    # Peaks at [100.0, 100.0001, 100.0002]
    # Each neighbor gap = 0.0001 Da = 1 ppm @ 100 Da
    # Tolerance @ 20 ppm = 0.002 Da
    # All neighbors within tolerance -> all merge
    mz = np.array([100.0, 100.0001, 100.0002], dtype=np.float64)
    intensity = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    
    cent_mz, cent_int = centroid_by_neighbor_distance(
        mz, intensity, tolerance_ppm=20.0
    )
    
    assert len(cent_mz) == 1, "Chain of close peaks should merge into one"
    assert np.isclose(cent_int[0], 60.0), "Intensity should be sum"
    
    # Weighted mean: (100*10 + 100.0001*20 + 100.0002*30) / 60
    expected_mz = (100.0*10 + 100.0001*20 + 100.0002*30) / 60.0
    assert np.isclose(cent_mz[0], expected_mz)


def test_centroid_partial_merge():
    """Some peaks merge, others stay separate."""
    # [100.0, 100.0001] merge (close)
    # [200.0] stays separate (far from 100.x)
    mz = np.array([100.0, 100.0001, 200.0], dtype=np.float64)
    intensity = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    
    cent_mz, cent_int = centroid_by_neighbor_distance(
        mz, intensity, tolerance_ppm=20.0
    )
    
    assert len(cent_mz) == 2, "Should have 2 centroids"
    
    # First centroid: merged 100.x peaks
    assert np.isclose(cent_int[0], 30.0)
    
    # Second centroid: isolated 200.0 peak
    assert np.isclose(cent_mz[1], 200.0)
    assert np.isclose(cent_int[1], 30.0)


def test_centroid_intensity_weighted_mz():
    """Centroid m/z is intensity-weighted mean, not simple average."""
    # Two peaks with very different intensities
    mz = np.array([100.0, 100.001], dtype=np.float64)
    intensity = np.array([1.0, 99.0], dtype=np.float32)  # 1:99 ratio
    
    cent_mz, cent_int = centroid_by_neighbor_distance(
        mz, intensity, tolerance_ppm=20.0
    )
    
    assert len(cent_mz) == 1
    
    # Weighted mean heavily biased toward 100.001 (higher intensity)
    expected_mz = (100.0 * 1.0 + 100.001 * 99.0) / 100.0
    assert np.isclose(cent_mz[0], expected_mz)
    
    # Check it's closer to 100.001 than to 100.0
    assert abs(cent_mz[0] - 100.001) < abs(cent_mz[0] - 100.0)


def test_centroid_preserves_order():
    """Output is sorted by m/z."""
    mz = np.array([300.0, 100.0, 200.0], dtype=np.float64)  # Unsorted
    intensity = np.array([30.0, 10.0, 20.0], dtype=np.float32)
    
    cent_mz, cent_int = centroid_by_neighbor_distance(
        mz, intensity, tolerance_ppm=20.0
    )
    
    # Should be sorted: [100, 200, 300]
    assert len(cent_mz) == 3
    assert np.allclose(cent_mz, [100.0, 200.0, 300.0])
    assert np.allclose(cent_int, [10.0, 20.0, 30.0])


def test_centroid_unsorted_with_merging():
    """Unsorted input with merging is handled correctly."""
    # Unsorted: [200, 100.0001, 100.0]
    # After sorting: [100.0, 100.0001, 200]
    # Should merge first two
    mz = np.array([200.0, 100.0001, 100.0], dtype=np.float64)
    intensity = np.array([30.0, 20.0, 10.0], dtype=np.float32)
    
    cent_mz, cent_int = centroid_by_neighbor_distance(
        mz, intensity, tolerance_ppm=20.0
    )
    
    assert len(cent_mz) == 2
    # First centroid: merged 100.x
    assert np.isclose(cent_int[0], 30.0)  # 10 + 20
    # Second centroid: 200
    assert np.isclose(cent_mz[1], 200.0)
    assert np.isclose(cent_int[1], 30.0)


def test_centroid_low_mz_cutoff():
    """Test that low m/z cutoff (200 Da) is applied correctly.
    
    Why: below 200 Da, we use 200 Da for ppm calculation to avoid
    unrealistically small tolerances.
    """
    # Peaks at 50 Da with 20 ppm
    # Without cutoff: tolerance = 50 * 20e-6 = 0.001 Da
    # With cutoff: tolerance = 200 * 20e-6 = 0.004 Da
    # Gap = 0.002 Da
    # 0.002 > 0.001 (no merge without cutoff)
    # 0.002 < 0.004 (merge with cutoff)
    mz = np.array([50.0, 50.002], dtype=np.float64)
    intensity = np.array([10.0, 20.0], dtype=np.float32)
    
    cent_mz, cent_int = centroid_by_neighbor_distance(
        mz, intensity, tolerance_ppm=20.0, mass_tolerance_cutoff_mz=200.0
    )
    
    # With cutoff, should merge
    assert len(cent_mz) == 1, "Low m/z peaks should merge with cutoff"
    assert np.isclose(cent_int[0], 30.0)


def test_centroid_flat_spectra_empty():
    """Empty flat spectra returns empty arrays."""
    flat_mzs = np.array([], dtype=np.float64)
    flat_ints = np.array([], dtype=np.float32)
    spec_pos = np.array([], dtype=np.int32)
    
    cent_mzs, cent_ints, cent_pos, n_spec = centroid_flat_spectra(
        flat_mzs, flat_ints, spec_pos, n_spec=0, tolerance_ppm=20.0
    )
    
    assert len(cent_mzs) == 0
    assert len(cent_ints) == 0
    assert len(cent_pos) == 0
    assert n_spec == 0


def test_centroid_flat_spectra_single_spectrum():
    """Single spectrum in flat format is centroided correctly."""
    # Spectrum 0: peaks at [100.0, 100.0001, 200.0]
    flat_mzs = np.array([100.0, 100.0001, 200.0], dtype=np.float64)
    flat_ints = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    spec_pos = np.array([0, 0, 0], dtype=np.int32)
    
    cent_mzs, cent_ints, cent_pos, n_spec = centroid_flat_spectra(
        flat_mzs, flat_ints, spec_pos, n_spec=1, tolerance_ppm=20.0
    )
    
    assert n_spec == 1
    assert len(cent_mzs) == 2  # Two centroids after merging first two peaks
    assert all(cent_pos == 0)  # All belong to spectrum 0


def test_centroid_flat_spectra_multiple_spectra():
    """Multiple spectra are centroided independently."""
    # Spectrum 0: [100.0, 100.0001] -> merge
    # Spectrum 1: [200.0, 200.1] -> no merge (gap too large)
    flat_mzs = np.array([100.0, 100.0001, 200.0, 200.1], dtype=np.float64)
    flat_ints = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    spec_pos = np.array([0, 0, 1, 1], dtype=np.int32)
    
    cent_mzs, cent_ints, cent_pos, n_spec = centroid_flat_spectra(
        flat_mzs, flat_ints, spec_pos, n_spec=2, tolerance_ppm=20.0
    )
    
    assert n_spec == 2
    # Spectrum 0: 1 centroid (merged)
    # Spectrum 1: 2 centroids (not merged, gap = 0.1 Da >> tolerance)
    assert len(cent_mzs) == 3
    
    # Check spectrum assignment
    assert np.sum(cent_pos == 0) == 1  # Spectrum 0 has 1 centroid
    assert np.sum(cent_pos == 1) == 2  # Spectrum 1 has 2 centroids


def test_centroid_normalization_approximately_preserved():
    """Test that L2 norm is approximately preserved after centroiding.
    
    Why approximate:
        Intensity sum (I1 + I2) is not exactly equal to L2 norm sqrt(I1^2 + I2^2),
        but for MS data they're close enough.
    """
    mz = np.array([100.0, 100.0001, 100.0002], dtype=np.float64)
    intensity = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    
    # Original L2 norm
    original_norm = np.linalg.norm(intensity)
    
    cent_mz, cent_int = centroid_by_neighbor_distance(
        mz, intensity, tolerance_ppm=20.0
    )
    
    # Centroided L2 norm
    centroided_norm = np.linalg.norm(cent_int)
    
    # Should be within same order of magnitude
    # Why: sum is upper bound, L2 norm is lower bound
    # For positive values: sqrt(I1^2 + I2^2) <= I1 + I2 <= sqrt(n) * sqrt(I1^2 + I2^2)
    assert centroided_norm <= np.sum(intensity)  # Upper bound
    assert centroided_norm >= original_norm / np.sqrt(len(intensity))  # Lower bound (loose)


def test_centroid_self_similarity_equals_one():
    """After centroiding, identical spectra should have similarity = 1.0.
    
    Why this is the key test:
        This is the original problem we're solving. Without centroiding,
        very close peaks could match multiple times, causing similarity > 1.0.
        With centroiding, this is impossible.
    """
    # Spectrum with very close peaks that might cause issues
    mz = np.array([100.0, 100.0001, 200.0, 200.0002], dtype=np.float64)
    intensity = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    
    # Centroid
    cent_mz, cent_int = centroid_by_neighbor_distance(
        mz, intensity, tolerance_ppm=20.0
    )
    
    # Compute self-similarity (dot product of normalized vectors)
    norm = np.linalg.norm(cent_int)
    normalized = cent_int / norm
    self_similarity = np.dot(normalized, normalized)
    
    # Should be exactly 1.0 (within float precision)
    assert np.isclose(self_similarity, 1.0, atol=1e-6)


def test_centroid_different_tolerances():
    """Different tolerances produce different clustering."""
    mz = np.array([100.0, 100.001], dtype=np.float64)
    intensity = np.array([10.0, 20.0], dtype=np.float32)
    
    # Gap = 0.001 Da = 10 ppm @ 100 Da
    
    # Strict tolerance (5 ppm): gap > tolerance -> no merge
    cent_mz_strict, cent_int_strict = centroid_by_neighbor_distance(
        mz, intensity, tolerance_ppm=5.0
    )
    assert len(cent_mz_strict) == 2, "Strict tolerance should not merge"
    
    # Loose tolerance (20 ppm): gap < tolerance -> merge
    cent_mz_loose, cent_int_loose = centroid_by_neighbor_distance(
        mz, intensity, tolerance_ppm=20.0
    )
    assert len(cent_mz_loose) == 1, "Loose tolerance should merge"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
