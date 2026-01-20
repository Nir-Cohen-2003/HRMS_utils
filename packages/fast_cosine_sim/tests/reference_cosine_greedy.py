"""
Reference implementation of greedy cosine similarity for mass spectrometry data.

This module provides a correct reference implementation of greedy cosine similarity
that matches the algorithm used in matchms.CosineGreedy, but with ppm-based tolerance
handling consistent with the fast_cosine_sim package.

Key differences from matchms:
- Uses ppm-based tolerance instead of absolute Da tolerance
- Below MASS_TOLERANCE_CUTOFF (200 Da), uses tolerance = 200 * ppm * 1e-6
- Above cutoff, uses tolerance = mz * ppm * 1e-6

Algorithm:
1. Find all peak pairs within tolerance
2. Compute scores (intensity products with optional mz/intensity powers)
3. Sort pairs by score (descending)
4. Greedily assign matches: highest score first, each peak matched at most once
5. Compute normalized cosine score from matched pairs
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from fast_cosine_sim.config import MASS_TOLERANCE_CUTOFF


def collect_peak_pairs_ppm(
    mz1: NDArray[np.float64],
    intensity1: NDArray[np.float32],
    mz2: NDArray[np.float64],
    intensity2: NDArray[np.float32],
    tolerance_ppm: float,
    mz_power: float = 0.0,
    intensity_power: float = 1.0,
) -> NDArray[np.float64] | None:
    """
    Find all pairs of peaks within ppm-based tolerance and compute their scores.
    
    Args:
        mz1: m/z values of first spectrum (shape: (n_peaks1,))
        intensity1: intensity values of first spectrum (shape: (n_peaks1,))
        mz2: m/z values of second spectrum (shape: (n_peaks2,))
        intensity2: intensity values of second spectrum (shape: (n_peaks2,))
        tolerance_ppm: tolerance in ppm for peak matching
        mz_power: power to raise m/z to in score calculation (default: 0.0)
        intensity_power: power to raise intensities to in score calculation (default: 1.0)
    
    Returns:
        Array of shape (n_pairs, 3) where each row is [idx1, idx2, score]
        Returns None if no pairs found within tolerance
        
    Why ppm-based tolerance:
        - For mz >= MASS_TOLERANCE_CUTOFF (200 Da): tolerance_da = mz * tolerance_ppm * 1e-6
        - For mz < MASS_TOLERANCE_CUTOFF: tolerance_da = MASS_TOLERANCE_CUTOFF * tolerance_ppm * 1e-6
        This prevents unrealistically small tolerances at low m/z values.
    """
    assert mz1.ndim == 1, f"mz1 must be 1D, got {mz1.ndim}D"
    assert intensity1.ndim == 1, f"intensity1 must be 1D, got {intensity1.ndim}D"
    assert mz2.ndim == 1, f"mz2 must be 1D, got {mz2.ndim}D"
    assert intensity2.ndim == 1, f"intensity2 must be 1D, got {intensity2.ndim}D"
    assert mz1.shape[0] == intensity1.shape[0], (
        f"mz1 and intensity1 must have same length, got {mz1.shape[0]} vs {intensity1.shape[0]}"
    )
    assert mz2.shape[0] == intensity2.shape[0], (
        f"mz2 and intensity2 must have same length, got {mz2.shape[0]} vs {intensity2.shape[0]}"
    )
    
    if mz1.size == 0 or mz2.size == 0:
        return None
    
    # Apply intensity power transformation
    # Why: matchms applies this before computing products
    int1_transformed = np.power(intensity1, intensity_power, dtype=np.float64)
    int2_transformed = np.power(intensity2, intensity_power, dtype=np.float64)
    
    pairs = []
    
    # For each peak in spectrum 1, find all matching peaks in spectrum 2
    for i, (m1, int1) in enumerate(zip(mz1, int1_transformed)):
        # Compute tolerance for this m/z value
        # Why: ppm tolerance varies with m/z, but we enforce a minimum based on cutoff
        effective_mz = max(float(m1), float(MASS_TOLERANCE_CUTOFF))
        tolerance_da = effective_mz * float(tolerance_ppm) * 1e-6
        
        # Find peaks in spectrum 2 within tolerance
        mz_diff = np.abs(mz2 - m1)
        within_tolerance = mz_diff <= tolerance_da
        
        if not np.any(within_tolerance):
            continue
        
        # For each matching peak, compute score
        for j in np.where(within_tolerance)[0]:
            m2 = mz2[j]
            int2 = int2_transformed[j]
            
            # Compute score: intensity product * mz^mz_power
            # Why: matchms uses this formula for peak pair scoring
            if mz_power == 0.0:
                # Optimize common case: no mz weighting
                score = float(int1 * int2)
            else:
                # Use geometric mean of m/z values for mz weighting
                # Why: this is how matchms handles mz_power
                mz_avg = (m1 + m2) / 2.0
                score = float(int1 * int2 * np.power(mz_avg, mz_power))
            
            pairs.append([i, j, score])
    
    if not pairs:
        return None
    
    return np.array(pairs, dtype=np.float64)


def score_best_matches(
    matching_pairs: NDArray[np.float64],
    mz1: NDArray[np.float64],
    intensity1: NDArray[np.float32],
    mz2: NDArray[np.float64],
    intensity2: NDArray[np.float32],
    mz_power: float = 0.0,
    intensity_power: float = 1.0,
) -> Tuple[float, int]:
    """
    Greedily select best matches and compute normalized cosine score.
    
    Algorithm:
    1. Sort pairs by score (descending)
    2. Greedily assign matches: highest score first
    3. Each peak can be matched at most once
    4. Compute normalized cosine: sum(matched_products) / (norm(spec1) * norm(spec2))
    
    Args:
        matching_pairs: Array of shape (n_pairs, 3) with [idx1, idx2, score] per row
        mz1: m/z values of first spectrum
        intensity1: intensity values of first spectrum
        mz2: m/z values of second spectrum
        intensity2: intensity values of second spectrum
        mz_power: power to raise m/z to in normalization
        intensity_power: power to raise intensities to in normalization
    
    Returns:
        Tuple of (cosine_score, num_matches)
        
    Why greedy:
        The Hungarian algorithm guarantees optimal assignment, but greedy matching
        is much faster and produces very similar results in practice, especially
        for small tolerances typical in HRMS.
    """
    # Sort pairs by score (descending)
    # Why: greedy algorithm processes highest-scoring pairs first
    sorted_indices = np.argsort(matching_pairs[:, 2])[::-1]
    sorted_pairs = matching_pairs[sorted_indices]
    
    # Track which peaks have been matched
    used1 = set()
    used2 = set()
    
    matched_product_sum = 0.0
    num_matches = 0
    
    # Greedily assign matches
    for idx1, idx2, score in sorted_pairs:
        i = int(idx1)
        j = int(idx2)
        
        # Skip if either peak already matched
        if i in used1 or j in used2:
            continue
        
        # Assign match
        used1.add(i)
        used2.add(j)
        matched_product_sum += score
        num_matches += 1
    
    if num_matches == 0:
        return (0.0, 0)
    
    # Compute normalization factors
    # Why: standard cosine similarity normalizes by vector norms
    int1_transformed = np.power(intensity1, intensity_power, dtype=np.float64)
    int2_transformed = np.power(intensity2, intensity_power, dtype=np.float64)
    
    if mz_power == 0.0:
        # Common case: no mz weighting
        norm1 = np.linalg.norm(int1_transformed)
        norm2 = np.linalg.norm(int2_transformed)
    else:
        # Apply mz weighting
        mz1_weighted = np.power(mz1, mz_power / 2.0, dtype=np.float64)
        mz2_weighted = np.power(mz2, mz_power / 2.0, dtype=np.float64)
        norm1 = np.linalg.norm(int1_transformed * mz1_weighted)
        norm2 = np.linalg.norm(int2_transformed * mz2_weighted)
    
    # Compute normalized cosine score
    if norm1 == 0.0 or norm2 == 0.0:
        return (0.0, num_matches)
    
    cosine_score = matched_product_sum / (norm1 * norm2)
    
    return (float(cosine_score), int(num_matches))


def cosine_greedy_ppm(
    mz1: NDArray[np.float64],
    intensity1: NDArray[np.float32],
    mz2: NDArray[np.float64],
    intensity2: NDArray[np.float32],
    tolerance_ppm: float,
    mz_power: float = 0.0,
    intensity_power: float = 1.0,
    apply_centroiding: bool = True,
) -> Tuple[float, int]:
    """
    Compute greedy cosine similarity between two mass spectra using ppm tolerance.
    
    This is the main entry point for the reference implementation.
    
    Args:
        mz1: m/z values of first spectrum
        intensity1: intensity values of first spectrum
        mz2: m/z values of second spectrum
        intensity2: intensity values of second spectrum
        tolerance_ppm: tolerance in ppm for peak matching
        mz_power: power to raise m/z to in score calculation (default: 0.0)
        intensity_power: power to raise intensities to in score calculation (default: 1.0)
        apply_centroiding: if True, centroid spectra before matching (default: True)
    
    Returns:
        Tuple of (cosine_score, num_matches)
        
    Why centroiding by default:
        Centroiding prevents one-to-many peak matching (which causes similarities > 1.0).
        Both reference and GPU implementations should see the same centroided data
        for fair comparison. Centroiding is enabled by default to match GPU pipeline.
        
    Example:
        >>> mz1 = np.array([100.0, 200.0, 300.0], dtype=np.float64)
        >>> intensity1 = np.array([1.0, 0.5, 0.3], dtype=np.float32)
        >>> mz2 = np.array([100.0, 200.0, 300.0], dtype=np.float64)
        >>> intensity2 = np.array([1.0, 0.5, 0.3], dtype=np.float32)
        >>> score, matches = cosine_greedy_ppm(mz1, intensity1, mz2, intensity2, tolerance_ppm=20.0)
        >>> assert np.isclose(score, 1.0)  # Identical spectra
        >>> assert matches == 3
    """
    assert mz1.ndim == 1 and intensity1.ndim == 1, "Inputs must be 1D arrays"
    assert mz2.ndim == 1 and intensity2.ndim == 1, "Inputs must be 1D arrays"
    assert mz1.shape[0] == intensity1.shape[0], "mz1 and intensity1 must have same length"
    assert mz2.shape[0] == intensity2.shape[0], "mz2 and intensity2 must have same length"
    assert float(tolerance_ppm) > 0.0, f"tolerance_ppm must be positive, got {tolerance_ppm}"
    
    # Apply centroiding if requested (default: True)
    # Why: ensures both reference and GPU implementations see the same data
    if apply_centroiding:
        from fast_cosine_sim.centroiding import centroid_by_neighbor_distance
        
        mz1, intensity1 = centroid_by_neighbor_distance(
            mz1,
            intensity1,
            tolerance_ppm=tolerance_ppm,
            mass_tolerance_cutoff_mz=MASS_TOLERANCE_CUTOFF,
        )
        mz2, intensity2 = centroid_by_neighbor_distance(
            mz2,
            intensity2,
            tolerance_ppm=tolerance_ppm,
            mass_tolerance_cutoff_mz=MASS_TOLERANCE_CUTOFF,
        )
    
    # Find all matching pairs within tolerance
    matching_pairs = collect_peak_pairs_ppm(
        mz1, intensity1, mz2, intensity2,
        tolerance_ppm=tolerance_ppm,
        mz_power=mz_power,
        intensity_power=intensity_power,
    )
    
    if matching_pairs is None:
        return (0.0, 0)
    
    # Greedily select best matches and compute score
    score, num_matches = score_best_matches(
        matching_pairs, mz1, intensity1, mz2, intensity2,
        mz_power=mz_power,
        intensity_power=intensity_power,
    )
    
    return (score, num_matches)
