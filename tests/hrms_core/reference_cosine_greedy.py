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

# Mass tolerance cutoff in Da - below this value, tolerance is computed as cutoff * ppm * 1e-6
MASS_TOLERANCE_CUTOFF = 200.0


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
        # Why: matchms applies mz_power to individual peaks, then multiplies
        # Formula: score = (mz1^p * int1^q) * (mz2^p * int2^q)
        for j in np.where(within_tolerance)[0]:
            m2 = mz2[j]
            int2 = int2_transformed[j]
            
            # Compute weighted intensities for each peak separately
            power_prod_spec1 = (m1 ** mz_power) * int1
            power_prod_spec2 = (m2 ** mz_power) * int2
            score = float(power_prod_spec1 * power_prod_spec2)
            
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
    # Use mergesort (stable) to match matchms exactly
    sorted_indices = np.argsort(matching_pairs[:, 2], kind="mergesort")[::-1]
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
    # Why: matchms computes spec_power = mz^mz_power * intensity^intensity_power
    # then normalizes by the L2 norm: sqrt(sum(spec_power^2))
    int1_transformed = np.power(intensity1, intensity_power, dtype=np.float64)
    int2_transformed = np.power(intensity2, intensity_power, dtype=np.float64)
    
    # Apply mz power to match matchms: spec_power = mz^mz_power * intensity^intensity_power
    spec1_power = np.power(mz1, mz_power, dtype=np.float64) * int1_transformed
    spec2_power = np.power(mz2, mz_power, dtype=np.float64) * int2_transformed
    
    norm1 = np.linalg.norm(spec1_power)
    norm2 = np.linalg.norm(spec2_power)
    
    # Compute normalized cosine score
    if norm1 == 0.0 or norm2 == 0.0:
        return (0.0, num_matches)
    
    cosine_score = matched_product_sum / (norm1 * norm2)
    
    return (float(cosine_score), int(num_matches))


def _need_centroid(
    mz: NDArray[np.float64],
    tolerance_ppm: float,
) -> bool:
    """
    Check if spectrum needs centroiding based on C's logic.
    
    Args:
        mz: m/z values (must be sorted)
        tolerance_ppm: tolerance in ppm for peak merging
        
    Returns:
        True if any adjacent peaks are within ppm tolerance
    """
    if mz.size < 2 or tolerance_ppm <= 0.0:
        return False
    
    for i in range(len(mz) - 1):
        tolerance_da = mz[i + 1] * tolerance_ppm * 1e-6
        if mz[i + 1] - mz[i] <= tolerance_da:
            return True
    return False


def _centroid_spectrum(
    mz: NDArray[np.float64],
    intensity: NDArray[np.float32],
    tolerance_ppm: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float32]]:
    """
    Centroid a spectrum using the logic from the C code.
    
    Merges peaks within ppm tolerance using intensity-weighted average m/z.
    
    Args:
        mz: m/z values
        intensity: intensity values
        tolerance_ppm: tolerance in ppm for peak merging
        
    Returns:
        Tuple of (centroided_mz, centroided_intensity)
    """
    if mz.size == 0:
        return mz, intensity
    
    # Work with copies
    peaks_mz = mz.copy()
    peaks_int = intensity.copy().astype(np.float64)
    
    # Get indices sorted by intensity descending
    argsort = np.argsort(peaks_int)[::-1]
    
    for i in range(len(argsort)):
        idx = argsort[i]
        
        if peaks_int[idx] <= 0.0:
            continue  # Already merged
        
        current_mz = peaks_mz[idx]
        
        # Calculate tolerance windows
        if tolerance_ppm > 0.0:
            mz_delta_left = current_mz * tolerance_ppm * 1e-6
            mz_delta_right = current_mz * tolerance_ppm / (1e6 - tolerance_ppm)
        else:
            mz_delta_left = 0.0
            mz_delta_right = 0.0
        
        # Find range of peaks to potentially merge
        idx_left = idx
        while idx_left > 0 and (current_mz - peaks_mz[idx_left - 1]) <= mz_delta_left:
            idx_left -= 1
        
        idx_right = idx
        while idx_right < len(peaks_mz) - 1 and (peaks_mz[idx_right + 1] - current_mz) <= mz_delta_right:
            idx_right += 1
        
        # Count merge candidates
        merge_candidates = 0
        for j in range(idx_left, idx_right + 1):
            if peaks_int[j] > 0.0:
                merge_candidates += 1
        
        # Merge if there are multiple peaks in range
        if merge_candidates > 1:
            intensity_sum = 0.0
            intensity_weighted_mz_sum = 0.0
            
            for j in range(idx_left, idx_right + 1):
                if peaks_int[j] > 0.0:
                    intensity_sum += peaks_int[j]
                    intensity_weighted_mz_sum += peaks_int[j] * peaks_mz[j]
                    peaks_int[j] = 0.0  # Mark as merged
            
            if intensity_sum > 0.0:
                peaks_mz[idx] = intensity_weighted_mz_sum / intensity_sum
                peaks_int[idx] = intensity_sum
    
    # Remove merged peaks (intensity <= 0)
    mask = peaks_int > 0.0
    peaks_mz = peaks_mz[mask]
    peaks_int = peaks_int[mask]
    
    # Sort by m/z
    sort_idx = np.argsort(peaks_mz)
    peaks_mz = peaks_mz[sort_idx]
    peaks_int = peaks_int[sort_idx].astype(np.float32)
    
    return peaks_mz, peaks_int


def _apply_centroiding_if_needed(
    mz: NDArray[np.float64],
    intensity: NDArray[np.float32],
    tolerance_ppm: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float32]]:
    """
    Apply centroiding to spectrum if needed.
    
    Repeatedly applies centroiding until no more peaks need merging.
    
    Args:
        mz: m/z values
        intensity: intensity values
        tolerance_ppm: tolerance in ppm for peak merging
        
    Returns:
        Tuple of (centroided_mz, centroided_intensity)
    """
    if tolerance_ppm <= 0.0 or mz.size < 2:
        return mz, intensity
    
    current_mz = mz.copy()
    current_intensity = intensity.copy()
    
    # Iteratively centroid until no more peaks need merging
    while _need_centroid(current_mz, tolerance_ppm):
        current_mz, current_intensity = _centroid_spectrum(
            current_mz, current_intensity, tolerance_ppm
        )
    
    return current_mz, current_intensity


def cosine_greedy_ppm(
    mz1: NDArray[np.float64],
    intensity1: NDArray[np.float32],
    mz2: NDArray[np.float64],
    intensity2: NDArray[np.float32],
    tolerance_ppm: float,
    mz_power: float = 0.0,
    intensity_power: float = 1.0,
    apply_centroiding: bool = False,
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
        apply_centroiding: whether to apply centroiding before matching (default: False)
    
    Returns:
        Tuple of (cosine_score, num_matches)
        
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
    
    # Apply centroiding if requested
    if apply_centroiding:
        mz1, intensity1 = _apply_centroiding_if_needed(mz1, intensity1, 2.0 * tolerance_ppm)
        mz2, intensity2 = _apply_centroiding_if_needed(mz2, intensity2, 2.0 * tolerance_ppm)
    
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
