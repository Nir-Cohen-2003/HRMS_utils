"""
Centroiding (peak merging) preprocessing for mass spectrometry data.

This module implements standard MS centroiding: merging consecutive peaks
within a tolerance window into single centroid peaks. This is essential for:

1. Preventing one-to-many peak matching (which causes similarities > 1.0)
2. Converting profile-mode data to centroid-mode data
3. Reducing data size and improving performance

Algorithm: Single-linkage clustering along the m/z axis
- Walk through sorted peaks
- Merge consecutive peaks whose m/z gap is below tolerance
- Compute intensity-weighted m/z mean and intensity sum for each cluster

This is the standard approach used in MS data processing (e.g., msconvert).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def centroid_by_neighbor_distance(
    mz: NDArray[np.float64],
    intensity: NDArray[np.float32],
    *,
    tolerance_ppm: float,
    mass_tolerance_cutoff_mz: float = 200.0,
) -> tuple[NDArray[np.float64], NDArray[np.float32]]:
    """
    Centroid a spectrum by merging consecutive peaks within tolerance.
    
    Algorithm (standard MS centroiding):
    1. Sort peaks by m/z (if not already sorted)
    2. Walk through consecutive peaks (i, i+1)
    3. If |mz[i+1] - mz[i]| < tolerance_da(mz[i]):
       - Add peak i+1 to current cluster
    4. Else:
       - Finalize current cluster as a centroid
       - Start new cluster at peak i+1
    5. For each cluster [(m1,I1), (m2,I2), ...]:
       - m_centroid = Σ(mi * Ii) / Σ(Ii)  [intensity-weighted mean]
       - I_centroid = Σ(Ii)                [sum of intensities]
    
    Why contiguous neighbors (not all-pairs):
        If peaks are at [100.0, 100.0005, 100.001]:
        - Check only: |100.0005 - 100.0| and |100.001 - 100.0005|
        - Transitive: if both are within tolerance, all 3 merge
        - But we don't check |100.001 - 100.0| directly
        - This is single-linkage clustering along the m/z axis
        - Standard in MS data processing (msconvert, etc.)
    
    Why intensity-weighted m/z:
        Approximates the mode/center of the peak profile in profile-mode data.
        More representative than simple mean when intensities vary.
    
    Why sum intensities:
        Represents the total area under the peak (total signal energy).
        Preserves normalization approximately.
    
    Complexity: O(n log n) for sorting, O(n) for clustering -> O(n log n) total
    
    Args:
        mz: m/z values (shape: (n_peaks,))
        intensity: intensity values (shape: (n_peaks,))
        tolerance_ppm: MS2 tolerance in ppm for merging
        mass_tolerance_cutoff_mz: minimum m/z for ppm calculation (default 200 Da)
    
    Returns:
        (centroided_mz, centroided_intensity) - sorted by m/z
        
    Example:
        >>> mz = np.array([100.0, 100.0001, 200.0], dtype=np.float64)
        >>> intensity = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        >>> cent_mz, cent_int = centroid_by_neighbor_distance(
        ...     mz, intensity, tolerance_ppm=20.0
        ... )
        >>> # First two peaks merge (gap=0.0001 Da = 1 ppm < 20 ppm tolerance)
        >>> len(cent_mz)  # 2 centroids
        2
        >>> cent_int[0]  # Sum of first two: 10 + 20
        30.0
    """
    assert mz.ndim == 1, f"mz must be 1D, got {mz.ndim}D"
    assert intensity.ndim == 1, f"intensity must be 1D, got {intensity.ndim}D"
    assert mz.shape[0] == intensity.shape[0], (
        f"mz and intensity must have same length, got {mz.shape[0]} vs {intensity.shape[0]}"
    )
    assert float(tolerance_ppm) > 0.0, (
        f"tolerance_ppm must be positive, got {tolerance_ppm}"
    )
    assert float(mass_tolerance_cutoff_mz) > 0.0, (
        f"mass_tolerance_cutoff_mz must be positive, got {mass_tolerance_cutoff_mz}"
    )
    
    n = len(mz)
    if n == 0:
        return (mz.copy(), intensity.copy())
    
    if n == 1:
        return (mz.copy(), intensity.copy())
    
    # Ensure sorted by m/z
    # Why: clustering algorithm assumes sorted order for efficiency
    if not np.all(mz[:-1] <= mz[1:]):
        sort_idx = np.argsort(mz)
        mz = mz[sort_idx]
        intensity = intensity[sort_idx]
    
    # Build clusters by walking through consecutive peaks
    # Why: use list of tuples for clusters, convert to arrays at end
    clusters_mz = []
    clusters_intensity = []
    
    cluster_start = 0
    
    for i in range(n - 1):
        # Compute tolerance for current peak
        # Why: use ppm-based tolerance with cutoff to match rest of package
        effective_mz = max(float(mz[i]), float(mass_tolerance_cutoff_mz))
        tolerance_da = effective_mz * float(tolerance_ppm) * 1e-6
        
        # Check gap to next peak
        gap = float(mz[i + 1] - mz[i])
        
        if gap > tolerance_da:
            # End current cluster, compute centroid
            _finalize_cluster(
                mz[cluster_start:i + 1],
                intensity[cluster_start:i + 1],
                clusters_mz,
                clusters_intensity,
            )
            cluster_start = i + 1
    
    # Don't forget the last cluster
    _finalize_cluster(
        mz[cluster_start:n],
        intensity[cluster_start:n],
        clusters_mz,
        clusters_intensity,
    )
    
    return (
        np.array(clusters_mz, dtype=np.float64),
        np.array(clusters_intensity, dtype=np.float32),
    )


def _finalize_cluster(
    cluster_mz: NDArray[np.float64],
    cluster_int: NDArray[np.float32],
    out_mz: list[float],
    out_intensity: list[float],
) -> None:
    """
    Compute centroid for a cluster and append to output lists.
    
    Why a separate function:
        - Avoids code duplication (called twice in main loop)
        - Keeps main function focused on clustering logic
        - Easier to test centroid computation independently
    
    Args:
        cluster_mz: m/z values in this cluster
        cluster_int: intensity values in this cluster
        out_mz: output list to append centroid m/z
        out_intensity: output list to append centroid intensity
    """
    # Intensity sum
    total_intensity = float(np.sum(cluster_int))
    
    # Intensity-weighted m/z mean
    if total_intensity > 0:
        # Use float64 for weighted mean to avoid precision loss
        weighted_mz = float(
            np.sum(cluster_mz.astype(np.float64) * cluster_int.astype(np.float64)) 
            / total_intensity
        )
    else:
        # Fallback: simple mean (shouldn't happen with valid MS data)
        # Why: all MS intensities should be positive
        weighted_mz = float(np.mean(cluster_mz))
    
    out_mz.append(weighted_mz)
    out_intensity.append(total_intensity)


def centroid_flat_spectra(
    flat_mzs: NDArray[np.float64],
    flat_ints: NDArray[np.float32],
    spec_pos: NDArray[np.int32],
    n_spec: int,
    *,
    tolerance_ppm: float,
    mass_tolerance_cutoff_mz: float = 200.0,
) -> tuple[NDArray[np.float64], NDArray[np.float32], NDArray[np.int32], int]:
    """
    Centroid multiple spectra in flattened format.
    
    Why this function exists:
        The GPU pipeline works with flattened arrays where all spectra
        are concatenated (from polars explode). This avoids unpacking/repacking
        overhead and allows efficient processing of many spectra.
    
    Algorithm:
    1. Find boundaries for each spectrum using spec_pos
    2. For each spectrum, apply centroid_by_neighbor_distance()
    3. Concatenate results back into flat arrays
    
    Why use split instead of loop+mask:
        spec_pos is sorted by construction (from explode operation), so we can
        use np.split which is more efficient than creating n_spec boolean masks.
    
    Args:
        flat_mzs: concatenated m/z values (shape: (total_peaks,))
        flat_ints: concatenated intensities (shape: (total_peaks,))
        spec_pos: spectrum index for each peak (shape: (total_peaks,))
        n_spec: number of spectra
        tolerance_ppm: MS2 tolerance in ppm
        mass_tolerance_cutoff_mz: m/z cutoff for ppm calculation
    
    Returns:
        (centroided_mzs, centroided_ints, centroided_spec_pos, n_spec)
        Same format as input, but with merged peaks
        
    Performance:
        O(n log n) per spectrum for sorting + O(n) for clustering
        Total: O(N log(N/S)) where N=total peaks, S=num spectra
    """
    assert flat_mzs.ndim == 1, f"flat_mzs must be 1D, got {flat_mzs.ndim}D"
    assert flat_ints.ndim == 1, f"flat_ints must be 1D, got {flat_ints.ndim}D"
    assert spec_pos.ndim == 1, f"spec_pos must be 1D, got {spec_pos.ndim}D"
    assert flat_mzs.shape[0] == flat_ints.shape[0] == spec_pos.shape[0], (
        f"flat_mzs, flat_ints, spec_pos must have same length, "
        f"got {flat_mzs.shape[0]}, {flat_ints.shape[0]}, {spec_pos.shape[0]}"
    )
    assert int(n_spec) >= 0, f"n_spec must be non-negative, got {n_spec}"
    
    if n_spec == 0 or len(flat_mzs) == 0:
        return (
            flat_mzs.copy(),
            flat_ints.copy(),
            spec_pos.copy(),
            n_spec,
        )
    
    # Find boundaries where spectrum index changes
    # Why: spec_pos is sorted (e.g., [0,0,0,1,1,2,2,2,2]), so we find change points
    boundaries = np.where(np.diff(spec_pos) != 0)[0] + 1
    boundaries = np.concatenate([[0], boundaries, [len(spec_pos)]])
    
    # Split arrays at boundaries
    mz_per_spectrum = np.split(flat_mzs, boundaries[1:-1])
    int_per_spectrum = np.split(flat_ints, boundaries[1:-1])
    
    centroided_mzs_list = []
    centroided_ints_list = []
    centroided_spec_pos_list = []
    
    for spec_idx, (spec_mz, spec_int) in enumerate(zip(mz_per_spectrum, int_per_spectrum)):
        if len(spec_mz) == 0:
            # Empty spectrum, skip
            continue
        
        # Centroid this spectrum
        cent_mz, cent_int = centroid_by_neighbor_distance(
            spec_mz,
            spec_int,
            tolerance_ppm=tolerance_ppm,
            mass_tolerance_cutoff_mz=mass_tolerance_cutoff_mz,
        )
        
        # Add to output
        centroided_mzs_list.append(cent_mz)
        centroided_ints_list.append(cent_int)
        centroided_spec_pos_list.append(
            np.full(len(cent_mz), spec_idx, dtype=np.int32)
        )
    
    if not centroided_mzs_list:
        # All spectra were empty
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.int32),
            n_spec,
        )
    
    return (
        np.concatenate(centroided_mzs_list),
        np.concatenate(centroided_ints_list),
        np.concatenate(centroided_spec_pos_list),
        n_spec,
    )
