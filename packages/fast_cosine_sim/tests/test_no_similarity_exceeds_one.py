"""
Additional test to verify that similarities never exceed 1.0 after centroiding.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.utils import (
    SpectrumPair,
    run_fast_cosine_sim,
    run_reference_cosine_greedy,
    skip_if_no_gpu,
)


@pytest.mark.runtime
def test_no_similarities_exceed_one():
    """
    Verify that all similarities are <= 1.0 after centroiding.
    
    Why this test:
        The original problem was that self-comparison could yield
        similarities > 1.0 due to one-to-many peak matching.
        Centroiding prevents this by ensuring each peak can only
        match once.
    """
    skip_if_no_gpu()
    
    # Create problematic test cases with very close peaks
    test_pairs = []
    
    # Case 1: Two identical very close peaks (self-comparison)
    for i, offset in enumerate([0.0, 0.0001, 0.0005, 0.001]):
        mz = np.array([100.0, 100.0 + offset, 200.0, 200.0 + offset], dtype=np.float64)
        intensity = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        
        test_pairs.append(
            SpectrumPair(
                idx_left=i,
                idx_right=i,
                mz_left=mz.copy(),
                intensity_left=intensity.copy(),
                mz_right=mz.copy(),
                intensity_right=intensity.copy(),
                description=f"self_comparison_offset_{offset}",
            )
        )
    
    # Case 2: Dense clusters of peaks
    for i, n_peaks in enumerate([5, 10, 20]):
        base_mz = 300.0
        # Create n_peaks very close together (within 0.001 Da)
        mz = np.array([base_mz + j * 0.0001 for j in range(n_peaks)], dtype=np.float64)
        intensity = np.ones(n_peaks, dtype=np.float32)
        
        test_pairs.append(
            SpectrumPair(
                idx_left=10 + i,
                idx_right=10 + i,
                mz_left=mz.copy(),
                intensity_left=intensity.copy(),
                mz_right=mz.copy(),
                intensity_right=intensity.copy(),
                description=f"dense_cluster_{n_peaks}_peaks",
            )
        )
    
    # Run both implementations with various bin sizes
    for bin_size in [1e-5, 3e-5, 1e-4]:
        # GPU implementation
        fast_scores = run_fast_cosine_sim(
            test_pairs,
            bin_size=bin_size,
            tolerance_ppm=20.0,
            comparison_mode="cross",
        )
        
        # Reference implementation
        ref_scores = run_reference_cosine_greedy(
            test_pairs,
            tolerance_ppm=20.0,
            apply_centroiding=True,
        )
        
        # Check that all similarities are <= 1.0 (with float32 tolerance)
        # Why: float32 arithmetic can produce values like 1.0000001 due to rounding
        float32_epsilon = 1e-6
        
        for key, score in fast_scores.items():
            assert score <= 1.0 + float32_epsilon, (
                f"GPU similarity > 1.0: {score:.10f} for pair {key} "
                f"with bin_size={bin_size}"
            )
            # Should be very close to 1.0 for self-comparison
            assert score >= 0.99, (
                f"GPU self-similarity unexpectedly low: {score:.6f} for pair {key}"
            )
        
        for key, score in ref_scores.items():
            assert score <= 1.0 + float32_epsilon, (
                f"Reference similarity > 1.0: {score:.10f} for pair {key}"
            )
            assert score >= 0.99, (
                f"Reference self-similarity unexpectedly low: {score:.6f} for pair {key}"
            )
        
        print(f"\nbin_size={bin_size:.2e}: All {len(fast_scores)} similarities in [0.99, 1.0] ✓")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
