"""
GPU batched approximate cosine similarity (approximate stage only).

This package intentionally does NOT depend on `hrms_utils`. It provides the
GPU-batched approximate candidate-pair generation step up to (and excluding)
the exact similarity computation/refinement.
"""

from __future__ import annotations

from .gpu_approximate_similarity import (
    GPUApproximateConfig,
    batched_approximate_similarity_gpu,
)

# Backward compatibility alias for old function name
compute_gpu_batched_approximate_similarity_pairs = batched_approximate_similarity_gpu

__all__ = [
    "GPUApproximateConfig",
    "batched_approximate_similarity_gpu",
    "compute_gpu_batched_approximate_similarity_pairs",
]
