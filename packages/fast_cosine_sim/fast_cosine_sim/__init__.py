"""
GPU batched approximate cosine similarity (approximate stage only).

This package intentionally does NOT depend on `hrms_utils`. It provides the
GPU-batched approximate candidate-pair generation step up to (and excluding)
the exact similarity computation/refinement.
"""

from __future__ import annotations

from .config import (
    ApproximateGpuBatchedSimilarityConfig,
    ApproximateGpuDtypesConfig,
    BatchSizingConfig,
    IntensityTransformConfig,
    LoggingConfig,
    OutputParquetConfig,
)
from .gpu_batched_approximate import compute_gpu_batched_approximate_similarity_pairs

__all__ = [
    "ApproximateGpuBatchedSimilarityConfig",
    "ApproximateGpuDtypesConfig",
    "BatchSizingConfig",
    "IntensityTransformConfig",
    "LoggingConfig",
    "OutputParquetConfig",
    "compute_gpu_batched_approximate_similarity_pairs",
]
