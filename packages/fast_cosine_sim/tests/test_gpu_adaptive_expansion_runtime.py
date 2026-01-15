"""
Runtime test: verify GPU adaptive expansion enables matching within MS2 tolerance
across adjacent bins.

This test is intentionally "integration-ish": it calls the public API
`compute_gpu_batched_approximate_similarity_pairs` and checks that when
`ms2_tolerance_ppm` is enabled, a pair that *should* match within tolerance is
returned, and when it's disabled, the same pair is *not* returned.

Notes / assumptions:
- We construct spectra that differ by ~0.01 Da around m/z ~500, which is 20 ppm.
- With bin_size=0.01, these peaks land in adjacent bins (500.00 vs 500.01),
  so without expansion they won't match, but with adaptive expansion they should.
- The approximate stage bins, L2-normalizes, optionally expands the RHS, then
  computes sparse dot-product similarities.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from fast_cosine_sim import ApproximateGpuBatchedSimilarityConfig
from fast_cosine_sim.gpu_batched_approximate import (
    compute_gpu_batched_approximate_similarity_pairs,
)


def _skip_if_no_gpu() -> None:
    """
    Skip helper that reports *why* CUDA/CuPy isn't usable in this environment.

    Rationale:
    - In many CI/dev setups you "have a GPU" but the Python env can't see it due to
      missing CUDA runtime, incompatible drivers, wrong CuPy build, or CUDA
      visibility config (e.g. CUDA_VISIBLE_DEVICES).
    - Pytest skips should be actionable, not silent.
    """
    try:
        import cupy as cp
    except Exception as exc:
        pytest.skip(f"CuPy import failed: {exc!r}")

    diagnostics: list[str] = []

    # Basic environment visibility info
    try:
        import os

        diagnostics.append(
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
        )
    except Exception as exc:
        diagnostics.append(f"env diagnostics failed: {exc!r}")

    # CUDA runtime visibility
    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
        diagnostics.append(f"cp.cuda.runtime.getDeviceCount()={device_count}")
        if device_count <= 0:
            pytest.skip("CUDA runtime reports 0 devices. " + " | ".join(diagnostics))
    except Exception as exc:
        diagnostics.append(f"cp.cuda.runtime.getDeviceCount() raised {exc!r}")

        # Try a second, often-more-informative probe
        try:
            _ = cp.cuda.Device(0).compute_capability
            diagnostics.append("cp.cuda.Device(0) is accessible")
        except Exception as exc2:
            diagnostics.append(f"cp.cuda.Device(0) probe raised {exc2!r}")

        pytest.skip("CUDA device probe failed. " + " | ".join(diagnostics))

    # If we got here, CuPy imported and CUDA sees at least one device.


def _make_single_peak_df(*, idx: int, mz: float, intensity: float) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "idx": [idx],
            # list-valued columns: one spectrum with one peak
            "mz": [[float(mz)]],
            "intensity": [[float(intensity)]],
        }
    )


@pytest.mark.runtime
def test_gpu_adaptive_expansion_enables_adjacent_bin_match() -> None:
    _skip_if_no_gpu()

    # Left has 500.00; Right has 500.01 (adjacent bins for bin_size=0.01 with rint rule)
    left = _make_single_peak_df(idx=1, mz=500.00, intensity=1.0)
    right = _make_single_peak_df(idx=2, mz=500.001, intensity=1.0)

    # Common base config
    # - upper_mass_bound must exceed the peak m/z
    # - approx_threshold=1.0 because single identical normalized vectors should dot to 1
    base_kwargs = dict(
        upper_mass_bound=1000.0,
        bin_size=0.001,
        approx_threshold=1.0,
        comparison_mode="cross",
        spectrum_id_column="idx",
        mz_column="mz",
        intensity_column="intensity",
    )

    # Case A: expansion disabled => should NOT match (peaks in different bins)
    config_no_expand = ApproximateGpuBatchedSimilarityConfig(
        **base_kwargs,
        ms2_tolerance_ppm=5.0,
    )
    out_no = compute_gpu_batched_approximate_similarity_pairs(
        left, config_no_expand, right=right
    )
    assert out_no.height == 0, (
        "Expected no candidates without adaptive expansion, but got "
        f"{out_no.height} rows: {out_no}"
    )

    # Case B: expansion enabled with 20 ppm at m/z~500 => tolerance_da=0.01 => window=ceil(0.01/0.01)=1 bin.
    # This should expand RHS peak into adjacent bins and enable matching.
    config_expand = ApproximateGpuBatchedSimilarityConfig(
        **base_kwargs,
        ms2_tolerance_ppm=20.0,
        # Use the experiments cutoff behavior (default is 200.0); keep explicit for test clarity.
        mass_tolerance_cutoff_mz=200.0,
    )
    out_yes = compute_gpu_batched_approximate_similarity_pairs(
        left, config_expand, right=right
    )

    assert out_yes.height == 1, (
        "Expected exactly 1 candidate with adaptive expansion enabled, "
        f"but got {out_yes.height} rows: {out_yes}"
    )

    row = out_yes.row(0, named=True)
    assert int(row["idx_left"]) == 1
    assert int(row["idx_right"]) == 2
    assert np.isclose(float(row["approx_similarity"]), 1.0, atol=1e-6), (
        "Expected similarity ~1.0 for single-peak match after normalization/expansion, "
        f"got {row['approx_similarity']}"
    )
