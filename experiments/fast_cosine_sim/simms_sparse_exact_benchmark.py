#!/usr/bin/env python
"""
Benchmark approximate (batched GPU) + exact pipelines (CPU, dense GPU, SimMS sparse GPU).

This module moves the benchmarking logic out of `simms_sparse_exact.py` and
switches the approximate stage to the batched GPU path defined in `bathced_gpu.py`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import cupy as cp
import cupyx.scipy.sparse as cps
import numpy as np
import polars as pl
from approximate_similarity import (
    SimilarityConfig,
    _expand_csr_horizontal_adaptive_gpu,
    _normalize_csr_rows_inplace_gpu,
    _sparse_bin_spectra_df_to_csr,
)
from bathced_gpu import BatchedGPUConfig, _yield_batches_dynamic
from numba import cuda
from optimized_cosine import run_greedy_cosine_fast
from simms_sparse_exact import SparseExactConfig, sparse_exact_cosine_from_pairs_gpu


@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark pipeline."""

    threshold: float = 0.5
    ms2_tolerance_ppm: float = 10.0
    approx_bin_size: float = 0.0001
    approx_upper: float = 1000.0
    verbose: bool = True
    # Approximate batching
    batch_size: int = 1000
    target_gpu_mem_ratio: float = 0.6
    max_peaks_per_batch: int | None = None
    gpu_batch_write_interval: int = 10
    # SimMS exact settings
    match_limit: int = 1024
    pair_batch_size: int = 65_536


def _diff_stats(name: str, cpu_scores: np.ndarray, test_scores: np.ndarray) -> None:
    diff = np.abs(cpu_scores - test_scores)
    print(
        f"[{name}] avg_abs_diff={diff.mean():.6f} "
        f">1e-2={(diff > 1e-2).mean() * 100:.2f}% "
        f">1e-3={(diff > 1e-3).mean() * 100:.2f}%"
    )


def _extract_lists_from_df(
    df: pl.DataFrame,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Extract mz/intensity lists from a Polars dataframe."""
    assert "cleaned_normalized_mz" in df.columns, (
        "cleaned_normalized_mz column is required"
    )
    assert "cleaned_normalized_intensity" in df.columns, (
        "cleaned_normalized_intensity column is required"
    )
    return df["cleaned_normalized_mz"].to_list(), df[
        "cleaned_normalized_intensity"
    ].to_list()


def _gather_dense_for_pairs(
    mz_seq: Sequence[np.ndarray], int_seq: Sequence[np.ndarray], idx_seq: np.ndarray
) -> tuple[
    cuda.cudadrv.devicearray.DeviceNDArray,
    cuda.cudadrv.devicearray.DeviceNDArray,
    cuda.cudadrv.devicearray.DeviceNDArray,
]:
    """Materialize dense padded arrays for a set of pair indices and move them to device."""
    lengths = np.array([len(mz_seq[i]) for i in idx_seq], dtype=np.int32)
    max_peaks = int(np.max(lengths))
    mz_arr = np.zeros((len(idx_seq), max_peaks), dtype=np.float32)
    int_arr = np.zeros_like(mz_arr)
    for k, sidx in enumerate(idx_seq):
        l_i = lengths[k]
        mz_arr[k, :l_i] = mz_seq[sidx][:l_i]
        int_arr[k, :l_i] = int_seq[sidx][:l_i]
    return cuda.to_device(mz_arr), cuda.to_device(int_arr), cuda.to_device(lengths)


def _compute_dynamic_max_peaks(
    approx_cfg: SimilarityConfig,
    target_gpu_mem_ratio: float,
    user_max_peaks: int | None,
) -> int:
    """Estimate peaks per batch to respect GPU memory."""
    free_mem, _ = cp.cuda.Device(0).mem_info
    target_mem = free_mem * target_gpu_mem_ratio
    expansion_factor = 1.0
    if approx_cfg.ms2_tolerance_ppm is not None:
        window_da = approx_cfg.upper_mass_bound * approx_cfg.ms2_tolerance_ppm * 1e-6
        expansion_factor = max(1.0, (2 * window_da) / approx_cfg.bin_size)
        expansion_factor *= 2.0
    bytes_per_peak = 12  # float32 + int32 + overhead
    estimated_max_peaks = int(target_mem / (bytes_per_peak * expansion_factor))
    max_peaks = (
        estimated_max_peaks
        if user_max_peaks is None
        else min(estimated_max_peaks, user_max_peaks)
    )
    return max(max_peaks, 100_000)


def _run_batched_approximate_gpu(
    df: pl.DataFrame,
    approx_cfg: SimilarityConfig,
    batched_cfg: BatchedGPUConfig,
    verbose: bool,
) -> pl.DataFrame:
    """
    Run batched GPU approximate similarity and return candidate pairs as a Polars DataFrame.
    """
    assert len(df) > 0, "Input dataframe must be non-empty for approximate search"
    # Build CSR
    left_csr, _ = _sparse_bin_spectra_df_to_csr(
        df,
        "cleaned_normalized_mz",
        "cleaned_normalized_intensity",
        upper_bound=approx_cfg.upper_mass_bound,
        intensity_power=approx_cfg.intensity_power,
        bin_size=approx_cfg.bin_size,
    )
    global_idxs = df["idx"].cast(pl.Int64).to_numpy()
    max_peaks = _compute_dynamic_max_peaks(
        approx_cfg, batched_cfg.target_gpu_mem_ratio, batched_cfg.max_peaks_per_batch
    )
    batches = list(
        _yield_batches_dynamic(
            left_csr, global_idxs, max_peaks, min_batch_size=batched_cfg.batch_size
        )
    )
    total_pairs: list[np.ndarray] = []
    total_pairs_right: list[np.ndarray] = []
    total_scores: list[np.ndarray] = []
    approx_threshold = approx_cfg.approx_threshold
    t0 = time.perf_counter()
    for j, (r_start, r_end, r_csr, r_idxs) in enumerate(batches):
        R_gpu = cps.csr_matrix(r_csr)
        _normalize_csr_rows_inplace_gpu(R_gpu)
        if approx_cfg.ms2_tolerance_ppm is not None:
            R_gpu = _expand_csr_horizontal_adaptive_gpu(
                R_gpu,
                approx_cfg.bin_size,
                approx_cfg.ms2_tolerance_ppm,
                approx_cfg.nbins,
            )
        for i, (l_start, l_end, l_csr, l_idxs) in enumerate(batches):
            if i > j:
                continue
            L_gpu = cps.csr_matrix(l_csr)
            _normalize_csr_rows_inplace_gpu(L_gpu)
            sim = L_gpu.dot(R_gpu.T)
            mask = sim.data >= approx_threshold
            if int(mask.sum()) == 0:
                continue
            out_data = sim.data[mask]
            out_cols = sim.indices[mask]
            indices_in_data = cp.nonzero(mask)[0]
            out_rows = cp.searchsorted(sim.indptr, indices_in_data, side="right") - 1
            left_pairs = l_idxs[cp.asnumpy(out_rows).astype(np.int64)]
            right_pairs = r_idxs[cp.asnumpy(out_cols).astype(np.int64)]
            scores = cp.asnumpy(out_data).astype(np.float32)
            if i == j:
                diag_mask = left_pairs != right_pairs
                left_pairs = left_pairs[diag_mask]
                right_pairs = right_pairs[diag_mask]
                scores = scores[diag_mask]
                upper_mask = left_pairs < right_pairs
                left_pairs = left_pairs[upper_mask]
                right_pairs = right_pairs[upper_mask]
                scores = scores[upper_mask]
            if len(left_pairs) == 0:
                continue
            total_pairs.append(left_pairs)
            total_pairs_right.append(right_pairs)
            total_scores.append(scores)
        cp.get_default_memory_pool().free_all_blocks()
        if verbose:
            elapsed = time.perf_counter() - t0
            print(
                f"[approx-batched] processed right batch {j + 1}/{len(batches)} in {elapsed:.3f}s"
            )
    if not total_pairs:
        return pl.DataFrame({"idx": [], "idx_right": [], "proximate_similarity": []})
    return pl.DataFrame(
        {
            "idx": np.concatenate(total_pairs),
            "idx_right": np.concatenate(total_pairs_right),
            "proximate_similarity": np.concatenate(total_scores),
        }
    )


def benchmark_batched_simms_exact(
    df: pl.DataFrame, config: BenchmarkConfig | None = None
) -> None:
    """Run approximate (batched GPU) then exact benchmarks (CPU, dense GPU, SimMS sparse GPU)."""
    if config is None:
        config = BenchmarkConfig()
    assert 0.0 <= config.threshold <= 1.0, "threshold must be within [0, 1]"
    df_idx = df.with_row_index("idx").with_columns(pl.col("idx").cast(pl.Int64))
    approx_cfg = SimilarityConfig(
        upper_mass_bound=config.approx_upper,
        bin_size=config.approx_bin_size,
        ms2_tolerance_ppm=config.ms2_tolerance_ppm,
        intensity_power=0.5,
        threshold=config.threshold,
    )
    batched_cfg = BatchedGPUConfig(
        batch_size=config.batch_size,
        threshold=config.threshold,
        gpu_batch_write_interval=config.gpu_batch_write_interval,
        target_gpu_mem_ratio=config.target_gpu_mem_ratio,
        max_peaks_per_batch=config.max_peaks_per_batch,
        approx_config=approx_cfg,
    )
    if config.verbose:
        print(
            f"Running batched approximate with batch_size={batched_cfg.batch_size}, "
            f"max_peaks_per_batch={batched_cfg.max_peaks_per_batch}, "
            f"approx_threshold={approx_cfg.approx_threshold}"
        )
    t_approx0 = time.perf_counter()
    approx_pairs = _run_batched_approximate_gpu(
        df_idx, approx_cfg, batched_cfg, config.verbose
    )
    t_approx1 = time.perf_counter()
    if config.verbose:
        print(
            f"[approx-batched] pairs={len(approx_pairs)} time={t_approx1 - t_approx0:.3f}s "
            f"(approx_threshold={approx_cfg.approx_threshold})"
        )
    if len(approx_pairs) == 0:
        print("No candidate pairs from approximate stage; skipping exact benchmarks.")
        return
    pair_left = approx_pairs["idx"].to_numpy()
    pair_right = approx_pairs["idx_right"].to_numpy()
    mz_left, int_left = _extract_lists_from_df(df_idx)
    mz_right, int_right = mz_left, int_left
    pairs_order = np.arange(len(pair_left), dtype=np.int64)
    pairs_df = pl.DataFrame(
        {"idx": pair_left, "idx_right": pair_right, "order": pairs_order}
    ).lazy()
    df_lazy = df_idx.lazy()
    pairs_struct = (
        pairs_df.join(df_lazy, on="idx")
        .join(df_lazy, left_on="idx_right", right_on="idx", suffix="_right")
        .with_columns(
            spectra=pl.struct(
                mz1=pl.col("cleaned_normalized_mz"),
                intensities1=pl.col("cleaned_normalized_intensity"),
                mz2=pl.col("cleaned_normalized_mz_right"),
                intensities2=pl.col("cleaned_normalized_intensity_right"),
                precursor_mz1=pl.col("precursor_mz"),
                precursor_mz2=pl.col("precursor_mz_right"),
            )
        )
        .with_columns(
            dotprod=pl.col("spectra").spectral_similarity.dotprod_similarity(
                ms2_tolerance_in_ppm=config.ms2_tolerance_ppm,
                clean_spectra_first=False,
                ignore_precursor=True,
            )
        )
        .select(["order", "dotprod"])
    )
    t_cpu0 = time.perf_counter()
    cpu_scores = (
        pairs_struct.sort("order")
        .collect()
        .get_column("dotprod")
        .to_numpy()
        .astype(np.float32)
    )
    t_cpu1 = time.perf_counter()
    if config.verbose:
        print(f"[exact-cpu] pairs={len(cpu_scores)} time={t_cpu1 - t_cpu0:.3f}s")
    d_mz_a, d_int_a, d_len_a = _gather_dense_for_pairs(mz_left, int_left, pair_left)
    d_mz_b, d_int_b, d_len_b = _gather_dense_for_pairs(mz_right, int_right, pair_right)
    _ = run_greedy_cosine_fast(
        d_mz_a,
        d_int_a,
        d_len_a,
        d_mz_b,
        d_int_b,
        d_len_b,
        tolerance=config.ms2_tolerance_ppm,
        shift=0.0,
        mz_power=0.0,
        int_power=0.5,
    )
    cuda.synchronize()
    t_opt0 = time.perf_counter()
    scores_opt_dev = run_greedy_cosine_fast(
        d_mz_a,
        d_int_a,
        d_len_a,
        d_mz_b,
        d_int_b,
        d_len_b,
        tolerance=config.ms2_tolerance_ppm,
        shift=0.0,
        mz_power=0.0,
        int_power=0.5,
    )
    cuda.synchronize()
    scores_opt = scores_opt_dev.copy_to_host().astype(np.float32)
    t_opt1 = time.perf_counter()
    if config.verbose:
        print(f"[exact-optimized] pairs={len(scores_opt)} time={t_opt1 - t_opt0:.3f}s")
    simms_cfg = SparseExactConfig(
        tolerance_ppm=config.ms2_tolerance_ppm,
        mz_power=0.0,
        intensity_power=0.5,
        match_limit=config.match_limit,
        n_max_peaks=None,
        pair_batch_size=config.pair_batch_size,
    )
    t_simms0 = time.perf_counter()
    coo_scores = sparse_exact_cosine_from_pairs_gpu(
        mz_left, int_left, mz_right, int_right, pair_left, pair_right, config=simms_cfg
    )
    t_simms1 = time.perf_counter()
    simms_scores = coo_scores.data.astype(np.float32)
    if config.verbose:
        print(
            f"[exact-simms] pairs={len(simms_scores)} time={t_simms1 - t_simms0:.3f}s"
        )

    _diff_stats("optimized", cpu_scores, scores_opt)
    _diff_stats("simms_sparse", cpu_scores, simms_scores)
    print(
        "\nSummary timings (s): "
        f"approx={t_approx1 - t_approx0:.3f}, "
        f"exact_cpu={t_cpu1 - t_cpu0:.3f}, "
        f"exact_optimized_gpu={t_opt1 - t_opt0:.3f}, "
        f"exact_simms_sparse_gpu={t_simms1 - t_simms0:.3f}"
    )


if __name__ == "__main__":
    spectra = (
        pl.scan_parquet("/gpfs01/work/nircoh/HRMS_utils/data/fraghub.parquet")
        .head(50_000)
        .collect()
    )
    benchmark_batched_simms_exact(spectra)
