#!/usr/bin/env python
"""
SimMS-based sparse exact cosine kernel and batching helper.

This module uses a lightweight CUDA kernel inspired by SimMS' cosine kernel to
compute exact cosine similarities only for candidate (row, col) pairs produced
by the approximate stage. It returns a COO sparse matrix with the filled
scores, enabling a two-stage pipeline: approximate (fast, sparse) -> exact
(focused on candidates).

Key choices:
- Uses numba.cuda directly; assumes spectra are individually sorted by m/z.
- Omits precursor shift and mass-shift logic for simplicity per request.
- Fails fast when required inputs are missing.
- Provides a benchmark that runs: approximate -> CPU exact (baseline) ->
  optimized_cosine kernel -> new SimMS sparse kernel, and compares timings and
  score deltas.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Sequence

import numba
import numpy as np
import polars as pl
import scipy.sparse as sp
from approximate_similarity import proximate_all_vs_all_pairs
from numba import cuda
from optimized_cosine import run_greedy_cosine_fast


@dataclass
class SparseExactConfig:
    """Configuration for sparse exact cosine on GPU.

    Attributes
    ----------
    tolerance_ppm : float
        MS2 tolerance in ppm for matching peaks.
    mz_power : float
        Exponent for m/z scaling in cosine computation.
    intensity_power : float
        Exponent for intensity scaling in cosine computation.
    match_limit : int
        Maximum matches per pair (bounds local buffers).
    n_max_peaks : int | None
        Max peaks per spectrum; if None uses per-side max length.
    pair_batch_size : int
        Number of candidate pairs per GPU launch to bound memory.
    """

    tolerance_ppm: float = 10.0
    mz_power: float = 0.0
    intensity_power: float = 0.5
    match_limit: int = 1024
    n_max_peaks: int | None = None
    pair_batch_size: int = 65_536  # ~64k pairs per launch

    def __post_init__(self) -> None:
        assert self.tolerance_ppm > 0, "tolerance_ppm must be positive"
        assert self.match_limit > 0, "match_limit must be positive"
        assert self.pair_batch_size > 0, "pair_batch_size must be positive"


def _pad_spectra_to_dense(
    mz_list: Sequence[np.ndarray],
    int_list: Sequence[np.ndarray],
    n_max_peaks: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Pad spectra to dense arrays.

    Returns
    -------
    mz_dense : np.ndarray, shape (n_spectra, max_peaks)
    int_dense : np.ndarray, shape (n_spectra, max_peaks)
    lengths : np.ndarray, shape (n_spectra,)
    max_peaks : int
    """
    assert len(mz_list) == len(int_list), "mz_list and int_list length mismatch"
    lengths = np.array([len(m) for m in mz_list], dtype=np.int32)
    max_peaks = int(np.max(lengths)) if lengths.size > 0 else 0
    if n_max_peaks is not None:
        max_peaks = min(max_peaks, int(n_max_peaks))
    mz_dense = np.zeros((len(mz_list), max_peaks), dtype=np.float32)
    int_dense = np.zeros((len(int_list), max_peaks), dtype=np.float32)
    for i, (mz, inten) in enumerate(zip(mz_list, int_list)):
        l_i = min(len(mz), max_peaks)
        mz_dense[i, :l_i] = mz[:l_i]
        int_dense[i, :l_i] = inten[:l_i]
        lengths[i] = l_i
    return mz_dense, int_dense, lengths, max_peaks


def _compute_norms(
    mz_dense: np.ndarray,
    int_dense: np.ndarray,
    lengths: np.ndarray,
    mz_power: float,
    intensity_power: float,
) -> np.ndarray:
    """Compute cosine norms for each spectrum."""
    weights = (mz_dense**mz_power) * (int_dense**intensity_power)
    sq = (weights**2).sum(axis=1)
    norms = np.sqrt(sq).astype(np.float32)
    assert np.all(norms > 0), "All spectra must have non-zero norm"
    return norms


def _build_sparse_exact_cosine_kernel(
    match_limit_const: int,
    max_peaks_const: int,
):
    assert match_limit_const > 0, "match_limit_const must be positive"
    assert max_peaks_const > 0, "max_peaks_const must be positive"

    @cuda.jit
    def _kernel(
        mz_left,
        int_left,
        len_left,
        mz_right,
        int_right,
        len_right,
        norms_left,
        norms_right,
        pair_left_idx,
        pair_right_idx,
        tolerance_ppm,
        mz_power,
        int_power,
        out_scores,
        out_matches,
    ):
        """Compute exact cosine for candidate pairs (1D grid).

        Matches up to `match_limit_const` peaks per pair; overflow is signaled
        by returning a negative match count.
        """
        idx = cuda.grid(1)
        n_pairs = pair_left_idx.size
        if idx >= n_pairs:
            return

        li = pair_left_idx[idx]
        ri = pair_right_idx[idx]

        l_len = len_left[li]
        r_len = len_right[ri]
        if l_len == 0 or r_len == 0:
            out_scores[idx] = 0.0
            out_matches[idx] = 0
            return

        match_r = cuda.local.array(match_limit_const, numba.int32)
        match_q = cuda.local.array(match_limit_const, numba.int32)
        match_vals = cuda.local.array(match_limit_const, numba.float32)
        used_r = cuda.local.array(max_peaks_const, numba.boolean)
        used_q = cuda.local.array(max_peaks_const, numba.boolean)

        for k in range(max_peaks_const):
            used_r[k] = False
            used_q[k] = False

        count = numba.int32(0)
        overflow = numba.int32(0)

        i = numba.int32(0)
        j = numba.int32(0)
        while i < l_len and j < r_len:
            mz_l = mz_left[li, i]
            mz_r = mz_right[ri, j]

            base = mz_l
            if base < 200.0:
                base = 200.0
            tol_da = base * tolerance_ppm * 1e-6

            diff = mz_l - mz_r
            if diff > tol_da:
                j += 1
                continue
            if diff < -tol_da:
                i += 1
                continue

            int_l = int_left[li, i]
            int_r = int_right[ri, j]
            amp_l = math.pow(mz_l, mz_power) * math.pow(int_l, int_power)
            amp_r = math.pow(mz_r, mz_power) * math.pow(int_r, int_power)

            if count < match_limit_const:
                match_r[count] = i
                match_q[count] = j
                match_vals[count] = amp_l * amp_r
            else:
                overflow = 1
                break
            count += 1
            i += 1
            j += 1

        score = numba.float32(0.0)
        used_matches = numba.int32(0)
        for k in range(count):
            r_idx = match_r[k]
            q_idx = match_q[k]
            if (not used_r[r_idx]) and (not used_q[q_idx]):
                used_r[r_idx] = True
                used_q[q_idx] = True
                score += match_vals[k]
                used_matches += 1

        norm = norms_left[li] * norms_right[ri]
        if norm > 0:
            out_scores[idx] = score / norm
        else:
            out_scores[idx] = 0.0
        out_matches[idx] = -used_matches if overflow else used_matches

    return _kernel


def sparse_exact_cosine_from_pairs_gpu(
    left_mz: Sequence[np.ndarray],
    left_int: Sequence[np.ndarray],
    right_mz: Sequence[np.ndarray],
    right_int: Sequence[np.ndarray],
    pair_left_idx: np.ndarray,
    pair_right_idx: np.ndarray,
    config: SparseExactConfig | None = None,
) -> sp.coo_matrix:
    """Compute sparse exact cosine on GPU for provided candidate pairs.

    Parameters
    ----------
    left_mz, left_int, right_mz, right_int : sequences of np.ndarray
        Sorted spectra (per spectrum) for left/right sides.
    pair_left_idx, pair_right_idx : np.ndarray[int]
        Indices into left/right spectra (same length).
    config : SparseExactConfig
        Kernel and batching settings.

    Returns
    -------
    scipy.sparse.coo_matrix
        COO matrix with shape (len(left_mz), len(right_mz)) containing scores.
    """
    assert pair_left_idx.shape == pair_right_idx.shape, "pair index shape mismatch"
    if config is None:
        config = SparseExactConfig()

    n_left = len(left_mz)
    n_right = len(right_mz)
    assert n_left > 0 and n_right > 0, "left/right spectra must be non-empty"
    if pair_left_idx.size == 0:
        return sp.coo_matrix((n_left, n_right))
    min_left = int(pair_left_idx.min())
    max_left = int(pair_left_idx.max())
    min_right = int(pair_right_idx.min())
    max_right = int(pair_right_idx.max())
    assert min_left >= 0 and min_right >= 0, "pair indices must be non-negative"
    assert max_left < n_left, "pair_left_idx has out-of-bounds entries"
    assert max_right < n_right, "pair_right_idx has out-of-bounds entries"

    mz_l, int_l, len_l, max_peaks_l = _pad_spectra_to_dense(
        left_mz, left_int, config.n_max_peaks
    )
    mz_r, int_r, len_r, max_peaks_r = _pad_spectra_to_dense(
        right_mz, right_int, config.n_max_peaks
    )
    assert max_peaks_l == max_peaks_r, (
        "Left and right must share padded peak dimension for the kernel"
    )

    norms_l = _compute_norms(
        mz_l, int_l, len_l, config.mz_power, config.intensity_power
    )
    norms_r = _compute_norms(
        mz_r, int_r, len_r, config.mz_power, config.intensity_power
    )

    d_mz_l = cuda.to_device(mz_l)
    d_int_l = cuda.to_device(int_l)
    d_len_l = cuda.to_device(len_l)
    d_mz_r = cuda.to_device(mz_r)
    d_int_r = cuda.to_device(int_r)
    d_len_r = cuda.to_device(len_r)
    d_norms_l = cuda.to_device(norms_l)
    d_norms_r = cuda.to_device(norms_r)

    kernel = _build_sparse_exact_cosine_kernel(
        match_limit_const=config.match_limit,
        max_peaks_const=max_peaks_l,
    )
    scores: list[np.ndarray] = []
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []

    threads_per_block = 256
    n_pairs_total = pair_left_idx.size
    batch_size = config.pair_batch_size

    for start in range(0, n_pairs_total, batch_size):
        end = min(start + batch_size, n_pairs_total)
        batch_left = pair_left_idx[start:end].astype(np.int32, copy=False)
        batch_right = pair_right_idx[start:end].astype(np.int32, copy=False)

        d_left_idx = cuda.to_device(batch_left)
        d_right_idx = cuda.to_device(batch_right)
        d_scores = cuda.device_array(shape=batch_left.shape, dtype=np.float32)
        d_matches = cuda.device_array(shape=batch_left.shape, dtype=np.int32)

        blocks = (batch_left.size + threads_per_block - 1) // threads_per_block
        kernel[blocks, threads_per_block](
            d_mz_l,
            d_int_l,
            d_len_l,
            d_mz_r,
            d_int_r,
            d_len_r,
            d_norms_l,
            d_norms_r,
            d_left_idx,
            d_right_idx,
            np.float32(config.tolerance_ppm),
            np.float32(config.mz_power),
            np.float32(config.intensity_power),
            d_scores,
            d_matches,
        )

        scores.append(d_scores.copy_to_host())
        rows.append(batch_left.astype(np.int64, copy=False))
        cols.append(batch_right.astype(np.int64, copy=False))

    data = np.concatenate(scores) if scores else np.array([], dtype=np.float32)
    row = np.concatenate(rows) if rows else np.array([], dtype=np.int64)
    col = np.concatenate(cols) if cols else np.array([], dtype=np.int64)

    return sp.coo_matrix((data, (row, col)), shape=(len(left_mz), len(right_mz)))


def _extract_lists_from_df(
    df: pl.DataFrame,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Extract mz/intensity lists from a Polars dataframe."""
    assert "cleaned_normalized_mz" in df.columns, "mz column missing"
    assert "cleaned_normalized_intensity" in df.columns, "intensity column missing"
    return df["cleaned_normalized_mz"].to_list(), df[
        "cleaned_normalized_intensity"
    ].to_list()


def benchmark_sparse_exact_pipeline(
    df: pl.DataFrame,
    threshold: float = 0.5,
    ms2_tolerance_ppm: float = 10.0,
    approx_bin_size: float = 0.0001,
    approx_upper: float = 1000.0,
    verbose: bool = True,
) -> None:
    """Benchmark approximate stage + three exact methods."""
    assert 0.0 <= threshold <= 1.0, "threshold must be in [0,1]"
    df = df.with_row_index("idx").with_columns(pl.col("idx").cast(pl.Int64))

    # Approximate stage
    t0 = time.perf_counter()
    approx_res, approx_timings = proximate_all_vs_all_pairs(
        df,
        df,
        threshold=threshold,
        ms2_tolerance_ppm=ms2_tolerance_ppm,
        proximate_bin_size=approx_bin_size,
        proximate_bin_upper=approx_upper,
        use_gpu=True,
        return_timings=True,
    )
    t1 = time.perf_counter()
    time_approx = t1 - t0
    if verbose:
        print(
            f"[approx] pairs={len(approx_res)} time={time_approx:.3f}s timings={approx_timings}"
        )

    if len(approx_res) == 0:
        print("No candidate pairs from approximate stage; skipping exact benchmarks.")
        return

    pair_left = approx_res["idx"].to_numpy()
    pair_right = approx_res["idx_right"].to_numpy()

    # Prepare spectra lists
    mz_left, int_left = _extract_lists_from_df(df)
    mz_right, int_right = mz_left, int_left  # symmetric benchmark

    # Exact CPU baseline (dotprod via Polars extension)
    pairs_df = pl.DataFrame(
        {
            "idx": pair_left,
            "idx_right": pair_right,
            "order": np.arange(len(pair_left), dtype=np.int64),
        }
    ).lazy()
    df_lazy = df.lazy()
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
                ms2_tolerance_in_ppm=ms2_tolerance_ppm,
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
    time_exact_cpu = t_cpu1 - t_cpu0
    if verbose:
        print(f"[exact-cpu] pairs={len(cpu_scores)} time={time_exact_cpu:.3f}s")

    # Exact via optimized_cosine kernel (GPU dense)
    def _gather_dense_for_pairs(
        mz_seq: Sequence[np.ndarray], int_seq: Sequence[np.ndarray], idx_seq: np.ndarray
    ) -> tuple[
        cuda.cudadrv.devicearray.DeviceNDArray,
        cuda.cudadrv.devicearray.DeviceNDArray,
        cuda.cudadrv.devicearray.DeviceNDArray,
    ]:
        lens = np.array([len(mz_seq[i]) for i in idx_seq], dtype=np.int32)
        max_peaks = int(np.max(lens))
        mz_arr = np.zeros((len(idx_seq), max_peaks), dtype=np.float32)
        int_arr = np.zeros_like(mz_arr)
        for k, sidx in enumerate(idx_seq):
            l = lens[k]
            mz_arr[k, :l] = mz_seq[sidx][:l]
            int_arr[k, :l] = int_seq[sidx][:l]
        return cuda.to_device(mz_arr), cuda.to_device(int_arr), cuda.to_device(lens)

    d_mz_a, d_int_a, d_len_a = _gather_dense_for_pairs(mz_left, int_left, pair_left)
    d_mz_b, d_int_b, d_len_b = _gather_dense_for_pairs(mz_right, int_right, pair_right)

    # Warm-up
    _ = run_greedy_cosine_fast(
        d_mz_a,
        d_int_a,
        d_len_a,
        d_mz_b,
        d_int_b,
        d_len_b,
        tolerance=ms2_tolerance_ppm,
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
        tolerance=ms2_tolerance_ppm,
        shift=0.0,
        mz_power=0.0,
        int_power=0.5,
    )
    cuda.synchronize()
    scores_opt = scores_opt_dev.copy_to_host().astype(np.float32)
    t_opt1 = time.perf_counter()
    time_exact_opt = t_opt1 - t_opt0
    if verbose:
        print(f"[exact-optimized] pairs={len(scores_opt)} time={time_exact_opt:.3f}s")

    # Exact via new sparse SimMS-style kernel
    simms_cfg = SparseExactConfig(
        tolerance_ppm=ms2_tolerance_ppm,
        mz_power=0.0,
        intensity_power=0.5,
        match_limit=1024,
        n_max_peaks=None,
        pair_batch_size=65_536,
    )
    t_simms0 = time.perf_counter()
    coo_scores = sparse_exact_cosine_from_pairs_gpu(
        mz_left, int_left, mz_right, int_right, pair_left, pair_right, config=simms_cfg
    )
    t_simms1 = time.perf_counter()
    time_exact_simms = t_simms1 - t_simms0
    simms_scores = coo_scores.data.astype(np.float32)
    if verbose:
        print(f"[exact-simms] pairs={len(simms_scores)} time={time_exact_simms:.3f}s")

    # Metrics vs CPU baseline
    def _diff_stats(name: str, test_scores: np.ndarray) -> None:
        diff = np.abs(cpu_scores - test_scores)
        print(
            f"[{name}] avg_abs_diff={diff.mean():.6f} >1e-2={(diff > 1e-2).mean() * 100:.2f}% >1e-3={(diff > 1e-3).mean() * 100:.2f}%"
        )

    _diff_stats("optimized", scores_opt)
    _diff_stats("simms_sparse", simms_scores)

    print(
        "\nSummary timings (s): "
        f"approx={time_approx:.3f}, "
        f"exact_cpu={time_exact_cpu:.3f}, "
        f"exact_optimized_gpu={time_exact_opt:.3f}, "
        f"exact_simms_sparse_gpu={time_exact_simms:.3f}"
    )


if __name__ == "__main__":
    # Example: load a small parquet via Polars before running.
    spectra = (
        pl.scan_parquet("/gpfs01/work/nircoh/HRMS_utils/data/fraghub.parquet")
        .head(20000)
        .collect()
    )
    benchmark_sparse_exact_pipeline(spectra)
