#!/usr/bin/env python
"""
Batched approximate + exact cosine similarity pipeline with dynamic GPU batching.

This module provides a complete pipeline that:
1. Runs approximate (proximate) similarity on GPU with batching
2. Filters candidate pairs above a threshold
3. Runs exact cosine similarity either on CPU or GPU with dynamic batching

The GPU exact path uses dynamic batching that accounts for the number of peaks
in spectra to efficiently utilize GPU memory.
"""

from __future__ import annotations

import time
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
from batched_utils import BatchedGPUConfig, _log_message_to_file, _yield_batches_dynamic
from numba import cuda
from numpy.typing import NDArray
from optimized_cosine import run_greedy_cosine_fast

INT32_MAX = np.iinfo(np.int32).max


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
    """
    Materialize dense padded arrays for a set of pair indices and move them to device.

    Why: The optimized_cosine kernels require dense padded arrays on device.
    This gathers and pads spectra for a given set of indices.
    """
    lengths = np.array([len(mz_seq[i]) for i in idx_seq], dtype=np.int32)
    max_peaks = int(np.max(lengths))
    mz_arr = np.zeros((len(idx_seq), max_peaks), dtype=np.float32)
    int_arr = np.zeros_like(mz_arr)
    for k, sidx in enumerate(idx_seq):
        l_i = lengths[k]
        mz_arr[k, :l_i] = mz_seq[sidx][:l_i]
        int_arr[k, :l_i] = int_seq[sidx][:l_i]
    return cuda.to_device(mz_arr), cuda.to_device(int_arr), cuda.to_device(lengths)


def _compute_dynamic_max_peaks_approximate(
    approx_cfg: SimilarityConfig,
    target_gpu_mem_ratio: float,
    user_max_peaks: int | None,
) -> int:
    """
    Estimate max peaks per batch for approximate stage (binned CSR matrices).

    Why: Approximate stage uses binned sparse matrices with expansion from tolerance windows.
    GPU memory is limited, so we need to estimate how many peaks can fit in a batch.
    """
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


def _compute_dynamic_max_peaks_exact(
    target_gpu_mem_ratio: float,
    user_max_peaks: int | None,
) -> int:
    """
    Estimate max total peaks per batch for exact GPU stage (unbinned data).

    Why: Exact stage uses dense padded arrays of unbinned m/z and intensity values.
    Memory usage = n_spectra × max_peaks_per_spectrum × 2 arrays × 4 bytes (float32)
    We track unique spectra per batch, so this limits total peaks from unique spectra.

    This is different from approximate stage which works with sparse binned matrices.
    """
    free_mem, _ = cp.cuda.Device(0).mem_info
    target_mem = free_mem * target_gpu_mem_ratio

    # For exact cosine: 2 dense arrays (mz, intensity) per spectrum
    # Each array is float32 (4 bytes) and padded to max_peaks
    # Plus length array (int32, 4 bytes per spectrum)
    # Memory = n_unique_spectra × max_peaks × 8 bytes + n_unique_spectra × 4 bytes
    # Conservatively estimate: ~10 bytes per peak (accounting for padding and overhead)
    bytes_per_peak = 10

    estimated_max_peaks = int(target_mem / bytes_per_peak)

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

    Why: Approximate similarity on GPU with batching handles large libraries that don't
    fit in GPU memory all at once. Dynamic batching accounts for variable spectrum sizes.
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

    # Use int32 global indices for memory efficiency; fail fast if overflow would occur.
    # Why: IDs are "per spectrum", and for up to ~10M spectra int32 is sufficient and
    # materially reduces memory footprint for large candidate pair sets.
    idx_max = df.select(pl.col("idx").max()).item()
    assert idx_max is not None, (
        "idx max was None; dataframe appears empty or idx missing unexpectedly"
    )
    assert int(idx_max) <= INT32_MAX, (
        f"idx values must fit in int32 (<= {INT32_MAX}), got max idx={idx_max}. "
        "Reduce library size, shard input, or change index dtype policy."
    )
    global_idxs = df["idx"].cast(pl.Int32).to_numpy().astype(np.int32, copy=False)

    max_peaks = _compute_dynamic_max_peaks_approximate(
        approx_cfg, batched_cfg.target_gpu_mem_ratio, batched_cfg.max_peaks_per_batch
    )
    batches = list(
        _yield_batches_dynamic(
            left_csr,
            global_idxs.astype(np.int32, copy=False),
            max_peaks,
            min_batch_size=batched_cfg.batch_size,
        )
    )
    total_pairs: list[np.ndarray] = []
    total_pairs_right: list[np.ndarray] = []
    total_scores: list[np.ndarray] = []
    approx_threshold = approx_cfg.approx_threshold
    t0 = time.perf_counter()
    for j, (r_start, r_end, r_csr, r_idxs) in enumerate(batches):
        # Ensure batch id arrays are int32 (they may come in as int64 depending on upstream)
        r_idxs = np.asarray(r_idxs, dtype=np.int32)

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
            # Ensure batch id arrays are int32 (they may come in as int64 depending on upstream)
            l_idxs = np.asarray(l_idxs, dtype=np.int32)

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

            # These are row/col positions inside the (batch-local) similarity matrix.
            # Use int32 throughout when indexing the batch-global id arrays.
            left_pairs = l_idxs[cp.asnumpy(out_rows).astype(np.int32)]
            right_pairs = r_idxs[cp.asnumpy(out_cols).astype(np.int32)]
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

            # Ensure stored pair ids remain int32 (avoid accidental upcast from numpy ops)
            total_pairs.append(np.asarray(left_pairs, dtype=np.int32))
            total_pairs_right.append(np.asarray(right_pairs, dtype=np.int32))
            total_scores.append(scores)

        cp.get_default_memory_pool().free_all_blocks()
        if verbose:
            elapsed = time.perf_counter() - t0
            _log_message_to_file(
                f"[approx-batched] processed right batch {j + 1}/{len(batches)} in {elapsed:.3f}s"
            )
    if not total_pairs:
        return pl.DataFrame({"idx": [], "idx_right": [], "proximate_similarity": []})
    return pl.DataFrame(
        {
            "idx": np.concatenate(total_pairs).astype(np.int32, copy=False),
            "idx_right": np.concatenate(total_pairs_right).astype(np.int32, copy=False),
            "proximate_similarity": np.concatenate(total_scores),
        }
    )


def _batch_pairs_by_unique_spectra_peaks(
    pair_left: np.ndarray,
    pair_right: np.ndarray,
    mz_left: list[np.ndarray],
    mz_right: list[np.ndarray],
    max_total_peaks: int,
    min_pairs_per_batch: int = 100,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Batch pairs dynamically based on unique spectra peak counts for exact GPU computation.

    Why: For exact cosine, we work with unbinned m/z arrays (not CSR matrices). GPU memory
    usage depends on the unique spectra referenced by pairs in a batch, not the pairs themselves.
    Multiple pairs can reference the same spectra, so we need to track unique spectra per batch.

    This is different from approximate stage batching which works with binned CSR matrices.

    Args:
        pair_left: Array of left spectrum indices
        pair_right: Array of right spectrum indices
        mz_left: List of m/z arrays for left library
        mz_right: List of m/z arrays for right library
        max_total_peaks: Maximum total peaks from unique spectra per batch
        min_pairs_per_batch: Minimum pairs per batch (unless end of data)

    Returns:
        List of (left_indices, right_indices) tuples for each batch.
    """
    n_pairs = len(pair_left)
    assert n_pairs == len(pair_right), (
        f"pair_left and pair_right must have same length, got {n_pairs} vs {len(pair_right)}"
    )

    if n_pairs == 0:
        return []

    batches = []
    start_idx = 0

    while start_idx < n_pairs:
        end_idx = start_idx
        unique_left = set()
        unique_right = set()
        cumsum_peaks = 0

        # Greedily add pairs while tracking unique spectra peaks
        while end_idx < n_pairs:
            left_idx = pair_left[end_idx]
            right_idx = pair_right[end_idx]

            # Calculate additional peaks if we add this pair
            additional_peaks = 0
            if left_idx not in unique_left:
                additional_peaks += len(mz_left[left_idx])
            if right_idx not in unique_right:
                additional_peaks += len(mz_right[right_idx])

            # Check if adding this pair would exceed limit
            if cumsum_peaks + additional_peaks > max_total_peaks:
                # Only stop if we have at least min_pairs_per_batch
                if end_idx - start_idx >= min_pairs_per_batch:
                    break

            # Add this pair
            unique_left.add(left_idx)
            unique_right.add(right_idx)
            cumsum_peaks += additional_peaks
            end_idx += 1

            # Stop if we've accumulated enough pairs and would exceed on next iteration
            if (
                end_idx - start_idx >= min_pairs_per_batch
                and cumsum_peaks > max_total_peaks * 0.8
            ):
                break

        # Ensure at least one pair per batch
        if end_idx == start_idx:
            end_idx = start_idx + 1

        batches.append((pair_left[start_idx:end_idx], pair_right[start_idx:end_idx]))
        start_idx = end_idx

    return batches


def _run_exact_cosine_gpu_batched(
    pair_left: np.ndarray,
    pair_right: np.ndarray,
    mz_left: list[np.ndarray],
    int_left: list[np.ndarray],
    mz_right: list[np.ndarray],
    int_right: list[np.ndarray],
    config: SimilarityConfig,
    max_peaks_per_batch: int = 10_000_000,
    verbose: bool = False,
) -> np.ndarray:
    """
    Run exact cosine similarity on GPU with dynamic batching by peak count.

    Why: GPU memory is limited. By batching pairs based on their total peak counts,
    we can process large numbers of pairs without OOM errors while maintaining
    high GPU utilization.

    Args:
        pair_left: Indices into left library
        pair_right: Indices into right library
        mz_left: List of m/z arrays for left library
        int_left: List of intensity arrays for left library
        mz_right: List of m/z arrays for right library
        int_right: List of intensity arrays for right library
        config: SimilarityConfig with ms2_tolerance_ppm and intensity_power
        max_peaks_per_batch: Maximum total peaks to process in one GPU batch
        verbose: Print timing information

    Returns:
        Array of exact cosine similarity scores for each pair
    """
    assert len(pair_left) == len(pair_right), (
        f"pair_left and pair_right must have same length, got {len(pair_left)} vs {len(pair_right)}"
    )
    assert config.ms2_tolerance_ppm is not None, (
        "ms2_tolerance_ppm must be set in config for exact cosine"
    )

    if len(pair_left) == 0:
        return np.array([], dtype=np.float32)

    # Batch pairs by unique spectra peak counts
    pair_batches = _batch_pairs_by_unique_spectra_peaks(
        pair_left,
        pair_right,
        mz_left,
        mz_right,
        max_peaks_per_batch,
        min_pairs_per_batch=100,
    )

    if verbose:
        # Count unique spectra across all pairs
        unique_left_all = len(np.unique(pair_left))
        unique_right_all = len(np.unique(pair_right))
        total_peaks_left = sum(len(mz_left[i]) for i in np.unique(pair_left))
        total_peaks_right = sum(len(mz_right[i]) for i in np.unique(pair_right))
        _log_message_to_file(
            f"[exact-gpu-batched] Processing {len(pair_left)} pairs in {len(pair_batches)} batches"
        )
        _log_message_to_file(
            f"  Unique spectra: left={unique_left_all} ({total_peaks_left} peaks), "
            f"right={unique_right_all} ({total_peaks_right} peaks)"
        )
        _log_message_to_file(f"  Max peaks per batch: {max_peaks_per_batch}")

    all_scores = []

    for batch_idx, (batch_left, batch_right) in enumerate(pair_batches):
        # For this batch, find unique spectra needed
        batch_left_unique, batch_left_inv = np.unique(batch_left, return_inverse=True)
        batch_right_unique, batch_right_inv = np.unique(
            batch_right, return_inverse=True
        )

        # Gather dense arrays for unique spectra in this batch
        d_mz_a, d_int_a, d_len_a = _gather_dense_for_pairs(
            mz_left, int_left, batch_left_unique
        )
        d_mz_b, d_int_b, d_len_b = _gather_dense_for_pairs(
            mz_right, int_right, batch_right_unique
        )

        # Map batch pairs to unique indices
        pair_a_indices = batch_left_inv.astype(np.int32)
        pair_b_indices = batch_right_inv.astype(np.int32)

        # Run exact cosine on GPU
        t0 = time.perf_counter()
        scores_dev = run_greedy_cosine_fast(
            d_mz_a,
            d_int_a,
            d_len_a,
            d_mz_b,
            d_int_b,
            d_len_b,
            tolerance=config.ms2_tolerance_ppm,
            shift=0.0,
            mz_power=0.0,
            int_power=config.intensity_power,
            pair_a_indices=pair_a_indices,
            pair_b_indices=pair_b_indices,
        )
        cuda.synchronize()
        scores = scores_dev.copy_to_host().astype(np.float32)
        t1 = time.perf_counter()

        if verbose:
            # Count unique spectra and their peaks in this batch
            batch_unique_left = len(batch_left_unique)
            batch_unique_right = len(batch_right_unique)
            batch_peaks_left = sum(len(mz_left[i]) for i in batch_left_unique)
            batch_peaks_right = sum(len(mz_right[i]) for i in batch_right_unique)
            _log_message_to_file(
                f"[exact-gpu-batched] batch {batch_idx + 1}/{len(pair_batches)}: "
                f"{len(batch_left)} pairs, "
                f"{batch_unique_left}+{batch_unique_right} unique spectra, "
                f"{batch_peaks_left + batch_peaks_right} total peaks, "
                f"{t1 - t0:.3f}s"
            )

        all_scores.append(scores)

    return np.concatenate(all_scores)


def _run_exact_cosine_cpu(
    pair_left: np.ndarray,
    pair_right: np.ndarray,
    df_idx: pl.DataFrame,
    ms2_tolerance_ppm: float,
) -> np.ndarray:
    """
    Run exact cosine similarity on CPU using Polars spectral_similarity.

    Why: CPU exact cosine is the baseline implementation. It's slower than GPU
    but requires no special memory management.
    """
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
                ms2_tolerance_in_ppm=ms2_tolerance_ppm,
                clean_spectra_first=False,
                ignore_precursor=True,
            )
        )
        .select(["order", "dotprod"])
    )
    cpu_scores = (
        pairs_struct.sort("order")
        .collect()
        .get_column("dotprod")
        .to_numpy()
        .astype(np.float32)
    )
    return cpu_scores


def run_approximate_and_exact_similarity(
    df: pl.DataFrame,
    config: SimilarityConfig,
    batch_size: int = 1000,
    target_gpu_mem_ratio: float = 0.6,
    max_peaks_per_batch: int | None = None,
    max_peaks_per_exact_batch: int | None = None,
    target_gpu_mem_ratio_exact: float = 0.5,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Run complete approximate + exact similarity pipeline on a single library (symmetric case).

    This function:
    1. Runs batched GPU approximate similarity to find candidate pairs
    2. Filters pairs above approx_threshold
    3. Runs exact cosine similarity (GPU or CPU based on config.use_gpu_exact_cosine)
    4. Returns DataFrame with idx, idx_right, proximate_similarity, exact_similarity

    Why: This provides a complete end-to-end pipeline that handles memory constraints
    through dynamic batching while allowing users to choose between CPU and GPU exact
    computation. Different memory ratios for approximate vs exact stages account for
    different data representations (binned CSR vs unbinned dense).

    Args:
        df: Polars DataFrame with columns 'cleaned_normalized_mz', 'cleaned_normalized_intensity', 'precursor_mz'
        config: SimilarityConfig with threshold, ms2_tolerance_ppm, intensity_power, use_gpu_exact_cosine
        batch_size: Minimum batch size for approximate stage
        target_gpu_mem_ratio: Fraction of free GPU memory for approximate stage (binned sparse)
        max_peaks_per_batch: Maximum peaks per batch for approximate stage (None = auto)
        max_peaks_per_exact_batch: Maximum total peaks from unique spectra per GPU exact batch (None = auto)
        target_gpu_mem_ratio_exact: Fraction of free GPU memory for exact stage (unbinned dense)
        verbose: Print timing and progress information

    Returns:
        DataFrame with columns: idx, idx_right, proximate_similarity, exact_similarity
    """
    assert len(df) > 0, "Input dataframe must be non-empty"
    assert "cleaned_normalized_mz" in df.columns, (
        "cleaned_normalized_mz column is required"
    )
    assert "cleaned_normalized_intensity" in df.columns, (
        "cleaned_normalized_intensity column is required"
    )
    assert "precursor_mz" in df.columns, "precursor_mz column is required"

    # Add row index
    df_idx = df.with_row_index("idx").with_columns(pl.col("idx").cast(pl.Int32))

    # Fail fast: idx must fit in int32 because we cast and store indices as int32 for memory efficiency.
    idx_max = df_idx.select(pl.col("idx").max()).item()
    assert idx_max is not None, (
        "idx max was None; dataframe appears empty or idx missing unexpectedly"
    )
    assert int(idx_max) <= INT32_MAX, (
        f"idx values must fit in int32 (<= {INT32_MAX}), got max idx={idx_max}. "
        "Reduce library size, shard input, or change index dtype policy."
    )

    # Use int32 for downstream arrays/joins
    df_idx = df_idx.with_columns(pl.col("idx").cast(pl.Int32))

    # Configure approximate stage
    batched_cfg = BatchedGPUConfig(
        batch_size=batch_size,
        gpu_batch_write_interval=10,
        target_gpu_mem_ratio=target_gpu_mem_ratio,
        max_peaks_per_batch=max_peaks_per_batch,
        approx_config=config,
    )

    if verbose:
        _log_message_to_file(
            f"Running approximate similarity with batch_size={batch_size}, "
            f"approx_threshold={config.approx_threshold}, "
            f"use_gpu_exact_cosine={config.use_gpu_exact_cosine}"
        )

    # Run approximate stage
    t_approx0 = time.perf_counter()
    approx_pairs = _run_batched_approximate_gpu(df_idx, config, batched_cfg, verbose)
    t_approx1 = time.perf_counter()

    if verbose:
        _log_message_to_file(
            f"[approx] Found {len(approx_pairs)} candidate pairs in {t_approx1 - t_approx0:.3f}s"
        )

    if len(approx_pairs) == 0:
        return pl.DataFrame(
            {
                "idx": [],
                "idx_right": [],
                "proximate_similarity": [],
                "exact_similarity": [],
            }
        )

    # Extract pairs
    pair_left = approx_pairs["idx"].to_numpy()
    pair_right = approx_pairs["idx_right"].to_numpy()

    # Run exact stage
    t_exact0 = time.perf_counter()

    if config.use_gpu_exact_cosine:
        # GPU exact path with dynamic batching
        mz_left, int_left = _extract_lists_from_df(df_idx)
        mz_right, int_right = mz_left, int_left

        # Compute or use provided max peaks for exact stage
        if max_peaks_per_exact_batch is None:
            max_peaks_per_exact_batch = _compute_dynamic_max_peaks_exact(
                target_gpu_mem_ratio_exact, None
            )
            if verbose:
                _log_message_to_file(
                    f"[exact-gpu] Auto-computed max_peaks_per_exact_batch={max_peaks_per_exact_batch}"
                )

        exact_scores = _run_exact_cosine_gpu_batched(
            pair_left,
            pair_right,
            mz_left,
            int_left,
            mz_right,
            int_right,
            config,
            max_peaks_per_batch=max_peaks_per_exact_batch,
            verbose=verbose,
        )
    else:
        # CPU exact path
        # Fail fast: CPU exact cosine requires an ms2 tolerance to be defined.
        assert config.ms2_tolerance_ppm is not None, (
            "ms2_tolerance_ppm must be set in config for CPU exact cosine"
        )
        exact_scores = _run_exact_cosine_cpu(
            pair_left, pair_right, df_idx, config.ms2_tolerance_ppm
        )

    t_exact1 = time.perf_counter()

    if verbose:
        device = "GPU" if config.use_gpu_exact_cosine else "CPU"
        _log_message_to_file(
            f"[exact-{device}] Computed {len(exact_scores)} exact scores in {t_exact1 - t_exact0:.3f}s"
        )

    # Combine results
    result = approx_pairs.with_columns(pl.Series("exact_similarity", exact_scores))

    # Filter by final threshold
    result = result.filter(pl.col("exact_similarity") >= config.threshold)

    if verbose:
        _log_message_to_file(
            f"[final] {len(result)} pairs above threshold={config.threshold} "
            f"(total time: {t_exact1 - t_approx0:.3f}s)"
        )

    return result
