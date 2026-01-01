import gc
import itertools
import logging
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np
import polars as pl
from numba import cuda

# NEVER REMOVE THIS
import hrms_utils

# Adjust path to import local modules
sys.path.append(str(Path.cwd()))

from approximate_similarity import SimilarityConfig, proximate_all_vs_all_pairs
from optimized_cosine import run_greedy_cosine_fast

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("speed_test")


def prepare_gpu_arrays_dense(mz_list, int_list):
    """
    Pads a list of spectra into a dense GPU array.
    """
    n_spectra = len(mz_list)
    lens = np.array([len(x) for x in mz_list], dtype=np.int32)
    max_peaks = np.max(lens) if n_spectra > 0 else 0

    mz_arr = np.zeros((n_spectra, max_peaks), dtype=np.float32)
    int_arr = np.zeros((n_spectra, max_peaks), dtype=np.float32)

    for i in range(n_spectra):
        l = lens[i]
        mz_arr[i, :l] = mz_list[i]
        int_arr[i, :l] = int_list[i]

    return (
        cuda.to_device(mz_arr),
        cuda.to_device(int_arr),
        cuda.to_device(lens),
        max_peaks,
    )


def main():
    # 1. Load Data
    parquet_path = "/home/analytit_admin/Data/spectral_libs/fraghub/fraghub.parquet"
    logger.info(f"Loading data from {parquet_path}...")

    # Read first 1000 spectra
    df = (
        pl.scan_parquet(parquet_path)
        .select(
            ["precursor_mz", "cleaned_normalized_mz", "cleaned_normalized_intensity"]
        )
        .head(1500)
        .collect()
    )

    # Ensure index
    df = df.with_row_index("idx").with_columns(pl.col("idx").cast(pl.Int64))
    n_spectra = len(df)
    logger.info(f"Loaded {n_spectra} spectra.")

    # ---------------------------------------------------------
    # APPROACH 1: Approximate (GPU) All-vs-All
    # ---------------------------------------------------------
    logger.info("\n--- Running Approach 1: Approximate Similarity (GPU) ---")

    ms2_tol = 10.0  # ppm
    # For speed comparison, we set threshold to 0.0 to force it to return EVERYTHING (or as much as possible)
    # But approximate method is sparse by nature.
    # To treat it as "running the pipeline", we use a low threshold.
    threshold = 0.5

    t0 = time.perf_counter()
    # This runs the full sparse pipeline: Binning -> Norm -> Expansion -> Matmul -> Threshold
    approx_res, approx_timings = proximate_all_vs_all_pairs(
        df,
        df,
        threshold=threshold,
        ms2_tolerance_ppm=ms2_tol,
        proximate_bin_size=0.0001,
        use_gpu=True,
        return_timings=True,
    )
    t1 = time.perf_counter()
    time_approx = t1 - t0
    logger.info(f"Approximate Time: {time_approx:.4f}s")
    logger.info(f"Approximate Pairs Found: {len(approx_res)}")

    # ---------------------------------------------------------
    # PREPARE DENSE DATA FOR EXACT METHODS
    # ---------------------------------------------------------
    logger.info("\n--- Preparing Data for Exact All-vs-All ---")
    # We want 1000x1000 = 1M pairs.

    # Extract lists
    mz_list = df["cleaned_normalized_mz"].to_list()
    int_list = df["cleaned_normalized_intensity"].to_list()

    # Create All-vs-All indices
    # We can create inputs for the kernel by repeating arrays.
    # A: [0, 0, 0, 1, 1, 1 ...] (Repeat rows N times)
    # B: [0, 1, 2, 0, 1, 2 ...] (Tile rows N times)

    # Efficient way for GPU Kernel:
    # 1. Upload unique spectra to GPU ONCE (N x MaxPeaks).
    # 2. Pass index arrays to the kernel to avoid materializing 1M pairs.

    t_prep0 = time.perf_counter()

    # 1. Pad unique spectra first
    d_mz_unique, d_int_unique, d_len_unique, max_peaks = prepare_gpu_arrays_dense(
        mz_list, int_list
    )

    # 2. Create index map for 1M pairs
    # indices_a = [0,0...0, 1,1...1, ...]
    # indices_b = [0,1...N, 0,1...N, ...]
    indices = np.arange(n_spectra, dtype=np.int32)
    idx_a = np.repeat(indices, n_spectra)
    idx_b = np.tile(indices, n_spectra)

    cp.cuda.Stream.null.synchronize()
    t_prep1 = time.perf_counter()
    logger.info(f"Data Prep Time (1M pairs): {t_prep1 - t_prep0:.4f}s")
    # ---------------------------------------------------------
    # APPROACH 3: Exact (CPU) All-vs-All
    # ---------------------------------------------------------
    logger.info("\n--- Running Approach 3: Exact (CPU) ---")
    # We construct a Polars DF with all 1M pairs and use the dotprod extension.
    # This is how the "Exact" stage works in the pipeline.

    # Construct DF (careful with memory, 1M rows is fine)
    # We need to replicate the lists in Polars.
    # Replicating lists 1M times in Polars might be slow.
    # Strategy: Create DF with indices, then join.

    pairs_df = pl.DataFrame({"idx": idx_a, "idx_right": idx_b}).lazy()

    # Join spectra
    df_lazy = df.lazy()
    pairs_with_spectra = pairs_df.join(df_lazy, on="idx").join(
        df_lazy, left_on="idx_right", right_on="idx", suffix="_right"
    )

    # Prepare struct
    pairs_struct = pairs_with_spectra.with_columns(
        spectra=pl.struct(
            mz1=pl.col("cleaned_normalized_mz"),
            intensities1=pl.col("cleaned_normalized_intensity"),
            mz2=pl.col("cleaned_normalized_mz_right"),
            intensities2=pl.col("cleaned_normalized_intensity_right"),
            # Dummy precursors if needed by logic (approx_sim uses them? dotprod_similarity ignores them with ignore_precursor=True)
            precursor_mz1=pl.col("precursor_mz"),
            precursor_mz2=pl.col("precursor_mz"),  # Dummy
        )
    )

    t_cpu0 = time.perf_counter()
    res_cpu = pairs_struct.select(
        pl.col("spectra").spectral_similarity.dotprod_similarity(
            ms2_tolerance_in_ppm=10.0, clean_spectra_first=False, ignore_precursor=True
        )
    ).collect()
    t_cpu1 = time.perf_counter()

    time_exact_cpu = t_cpu1 - t_cpu0
    logger.info(f"Exact CPU Time: {time_exact_cpu:.4f}s")

    scores_cpu = res_cpu.to_series().to_numpy()
    # ---------------------------------------------------------
    # APPROACH 2: Exact (GPU) All-vs-All
    # ---------------------------------------------------------
    logger.info("\n--- Running Approach 2: Exact Optimized Kernel (GPU) ---")
    # warming run
    scores_gpu_dev = run_greedy_cosine_fast(
        d_mz_unique,
        d_int_unique,
        d_len_unique,
        d_mz_unique,
        d_int_unique,
        d_len_unique,
        tolerance=ms2_tol,  # PPM
        shift=0.0,
        mz_power=0.0,
        int_power=0.5,
        pair_a_indices=idx_a,
        pair_b_indices=idx_b,
    )
    cp.cuda.Stream.null.synchronize()
    del scores_gpu_dev
    gc.collect()
    t_gpu0 = time.perf_counter()
    scores_gpu_dev = run_greedy_cosine_fast(
        d_mz_unique,
        d_int_unique,
        d_len_unique,
        d_mz_unique,
        d_int_unique,
        d_len_unique,
        tolerance=ms2_tol,  # PPM
        shift=0.0,
        mz_power=0.0,
        int_power=0.5,
        pair_a_indices=idx_a,
        pair_b_indices=idx_b,
    )
    cp.cuda.Stream.null.synchronize()
    t_gpu1 = time.perf_counter()
    time_exact_gpu = t_gpu1 - t_gpu0
    logger.info(f"Exact GPU Time: {time_exact_gpu:.4f}s")
    t_copy = time.perf_counter()
    scores_gpu = scores_gpu_dev.copy_to_host()
    t_copy1 = time.perf_counter()
    logger.info(f"Exact GPU Copy Time: {t_copy1 - t_copy:.4f}s")

    # ---------------------------------------------------------
    # METRICS
    # ---------------------------------------------------------
    # Filter valid scores (some might be None)
    # CPU scores might have Nones? GPU scores are -1 or 0.

    valid_mask = ~np.isnan(scores_cpu)
    s_cpu = scores_cpu[valid_mask]
    s_gpu = scores_gpu[valid_mask]

    # Approximate scores: We need to align them.
    # Approx result is sparse. We map it to the dense grid.
    # Map (idx, idx_right) -> score
    approx_map = {}
    if len(approx_res) > 0:
        a_idx = approx_res["idx"].to_numpy()
        a_ridx = approx_res["idx_right"].to_numpy()
        a_sim = approx_res["proximate_similarity"].to_numpy()
        for i in range(len(a_idx)):
            approx_map[(a_idx[i], a_ridx[i])] = a_sim[i]

    # Create dense approx array corresponding to s_cpu
    s_approx = np.zeros_like(s_cpu)

    # Reconstruct indices for valid mask
    v_idx_a = idx_a[valid_mask]
    v_idx_b = idx_b[valid_mask]

    for i in range(len(s_approx)):
        pair = (v_idx_a[i], v_idx_b[i])
        s_approx[i] = approx_map.get(pair, 0.0)  # Default 0 if not found by approx

    def print_metrics(name, base_scores, test_scores):
        diff = np.abs(base_scores - test_scores)
        avg_diff = np.mean(diff)
        pct_0_1 = np.mean(diff > 0.1) * 100
        pct_0_01 = np.mean(diff > 0.01) * 100
        pct_0_001 = np.mean(diff > 0.001) * 100

        print(f"\n--- {name} vs CPU Exact ---")
        print(f"Avg Abs Diff:       {avg_diff:.6f}")
        print(f"% Diff > 0.1:       {pct_0_1:.2f}%")
        print(f"% Diff > 0.01:      {pct_0_01:.2f}%")
        print(f"% Diff > 0.001:     {pct_0_001:.2f}%")

    print(f"\n{'=' * 20} RESULTS (1000x1000 = 1M pairs) {'=' * 20}")
    print(f"Time - Approx (GPU): {time_approx:.4f}s")
    print(f"Time - Exact (CPU):  {time_exact_cpu:.4f}s")
    print(
        f"Time - Exact (GPU):  {time_exact_gpu:.4f}s (Speedup vs CPU: {time_exact_cpu / time_exact_gpu:.1f}x)"
    )

    print_metrics("Approximate (Binned)", s_cpu, s_approx)
    print_metrics("Exact (GPU Kernel)", s_cpu, s_gpu)


if __name__ == "__main__":
    main()
