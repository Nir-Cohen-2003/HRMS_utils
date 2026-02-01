#!/usr/bin/env python
"""
Comparison Benchmark: fast_cosine_sim (SpMM) vs FlashEntropySearch (Hybrid GPU)

This script compares the throughput and execution time of:
1. fast_cosine_sim: Batched SpMM-based Cosine Similarity
2. FlashEntropySearch: Hybrid Search (GPU) using "Shooting" algorithm

Note: FlashEntropySearch's GPU implementation is primarily "Hybrid Search" (Product + Neutral Loss).
We use CosineHybridSearchCore as the closest GPU-accelerated equivalent, but keep in mind 
it performs extra work (Neutral Loss matching) compared to pure Cosine.

Usage:
    python benchmarks/compare_flash_entropy_cosine.py --n-spectra 10000 --batch-size 1000
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import polars as pl
import cupy as cp

# Ensure fast_cosine_sim is in path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Ensure FlashEntropySearch is in path
# Assumes folder structure: packages/fast_cosine_sim/FlashEntropySearch/src
FLASH_ENTROPY_PATH = Path(__file__).parent.parent / "FlashEntropySearch" / "src"
if FLASH_ENTROPY_PATH.exists():
    sys.path.append(str(FLASH_ENTROPY_PATH))
else:
    print(f"WARNING: FlashEntropySearch source not found at {FLASH_ENTROPY_PATH}")
    print("Please clone/place FlashEntropySearch in the package root.")

try:
    from fast_cosine_sim import GPUApproximateConfig, batched_approximate_similarity_gpu
    from fast_cosine_sim.binning import sparse_bin_spectra_df_to_csr
    from fast_cosine_sim.gpu_operations import construct_expansion_matrix_gpu, normalize_csr_rows_inplace_gpu
except ImportError as e:
    print(f"Error importing fast_cosine_sim: {e}")
    sys.exit(1)

try:
    # We use CosineHybridSearchCore as it supports GPU (unlike CosineSearchCore)
    from library.hybrid_search import CosineHybridSearchCore
    from mimas.spectra.similarity.tools import clean_spectrum
    FLASH_ENTROPY_AVAILABLE = True
except ImportError as e:
    print(f"Error importing FlashEntropySearch: {e}")
    print("FlashEntropySearch benchmark will be skipped.")
    FLASH_ENTROPY_AVAILABLE = False


# =============================================================================
# Data Generation
# =============================================================================

def generate_synthetic_spectra(
    n_spectra: int = 1000,
    n_peaks: int = 100,
    mz_range: Tuple[float, float] = (100.0, 1000.0),
    seed: int = 42
) -> pl.DataFrame:
    """Generate synthetic spectra in Polars DataFrame format."""
    rng = np.random.RandomState(seed)
    
    idx_list = []
    mz_list = []
    int_list = []

    for i in range(n_spectra):
        # Random peaks
        mz = np.sort(rng.uniform(mz_range[0], mz_range[1], n_peaks)).astype(np.float64)
        intensity = rng.exponential(1.0, n_peaks).astype(np.float32)
        intensity /= intensity.max()  # Normalize to 0-1
        
        idx_list.append(i)
        mz_list.append(mz)
        int_list.append(intensity)

    return pl.DataFrame({
        "idx": idx_list,
        "mz": mz_list,
        "intensity": int_list
    })

def convert_to_flash_entropy_format(df: pl.DataFrame):
    """Convert Polars DF to format expected by FlashEntropySearch (list of dicts/arrays)."""
    # FlashEntropySearch build_index expects:
    # all_precursor_mz_list (list of floats)
    # all_peaks_list (list of np.ndarray shape (N, 2))
    
    n_spectra = len(df)
    
    # Fake precursor m/z (just max m/z + 10)
    # Hybrid search requires precursor m/z for neutral loss calculation
    mzs = df["mz"].to_list()
    ints = df["intensity"].to_list()
    
    all_peaks_list = []
    all_precursor_mz_list = []
    
    for i in range(n_spectra):
        mz_arr = np.array(mzs[i], dtype=np.float32)
        int_arr = np.array(ints[i], dtype=np.float32)
        
        # FlashEntropy requires sum(intensity) == 1
        s = np.sum(int_arr)
        if s > 0:
            int_arr /= s
        
        # Combine into (N, 2)
        peaks = np.column_stack((mz_arr, int_arr))

        if FLASH_ENTROPY_AVAILABLE:
            peaks = clean_spectrum(peaks, ms2_da=0.05)
        
        all_peaks_list.append(peaks)
        
        # Precursor m/z
        pre_mz = mz_arr[-1] + 10.0
        all_precursor_mz_list.append(pre_mz)
        
    return all_precursor_mz_list, all_peaks_list

# =============================================================================
# Runners
# =============================================================================

class FastCosineRunner:
    def __init__(self, config: GPUApproximateConfig, batch_size: int):
        self.config = config
        self.batch_size = batch_size
        self.csr_matrix = None
        self.expansion_matrix = None
        self.df = None

    def setup(self, df: pl.DataFrame):
        self.df = df
        t0 = time.perf_counter()
        
        # 1. Convert to CSR
        self.csr_matrix = sparse_bin_spectra_df_to_csr(
            df,
            self.config.mz_col,
            self.config.intensity_col,
            upper_bound=self.config.upper_mass_bound,
            intensity_power=self.config.intensity_power,
            bin_size=self.config.bin_size,
            apply_centroiding=self.config.centroiding_enabled,
            tolerance_ppm=self.config.ms2_tolerance_ppm,
            mass_tolerance_cutoff_mz=self.config.mass_tolerance_cutoff_mz,
        )

        # 2. Expansion Matrix
        nbins = int(self.config.upper_mass_bound / self.config.bin_size) + 1
        self.expansion_matrix = construct_expansion_matrix_gpu(
            self.config.bin_size, 
            self.config.ms2_tolerance_ppm, 
            nbins, 
            self.config.upper_mass_bound
        )
        
        t1 = time.perf_counter()
        return t1 - t0

    def run(self, n_queries: int):
        # We manually invoke batched_approximate_similarity_gpu logic or similar
        # But batched_approximate_similarity_gpu is high level and handles everything.
        # To make it fair and avoid re-doing setup, we should use the internal batching logic
        # OR just call the public API if we want to measure "API usage time".
        # However, we already did setup.
        # Let's write a minimal loop that uses the pre-computed matrices, similar to benchmark_throughput.py
        
        # Using the internal implementation style from benchmark_throughput for pure kernel speed
        # But we want to simulate the real "search" process.
        
        # Let's just use the `batched_approximate_similarity_gpu` function but assume 
        # we pass the dataframe. Wait, that function effectively does the setup inside.
        # To separate setup time, we should probably stick to the benchmark_throughput style loop.
        
        import cupyx.scipy.sparse as cps

        n_spectra = self.csr_matrix.shape[0]
        n_queries = min(n_queries, n_spectra)
        
        t0 = time.perf_counter()
        
        # Pre-move expansion matrix
        # Note: In real usage, this is cached.
        
        # Loop
        for i in range(0, n_queries, self.batch_size):
            left_end = min(i + self.batch_size, n_queries)
            left_csr = self.csr_matrix[i:left_end]
            
            # Transfer Left
            left_gpu = cps.csr_matrix(left_csr, dtype=np.float32)
            normalize_csr_rows_inplace_gpu(left_gpu)
            
            # Match against all DB (batched)
            for j in range(0, n_spectra, self.batch_size):
                right_end = min(j + self.batch_size, n_spectra)
                right_csr = self.csr_matrix[j:right_end]
                
                # Transfer Right
                right_gpu = cps.csr_matrix(right_csr, dtype=np.float32)
                normalize_csr_rows_inplace_gpu(right_gpu)
                
                # Expand
                right_expanded = right_gpu.dot(self.expansion_matrix)
                
                # Matmul
                sim = left_gpu.dot(right_expanded.T)
                
                # Threshold
                # (Simulated extraction)
                mask = sim.data >= self.config.approx_threshold
                _ = sim.data[mask]
                
                cp.cuda.Stream.null.synchronize()

        t1 = time.perf_counter()
        return t1 - t0


class FlashEntropyRunner:
    def __init__(self, tolerance_da: float):
        self.tolerance_da = tolerance_da
        self.searcher = None
        self.all_precursor_mz = None
        self.all_peaks = None

    def setup(self, df: pl.DataFrame):
        if not FLASH_ENTROPY_AVAILABLE:
            return 0.0
            
        t0 = time.perf_counter()
        
        # Conversion
        self.all_precursor_mz, self.all_peaks = convert_to_flash_entropy_format(df)
        
        # Initialize
        # Note: CosineHybridSearchCore performs Hybrid search (Product + Neutral Loss)
        self.searcher = CosineHybridSearchCore(
            path_array=None, # In-memory
            mz_index_step=0.0001, 
            max_ms2_tolerance_in_da=max(0.025, self.tolerance_da * 2) 
        )
        
        # Build Index
        self.searcher.build_index(self.all_precursor_mz, self.all_peaks)
        
        t1 = time.perf_counter()
        return t1 - t0

    def run(self, n_queries: int):
        if not FLASH_ENTROPY_AVAILABLE:
            return 0.0
            
        n_queries = min(n_queries, len(self.all_peaks))
        t0 = time.perf_counter()
        
        # Iterate query by query
        for i in range(n_queries):
            peaks = self.all_peaks[i]
            precursor_mz = self.all_precursor_mz[i]
            
            # Search
            # CosineHybridSearchCore.search(...)
            _ = self.searcher.search(
                precursor_mz=precursor_mz,
                peaks=peaks,
                ms2_tolerance_in_da=self.tolerance_da,
                target="gpu"
            )
            # Sync to ensure GPU is done before moving to next (or just let it queue)
            # FlashEntropySearch code ends with .get() which does sync/copy back.
            
        t1 = time.perf_counter()
        return t1 - t0

# =============================================================================
# Main Benchmark
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark: fast_cosine_sim vs FlashEntropySearch")
    parser.add_argument("--n-spectra", type=int, default=10000, help="Library size")
    parser.add_argument("--n-queries", type=int, default=1000, help="Number of queries")
    parser.add_argument("--n-peaks", type=int, default=100, help="Peaks per spectrum")
    parser.add_argument("--batch-size", type=int, default=2000, help="Batch size for fast_cosine_sim")
    parser.add_argument("--tolerance-ppm", type=float, default=10.0, help="Tolerance in PPM")
    parser.add_argument("--bin-size", type=float, default=0.0001, help="Bin size for fast_cosine_sim")
    
    args = parser.parse_args()
    
    # Physics Conversion
    # FlashEntropySearch needs Da. We use the middle of the mass range (500 Da) 
    # as a reference point for converting PPM to Da.
    # 10 ppm at 500 Da = 0.005 Da
    ref_mass = 500.0
    tolerance_da = ref_mass * args.tolerance_ppm * 1e-6
    
    print("="*60)
    print("COMPARISON BENCHMARK")
    print("="*60)
    print(f"Library Size:  {args.n_spectra}")
    print(f"Queries:       {args.n_queries}")
    print(f"Batch Size:    {args.batch_size} (fast_cosine_sim)")
    print(f"Tolerance:     {args.tolerance_ppm} ppm (~{tolerance_da:.5f} Da @ 500 m/z)")
    print("-" * 60)

    # Generate Data
    print("Generating data...")
    df = generate_synthetic_spectra(args.n_spectra, args.n_peaks)
    
    # --- Fast Cosine Sim ---
    print("\n[1] Running fast_cosine_sim (SpMM)...")
    config = GPUApproximateConfig(
        upper_mass_bound=1000.0,
        bin_size=args.bin_size,
        ms2_tolerance_ppm=args.tolerance_ppm,
        approx_threshold=0.6, # Low threshold to ensure some hits
    )
    
    runner_fast = FastCosineRunner(config, args.batch_size)
    
    try:
        t_setup_fast = runner_fast.setup(df)
        print(f"  Setup Time: {t_setup_fast:.4f} s")
        
        t_run_fast = runner_fast.run(args.n_queries)
        print(f"  Search Time: {t_run_fast:.4f} s")
        
        t_total_fast = t_setup_fast + t_run_fast
        throughput_fast = (args.n_queries * args.n_spectra) / t_run_fast
        
    except Exception as e:
        print(f"  FAILED: {e}")
        t_setup_fast = t_run_fast = throughput_fast = 0
        
    # --- FlashEntropySearch ---
    print("\n[2] Running FlashEntropySearch (Hybrid GPU)...")
    if FLASH_ENTROPY_AVAILABLE:
        runner_flash = FlashEntropyRunner(tolerance_da)
        
        try:
            t_setup_flash = runner_flash.setup(df)
            print(f"  Setup Time: {t_setup_flash:.4f} s")
            
            t_run_flash = runner_flash.run(args.n_queries)
            print(f"  Search Time: {t_run_flash:.4f} s")
            
            t_total_flash = t_setup_flash + t_run_flash
            throughput_flash = (args.n_queries * args.n_spectra) / t_run_flash
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  FAILED: {e!r}")
            t_setup_flash = t_run_flash = throughput_flash = 0
    else:
        print("  Skipped (Not available)")
        t_setup_flash = t_run_flash = throughput_flash = 0

    # --- Summary ---
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"{'Metric':<20} {'fast_cosine_sim':<20} {'FlashEntropy (Hybrid)':<20}")
    print("-" * 60)
    print(f"{'Setup (s)':<20} {t_setup_fast:<20.4f} {t_setup_flash:<20.4f}")
    print(f"{'Search (s)':<20} {t_run_fast:<20.4f} {t_run_flash:<20.4f}")
    print(f"{'Throughput (pairs/s)':<20} {int(throughput_fast):<20,d} {int(throughput_flash):<20,d}")
    
    if t_run_flash > 0:
        speedup = t_run_flash / t_run_fast
        print("-" * 60)
        print(f"Speedup (fast_cosine_sim vs FlashEntropy): {speedup:.2f}x")
    
    print("="*60)
    print("Note: FlashEntropySearch result is for 'Hybrid Search' (Product + Neutral Loss).")
    print("      It performs more work per pair than pure Cosine.")

if __name__ == "__main__":
    main()
