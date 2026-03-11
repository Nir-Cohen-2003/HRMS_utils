#!/usr/bin/env python
"""
GPU-accelerated batched approximate similarity computation.

This module provides the main orchestration for computing pairwise
approximate (binned dot-product) similarities on GPU with efficient memory
management and batching.

Key features:
- GPU-only computation using CuPy
- Self-comparison mode with upper-triangular optimization (exploits ij=ji symmetry)
- Cross-library comparison mode (full NxM)
- Dynamic batching based on GPU memory and peak counts
- Configurable via GPUApproximateConfig dataclass
- Async parquet writing with memory-limited queue
- Efficient polars -> numpy -> GPU arrays -> CSR -> COO pipeline

Architecture:
- config.py: Constants and dataclasses
- binning.py: CPU preprocessing (DataFrame -> CSR matrix)
- gpu_operations.py: GPU kernels (normalize, expand, SpMM matrix)
- batching.py: Dynamic batch size calculation and generation
- async_writer.py: Memory-limited queue and writer thread
- gpu_approximate_similarity.py (this file): Main orchestration

Usage:
    config = GPUApproximateConfig(
        upper_mass_bound=1000.0,
        bin_size=0.0001,
        ms2_tolerance_ppm=10.0,
        approx_threshold=0.65,
        target_gpu_mem_ratio=0.6,
    )

    # Self-comparison (upper triangular)
    result = batched_approximate_similarity_gpu(df, config)

    # Cross-library comparison
    result = batched_approximate_similarity_gpu(df1, config, right_df=df2)

    # Write to parquet with async I/O
    batched_approximate_similarity_gpu(df, config, output_path="output.parquet")
"""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Optional

import cupy as cp
import cupyx.scipy.sparse as cps
import numpy as np
import polars as pl

# Import from submodules
from .async_writer import AsyncParquetWriter, ResultBuffer
from .batching import compute_dynamic_max_peaks, yield_batches_dynamic
from .binning import select_and_collect, sparse_bin_spectra_df_to_csr
from .config import (
    APPROX_INTENSITY_DTYPE_NP,
    INDEX_DTYPE_NP,
    AggregatedKernelTimings,
    GPUApproximateConfig,
)
from .gpu_operations import (
    construct_expansion_matrix_gpu,
    expand_csr_horizontal_adaptive_gpu,
    normalize_csr_rows_inplace_gpu,
)

# Re-export for backward compatibility
__all__ = [
    "GPUApproximateConfig",
    "AggregatedKernelTimings",
    "batched_approximate_similarity_gpu",
]


def batched_approximate_similarity_gpu(
    left_df: pl.DataFrame | pl.LazyFrame,
    config: GPUApproximateConfig,
    right_df: Optional[pl.DataFrame | pl.LazyFrame] = None,
    output_path: Optional[Path | str] = None,
    logger: Optional[logging.Logger] = None,
    log_timings: bool = False,
) -> (
    pl.DataFrame
    | pl.LazyFrame
    | tuple[pl.DataFrame | pl.LazyFrame, AggregatedKernelTimings]
):
    """
    Compute batched approximate similarity on GPU with optimized memory management.

    This function performs binned dot-product similarity (cosine similarity on
    binned spectra) using GPU acceleration. It supports two modes:

    1. Self-comparison (right_df=None, config.comparison_mode="self"):
       Computes upper-triangular similarity matrix, exploiting symmetry (ij = ji)
       to reduce computation by ~2x.

    2. Cross-library comparison (right_df provided, config.comparison_mode="cross"):
       Computes full NxM similarity matrix between two libraries.

    The algorithm:
    1. Optional: Centroid spectra (merge peaks within tolerance, prevents similarities > 1.0)
    2. Convert DataFrames to binned CSR matrices (CPU)
    3. Dynamically batch based on GPU memory and peak counts
    4. For each batch pair:
       - Transfer to GPU
       - Expand left matrix (tolerance window) - done once per left batch, reused
       - Normalize right matrix rows (L2)
       - Compute expanded_left @ normalized_right.T (sparse matmul)
       - Threshold and extract pairs above threshold
       - Accumulate results
    5. Either return DataFrame or write to parquet with async I/O

    Why expand left:
    - The expanded matrix is larger (2-3x more non-zeros due to tolerance windows)
    - Streaming the expanded matrix row-wise in SpMM is more cache-friendly
    - The smaller normalized matrix is accessed repeatedly in inner loop and stays in cache
    - This reduces memory bandwidth pressure on the sparse matmul operation

    Args:
        left_df: DataFrame or LazyFrame with list columns specified by
                 config.mz_col and config.intensity_col
        config: GPUApproximateConfig instance with all parameters
        right_df: Optional second library for cross-comparison (None = self-comparison)
        output_path: Optional path for parquet output (None = return DataFrame)
        logger: Optional logger for progress reporting
        log_timings: If True, return (result, timings) tuple with detailed GPU profiling data

    Returns:
        If log_timings is False:
            DataFrame with columns ['idx_left', 'idx_right', 'similarity'] if output_path is None,
            LazyFrame (scan of written parquet) if output_path is provided
        If log_timings is True:
            Tuple of (DataFrame/LazyFrame, AggregatedKernelTimings)

    Raises:
        AssertionError: If inputs are invalid (with detailed messages)
        RuntimeError: If GPU operations fail
    """
    # =========================================================================
    # 1. Validate Inputs
    # =========================================================================

    # Collect LazyFrames if needed
    left_df = select_and_collect(left_df, config)
    if right_df is not None:
        right_df = select_and_collect(right_df, config)

    assert len(left_df) > 0, "left_df is empty. Provide at least one spectrum."

    assert config.mz_col in left_df.columns, (
        f"Column '{config.mz_col}' not found in left_df. "
        f"Available columns: {left_df.columns}. "
        f"Set config.mz_col to the correct m/z column name."
    )

    assert config.intensity_col in left_df.columns, (
        f"Column '{config.intensity_col}' not found in left_df. "
        f"Available columns: {left_df.columns}. "
        f"Set config.intensity_col to the correct intensity column name."
    )

    # Determine mode and validate consistency
    is_cross_library = right_df is not None

    if is_cross_library:
        assert config.comparison_mode == "cross", (
            f"When right_df is provided, config.comparison_mode must be 'cross', "
            f"got '{config.comparison_mode}'"
        )
        assert right_df is not None, (
            "right_df must not be None in cross-library mode (logic error)"
        )
        assert len(right_df) > 0, (
            "right_df is empty. Provide at least one spectrum for cross-comparison."
        )
        assert config.mz_col in right_df.columns, (
            f"Column '{config.mz_col}' not found in right_df. "
            f"Available columns: {right_df.columns}."
        )
        assert config.intensity_col in right_df.columns, (
            f"Column '{config.intensity_col}' not found in right_df. "
            f"Available columns: {right_df.columns}."
        )
    else:
        assert config.comparison_mode == "self", (
            f"When right_df is None, config.comparison_mode must be 'self', "
            f"got '{config.comparison_mode}'"
        )

    if logger:
        mode_str = (
            "cross-library"
            if is_cross_library
            else "self-comparison (upper triangular)"
        )
        logger.info(f"Starting batched GPU approximate similarity: mode={mode_str}")
        logger.info(
            f"  Config: threshold={config.approx_threshold}, "
            f"bin_size={config.bin_size}, tolerance={config.ms2_tolerance_ppm} ppm, "
            f"centroiding={'enabled' if config.centroiding_enabled else 'disabled'}"
        )

    # =========================================================================
    # 2. Add Row Indices (if not present)
    # =========================================================================

    # Add 0..N-1 indices to left if needed
    if config.spectrum_id_col not in left_df.columns:
        left_df_idx = left_df.with_row_index(config.spectrum_id_col).with_columns(
            pl.col(config.spectrum_id_col).cast(pl.Int32)
        )
    else:
        left_df_idx = left_df.with_columns(
            pl.col(config.spectrum_id_col).cast(pl.Int32)
        )

    # Validate int32 range
    idx_max = left_df_idx.select(pl.col(config.spectrum_id_col).max()).item()
    assert idx_max is not None, (
        f"{config.spectrum_id_col} max was None; left_df appears empty unexpectedly"
    )
    assert int(idx_max) <= np.iinfo(np.int32).max, (
        f"Index overflow: max {config.spectrum_id_col}={idx_max} exceeds int32 limit "
        f"({np.iinfo(np.int32).max}). "
        f"Reduce library size or change index dtype policy. "
        f"Current library size: {len(left_df_idx)} spectra."
    )

    n_spectra_left = len(left_df_idx)

    # Add indices to right if cross-library
    if is_cross_library:
        assert right_df is not None
        if config.spectrum_id_col not in right_df.columns:
            right_df_idx = right_df.with_row_index(config.spectrum_id_col).with_columns(
                pl.col(config.spectrum_id_col).cast(pl.Int32)
            )
        else:
            right_df_idx = right_df.with_columns(
                pl.col(config.spectrum_id_col).cast(pl.Int32)
            )

        idx_max_right = right_df_idx.select(pl.col(config.spectrum_id_col).max()).item()
        assert idx_max_right is not None, (
            f"{config.spectrum_id_col} max was None; right_df appears empty unexpectedly"
        )
        assert int(idx_max_right) <= np.iinfo(np.int32).max, (
            f"Index overflow in right_df: max {config.spectrum_id_col}={idx_max_right} "
            f"exceeds int32 limit. Reduce library size. "
            f"Current library size: {len(right_df_idx)} spectra."
        )

        n_spectra_right = len(right_df_idx)
    else:
        right_df_idx = None
        n_spectra_right = n_spectra_left

    if logger:
        if is_cross_library:
            logger.info(f"  Left library: {n_spectra_left} spectra")
            logger.info(f"  Right library: {n_spectra_right} spectra")
        else:
            logger.info(f"  Library: {n_spectra_left} spectra")

    # =========================================================================
    # 3. Initialize Resources
    # =========================================================================

    # SpMM Expansion Matrix (initialized lazily if enabled)
    expansion_matrix: Optional[cps.csr_matrix] = None

    # Construct expansion matrix if enabled and tolerance > 0
    if config.enable_spmm_expansion and config.ms2_tolerance_ppm > 0.0:
        if logger:
            logger.info("Attempting to construct SpMM expansion matrix...")

        expansion_matrix = construct_expansion_matrix_gpu(
            bin_size=config.bin_size,
            ms2_tolerance_ppm=config.ms2_tolerance_ppm,
            nbins=config.nbins,
            upper_mass_bound=config.upper_mass_bound,
            logger=logger,
        )

        if expansion_matrix is None:
            # Fallback message already logged by construct_expansion_matrix_gpu
            if logger:
                logger.info("Using element-wise adaptive expansion (fallback).")
        else:
            if logger:
                logger.info(
                    f"SpMM expansion matrix constructed successfully "
                    f"({expansion_matrix.nnz} elements)."
                )

    # =========================================================================
    # 4. Convert to CSR Matrices (with optional centroiding)
    # =========================================================================

    t_bin = perf_counter()

    if logger:
        logger.info("  Binning left library...")

    left_csr_matrix = sparse_bin_spectra_df_to_csr(
        left_df_idx,
        config.mz_col,
        config.intensity_col,
        upper_bound=config.upper_mass_bound,
        intensity_power=config.intensity_power,
        bin_size=config.bin_size,
        weight_col=config.weight_col,
        weight_power=config.weight_power,
        apply_centroiding=config.centroiding_enabled,
        tolerance_ppm=config.ms2_tolerance_ppm,
        mass_tolerance_cutoff_mz=config.mass_tolerance_cutoff_mz,
    )

    left_global_idxs = (
        left_df_idx[config.spectrum_id_col]
        .to_numpy()
        .astype(INDEX_DTYPE_NP, copy=False)
    )

    if is_cross_library:
        if logger:
            logger.info("  Binning right library...")

        assert right_df_idx is not None
        right_csr_matrix = sparse_bin_spectra_df_to_csr(
            right_df_idx,
            config.mz_col,
            config.intensity_col,
            upper_bound=config.upper_mass_bound,
            intensity_power=config.intensity_power,
            bin_size=config.bin_size,
            weight_col=config.weight_col,
            weight_power=config.weight_power,
            apply_centroiding=config.centroiding_enabled,
            tolerance_ppm=config.ms2_tolerance_ppm,
            mass_tolerance_cutoff_mz=config.mass_tolerance_cutoff_mz,
        )

        right_global_idxs = (
            right_df_idx[config.spectrum_id_col]
            .to_numpy()
            .astype(INDEX_DTYPE_NP, copy=False)
        )
    else:
        right_csr_matrix = left_csr_matrix
        right_global_idxs = left_global_idxs

    bin_time = perf_counter() - t_bin

    if logger:
        logger.info(
            f"  Binning complete in {bin_time:.3f}s. "
            f"Left: {left_csr_matrix.shape}, Right: {right_csr_matrix.shape}"
        )
        logger.info(
            f"  Left nnz: {left_csr_matrix.nnz:_}, Right nnz: {right_csr_matrix.nnz:_}"
        )

    # =========================================================================
    # 5. Dynamic Batching
    # =========================================================================

    avg_peaks_left = left_csr_matrix.nnz / max(n_spectra_left, 1)
    max_peaks = compute_dynamic_max_peaks(config, avg_peaks_left)

    if logger:
        free_mem, total_mem = cp.cuda.Device(0).mem_info
        logger.info(
            f"  GPU Memory: {free_mem / 1e9:.2f} GB free / {total_mem / 1e9:.2f} GB total"
        )
        logger.info(
            f"  Target usage: {config.target_gpu_mem_ratio:.0%} x {config.safety_factor} safety = "
            f"{free_mem * config.target_gpu_mem_ratio * config.safety_factor / 1e9:.2f} GB"
        )
        logger.info(f"  Avg peaks/spectrum: {avg_peaks_left:.1f}")
        logger.info(f"  Max peaks/batch: {max_peaks:_}")

    # Generate batches
    batches_left = list(
        yield_batches_dynamic(
            left_csr_matrix, left_global_idxs, max_peaks, min_batch_size=100
        )
    )

    if is_cross_library:
        avg_peaks_right = right_csr_matrix.nnz / max(n_spectra_right, 1)
        max_peaks_right = compute_dynamic_max_peaks(config, avg_peaks_right)
        batches_right = list(
            yield_batches_dynamic(
                right_csr_matrix, right_global_idxs, max_peaks_right, min_batch_size=100
            )
        )
    else:
        batches_right = batches_left

    num_batches_left = len(batches_left)
    num_batches_right = len(batches_right)

    if logger:
        logger.info(
            f"  Created {num_batches_left} left batches, {num_batches_right} right batches"
        )
        total_batch_pairs = num_batches_left * num_batches_right
        if not is_cross_library:
            # Upper triangular: sum(1..N) = N(N+1)/2
            total_batch_pairs = num_batches_left * (num_batches_left + 1) // 2
        logger.info(f"  Total batch pairs to process: {total_batch_pairs:_}")

    # =========================================================================
    # 6. Setup Output (Writer or Buffer)
    # =========================================================================

    writer: Optional[AsyncParquetWriter] = None
    if output_path is not None:
        output_path = Path(output_path)
        writer = AsyncParquetWriter(
            output_path,
            max_queue_memory_bytes=config.writer_max_queue_memory_bytes,
            memory_safety_ratio=config.writer_memory_safety_ratio,
            logger=logger,
        )
        writer.start()
        if logger:
            logger.info(f"  Writing results to: {output_path}")

    buffer = ResultBuffer()

    # =========================================================================
    # 7. Batch Processing Loop
    # =========================================================================

    total_pairs = 0
    gpu_batch_count = 0
    aggregated_timings = AggregatedKernelTimings()
    t_compute_start = perf_counter()

    # Outer loop: Left batches (expanded and reused)
    # Why left-outer: expand the larger matrix (expanded) once and stream it in SpMM
    for i, (l_start, l_end, l_csr, l_idxs) in enumerate(batches_left):
        t_left_batch = perf_counter()

        # Log GPU memory before processing
        free_before, total = cp.cuda.Device(0).mem_info
        if logger:
            logger.info(
                f"  [Left batch {i + 1}/{num_batches_left}] GPU mem before: "
                f"{free_before / 1e9:.2f} GB free"
            )

        # Create events for timing
        evt_start = cp.cuda.Event() if log_timings else None
        evt_xfer = cp.cuda.Event() if log_timings else None
        evt_norm = cp.cuda.Event() if log_timings else None
        evt_expand = cp.cuda.Event() if log_timings else None

        if log_timings:
            evt_start.record()

        # Transfer left batch to GPU
        l_data_gpu = cp.asarray(
            np.asarray(l_csr.data).astype(APPROX_INTENSITY_DTYPE_NP, copy=False)
        )
        l_indices_gpu = cp.asarray(np.asarray(l_csr.indices))
        l_indptr_gpu = cp.asarray(np.asarray(l_csr.indptr))
        L_gpu = cps.csr_matrix(
            (l_data_gpu, l_indices_gpu, l_indptr_gpu), shape=l_csr.shape
        )
        del l_data_gpu, l_indices_gpu, l_indptr_gpu

        if log_timings:
            evt_xfer.record()

        # Normalize in-place
        _ = normalize_csr_rows_inplace_gpu(L_gpu)

        if log_timings:
            evt_norm.record()

        # Expand Left (if using tolerance)
        if config.ms2_tolerance_ppm > 0.0:
            if expansion_matrix is not None:
                # Use fast SpMM expansion
                L_gpu = L_gpu.dot(expansion_matrix)
            else:
                # Fallback to element-wise expansion
                L_gpu = expand_csr_horizontal_adaptive_gpu(
                    L_gpu,
                    config.bin_size,
                    config.ms2_tolerance_ppm,
                    config.nbins,
                )

        if log_timings:
            evt_expand.record()
            evt_expand.synchronize()
            aggregated_timings.transfer_to_gpu_ms += cp.cuda.get_elapsed_time(
                evt_start, evt_xfer
            )
            aggregated_timings.normalize_left_ms += cp.cuda.get_elapsed_time(
                evt_xfer, evt_norm
            )
            aggregated_timings.expand_ms += cp.cuda.get_elapsed_time(
                evt_norm, evt_expand
            )
            aggregated_timings.total_ms += cp.cuda.get_elapsed_time(
                evt_start, evt_expand
            )

        # Inner loop: Right batches
        for j, (r_start, r_end, r_csr, r_idxs) in enumerate(batches_right):
            # Triangular check for self-comparison
            if not is_cross_library and j > i:
                continue

            # Timing events for inner loop
            evt_inner_start = cp.cuda.Event() if log_timings else None
            evt_r_xfer = cp.cuda.Event() if log_timings else None
            evt_r_norm = cp.cuda.Event() if log_timings else None
            evt_spmm = cp.cuda.Event() if log_timings else None
            evt_thresh = cp.cuda.Event() if log_timings else None
            evt_cpu = cp.cuda.Event() if log_timings else None

            if log_timings:
                evt_inner_start.record()

            # Transfer right batch to GPU
            r_data_gpu = cp.asarray(
                np.asarray(r_csr.data).astype(APPROX_INTENSITY_DTYPE_NP, copy=False)
            )
            r_indices_gpu = cp.asarray(np.asarray(r_csr.indices))
            r_indptr_gpu = cp.asarray(np.asarray(r_csr.indptr))
            R_gpu = cps.csr_matrix(
                (r_data_gpu, r_indices_gpu, r_indptr_gpu), shape=r_csr.shape
            )
            del r_data_gpu, r_indices_gpu, r_indptr_gpu

            if log_timings:
                evt_r_xfer.record()

            # Normalize
            _ = normalize_csr_rows_inplace_gpu(R_gpu)

            if log_timings:
                evt_r_norm.record()

            # Matmul: L @ R.T (expanded_left @ normalized_right)
            sim = L_gpu.dot(R_gpu.T)

            if log_timings:
                evt_spmm.record()

            # 3. Apply Threshold In-Place
            sim.data[sim.data < config.approx_threshold] = 0

            # 4. Prune Zeros
            sim.eliminate_zeros()

            if log_timings:
                evt_thresh.record()

            if sim.nnz > 0:
                # 5. Convert to COO
                sim_coo = sim.tocoo()

                # Transfer to CPU
                rows_cpu = cp.asnumpy(sim_coo.row)
                cols_cpu = cp.asnumpy(sim_coo.col)
                data_cpu = cp.asnumpy(sim_coo.data)

                if log_timings:
                    evt_cpu.record()

                # Map local indices to global indices
                global_left = l_idxs[rows_cpu]
                global_right = r_idxs[cols_cpu]

                # Accumulate
                buffer.add(global_left, global_right, data_cpu)
                total_pairs += len(data_cpu)

                del rows_cpu, cols_cpu, data_cpu, global_left, global_right, sim_coo
            else:
                if log_timings:
                    evt_cpu.record()

            del sim, R_gpu
            gpu_batch_count += 1

            if log_timings:
                evt_cpu.synchronize()
                aggregated_timings.transfer_to_gpu_ms += cp.cuda.get_elapsed_time(
                    evt_inner_start, evt_r_xfer
                )
                aggregated_timings.normalize_right_ms += cp.cuda.get_elapsed_time(
                    evt_r_xfer, evt_r_norm
                )
                aggregated_timings.spmm_ms += cp.cuda.get_elapsed_time(
                    evt_r_norm, evt_spmm
                )
                aggregated_timings.threshold_and_extract_ms += cp.cuda.get_elapsed_time(
                    evt_spmm, evt_thresh
                )
                aggregated_timings.transfer_to_cpu_ms += cp.cuda.get_elapsed_time(
                    evt_thresh, evt_cpu
                )
                aggregated_timings.total_ms += cp.cuda.get_elapsed_time(
                    evt_inner_start, evt_cpu
                )

            # Flush buffer to writer periodically
            if writer and not buffer.is_empty():
                if gpu_batch_count % config.write_buffer_batches == 0:
                    data = buffer.flush()
                    if data:
                        writer.write_batch(data)

        # Cleanup left batch
        del L_gpu
        # Force garbage collection to prevent fragmentation
        cp.get_default_memory_pool().free_all_blocks()

        if logger:
            elapsed = perf_counter() - t_left_batch
            rate = l_csr.shape[0] / elapsed
            logger.info(
                f"  [Left batch {i + 1}/{num_batches_left}] Done in {elapsed:.2f}s "
                f"({rate:.1f} spectra/s)"
            )

    t_compute = perf_counter() - t_compute_start

    # =========================================================================
    # 8. Finalize Output
    # =========================================================================

    if logger:
        logger.info(f"Computation complete in {t_compute:.2f}s")
        logger.info(f"Total matching pairs found: {total_pairs:_}")

    # Flush remaining buffer
    final_data = buffer.flush()

    if writer:
        if final_data:
            writer.write_batch(final_data)
        writer.stop()
        if logger:
            logger.info("Writer stopped.")

        # Return LazyFrame scan
        result = pl.scan_parquet(output_path)
    else:
        # Return in-memory DataFrame
        if final_data:
            result = pl.DataFrame(final_data)
        else:
            # Empty result
            result = pl.DataFrame(
                {
                    "idx_left": pl.Series([], dtype=pl.Int32),
                    "idx_right": pl.Series([], dtype=pl.Int32),
                    "similarity": pl.Series([], dtype=pl.Float32),
                }
            )

    if log_timings:
        # Add wall time (convert seconds to milliseconds)
        aggregated_timings.wall_time_ms = t_compute * 1000.0
        return result, aggregated_timings
    else:
        return result
