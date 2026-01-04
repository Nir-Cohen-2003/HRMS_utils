"""
Batched GPU-accelerated proximate similarity computation with optimized memory transfers.

This module implements an efficient batched approach for computing pairwise similarities
on GPU, with the following optimizations:
1. Upper triangular computation (batch i vs batches i..N) to avoid redundant transfers
2. Overlapped data transfer and computation where possible
3. Batched writing to intermediate files to avoid stalling GPU compute
4. Streaming exact similarity computation from intermediate results

Why: GPU similarity is memory-bound, so we maximize efficiency by minimizing data
transfers and overlapping I/O with compute.
"""

import gc
import logging
import math
import shutil
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import List, Union

import cupy as cp
import cupyx.scipy.sparse as cps
import numpy as np
import polars as pl
import scipy.sparse as sp
from approximate_similarity import (
    SimilarityConfig,
    _expand_csr_horizontal_adaptive_gpu,
    _normalize_csr_rows_inplace_gpu,
    _sparse_bin_spectra_df_to_csr,
    _sparse_proximate_similarity_pairs_above_threshold_gpu,
)
from batched_exact_cosine import (
    _compute_dynamic_max_peaks_exact,
    _extract_lists_from_df,
    _run_exact_cosine_gpu_batched,
)
from batched_utils import BatchedGPUConfig, _log_message_to_file, _yield_batches_dynamic
from numpy.dtypes import UShortDType
from numpy.typing import NDArray

import hrms_utils

logger = logging.getLogger(__name__)


def _write_batch_results_to_parquet(
    batch_results: List[
        tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float32]]
    ],
    output_path: Path,
    mode_name: str,
    batch_counter: int,
) -> int:
    """
    Write accumulated batch results to a single parquet file.

    Args:
        batch_results: List of (left_idxs, right_idxs, similarities) tuples
        output_path: Path to write parquet file
        mode_name: Ion mode name for filename
        batch_counter: Batch number for filename

    Returns:
        Total number of pairs written
    """
    if not batch_results:
        return 0

    # Concatenate all results
    all_left = np.concatenate([r[0] for r in batch_results])
    all_right = np.concatenate([r[1] for r in batch_results])
    all_sims = np.concatenate([r[2] for r in batch_results])

    n_pairs = len(all_left)

    if n_pairs == 0:
        return 0

    # Create dataframe and write
    df = pl.DataFrame(
        {
            "idx": all_left,
            "idx_right": all_right,
            "proximate_similarity": all_sims,
        }
    )

    chunk_path = output_path / f"pairs_{mode_name}_batch_{batch_counter}.parquet"
    df.write_parquet(chunk_path)

    return n_pairs


# `_yield_batches_dynamic` implementation moved to `batched_utils._yield_batches_dynamic`.
# Refer to that module for the implementation that yields batches based on non-zero counts.


def _compute_batched_gpu_similarity_single_mode(
    mode_df: pl.DataFrame,
    mode_name: str,
    config: BatchedGPUConfig,
    temp_dir: Path,
    log_path: Path,
    right_mode_df: pl.DataFrame | None = None,
) -> int:
    """
    Compute all-vs-all proximate similarity for a single ion mode using batched GPU.

    Optimizations:
    1. Dynamic batching based on GPU memory and peak counts.
    2. Loop reordering: Iterate Right batches (outer) vs Left batches (inner).
       This allows expanding the Right batch once and reusing it against all Left batches.
    """
    n_spectra_left = len(mode_df)

    if n_spectra_left == 0:
        return 0

    # Determine if cross-library comparison
    is_cross_library = right_mode_df is not None

    if is_cross_library:
        n_spectra_right = len(right_mode_df)
        if n_spectra_right == 0:
            return 0
        _log_message_to_file(
            f"  Cross-library mode: {n_spectra_left} left vs {n_spectra_right} right spectra",
            log_path,
        )
    else:
        n_spectra_right = n_spectra_left
        _log_message_to_file(
            f"  Binning {n_spectra_left} spectra for mode {mode_name} (GPU batched)...",
            log_path,
        )

    # Ensure the approximate configuration is present and non-None
    assert config.approx_config is not None, (
        "approx_config must be provided on BatchedGPUConfig"
    )
    approx_config = config.approx_config

    # Convert left library to CSR matrix
    t_bin = perf_counter()
    left_csr_matrix, _ = _sparse_bin_spectra_df_to_csr(
        mode_df,
        "cleaned_normalized_mz",
        "cleaned_normalized_intensity",
        upper_bound=approx_config.upper_mass_bound,
        intensity_power=approx_config.intensity_power,
        bin_size=approx_config.bin_size,
    )
    left_global_idxs = mode_df["idx"].cast(pl.Int64).to_numpy()

    # Convert right library to CSR matrix if cross-library
    if is_cross_library:
        assert right_mode_df is not None, (
            "right_mode_df must not be None in cross-library mode"
        )
        right_csr_matrix, _ = _sparse_bin_spectra_df_to_csr(
            right_mode_df,
            "cleaned_normalized_mz",
            "cleaned_normalized_intensity",
            upper_bound=approx_config.upper_mass_bound,
            intensity_power=approx_config.intensity_power,
            bin_size=approx_config.bin_size,
        )
        right_global_idxs = right_mode_df["idx"].cast(pl.Int64).to_numpy()
    else:
        # Self-comparison: use same matrix for both sides
        right_csr_matrix = left_csr_matrix
        right_global_idxs = left_global_idxs

    _log_message_to_file(
        f"  Binning complete in {perf_counter() - t_bin:.3f}s. "
        f"Left: {left_csr_matrix.shape}, Right: {right_csr_matrix.shape}",
        log_path,
    )

    # --- Dynamic Batching Calculation ---
    # Estimate max peaks per batch based on available GPU memory
    free_mem, total_mem = cp.cuda.Device(0).mem_info
    target_mem = free_mem * config.target_gpu_mem_ratio

    # Memory model:
    # Why: The similarity matrix (N×N×4 bytes) dominates memory usage, not CSR storage.
    # During computation we have simultaneously in GPU memory:
    # 1. R_gpu (expanded): N_spectra * avg_peaks_per_spectrum * 12 bytes * expansion_factor
    # 2. L_gpu: N_spectra * avg_peaks_per_spectrum * 12 bytes
    # 3. sim matrix: N_spectra^2 * 4 bytes (dense/semi-dense similarity matrix)
    # 4. Temporary arrays during thresholding: ~20% overhead
    #
    # Total ≈ N * bytes_per_spectrum * (1 + expansion) + N^2 * 4 + overhead
    # This is quadratic in N due to the similarity matrix.

    bytes_per_peak = 12
    avg_peaks_per_spectrum = left_csr_matrix.nnz / max(n_spectra_left, 1)

    # Calculate expansion factor, due to expandign the right CSR matrix due to ms2 tolerance
    expansion_factor = 1.0
    if approx_config.ms2_tolerance_ppm is not None:
        # Estimate expansion: window_da / bin_size
        mz_max = approx_config.upper_mass_bound
        tol_ppm = approx_config.ms2_tolerance_ppm
        window_da = mz_max * tol_ppm * 1e-6
        # 2x window (plus/minus)
        expansion_factor = max(1.0, (2 * window_da) / approx_config.bin_size)
        # Add safety margin for density
        expansion_factor *= 2.0

    bytes_per_spectrum_csr = avg_peaks_per_spectrum * bytes_per_peak
    bytes_per_spectrum_expanded = bytes_per_spectrum_csr * expansion_factor

    # Solve quadratic equation for optimal batch size N:
    # target_mem * safety_factor = N^2 * 4 + N * (bytes_expanded + bytes_csr)
    # Rearranged: 4*N^2 + (bytes_expanded + bytes_csr)*N - target_mem*safety_factor = 0
    safety_factor = 0.8  # Use 80% of target to account for temporary allocations

    a = 4.0  # bytes per element in similarity matrix
    b = bytes_per_spectrum_expanded + bytes_per_spectrum_csr
    c = -target_mem * safety_factor

    discriminant = b**2 - 4 * a * c

    if discriminant > 0:
        # Quadratic formula: (-b + sqrt(b^2 - 4ac)) / 2a
        estimated_spectra_per_batch = int((-b + np.sqrt(discriminant)) / (2 * a))
    else:
        # Fallback: very conservative estimate
        estimated_spectra_per_batch = int(
            target_mem / (bytes_per_spectrum_expanded * 10)
        )

    # Ensure reasonable minimum batch size
    estimated_spectra_per_batch = max(100, estimated_spectra_per_batch)

    # Convert from spectra count to peak count for batching logic
    estimated_max_peaks = int(estimated_spectra_per_batch * avg_peaks_per_spectrum)

    # Clamp with user config if provided
    if config.max_peaks_per_batch is not None:
        max_peaks = min(estimated_max_peaks, config.max_peaks_per_batch)
    else:
        max_peaks = estimated_max_peaks

    # Ensure reasonable minimum (e.g. 100k peaks)
    max_peaks = max(max_peaks, 100_000)

    _log_message_to_file(
        f"  Dynamic Batching: Free GPU Mem={free_mem / 1e9:.2f}GB. "
        f"Target Usage={config.target_gpu_mem_ratio:.0%} (={target_mem / 1e9:.2f}GB). "
        f"Est. Expansion={expansion_factor:.1f}x. "
        f"Est. Spectra/Batch={estimated_spectra_per_batch:_}. "
        f"Max Peaks/Batch={max_peaks:_}",
        log_path,
    )

    # Generate batches
    # For self-comparison, we must use the exact same batch definitions for left and right
    # to correctly identify diagonal blocks.
    batches_left = list(
        _yield_batches_dynamic(left_csr_matrix, left_global_idxs, max_peaks)
    )
    if is_cross_library:
        batches_right = list(
            _yield_batches_dynamic(right_csr_matrix, right_global_idxs, max_peaks)
        )
    else:
        batches_right = batches_left

    num_batches_left = len(batches_left)
    num_batches_right = len(batches_right)

    _log_message_to_file(
        f"  Created {num_batches_left} left batches and {num_batches_right} right batches.",
        log_path,
    )

    total_pairs = 0
    batch_results: List[
        tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float32]]
    ] = []
    gpu_batch_count = 0
    file_counter = 0

    # Outer Loop: Right Batches (Expanded)
    # Why: We expand the right batch once and reuse it against all left batches.
    for j, (r_start, r_end, r_csr, r_idxs) in enumerate(batches_right):
        t_batch_right = perf_counter()

        # Transfer Right to GPU
        R_gpu = cps.csr_matrix(r_csr)
        # Normalize Right (before expansion)
        _ = _normalize_csr_rows_inplace_gpu(R_gpu)

        # Expand Right (if configured)
        if approx_config.ms2_tolerance_ppm is not None:
            R_gpu = _expand_csr_horizontal_adaptive_gpu(
                R_gpu,
                approx_config.bin_size,
                approx_config.ms2_tolerance_ppm,
                approx_config.nbins,
            )

        # Inner Loop: Left Batches
        for i, (l_start, l_end, l_csr, l_idxs) in enumerate(batches_left):
            # Triangular check for self-comparison
            # We want pairs (u, v) where u <= v.
            # If i > j, then u > v (mostly). Skip.
            if not is_cross_library and i > j:
                continue

            t_gpu = perf_counter()

            # Transfer Left to GPU
            L_gpu = cps.csr_matrix(l_csr)
            # Normalize Left
            _ = _normalize_csr_rows_inplace_gpu(L_gpu)

            # Matmul
            # L (normalized) * R (normalized & expanded).T
            sim = L_gpu.dot(R_gpu.T)

            # Thresholding & Extraction
            # Logic copied from _sparse_proximate_similarity_pairs_above_threshold_gpu
            # to avoid re-normalization issues.
            mask = sim.data >= config.approx_config.approx_threshold

            if int(mask.sum()) > 0:
                out_data = sim.data[mask]
                out_cols = sim.indices[mask]
                indices_in_data = cp.nonzero(mask)[0]
                out_rows = (
                    cp.searchsorted(sim.indptr, indices_in_data, side="right") - 1
                )

                # Transfer back to CPU
                li = cp.asnumpy(out_rows).astype(np.int64)
                ri = cp.asnumpy(out_cols).astype(np.int64)
                prox_sims_out = cp.asnumpy(out_data).astype(np.float32)

                left_pairs = l_idxs[li]
                right_pairs = r_idxs[ri]

                # Filter diagonal block for self-comparison
                if not is_cross_library and i == j:
                    # Remove self-matches
                    mask_diag = left_pairs != right_pairs
                    left_pairs = left_pairs[mask_diag]
                    right_pairs = right_pairs[mask_diag]
                    prox_sims_out = prox_sims_out[mask_diag]

                    # Keep upper triangle (u < v)
                    upper_mask = left_pairs < right_pairs
                    left_pairs = left_pairs[upper_mask]
                    right_pairs = right_pairs[upper_mask]
                    prox_sims_out = prox_sims_out[upper_mask]

                if len(left_pairs) > 0:
                    batch_results.append((left_pairs, right_pairs, prox_sims_out))
                    total_pairs += len(left_pairs)

            gpu_batch_count += 1

            # Periodic write
            if gpu_batch_count % config.gpu_batch_write_interval == 0 and batch_results:
                t_write = perf_counter()
                pairs_written = _write_batch_results_to_parquet(
                    batch_results,
                    temp_dir,
                    mode_name,
                    file_counter,
                )
                _log_message_to_file(
                    f"    Wrote {pairs_written} pairs to file {file_counter} in {perf_counter() - t_write:.3f}s",
                    log_path,
                )
                batch_results.clear()
                file_counter += 1
                gc.collect()

            # Free Left batch memory immediately
            del L_gpu, sim
            if j % 1 == 0:
                cp.get_default_memory_pool().free_all_blocks()  # Optional, might be slow

        # Free Right batch memory after inner loop
        del R_gpu
        cp.get_default_memory_pool().free_all_blocks()

        _log_message_to_file(
            f"  Completed Right Batch {j}/{num_batches_right} in {perf_counter() - t_batch_right:.3f}s",
            log_path,
        )

    # Write remaining
    if batch_results:
        t_write = perf_counter()
        pairs_written = _write_batch_results_to_parquet(
            batch_results,
            temp_dir,
            mode_name,
            file_counter,
        )
        _log_message_to_file(
            f"    Wrote final {pairs_written} pairs to file {file_counter} in {perf_counter() - t_write:.3f}s",
            log_path,
        )
        batch_results.clear()

    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()

    return total_pairs


def build_and_write_pairs_parquet_gpu_batched(
    parquet_paths: List[Path],
    output_path: Union[str, Path],
    batched_config: BatchedGPUConfig,
    num_spectra: int | None = None,
    mass_range: tuple[float, float] | None = None,
    right_parquet_paths: List[Path] | None = None,
    num_spectra_right: int | None = None,
) -> None:
    """
    Build all-vs-all proximate similarity pairs using batched GPU computation,
    then compute exact similarities via streaming.

    This function requires a fully-constructed `BatchedGPUConfig` which contains
    batching parameters and an embedded `SimilarityConfig` (`approx_config`).

    Mode is automatically determined: if `right_parquet_paths` is provided, uses cross-library
    mode with full NxM comparison. Otherwise uses single-library mode with upper triangular.

    Args:
        parquet_paths: List of paths to input parquet files (left library)
        output_path: Path for final output parquet
        batched_config: Required `BatchedGPUConfig` with batching and approximate settings
        num_spectra: Optional limit on number of spectra to process from left library
        mass_range: Optional (min_mz, max_mz) filter for precursor mass
        right_parquet_paths: Optional list of paths for right library. If provided,
            automatically enables cross-library comparison mode.
        num_spectra_right: Optional limit on number of spectra from right library
    """
    # Fail fast: batched_config must be provided and must contain an approx_config
    assert isinstance(batched_config, BatchedGPUConfig), (
        "batched_config must be a BatchedGPUConfig"
    )
    assert batched_config.approx_config is not None, (
        "batched_config.approx_config must be provided (upper_mass_bound and bin_size are required)"
    )

    # Determine cross-library mode automatically from right_parquet_paths
    is_cross_library = right_parquet_paths is not None

    output_path = Path(output_path)
    log_path = output_path.with_suffix(".log")

    comparison_mode = (
        "cross-library" if is_cross_library else "single-library (upper triangular)"
    )
    _log_message_to_file(
        f"Started GPU batched proximate similarity: output={output_path}\n"
        f"  Mode: {comparison_mode}\n"
        f"  Approx config: {batched_config.approx_config}\n"
        f"  Batch size: {batched_config.batch_size}\n"
        f"  Threshold: {batched_config.approx_config.threshold}\n",
        log_path,
        overwrite=True,
    )

    temp_dir = Path(tempfile.mkdtemp(prefix="gpu_batched_proximate_"))
    _log_message_to_file(f"Created temp dir: {temp_dir}", log_path)

    try:
        assert len(parquet_paths) > 0, "parquet_paths must contain at least one path"

        # --- Step 1: Load and Preprocess Left Library ---
        _log_message_to_file("Loading and preprocessing left library...", log_path)

        lf_list = []
        for parquet_path in parquet_paths:
            assert Path(parquet_path).exists(), (
                f"Parquet file does not exist: {parquet_path}"
            )
            lf = pl.scan_parquet(str(parquet_path))

            if mass_range is not None:
                min_mz, max_mz = mass_range
                lf = lf.filter(
                    pl.col("precursor_mz").is_between(float(min_mz), float(max_mz))
                )

            lf_list.append(lf)

        lf = pl.union(lf_list).filter(pl.col("clean_precursor"))

        if num_spectra is not None:
            lf = lf.limit(num_spectra)

        # Select required columns and add global index
        lf = (
            lf.select(
                [
                    "precursor_type",
                    "precursor_mz",
                    "precursor_formula_array",
                    "ion_mode",
                    "base_inchikey",
                    "spectral_information_score",
                    "cleaned_normalized_mz",
                    "cleaned_normalized_intensity",
                    "smiles",
                ]
            )
            .filter(pl.col("smiles").is_not_null())
            .with_row_index("idx")
            .with_columns(
                mol_idx=pl.col("idx").min().over(["base_inchikey", "ion_mode"])
            )
            .sort(["idx", "mol_idx"])
        )

        _log_message_to_file("Materializing left library...", log_path)
        t_load = perf_counter()
        df_source = lf.collect()
        load_time = perf_counter() - t_load

        _log_message_to_file(
            f"Loaded {len(df_source)} spectra (left) in {load_time:.3f}s",
            log_path,
        )

        if len(df_source) == 0:
            _log_message_to_file("Left library is empty, nothing to do.", log_path)
            return

        # Persist minimal left-library snapshot for downstream joins (idx/mol_idx + metadata)
        left_library_snapshot = df_source.select(
            [
                "idx",
                "mol_idx",
                "base_inchikey",
                "ion_mode",
                "smiles",
                "precursor_mz",
                "spectral_information_score",
            ]
        )
        left_snapshot_path = output_path.with_suffix(".left_library.parquet")
        left_library_snapshot.write_parquet(left_snapshot_path)
        _log_message_to_file(
            f"Wrote left library snapshot to {left_snapshot_path} (rows={len(left_library_snapshot)})",
            log_path,
        )

        # --- Step 1b: Load and Preprocess Right Library (if cross-library) ---
        df_source_right = None
        if is_cross_library:
            assert right_parquet_paths is not None
            _log_message_to_file("Loading and preprocessing right library...", log_path)

            lf_list_right = []
            for parquet_path in right_parquet_paths:
                assert Path(parquet_path).exists(), (
                    f"Right parquet file does not exist: {parquet_path}"
                )
                lf_r = pl.scan_parquet(str(parquet_path))

                if mass_range is not None:
                    min_mz, max_mz = mass_range
                    lf_r = lf_r.filter(
                        pl.col("precursor_mz").is_between(float(min_mz), float(max_mz))
                    )

                lf_list_right.append(lf_r)

            lf_right = pl.union(lf_list_right).filter(pl.col("clean_precursor"))

            if num_spectra_right is not None:
                lf_right = lf_right.limit(num_spectra_right)

            # Select required columns and add global index (offset by left library size)
            # Why: idx must be unique across both libraries for proper joining later
            left_size = len(df_source)
            lf_right = (
                lf_right.select(
                    [
                        "precursor_type",
                        "precursor_mz",
                        "precursor_formula_array",
                        "ion_mode",
                        "base_inchikey",
                        "spectral_information_score",
                        "cleaned_normalized_mz",
                        "cleaned_normalized_intensity",
                        "smiles",
                    ]
                )
                .filter(pl.col("smiles").is_not_null())
                .with_row_index("idx", offset=left_size)
                .with_columns(
                    mol_idx=pl.col("idx").min().over(["base_inchikey", "ion_mode"])
                )
                .sort(["idx", "mol_idx"])
            )

            _log_message_to_file("Materializing right library...", log_path)
            t_load_right = perf_counter()
            df_source_right = lf_right.collect()
            load_time_right = perf_counter() - t_load_right

            _log_message_to_file(
                f"Loaded {len(df_source_right)} spectra (right) in {load_time_right:.3f}s",
                log_path,
            )

            if len(df_source_right) == 0:
                _log_message_to_file("Right library is empty, nothing to do.", log_path)
                return

            # Persist minimal right-library snapshot for downstream joins (idx/mol_idx + metadata)
            right_library_snapshot = df_source_right.select(
                [
                    "idx",
                    "mol_idx",
                    "base_inchikey",
                    "ion_mode",
                    "smiles",
                    "precursor_mz",
                    "spectral_information_score",
                ]
            )
            right_snapshot_path = output_path.with_suffix(".right_library.parquet")
            right_library_snapshot.write_parquet(right_snapshot_path)
            _log_message_to_file(
                f"Wrote right library snapshot to {right_snapshot_path} (rows={len(right_library_snapshot)})",
                log_path,
            )

            # Concatenate for final exact computation
            # Why: Needed for efficient joining in streaming exact stage
            df_source_combined = pl.concat([df_source, df_source_right])
        else:
            df_source_combined = df_source

        # --- Step 2: GPU Batched Proximate Similarity ---
        _log_message_to_file("Starting GPU batched proximate similarity...", log_path)

        ion_modes = df_source["ion_mode"].unique().to_list()
        if is_cross_library:
            # Ensure the right-hand dataframe was materialized before subscripting it.
            # Why: In some failure paths df_source_right could be None even when
            # `is_cross_library` is True; fail fast with a clear message instead
            # of raising a less-descriptive TypeError when attempting to subscript None.
            assert df_source_right is not None, (
                "df_source_right must not be None when is_cross_library=True"
            )
            # Intersect with right library ion modes
            ion_modes_right = df_source_right["ion_mode"].unique().to_list()
            ion_modes = list(set(ion_modes) & set(ion_modes_right))
            _log_message_to_file(
                f"Processing common ion modes: {ion_modes} "
                f"(left: {df_source['ion_mode'].unique().to_list()}, "
                f"right: {ion_modes_right})",
                log_path,
            )
        else:
            _log_message_to_file(f"Processing ion modes: {ion_modes}", log_path)

        total_proximate_pairs = 0
        proximate_start = perf_counter()

        for mode in ion_modes:
            mode_df = df_source.filter(pl.col("ion_mode") == mode)
            n_mode = len(mode_df)

            if n_mode == 0:
                continue

            # Get right mode df if cross-library
            right_mode_df = None
            if is_cross_library:
                assert df_source_right is not None, (
                    "df_source_right must not be None in cross-library mode"
                )
                right_mode_df = df_source_right.filter(pl.col("ion_mode") == mode)
                n_mode_right = len(right_mode_df)

                if n_mode_right == 0:
                    _log_message_to_file(
                        f"Skipping mode {mode}: no right spectra",
                        log_path,
                    )
                    continue

                _log_message_to_file(
                    f"\nProcessing mode {mode}: {n_mode} left vs {n_mode_right} right spectra",
                    log_path,
                )
            else:
                _log_message_to_file(
                    f"\nProcessing mode {mode}: {n_mode} spectra",
                    log_path,
                )

            mode_pairs = _compute_batched_gpu_similarity_single_mode(
                mode_df,
                mode,
                batched_config,
                temp_dir,
                log_path,
                right_mode_df=right_mode_df,
            )

            total_proximate_pairs += mode_pairs
            _log_message_to_file(
                f"  Mode {mode} complete: {mode_pairs} proximate pairs",
                log_path,
            )

            # Free memory between modes
            del mode_df
            if right_mode_df is not None:
                del right_mode_df
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()

        proximate_time = perf_counter() - proximate_start
        _log_message_to_file(
            f"\nProximate stage complete: {total_proximate_pairs} total pairs in {proximate_time:.3f}s",
            log_path,
        )

        if total_proximate_pairs == 0:
            _log_message_to_file(
                "No pairs found exceeding proximate threshold. Exiting.",
                log_path,
            )
            return

        # --- Step 3: Exact Similarity Computation ---
        use_gpu_exact = batched_config.approx_config.use_gpu_exact_cosine
        exact_method = "GPU (batched)" if use_gpu_exact else "CPU (streaming)"
        _log_message_to_file(
            f"\nStarting exact similarity computation ({exact_method})...", log_path
        )
        exact_start = perf_counter()

        # Scan intermediate files
        lf_indices = pl.scan_parquet(str(temp_dir / "*.parquet"))

        # Prepare source for joining (use combined source if cross-library)
        lf_source = df_source_combined.lazy()

        # Join to get pair metadata and filter self-matches
        # 1. Join left spectra
        joined = lf_indices.join(lf_source, on="idx")

        # 2. Join right spectra
        joined = joined.join(
            lf_source,
            left_on="idx_right",
            right_on="idx",
            suffix="_right",
        )

        # 3. Filter self-matches (same molecule) - always filter these
        # Why: Even in cross-library mode, same molecule could appear in both libraries
        joined = joined.filter(pl.col("base_inchikey") != pl.col("base_inchikey_right"))

        # Ensure ms2 tolerance is set in the approximate config (fail fast if not).
        assert batched_config.approx_config.ms2_tolerance_ppm is not None, (
            "approx_config.ms2_tolerance_ppm must be provided on batched_config"
        )

        if use_gpu_exact:
            # GPU exact path: collect pairs and compute in batches
            _log_message_to_file(
                "Using GPU exact cosine with dynamic batching", log_path
            )

            # Collect pairs that need exact computation
            pairs_df = joined.select(
                [
                    "idx",
                    "idx_right",
                    "mol_idx",
                    "mol_idx_right",
                    "base_inchikey",
                    "base_inchikey_right",
                ]
            ).collect()

            n_pairs = len(pairs_df)
            _log_message_to_file(
                f"Computing exact similarity for {n_pairs} pairs on GPU", log_path
            )

            if n_pairs == 0:
                _log_message_to_file("No pairs to process after filtering", log_path)
                # Create empty output
                pl.DataFrame(
                    {
                        "idx": [],
                        "idx_right": [],
                        "mol_idx": [],
                        "mol_idx_right": [],
                        "dotprod_similarity": [],
                    }
                ).write_parquet(output_path)
            else:
                # Extract spectra
                mz_left, int_left = _extract_lists_from_df(df_source_combined)
                mz_right, int_right = mz_left, int_left

                pair_left = pairs_df["idx"].to_numpy()
                pair_right = pairs_df["idx_right"].to_numpy()

                # Compute max peaks per batch
                max_peaks_per_batch = _compute_dynamic_max_peaks_exact(
                    target_gpu_mem_ratio=0.5,
                    user_max_peaks=None,
                )
                _log_message_to_file(
                    f"GPU exact batch size: {max_peaks_per_batch} total peaks from unique spectra",
                    log_path,
                )

                # Run GPU exact cosine
                exact_scores = _run_exact_cosine_gpu_batched(
                    pair_left,
                    pair_right,
                    mz_left,
                    int_left,
                    mz_right,
                    int_right,
                    config=batched_config.approx_config,
                    max_peaks_per_batch=max_peaks_per_batch,
                    verbose=True,
                )

                # Add scores to pairs DataFrame and filter by threshold
                results = (
                    pairs_df.with_columns(
                        dotprod_similarity=pl.Series("dotprod_similarity", exact_scores)
                    )
                    .filter(
                        pl.col("dotprod_similarity").is_not_null(),
                        pl.col("dotprod_similarity").ge(
                            batched_config.approx_config.threshold
                        ),
                    )
                    .select(
                        [
                            "idx",
                            "idx_right",
                            "mol_idx",
                            "mol_idx_right",
                            "dotprod_similarity",
                        ]
                    )
                )

                # Write results
                results.write_parquet(output_path)

                _log_message_to_file(
                    f"GPU exact complete: {len(results)} pairs above threshold",
                    log_path,
                )
        else:
            # CPU exact path: streaming computation
            _log_message_to_file("Using CPU exact cosine (streaming)", log_path)

            # 4. Prepare struct for exact similarity computation
            joined = joined.with_columns(
                spectra=pl.struct(
                    mz1=pl.col("cleaned_normalized_mz"),
                    intensities1=pl.col("cleaned_normalized_intensity"),
                    mz2=pl.col("cleaned_normalized_mz_right"),
                    intensities2=pl.col("cleaned_normalized_intensity_right"),
                    precursor_mz1=pl.col("precursor_mz"),
                    precursor_mz2=pl.col("precursor_mz_right"),
                )
            )

            # 5. Compute exact dotprod similarity and filter
            results = (
                joined.with_columns(
                    dotprod_similarity=pl.col(
                        "spectra"
                    ).spectral_similarity.dotprod_similarity(  # type: ignore
                        ms2_tolerance_in_ppm=batched_config.approx_config.ms2_tolerance_ppm,
                        clean_spectra_first=False,
                        ignore_precursor=True,
                    )
                )
                .drop("spectra")
                .filter(
                    pl.col("dotprod_similarity").is_not_null(),
                    pl.col("dotprod_similarity").ge(
                        batched_config.approx_config.threshold
                    ),
                )
                .select(
                    "idx",
                    "idx_right",
                    "mol_idx",
                    "mol_idx_right",
                    "dotprod_similarity",
                )
            )

            # 6. Sink to output parquet (streaming)
            # Why sink: Processes and writes in streaming fashion, low memory footprint
            results.sink_parquet(output_path, maintain_order=False)

        exact_time = perf_counter() - exact_start
        _log_message_to_file(
            f"Exact computation complete in {exact_time:.3f}s",
            log_path,
        )

        # Read final count (this is lightweight, just metadata)
        final_count = pl.scan_parquet(output_path).select(pl.len()).collect().item()

        _log_message_to_file(
            f"\nFinal results: {final_count} pairs written to {output_path}",
            log_path,
        )
        _log_message_to_file(
            f"Total time: {perf_counter() - t_load + load_time:.3f}s "
            f"(load: {load_time:.3f}s, proximate: {proximate_time:.3f}s, exact: {exact_time:.3f}s)",
            log_path,
        )

    except Exception as e:
        _log_message_to_file(
            f"Error in build_and_write_pairs_parquet_gpu_batched: {e}\n{traceback.format_exc()}",
            log_path,
            level=logging.ERROR,
        )
        raise

    finally:
        # Cleanup temp directory
        if temp_dir.exists():
            _log_message_to_file(f"Cleaning up temp directory: {temp_dir}", log_path)
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # Example usage
    example_parquet_paths = [
        # Path("/home/analytit_admin/Data/spectral_libs/info_score/fraghub_100k.parquet"),
        Path("/home/analytit_admin/Data/spectral_libs/info_score/fraghub_300k.parquet"),
        # Path("/home/analytit_admin/Data/spectral_libs/info_score/fraghub_600k.parquet"),
        # Path("/home/analytit_admin/Data/spectral_libs/fraghub/fraghub.parquet"),
    ]
    output_parquet_path = Path("output_similarity_pairs.parquet")

    approx_cfg = SimilarityConfig(
        upper_mass_bound=1000.0,
        bin_size=0.0001,
        ms2_tolerance_ppm=10.0,
        intensity_power=0.5,
        threshold=0.8,
        use_gpu_exact_cosine=True,
    )

    # Construct the BatchedGPUConfig here (non-optional) and pass it to the
    # main function. This ensures the function receives a concrete config
    # object instead of building one internally.
    batched_cfg = BatchedGPUConfig(
        batch_size=10000,
        gpu_batch_write_interval=100,
        approx_config=approx_cfg,
        target_gpu_mem_ratio=0.1,
    )

    build_and_write_pairs_parquet_gpu_batched(
        parquet_paths=example_parquet_paths,
        right_parquet_paths=example_parquet_paths,
        output_path=output_parquet_path,
        batched_config=batched_cfg,
    )
