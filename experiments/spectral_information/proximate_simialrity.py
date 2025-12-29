import concurrent.futures
import gc
import logging
import math
import multiprocessing
import os
import shutil
import tempfile
import traceback
from pathlib import Path
from time import perf_counter
from typing import List, Union

import numpy as np
import polars as pl

# import pyarrow.parquet as pq
from utils import _log_message_to_file


def _numpy_proximate_similarity_pairs_above_threshold(
    left_mat: np.ndarray,
    right_mat: np.ndarray,
    threshold: float,
    left_global_idxs: np.ndarray,
    right_global_idxs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Proximate stage (optimized NumPy/BLAS):

    - Normalize rows of `left_mat` and `right_mat` (safe divide for zero rows)
    - Compute full similarity matrix via BLAS matmul
    - Return all (idx, idx_right, proximate_similarity) where similarity >= threshold

    Notes:
      - This intentionally uses NumPy so the heavy lifting is done by BLAS.
      - Extraction of values uses advanced indexing (fast in NumPy, unsupported in Numba).
    """
    if left_mat.size == 0 or right_mat.size == 0:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float32),
        )

    assert left_mat.ndim == 2, f"expected left_mat to be 2D, got shape={left_mat.shape}"
    assert right_mat.ndim == 2, (
        f"expected right_mat to be 2D, got shape={right_mat.shape}"
    )

    L = left_mat
    R = right_mat

    # Row-wise L2 normalization (safe for zero rows)
    lnorm = np.linalg.norm(L, axis=1, keepdims=True)
    rnorm = np.linalg.norm(R, axis=1, keepdims=True)
    lnorm_safe = np.where(lnorm > 0.0, lnorm, 1.0)
    rnorm_safe = np.where(rnorm > 0.0, rnorm, 1.0)

    Ln = L / lnorm_safe
    Rn = R / rnorm_safe

    # BLAS matmul
    sim_matrix = (Ln @ Rn.T).astype(np.float32, copy=False)

    li, ri = np.where(sim_matrix >= np.float32(threshold))
    if li.size == 0:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float32),
        )

    left_idxs_out = left_global_idxs[li]
    right_idxs_out = right_global_idxs[ri]
    prox_sims_out = sim_matrix[li, ri].astype(np.float32, copy=False)

    return left_idxs_out, right_idxs_out, prox_sims_out


def _process_proximate_block(
    left_mat: np.ndarray,
    right_mat: np.ndarray,
    threshold: float,
    left_global_idxs: np.ndarray,
    right_global_idxs: np.ndarray,
    chunk_path: Path,
    block_info: tuple[int, int],
) -> tuple[int, tuple[int, int]]:
    """
    Helper function to run proximate similarity on a block and write results to parquet.
    """
    # Enforce single-threaded execution for this worker to avoid oversubscription/deadlocks
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["POLARS_MAX_THREADS"] = "1"

    l_idxs, r_idxs, sims = _numpy_proximate_similarity_pairs_above_threshold(
        left_mat, right_mat, threshold, left_global_idxs, right_global_idxs
    )
    n_pairs = len(l_idxs)
    if n_pairs > 0:
        chunk_df = pl.DataFrame(
            {
                "idx": l_idxs,
                "idx_right": r_idxs,
                "proximate_similarity": sims,
            }
        )
        chunk_df.write_parquet(chunk_path)
    return n_pairs, block_info


def _flatten_spectra_to_numpy(
    df: pl.DataFrame, mz_col: str, int_col: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Flatten list-valued spectrum columns from `df` into:
      - flat_mzs: np.ndarray[np.float64] of all m/z values
      - flat_ints: np.ndarray[np.float32] of corresponding intensities
      - spec_idx: np.ndarray[np.int64] of the originating spectrum index for each flattened peak
      - n_spec: int number of spectra in `df`

    Uses Polars' fast C-backed `explode` and avoids Python loops. Empty spectra are
    handled gracefully (they produce no flattened rows, but `n_spec` preserves the
    original number of spectra).

    Note: timing/logging for this step is performed at a higher level to allow
    aggregation per-batch (not per-mode).
    """
    n_spec = len(df)
    if n_spec == 0:
        return (
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float32),
            np.asarray([], dtype=np.int64),
            0,
        )

    # Add a row index so we can recover which spectrum each flattened peak
    # belongs to (this mirrors the spec_idx construction from list lengths).
    df_idx = df.with_row_index("__spec_idx")

    # Explode the mz/intensity list columns. If a row has no peaks the explode
    # will simply not produce rows for it (we still preserve `n_spec`).
    exploded = df_idx.explode([mz_col, int_col])
    if len(exploded) == 0:
        return (
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float32),
            np.asarray([], dtype=np.int64),
            n_spec,
        )

    # Enforce dtypes in Polars so `.to_numpy()` can be a no-copy view into the underlying buffer.
    # Why: Numba kernels require stable dtypes; keyword-based astype inside Numba breaks compilation.
    exploded = exploded.with_columns(
        [
            pl.col(mz_col).cast(pl.Float32),
            pl.col(int_col).cast(pl.Float32),
            pl.col("__spec_idx").cast(pl.Int64),
        ]
    )

    # Export columns from Polars directly to NumPy arrays (CPU).
    flat_mzs = exploded.get_column(mz_col).to_numpy()
    flat_ints = exploded.get_column(int_col).to_numpy()
    spec_idx = exploded.get_column("__spec_idx").to_numpy()

    return flat_mzs, flat_ints, spec_idx, n_spec


def _bin_flat_spectra_to_matrix(
    flat_mzs: np.ndarray,
    flat_ints: np.ndarray,
    spec_idx: np.ndarray,
    n_spec: int,
    upper_bound: int = 1000,
    intensity_power: float = 0.5,
) -> np.ndarray:
    """
    Bin already-flattened spectra arrays into a dense 2D matrix of shape (n_spec, nbins).

    This mirrors `_bin_spectra_to_matrix`'s vectorized logic but accepts flattened
    inputs (so callers can use Polars' `explode`/`list.len` workflow before converting
    to NumPy).

    Note: timing/logging for this step is collected by callers and aggregated per-batch.
    """
    nbins = int(upper_bound) + 1
    if n_spec == 0:
        return np.zeros((0, nbins), dtype=np.float32)
    if flat_mzs.size == 0 or flat_ints.size == 0:
        return np.zeros((n_spec, nbins), dtype=np.float32)

    # Bin to nearest integer mass and apply bounds filter
    mass_bins = np.rint(flat_mzs).astype(np.int32)
    valid_mask = (mass_bins >= 0) & (mass_bins <= upper_bound) & (flat_ints > 0)
    if not np.any(valid_mask):
        return np.zeros((n_spec, nbins), dtype=np.float32)

    mass_bins = mass_bins[valid_mask]
    spec_idx = spec_idx[valid_mask]
    weights = np.asarray(flat_ints[valid_mask], dtype=np.float32) ** float(
        intensity_power
    )

    # Build 1D keys and use bincount for accumulation (vectorized)
    flat_keys = spec_idx * nbins + mass_bins
    accum = np.bincount(flat_keys, weights=weights, minlength=n_spec * nbins).astype(
        np.float32
    )
    matrix = accum.reshape((n_spec, nbins))
    return matrix


def _bin_spectra_df_to_matrix(
    df: pl.DataFrame,
    mz_col: str = "cleaned_normalized_mz",
    int_col: str = "cleaned_normalized_intensity",
    upper_bound: int = 1000,
    intensity_power: float = 0.5,
) -> tuple[np.ndarray, dict]:
    """
    Explode list-valued spectra columns in `df` and bin them into a dense matrix
    of shape (n_spectra, nbins) in a single, vectorized call.

    Returns:
      - matrix: np.ndarray of shape (n_spectra, nbins)
      - timings: dict with keys:
          - 'flatten_time', 'bin_time', 'n_spec', 'n_peaks_total', 'n_peaks_valid', 'nbins'
    """
    nbins = int(upper_bound) + 1
    timings: dict = {
        "flatten_time": 0.0,
        "bin_time": 0.0,
        "n_spec": 0,
        "n_peaks_total": 0,
        "n_peaks_valid": 0,
        "nbins": nbins,
    }

    # Flatten (timed)
    t_flat0 = perf_counter()
    flat_mzs, flat_ints, spec_idx, n_spec = _flatten_spectra_to_numpy(
        df, mz_col, int_col
    )
    timings["flatten_time"] = perf_counter() - t_flat0
    timings["n_spec"] = int(n_spec)
    timings["n_peaks_total"] = int(flat_mzs.size)

    # Quick exits (preserve timings)
    if n_spec == 0:
        return np.zeros((0, nbins), dtype=np.float32), timings
    if flat_mzs.size == 0 or flat_ints.size == 0:
        return np.zeros((n_spec, nbins), dtype=np.float32), timings

    # Compute counters for reporting
    mass_bins = np.rint(flat_mzs).astype(np.int64)
    valid_mask = (mass_bins >= 0) & (mass_bins <= upper_bound) & (flat_ints > 0)
    timings["n_peaks_valid"] = int(np.count_nonzero(valid_mask))

    # Binning (timed)
    t_bin0 = perf_counter()
    matrix = _bin_flat_spectra_to_matrix(
        flat_mzs, flat_ints, spec_idx, n_spec, upper_bound, intensity_power
    )
    timings["bin_time"] = perf_counter() - t_bin0

    return matrix, timings


def build_and_write_pairs_parquet(
    parquet_paths: List[Path],
    output_path: Union[str, Path],
    threshold: float = 0.8,
    num_spectra: int | None = None,
    ms2_tolerance_ppm: float = 10.0,
    batch_size: int = 1000,
    mass_range: tuple[float, float] | None = None,
    proximate_bin_upper: int = 1000,
    num_workers: int = 4,
) -> None:
    """
    Build unioned library LF, compute pairwise dot-product similarities (ignoring precursor),
    and write pairs with high similarity to parquet.

    Refactored workflow:
    1. Load and clean all spectra into memory (df_source).
    2. Bin/Normalize spectra globally per ion mode.
    3. Run proximate similarity (matrix mult) in parallel batches (ProcessPoolExecutor),
       writing candidate indices to intermediate parquet files.
    4. Run exact similarity using Polars streaming by joining intermediate indices with source data.

    Args:
      - parquet_paths: list of Path objects pointing at library parquet files
      - output_path: where to write the pairs with similarities (required)
      - threshold: float (default 0.8). Only pairs with dotprod_similarity >= threshold are saved.
      - num_spectra: Optional[int]. If provided, limit the number of molecules read from the
        unioned input using a lazy .limit(num_spectra) to avoid collecting the full dataset.
      - batch_size: int (default 1000). Tile size for the proximate matrix multiplication.
      - mass_range: Optional[tuple[float, float]]. Filter spectra by precursor_mz.
      - proximate_bin_upper: int (default 1000). Upper bound for integer binning.
      - num_workers: int (default 4). Number of parallel workers for proximate search.
    """
    output_path = Path(output_path)
    log_path = output_path.with_suffix(".log")
    _log_message_to_file(
        f"Started build_and_write_pairs_parquet (Parallel): output={str(output_path)} threshold={threshold} "
        f"num_spectra={num_spectra} batch_size={batch_size} proximate_bin_upper={proximate_bin_upper} "
        f"num_workers={num_workers}",
        log_path,
        overwrite=True,
    )

    try:
        assert len(parquet_paths) > 0, "parquet_paths must contain at least one path"

        # --- Step 1: Load and Preprocess ---
        lf_list = []
        for PARQUET_PATH in parquet_paths:
            assert Path(PARQUET_PATH).exists(), (
                f"Requested parquet does not exist: {PARQUET_PATH}"
            )
            lf = pl.scan_parquet(str(PARQUET_PATH))
            if mass_range is not None:
                min_mz, max_mz = mass_range
                lf = lf.filter(
                    pl.col("precursor_mz").is_between(float(min_mz), float(max_mz))
                )
            lf_list.append(lf)

        lf = pl.union(lf_list).filter(pl.col("clean_precursor"))

        if num_spectra is not None:
            lf = lf.limit(num_spectra)

        # Select necessary columns and add global index
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

        _log_message_to_file("Materializing source library...", log_path)
        t_load = perf_counter()
        df_source = lf.collect()
        _log_message_to_file(
            f"Loaded {len(df_source)} spectra in {perf_counter() - t_load:.3f}s",
            log_path,
        )

        if len(df_source) == 0:
            _log_message_to_file("Source library is empty, nothing to do.", log_path)
            return

        # --- Step 2 & 3: Global Binning & Proximate Search (Parallel) ---
        temp_dir = Path(tempfile.mkdtemp(prefix="proximate_indices_"))
        _log_message_to_file(
            f"Created temp dir for intermediate indices: {temp_dir}", log_path
        )

        ion_modes = df_source["ion_mode"].unique().to_list()
        _log_message_to_file(f"Processing ion modes: {ion_modes}", log_path)

        total_proximate_pairs = 0
        proximate_start = perf_counter()

        for mode in ion_modes:
            mode_df = df_source.filter(pl.col("ion_mode") == mode)
            n_mode = len(mode_df)
            if n_mode == 0:
                continue

            _log_message_to_file(f"Processing mode {mode}: {n_mode} spectra", log_path)

            # Binning (Global for this mode)
            t_bin = perf_counter()
            matrix, timings = _bin_spectra_df_to_matrix(
                mode_df,
                "cleaned_normalized_mz",
                "cleaned_normalized_intensity",
                upper_bound=proximate_bin_upper,
                intensity_power=0.5,
            )
            _log_message_to_file(
                f"  Binning complete in {perf_counter() - t_bin:.3f}s", log_path
            )

            if matrix.size == 0:
                continue

            # Ensure matrix is ready for processing
            global_idxs = (
                mode_df["idx"].cast(pl.Int64).to_numpy()
            )  # Use global idx from df_source

            # Blocked processing
            num_blocks = math.ceil(n_mode / batch_size)
            _log_message_to_file(
                f"  Launching {num_blocks * num_blocks} tasks (blocks) with {num_workers} workers...",
                log_path,
            )

            with concurrent.futures.ProcessPoolExecutor(
                max_workers=num_workers,
                mp_context=multiprocessing.get_context("spawn"),
            ) as executor:
                futures = []

                # Submit tasks
                for i in range(num_blocks):
                    start_i = i * batch_size
                    end_i = min(start_i + batch_size, n_mode)
                    left_mat = matrix[start_i:end_i]
                    # Note: We must slice global_idxs here to avoid passing the whole array if possible,
                    # but here we pass slices which is fine.
                    l_idxs_slice = global_idxs[start_i:end_i]

                    for j in range(num_blocks):
                        start_j = j * batch_size
                        end_j = min(start_j + batch_size, n_mode)
                        right_mat = matrix[start_j:end_j]
                        r_idxs_slice = global_idxs[start_j:end_j]

                        chunk_path = temp_dir / f"pairs_{mode}_{i}_{j}.parquet"

                        futures.append(
                            executor.submit(
                                _process_proximate_block,
                                left_mat,
                                right_mat,
                                threshold,
                                l_idxs_slice,
                                r_idxs_slice,
                                chunk_path,
                                (i, j),
                            )
                        )

                # Process results as they complete
                mode_pairs_written = 0
                for future in concurrent.futures.as_completed(futures):
                    try:
                        n_pairs, (bi, bj) = future.result()
                        mode_pairs_written += n_pairs
                        _log_message_to_file(
                            f"    Block ({bi}, {bj}) finished. Written {n_pairs} pairs.",
                            log_path,
                        )
                    except Exception as exc:
                        _log_message_to_file(
                            f"    Task generated an exception: {exc}\n{traceback.format_exc()}",
                            log_path,
                            level=logging.ERROR,
                        )
                        raise exc

            total_proximate_pairs += mode_pairs_written
            _log_message_to_file(
                f"  Mode {mode}: Found {mode_pairs_written} proximate pairs", log_path
            )

            # Free memory for this mode
            del matrix
            del mode_df
            gc.collect()

        _log_message_to_file(
            f"Proximate stage done. Total pairs: {total_proximate_pairs}. Time: {perf_counter() - proximate_start:.3f}s",
            log_path,
        )

        if total_proximate_pairs == 0:
            _log_message_to_file(
                "No pairs found exceeding proximate threshold.", log_path
            )
            shutil.rmtree(temp_dir)
            return

        # --- Step 4: Streaming Exact Computation ---
        _log_message_to_file("Starting exact computation (streaming)...", log_path)
        exact_start = perf_counter()

        # Scan intermediate files
        lf_indices = pl.scan_parquet(str(temp_dir / "*.parquet"))

        # Prepare source lazyframe for joining
        lf_source = df_source.lazy()

        # Build the query
        # 1. Join left spectra
        joined = lf_indices.join(lf_source, on="idx")
        # 2. Join right spectra
        joined = joined.join(
            lf_source, left_on="idx_right", right_on="idx", suffix="_right"
        )

        # 3. Filter self-matches (same molecule)
        joined = joined.filter(pl.col("base_inchikey") != pl.col("base_inchikey_right"))

        # 4. Prepare structs for dotprod
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

        # 5. Compute dotprod and filter
        results = (
            joined.with_columns(
                dotprod_similarity=pl.col(
                    "spectra"
                ).spectral_similarity.dotprod_similarity(  # type: ignore
                    ms2_tolerance_in_ppm=ms2_tolerance_ppm,
                    clean_spectra_first=False,
                    ignore_precursor=True,
                )
            )
            .drop("spectra")
            .filter(
                pl.col("dotprod_similarity").is_not_null(),
                pl.col("dotprod_similarity").ge(threshold),
            )
            .select(
                "idx",
                "idx_right",
                "mol_idx",
                "mol_idx_right",
                "base_inchikey",
                "ion_mode",
                "base_inchikey_right",
                "smiles",
                "smiles_right",
                "dotprod_similarity",
                "spectral_information_score",
                "spectral_information_score_right",
            )
        )

        # 6. Sink to parquet (streaming)
        results.sink_parquet(output_path, maintain_order=False)

        _log_message_to_file(
            f"Exact computation and write complete in {perf_counter() - exact_start:.3f}s",
            log_path,
        )

    except Exception as e:
        _log_message_to_file(
            f"Error in build_and_write_pairs_parquet: {e}\n{traceback.format_exc()}",
            log_path,
            level=logging.ERROR,
        )
        raise
    finally:
        # Cleanup
        if "temp_dir" in locals() and temp_dir.exists():
            shutil.rmtree(temp_dir)
