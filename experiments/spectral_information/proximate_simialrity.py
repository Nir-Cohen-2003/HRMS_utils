import gc
import logging
import math
import threading
import traceback
from pathlib import Path
from time import perf_counter
from typing import List, Union

import numpy as np
import polars as pl
import pyarrow.parquet as pq


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


# Tracks which log files have been truncated for this process so we only truncate once.
_initialized_log_paths: set[str] = set()
_initialized_log_paths_lock = threading.Lock()

# Module logger (use debug level for timing info)
logger = logging.getLogger(__name__)


def _log_message_to_file(
    message: str,
    log_path: Union[str, Path],
    level: int = logging.INFO,
    overwrite: bool = False,
) -> None:
    """
    Log a single message to a file by attaching a temporary FileHandler to the module logger.
    The handler is removed and closed after logging so repeated calls do not accumulate handlers.

    If `overwrite` is True, the file is opened in write mode ('w') for this write;
    otherwise it is opened in append mode ('a'). This lets the caller truncate the
    log at the start of a run and append for subsequent progress updates.
    """
    logger = logging.getLogger(__name__)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # If overwrite=True, only truncate on the first write for that canonical path
    # during this process. Subsequent writes in the same process will append.
    # This prevents accidental repeated truncation if the same function is invoked
    # multiple times in one run.
    canonical = str(Path(log_path).resolve(strict=False))
    if overwrite:
        with _initialized_log_paths_lock:
            if canonical in _initialized_log_paths:
                do_truncate = False
            else:
                do_truncate = True
                _initialized_log_paths.add(canonical)
    else:
        do_truncate = False

    mode = "w" if do_truncate else "a"
    handler = logging.FileHandler(str(log_path), mode=mode)
    handler.setLevel(level)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)

    # Some environments set the logger level to WARNING by default, which can
    # filter out INFO/DEBUG messages. Temporarily set the logger level to DEBUG
    # so the message will be emitted, then restore the original level.
    prev_level = logger.level
    logger.setLevel(logging.DEBUG)

    logger.addHandler(handler)
    try:
        logger.log(level, message)
    finally:
        logger.removeHandler(handler)
        handler.close()
        logger.setLevel(prev_level)


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


def _filter_pairs_by_proximate_and_precise(
    batch_df: pl.DataFrame,
    source_df: pl.DataFrame,
    *,
    ms2_tolerance_in_ppm: float,
    threshold: float,
    upper_bound: int = 1000,
    intensity_power: float = 0.5,
) -> tuple[pl.DataFrame, dict]:
    """
    Given an in-memory `batch_df` (left) and the full `source_df` (right), compute a
    proximate (integer-binned) similarity for all left×right spectra grouped by `ion_mode`
    using a NumPy matrix multiply on CPU, then only send pairs whose
    proximate similarity >= threshold to the precise (plugin) dot-product calculation.

    Note: this implementation no longer applies a per-spectrum peak-overlap
    prefilter. Instead, it relies on the proximate (integer-binned) similarity
    matrix as the fast, first-pass filter to prune candidate pairs before the
    more expensive precise dot-product calculation.

    Returns:
      - tuple:
          - candidates (pl.DataFrame): filtered candidate pairs with the usual
            selected columns (includes `dotprod_similarity` and is filtered by `threshold`)
          - timings (dict): aggregated timings (floats, seconds) for this batch.
            The timings dict contains the following keys:
              - 'flatten_time': total time spent flattening list-columns to arrays
              - 'bin_time': total time spent binning flattened arrays
              - 'proximate_time': total time spent computing the proximate similarity matrix
              - 'precise_time': total time spent computing the precise dot-product similarity
              - 'write_time': time spent writing this batch to disk (0.0 if nothing was written)
              - 'n_peaks_total': total number of input peaks encountered in this batch
              - 'n_peaks_valid': number of peaks that passed bounds/intensity filters
              - 'n_candidates': number of candidate rows (height of `candidates`)
              - 'nbins': number of integer bins used (upper_bound + 1)

    The returned `candidates` DataFrame has the same selection/filtering semantics as
    before and is suitable for downstream precise scoring and pair writing.
    """
    candidate_frames: List[pl.DataFrame] = []
    # Per-batch timing accumulators (aggregated across ion modes)
    batch_timings: dict = {
        "flatten_time": 0.0,
        "bin_time": 0.0,
        "proximate_time": 0.0,
        "precise_time": 0.0,
        "n_peaks_total": 0,
        "n_peaks_valid": 0,
        "n_candidates": 0,
        # Extra diagnostics to explain "missing time" at the caller:
        # includes overhead for mode splitting, joins, unique, array conversions, etc.
        "n_left_rows": int(len(batch_df)),
        "n_right_rows": int(len(source_df)),
        "n_ion_modes": 0,
        # Diagnostics: number of proximate candidates produced before exact-stage filtering.
        "candidate_pairs_proximate": 0,
        # Batch overhead timings. Timings inside the proximate workload are intentionally
        # not broken down further (they are considered approximate).
        "mode_split_time": 0.0,
        "concat_unique_time": 0.0,
        "join_prepare_time": 0.0,
        "join_time": 0.0,
        "precise_prepare_time": 0.0,
        "precise_polars_time": 0.0,
    }

    # Iterate ion_modes present in the batch - comparisons only make sense within an ion mode.
    t_mode0 = perf_counter()
    ion_modes = set(batch_df.get_column("ion_mode").to_list())
    batch_timings["mode_split_time"] += perf_counter() - t_mode0
    batch_timings["n_ion_modes"] = int(len(ion_modes))
    for mode in ion_modes:
        t_mode_filter0 = perf_counter()
        left_sub = batch_df.filter(pl.col("ion_mode") == mode)
        right_sub = source_df.filter(pl.col("ion_mode") == mode)
        batch_timings["mode_split_time"] += perf_counter() - t_mode_filter0
        if len(left_sub) == 0 or len(right_sub) == 0:
            continue

        # Binning will be performed directly from the DataFrames using
        # `_bin_spectra_df_to_matrix` (no explicit flattening in Python).

        # Peak-overlap prefilter removed: rely on the proximate (integer-binned)
        # similarity matrix (approximate search) to prune candidates instead.

        # Build binned matrices (vectorized, directly from DataFrames) and accumulate timings
        left_mat, left_t = _bin_spectra_df_to_matrix(
            left_sub,
            "cleaned_normalized_mz",
            "cleaned_normalized_intensity",
            upper_bound,
            intensity_power,
        )
        right_mat, right_t = _bin_spectra_df_to_matrix(
            right_sub,
            "cleaned_normalized_mz",
            "cleaned_normalized_intensity",
            upper_bound,
            intensity_power,
        )

        batch_timings["flatten_time"] += left_t.get("flatten_time", 0.0) + right_t.get(
            "flatten_time", 0.0
        )
        batch_timings["bin_time"] += left_t.get("bin_time", 0.0) + right_t.get(
            "bin_time", 0.0
        )
        batch_timings["n_peaks_total"] += left_t.get("n_peaks_total", 0) + right_t.get(
            "n_peaks_total", 0
        )
        batch_timings["n_peaks_valid"] += left_t.get("n_peaks_valid", 0) + right_t.get(
            "n_peaks_valid", 0
        )

        if left_mat.size == 0 or right_mat.size == 0:
            continue

        # Proximate similarity compute + threshold pair extraction (NumPy/BLAS)
        #
        # Timing for this entire block is considered "approximate": we keep only the
        # end-to-end proximate timing and avoid fine-grained sub-timings.
        #
        # Enforce index dtype in Polars so `.to_numpy()` exports int64 without an extra cast/copy.
        left_sub = left_sub.with_columns(pl.col("idx").cast(pl.Int64))
        right_sub = right_sub.with_columns(pl.col("idx").cast(pl.Int64))

        left_global_idxs = left_sub.get_column("idx").to_numpy()
        right_global_idxs = right_sub.get_column("idx").to_numpy()

        assert left_global_idxs.dtype == np.int64, (
            f"expected left_global_idxs to be int64 from Polars, got {left_global_idxs.dtype}; "
            "ensure Polars column `idx` is Int64 before exporting to NumPy"
        )
        assert right_global_idxs.dtype == np.int64, (
            f"expected right_global_idxs to be int64 from Polars, got {right_global_idxs.dtype}; "
            "ensure Polars column `idx` is Int64 before exporting to NumPy"
        )

        assert left_mat.dtype == np.float32, (
            f"expected left_mat to be float32, got {left_mat.dtype}; "
            "ensure binning produces float32 (no post-casts before calling Numba)"
        )
        assert right_mat.dtype == np.float32, (
            f"expected right_mat to be float32, got {right_mat.dtype}; "
            "ensure binning produces float32 (no post-casts before calling Numba)"
        )
        assert left_mat.flags["C_CONTIGUOUS"], (
            "expected left_mat to be C-contiguous; ensure matrix construction uses a contiguous float32 buffer"
        )
        assert right_mat.flags["C_CONTIGUOUS"], (
            "expected right_mat to be C-contiguous; ensure matrix construction uses a contiguous float32 buffer"
        )

        t_prox = perf_counter()
        left_idxs, right_idxs, prox_sims = (
            _numpy_proximate_similarity_pairs_above_threshold(
                left_mat,
                right_mat,
                float(threshold),
                left_global_idxs,
                right_global_idxs,
            )
        )
        batch_timings["proximate_time"] += perf_counter() - t_prox

        batch_timings["candidate_pairs_proximate"] += int(prox_sims.size)
        if prox_sims.size == 0:
            continue

        cand_df = pl.DataFrame(
            {
                "idx": left_idxs,
                "idx_right": right_idxs,
                "proximate_similarity": prox_sims,
            }
        )
        candidate_frames.append(cand_df)

    if not candidate_frames:
        # Return an empty DataFrame with the expected schema (keeps downstream logic simple)
        result_df = pl.DataFrame(
            {
                "idx": pl.Series(dtype=pl.Int64),
                "idx_right": pl.Series(dtype=pl.Int64),
                "mol_idx": pl.Series(dtype=pl.Int64),
                "mol_idx_right": pl.Series(dtype=pl.Int64),
                "base_inchikey": pl.Series(dtype=pl.Utf8),
                "ion_mode": pl.Series(dtype=pl.Utf8),
                "base_inchikey_right": pl.Series(dtype=pl.Utf8),
                "smiles": pl.Series(dtype=pl.Utf8),
                "smiles_right": pl.Series(dtype=pl.Utf8),
                "dotprod_similarity": pl.Series(dtype=pl.Float32),
                "spectral_information_score": pl.Series(dtype=pl.Float32),
                "spectral_information_score_right": pl.Series(dtype=pl.Float32),
            }
        )
        return result_df, batch_timings

    t_concat0 = perf_counter()
    candidates_all = pl.concat(candidate_frames, how="vertical").unique()
    batch_timings["concat_unique_time"] += perf_counter() - t_concat0

    # Join candidate index pairs back to the full left/right rows to build structural columns for precise calculation.
    t_join_prep0 = perf_counter()
    # Why: force materialization of these references outside the join timer so we can see their separate contribution.
    _ = candidates_all.height
    _ = batch_df.height
    _ = source_df.height
    batch_timings["join_prepare_time"] += perf_counter() - t_join_prep0

    t_join0 = perf_counter()
    joined = candidates_all.join(batch_df, on="idx").join(
        source_df, left_on="idx_right", right_on="idx", suffix="_right"
    )

    # Why: self-comparisons belong to the exact stage policy, not the proximate stage.
    joined = joined.filter(pl.col("base_inchikey") != pl.col("base_inchikey_right"))
    batch_timings["join_time"] += perf_counter() - t_join0

    # Compute the precise NIST-like dot-product on only the filtered candidate rows.
    t_precise_prep0 = perf_counter()
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
    batch_timings["precise_prepare_time"] += perf_counter() - t_precise_prep0

    t_precise = perf_counter()
    joined = (
        joined.with_columns(
            dotprod_similarity=pl.col("spectra").spectral_similarity.dotprod_similarity(  # type: ignore[missing-attribute]
                ms2_tolerance_in_ppm=ms2_tolerance_in_ppm,
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
    batch_timings["precise_polars_time"] += perf_counter() - t_precise
    batch_timings["precise_time"] += (
        batch_timings["precise_prepare_time"] + batch_timings["precise_polars_time"]
    )
    batch_timings["n_candidates"] = joined.height
    return joined, batch_timings


def build_and_write_pairs_parquet(
    parquet_paths: List[Path],
    output_path: Union[str, Path],
    threshold: float = 0.8,
    num_spectra: int | None = None,
    ms2_tolerance_ppm: float = 10.0,
    batch_size: int = 1000,
    mass_range: tuple[float, float] | None = None,
    proximate_bin_upper: int = 1000,
) -> None:
    """
    Build unioned library LF, compute pairwise dot-product similarities (ignoring precursor),
    and write pairs with high similarity to parquet.

    Notes:
      - Adds a proximate (integer-binned) dot-product prefilter which bins all spectra
        into integer m/z bins (rounded, e.g. 4.78 -> 5) up to `proximate_bin_upper` (default 1000),
        computes the pairwise similarity via a NumPy matrix multiply on CPU,
        and only sends pairs with proximate similarity >= `threshold` to the precise dot-product
        (plugin) computation. This reduces the number of expensive precise computations.
    Args:
      - parquet_paths: list of Path objects pointing at library parquet files
      - output_path: where to write the pairs with similarities (required)
      - threshold: float (default 0.8). Only pairs with dotprod_similarity >= threshold are saved.
      - num_spectra: Optional[int]. If provided, limit the number of molecules read from the
        unioned input using a lazy .limit(num_spectra) to avoid collecting the full dataset.
      - The function operates in tiled batch mode: the library is processed in blocks
        of shape `batch_size × batch_size` (left × right). The streaming-only path
        has been removed; tiled batching is always used.
      - mass_range: Optional[tuple[float, float]] (default None). If provided, spectra will be
        filtered per-input-parquet by `precursor_mz` such that only spectra with
        `min <= precursor_mz <= max` are retained. This filtering is applied before the union
        of input libraries (i.e. before pairwise computation).
      - proximate_bin_upper: int (default 1000). Upper bound for integer binning in the proximate prefilter.
    Returns:
      - None (writes parquet to output_path)
    """
    # Ensure we create the log file early so it exists even on early failures
    output_path = Path(output_path)
    log_path = output_path.with_suffix(".log")
    _log_message_to_file(
        f"Started build_and_write_pairs_parquet: output={str(output_path)} threshold={threshold} num_spectra={num_spectra} parquet_paths={parquet_paths} tiled_batching=True mass_range={mass_range}",
        log_path,
        overwrite=True,
    )

    try:
        assert len(parquet_paths) > 0, "parquet_paths must contain at least one path"
        # Load and union into a single lazyframe
        lf_list = []
        for PARQUET_PATH in parquet_paths:
            assert Path(PARQUET_PATH).exists(), (
                f"Requested parquet does not exist: {PARQUET_PATH}"
            )
            _log_message_to_file(f"Found parquet: {PARQUET_PATH}", log_path)

            # Read lazily, then optionally apply per-file mass filtering if requested.
            lf = pl.scan_parquet(PARQUET_PATH)
            if mass_range is not None:
                # Validate mass_range shape and types
                assert isinstance(mass_range, (tuple, list)) and len(mass_range) == 2, (
                    "mass_range must be a tuple/list of (min, max) or None"
                )
                min_mz, max_mz = mass_range
                assert (
                    isinstance(min_mz, (int, float))
                    and isinstance(max_mz, (int, float))
                    and min_mz <= max_mz
                ), (
                    f"mass_range elements must be numbers with min <= max, got {mass_range}"
                )

                # Fail fast if the parquet doesn't contain precursor_mz when a mass_range is requested.
                try:
                    pfp = pq.ParquetFile(PARQUET_PATH)
                    assert "precursor_mz" in pfp.schema.names, (
                        f"Parquet file {PARQUET_PATH} does not contain 'precursor_mz' required for mass_range filtering"
                    )
                except Exception as e:
                    raise AssertionError(
                        f"Could not validate parquet schema for {PARQUET_PATH}: {e}"
                    )

                _log_message_to_file(
                    f"Applying mass_range filter to {PARQUET_PATH}: precursor_mz between {min_mz} and {max_mz}",
                    log_path,
                )
                # Use polars is_between as requested (inclusive bounds).
                lf = lf.filter(
                    pl.col("precursor_mz").is_between(float(min_mz), float(max_mz))
                )

            lf_list.append(lf)

        # Keep only precursors that passed cleaning
        lf = pl.union(lf_list).filter(pl.col("clean_precursor"))

        # Optionally limit the number of molecules sampled from the unioned libraries.
        if num_spectra is not None:
            assert isinstance(num_spectra, int) and num_spectra > 0, (
                "num_spectra must be a positive integer or None"
            )
            lf = lf.limit(num_spectra)

        # Keep only necessary columns; add idx and nominal_mass to join on
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
            #     .with_columns(
            #         spectral_entropy=(
            #             pl.col("cleaned_normalized_intensity")
            #             / pl.col("cleaned_normalized_intensity").list.sum()
            #         )
            #         .list.eval(pl.element().log(base=math.e).mul(pl.element()))
            #         .list.sum()
            #         .neg(),
            #         num_clean_peaks=pl.col("cleaned_normalized_mz").list.len(),
            #         normalized_spectral_information_score=(
            #             # here we normalize the SIS per molecule+Ion mode, so its a fraction of the max possible SIS for that molecule
            #             pl.col("spectral_information_score").truediv(
            #                 pl.col("spectral_information_score")
            #                 .mean()
            #                 .over(["base_inchikey", "ion_mode"])
            #             )
            #         ),
            #     )
            #     .with_columns(
            #         most_informative=pl.col("normalized_spectral_information_score").eq(
            #             1.0
            #         ),
            #         normalized_spectral_entropy=pl.col("spectral_entropy").truediv(
            #             pl.col("spectral_entropy")
            #             .mean()
            #             .over(["base_inchikey", "ion_mode"])
            #         ),
            #         normalized_num_clean_peaks=pl.col("num_clean_peaks").truediv(
            #             pl.col("num_clean_peaks").mean().over(["base_inchikey", "ion_mode"])
            #         ),
            #     )
            .collect()
            .rechunk()
            .lazy()
        )

        start = perf_counter()

        # Always use tiled batching: process the library in blocks of size
        # `batch_size × batch_size` (left × right). The streaming path was removed.
        _log_message_to_file(
            "Materializing source library for tiled batch processing...", log_path
        )
        df_source = lf.collect()
        n_source = len(df_source)
        _log_message_to_file(f"Source library size: {n_source}", log_path)

        _log_message_to_file(
            f"Starting tiled pairwise similarity computation (batch_size={batch_size})",
            log_path,
        )

        writer = None
        total_written = 0
        total_timings = {
            "flatten_time": 0.0,
            "bin_time": 0.0,
            "proximate_time": 0.0,
            "precise_time": 0.0,
            "write_time": 0.0,
            "pairs_written": 0,
            "blocks_processed": 0,
        }

        try:
            num_blocks = max(1, math.ceil(n_source / batch_size))
            total_blocks = num_blocks * num_blocks
            flat_idx = 0
            for i_block in range(num_blocks):
                left_start = i_block * batch_size
                left_df = df_source.slice(left_start, batch_size)
                for j_block in range(num_blocks):
                    flat_idx += 1
                    right_start = j_block * batch_size
                    right_df = df_source.slice(right_start, batch_size)

                    block_t0 = perf_counter()
                    try:
                        pairs_block, batch_timings = (
                            _filter_pairs_by_proximate_and_precise(
                                left_df,
                                right_df,
                                ms2_tolerance_in_ppm=ms2_tolerance_ppm,
                                threshold=threshold,
                                upper_bound=proximate_bin_upper,
                                intensity_power=0.5,
                            )
                        )
                    except Exception:
                        _log_message_to_file(
                            f"Exception during tiled build loop (block {i_block},{j_block}):\n{traceback.format_exc()}",
                            log_path,
                            level=logging.ERROR,
                        )
                        raise

                    # Write results if any and capture write time
                    # Also profile the individual overhead components (Arrow conversion, writer init, GC).
                    if len(pairs_block) > 0:
                        batch_pairs_written = len(pairs_block)

                        t_arrow0 = perf_counter()
                        table = pairs_block.to_arrow()
                        arrow_dur = perf_counter() - t_arrow0
                        batch_timings["arrow_time"] = arrow_dur

                        writer_init_dur = 0.0
                        if writer is None:
                            t_writer0 = perf_counter()
                            writer = pq.ParquetWriter(output_path, table.schema)
                            writer_init_dur = perf_counter() - t_writer0
                        batch_timings["writer_init_time"] = writer_init_dur

                        t_write = perf_counter()
                        writer.write_table(table)
                        write_dur = perf_counter() - t_write
                        batch_timings["write_time"] = write_dur
                        total_timings["write_time"] += write_dur

                        total_written += batch_pairs_written
                        total_timings["pairs_written"] += batch_pairs_written

                        t_gc0 = perf_counter()
                        del pairs_block
                        del table
                        gc.collect()
                        batch_timings["gc_time"] = perf_counter() - t_gc0
                    else:
                        batch_pairs_written = 0
                        # Ensure write/overhead timing keys are present in batch_timings for consistency
                        batch_timings.setdefault("arrow_time", 0.0)
                        batch_timings.setdefault("writer_init_time", 0.0)
                        batch_timings.setdefault("write_time", 0.0)
                        batch_timings.setdefault("gc_time", 0.0)

                    # Accumulate totals and emit a single batch-level summary
                    #
                    # Why your subtotals don't add up:
                    # `_filter_pairs_by_proximate_and_precise()` times many internal steps now,
                    # but the outer loop still has overhead (slicing, Arrow/writer init/GC, and Python glue).
                    # We log a detailed, per-block breakdown so you can decide what to optimize next.
                    total_timings["flatten_time"] += batch_timings.get(
                        "flatten_time", 0.0
                    )
                    total_timings["bin_time"] += batch_timings.get("bin_time", 0.0)
                    total_timings["proximate_time"] += batch_timings.get(
                        "proximate_time", 0.0
                    )
                    total_timings["precise_time"] += batch_timings.get(
                        "precise_time", 0.0
                    )
                    total_timings["blocks_processed"] += 1

                    batch_summary = (
                        f"Batch {flat_idx}/{total_blocks} summary: "
                        f"flatten={batch_timings.get('flatten_time', 0):.4f}s "
                        f"bin={batch_timings.get('bin_time', 0):.4f}s "
                        f"proximate={batch_timings.get('proximate_time', 0):.4f}s "
                        f"precise={batch_timings.get('precise_time', 0):.4f}s "
                        f"write={batch_timings.get('write_time', 0):.4f}s "
                        f"pairs={batch_pairs_written}"
                    )
                    _log_message_to_file(batch_summary, log_path)

                    block_t1 = perf_counter()
                    block_total_time = block_t1 - block_t0
                    accounted = (
                        float(batch_timings.get("flatten_time", 0.0))
                        + float(batch_timings.get("bin_time", 0.0))
                        + float(batch_timings.get("proximate_time", 0.0))
                        + float(batch_timings.get("precise_time", 0.0))
                        + float(batch_timings.get("write_time", 0.0))
                    )
                    other_overhead = block_total_time - accounted

                    # Provide a detailed breakdown so "missing time" is attributable to specific steps.
                    _log_message_to_file(
                        (
                            f"Block {flat_idx}/{total_blocks} breakdown: "
                            f"total={block_total_time:.4f}s "
                            f"accounted={accounted:.4f}s "
                            f"other_overhead={other_overhead:.4f}s | "
                            f"core(flatten={batch_timings.get('flatten_time', 0):.4f}s "
                            f"bin={batch_timings.get('bin_time', 0):.4f}s "
                            f"proximate={batch_timings.get('proximate_time', 0):.4f}s "
                            f"precise_total={batch_timings.get('precise_time', 0):.4f}s "
                            f"write={batch_timings.get('write_time', 0):.4f}s) | "
                            f"overhead(mode_split={batch_timings.get('mode_split_time', 0):.4f}s "
                            f"concat_unique={batch_timings.get('concat_unique_time', 0):.4f}s "
                            f"join_prep={batch_timings.get('join_prepare_time', 0):.4f}s "
                            f"join={batch_timings.get('join_time', 0):.4f}s "
                            f"precise_prep={batch_timings.get('precise_prepare_time', 0):.4f}s "
                            f"precise_polars={batch_timings.get('precise_polars_time', 0):.4f}s "
                            f"arrow={batch_timings.get('arrow_time', 0):.4f}s "
                            f"writer_init={batch_timings.get('writer_init_time', 0):.4f}s "
                            f"gc={batch_timings.get('gc_time', 0):.4f}s) | "
                            f"stats(ion_modes={batch_timings.get('n_ion_modes', 0)} "
                            f"left_rows={batch_timings.get('n_left_rows', 0)} "
                            f"right_rows={batch_timings.get('n_right_rows', 0)} "
                            f"candidates={batch_timings.get('n_candidates', 0)} "
                            f"pairs_written={batch_pairs_written} "
                            f"peaks_total={batch_timings.get('n_peaks_total', 0)} "
                            f"peaks_valid={batch_timings.get('n_peaks_valid', 0)})"
                        ),
                        log_path,
                    )

                    _log_message_to_file(
                        f"Processed block {flat_idx}/{total_blocks} (left_block={i_block}, right_block={j_block}). "
                        f"Written {total_written} pairs so far. "
                        f"Time for block: {block_total_time:.3f}s",
                        log_path,
                    )

        except Exception:
            _log_message_to_file(
                f"Exception during tiled build_and_write_pairs_parquet loop:\n{traceback.format_exc()}",
                log_path,
                level=logging.ERROR,
            )
            raise
        finally:
            if writer:
                writer.close()

        end = perf_counter()
        _log_message_to_file(
            f"Wrote results of library search to file {str(output_path)} in time {end - start:.3f}s.",
            log_path,
        )

        # Emit a single, final total summary for the run (aggregated across batches)
        summary_msg = (
            f"TOTAL SUMMARY: blocks={total_timings['blocks_processed']} "
            f"pairs_written={total_timings['pairs_written']} "
            f"total_time={end - start:.3f}s "
            f"flatten={total_timings['flatten_time']:.3f}s "
            f"bin={total_timings['bin_time']:.3f}s "
            f"proximate={total_timings['proximate_time']:.3f}s "
            f"precise={total_timings['precise_time']:.3f}s "
            f"write={total_timings['write_time']:.3f}s"
        )
        _log_message_to_file(summary_msg, log_path)

    except Exception:
        _log_message_to_file(
            f"Exception during build_and_write_pairs_parquet:\n{traceback.format_exc()}",
            log_path,
            level=logging.ERROR,
        )
        raise
    return None
