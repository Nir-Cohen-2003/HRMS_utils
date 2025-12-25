import ctypes.util
import gc
import logging
import math
import os
import shutil
import subprocess
import threading
import traceback
from glob import glob as _glob
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, List, Union

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from nvmolkit.fingerprints import MorganFingerprintGenerator
from nvmolkit.similarity import crossTanimotoSimilarityMemoryConstrained
from pyarrow import ChunkedArray, Table
from rdkit import Chem

from hrms_utils.rdkit import sanitize_smiles

if TYPE_CHECKING:
    # Type-only imports so static analyzers know the JAX symbols and attributes.
    # These are only executed by type checkers and won't trigger imports at runtime.
    import jax  # type: ignore
    import jax.numpy as jnp  # type: ignore

# Define these at module scope so static analysis tools know the names exist even
# when JAX is not imported/used in this environment. They will be assigned real
# module objects when a GPU is detected and JAX is successfully imported.
jax = None  # type: ignore
jnp = None  # type: ignore


def _nvidia_gpu_available() -> bool:
    """
    Heuristic for detecting an NVIDIA GPU in the runtime environment.

    Why: we only want to import and configure JAX when an NVIDIA GPU is available.
    Importing JAX on systems where only TPU libraries are partially present can
    result in noisy INFO logs like "Unable to initialize backend 'tpu': Failed to
    open libtpu.so" even though the user has no TPU. By detecting a GPU first
    and setting `JAX_PLATFORM_NAME` before importing JAX we avoid those logs and
    ensure JAX will try to use the 'gpu' backend.
    """
    # If CUDA_VISIBLE_DEVICES is explicitly set to an empty string, treat that
    # as "no GPU available".
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip() == "":
        return False

    # If nvidia-smi exists and reports devices, that's a reliable indicator.
    nvsmi = shutil.which("nvidia-smi")
    if nvsmi:
        try:
            out = subprocess.run(
                [nvsmi, "-L"], capture_output=True, text=True, check=False
            )
            if out.returncode == 0 and out.stdout.strip() != "":
                return True
        except Exception:
            # Best-effort; fall through to other checks.
            pass

    # Check for common CUDA libraries via ctypes
    for libname in ("cuda", "cudart"):
        try:
            if ctypes.util.find_library(libname):
                return True
        except Exception:
            pass

    # Last resort: check common CUDA library locations.
    if os.path.exists("/usr/local/cuda/lib64/libcudart.so") or os.path.exists(
        "/usr/lib/x86_64-linux-gnu/libcuda.so"
    ):
        return True

    return False


jax_is_available = False
if _nvidia_gpu_available():
    # Force JAX to use the GPU backend. Must be set before importing JAX so
    # other backends (TPU) aren't probed/initialized and produce noisy logs.
    os.environ["JAX_PLATFORM_NAME"] = "gpu"
    try:
        import jax.numpy as jnp  # noqa: F401

        # Reinforce backend selection if JAX was previously imported.
        try:
            from jax import config as jax_config

            jax_config.update("jax_platform_name", "gpu")
        except Exception:
            # Non-fatal: if config update fails, JAX may already be configured.
            pass

        jax_is_available = True
        logging.getLogger(__name__).info(
            "JAX imported and configured to use the 'gpu' backend."
        )
    except Exception:
        # If JAX fails to import even when GPU heuristics succeeded, fall back.
        logging.getLogger(__name__).info(
            "JAX import failed despite NVIDIA GPU detection; falling back to NumPy."
        )
        jax_is_available = False
else:
    logging.getLogger(__name__).info(
        "No NVIDIA GPU detected; JAX will not be used (falling back to NumPy)."
    )
    jax_is_available = False
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


def _cross_join_and_score(
    newer: pl.LazyFrame,
    older: pl.LazyFrame,
    ms2_tolerance_in_ppm: float,
    threshold: float,
    require_peak_overlap: bool = False,
) -> pl.LazyFrame:
    """
    Centralize the join/crossing operation used by pairwise scoring.

    This function uses the definition of the first LazyFrame (`newer`) as the
    drive side of the join so the join result is deterministic. It returns a
    LazyFrame (no collection) with dot-product similarity computed and filtered
    by `threshold`. If `require_peak_overlap` is True, an additional pre-filter
    requiring at least one rounded m/z overlap is applied to reduce downstream
    compute for obvious non-matching pairs.
    """
    # Deterministic join on ion_mode; the suffix ensures columns from `older` are suffixed.
    joined = newer.join(other=older, on="ion_mode", suffix="_right")

    # Base filter: avoid self-comparisons by different inchikeys.
    filters = [pl.col("base_inchikey") != pl.col("base_inchikey_right")]

    if require_peak_overlap:
        filters.append(
            pl.col("cleaned_normalized_mz")
            .list.eval(pl.element().round(2))
            .list.set_intersection(
                pl.col("cleaned_normalized_mz_right").list.eval(pl.element().round(2))
            )
            .list.len()
            .ge(1)
        )

    lf = (
        joined.filter(*filters)
        .with_columns(
            spectra=pl.struct(
                mz1=pl.col("cleaned_normalized_mz").alias("mz1"),
                intensities1=pl.col("cleaned_normalized_intensity").alias(
                    "intensities1"
                ),
                mz2=pl.col("cleaned_normalized_mz_right").alias("mz2"),
                intensities2=pl.col("cleaned_normalized_intensity_right").alias(
                    "intensities2"
                ),
                precursor_mz1=pl.col("precursor_mz").alias("precursor_mz1"),
                precursor_mz2=pl.col("precursor_mz_right").alias("precursor_mz2"),
            )
        )
        .drop(
            [
                "cleaned_normalized_mz",
                "cleaned_normalized_intensity",
                "cleaned_normalized_mz_right",
                "cleaned_normalized_intensity_right",
                "precursor_mz",
                "precursor_mz_right",
            ]
        )
        .with_columns(
            dotprod_similarity=pl.col("spectra").spectral_similarity.dotprod_similarity(
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

    return lf


def _bin_spectra_to_matrix(
    mzs_list: List[list],
    ints_list: List[list],
    upper_bound: int = 1000,
    intensity_power: float = 0.5,
) -> np.ndarray:
    """
    Bin lists of spectra into a dense 2D matrix of shape (n_spectra, nbins).

    Implementation notes (why this approach):
      - Vectorized: flatten all spectra into 1D arrays, compute integer bin indices with
        rounding and build a single 1D `np.bincount` where the flattened key is
        `spec_idx * nbins + mass_bin`. This avoids Python loops over spectra while
        being memory efficient and fast for typical batch sizes.
      - We apply `intensity_power` (default 0.5 to match NIST-like dot-product weighting)
        prior to accumulation, so the binned matrix entries correspond to the weighted
        intensities used in similarity calculation.

    Returns:
      - np.ndarray(dtype=np.float32) shape (n_spectra, nbins)
    """
    nbins = int(upper_bound) + 1
    n_spec = len(mzs_list)
    if n_spec == 0:
        return np.zeros((0, nbins), dtype=np.float32)

    # Prepare lengths, handling None or empty lists gracefully
    lengths = [len(m) if m is not None else 0 for m in mzs_list]
    total_peaks = sum(lengths)
    if total_peaks == 0:
        return np.zeros((n_spec, nbins), dtype=np.float32)

    # Flatten arrays (fast, vectorized)
    mzs_arrs = [
        np.asarray(m, dtype=np.float64)
        if (m is not None and len(m) > 0)
        else np.asarray([], dtype=np.float64)
        for m in mzs_list
    ]
    ints_arrs = [
        np.asarray(i, dtype=np.float32)
        if (i is not None and len(i) > 0)
        else np.asarray([], dtype=np.float32)
        for i in ints_list
    ]

    flat_mzs = np.concatenate(mzs_arrs)
    flat_ints = np.concatenate(ints_arrs)
    spec_idx = np.repeat(np.arange(n_spec, dtype=np.int64), lengths)

    # Bin to nearest integer mass and apply bounds filter
    mass_bins = np.rint(flat_mzs).astype(np.int64)
    valid_mask = (mass_bins >= 0) & (mass_bins <= upper_bound) & (flat_ints > 0)
    if not np.any(valid_mask):
        return np.zeros((n_spec, nbins), dtype=np.float32)

    mass_bins = mass_bins[valid_mask]
    spec_idx = spec_idx[valid_mask]
    weights = np.asarray(flat_ints[valid_mask], dtype=np.float32) ** float(
        intensity_power
    )

    # Build 1D keys and use bincount for accumulation
    flat_keys = spec_idx * nbins + mass_bins
    accum = np.bincount(flat_keys, weights=weights, minlength=n_spec * nbins).astype(
        np.float32
    )
    matrix = accum.reshape((n_spec, nbins))
    return matrix


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
    """
    n_spec = len(df)
    if n_spec == 0:
        return (
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float32),
            np.asarray([], dtype=np.int64),
            0,
        )

    # Start timing the flattening operation.
    t0 = perf_counter()

    # Add a row index so we can recover which spectrum each flattened peak
    # belongs to (this mirrors the spec_idx construction from list lengths).
    df_idx = df.with_row_count("__spec_idx")

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

    # Export columns to NumPy arrays for fast, vectorized processing downstream.
    flat_mzs = np.asarray(exploded.get_column(mz_col).to_list(), dtype=np.float64)
    flat_ints = np.asarray(exploded.get_column(int_col).to_list(), dtype=np.float32)
    spec_idx = np.asarray(exploded.get_column("__spec_idx").to_list(), dtype=np.int64)

    duration = perf_counter() - t0
    logger.debug(
        "Flattened %d spectra into %d peaks in %.4f s",
        n_spec,
        int(flat_mzs.size),
        duration,
    )

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
    """
    nbins = int(upper_bound) + 1
    if n_spec == 0:
        return np.zeros((0, nbins), dtype=np.float32)
    if flat_mzs.size == 0 or flat_ints.size == 0:
        return np.zeros((n_spec, nbins), dtype=np.float32)

    t0 = perf_counter()

    # Bin to nearest integer mass and apply bounds filter
    mass_bins = np.rint(flat_mzs).astype(np.int64)
    valid_mask = (mass_bins >= 0) & (mass_bins <= upper_bound) & (flat_ints > 0)
    n_peaks_total = int(flat_mzs.size)
    n_peaks_valid = int(np.count_nonzero(valid_mask))
    if not np.any(valid_mask):
        duration = perf_counter() - t0
        logger.debug(
            "Binning aborted: no valid peaks after filtering (n_spec=%d, nbins=%d) in %.4f s",
            n_spec,
            nbins,
            duration,
        )
        return np.zeros((n_spec, nbins), dtype=np.float32)

    mass_bins = mass_bins[valid_mask]
    spec_idx = spec_idx[valid_mask].astype(np.int64)
    weights = np.asarray(flat_ints[valid_mask], dtype=np.float32) ** float(
        intensity_power
    )

    # Build 1D keys and use bincount for accumulation
    flat_keys = spec_idx * nbins + mass_bins
    accum = np.bincount(flat_keys, weights=weights, minlength=n_spec * nbins).astype(
        np.float32
    )
    matrix = accum.reshape((n_spec, nbins))
    duration = perf_counter() - t0
    logger.debug(
        "Binned %d total peaks (%d valid) into matrix %dx%d in %.4f s",
        n_peaks_total,
        n_peaks_valid,
        n_spec,
        nbins,
        duration,
    )
    return matrix


def _bin_spectra_df_to_matrix(
    df: pl.DataFrame,
    mz_col: str = "cleaned_normalized_mz",
    int_col: str = "cleaned_normalized_intensity",
    upper_bound: int = 1000,
    intensity_power: float = 0.5,
) -> np.ndarray:
    """
    Explode list-valued spectra columns in `df` and bin them into a dense matrix
    of shape (n_spectra, nbins) in a single, vectorized call.

    This function delegates to the lower-level helpers `_flatten_spectra_to_numpy`
    and `_bin_flat_spectra_to_matrix` to avoid duplicating the flatten/bin logic.
    """
    flat_mzs, flat_ints, spec_idx, n_spec = _flatten_spectra_to_numpy(
        df, mz_col, int_col
    )
    return _bin_flat_spectra_to_matrix(
        flat_mzs, flat_ints, spec_idx, n_spec, upper_bound, intensity_power
    )


def _compute_pairwise_proximate_similarity(
    left_mat: np.ndarray,
    right_mat: np.ndarray,
    use_jax_if_available: bool = True,
) -> np.ndarray:
    """
    Compute pairwise cosine-like similarity between two binned matrices:
      sim(i,j) = (left[i] · right[j]) / (||left[i]|| * ||right[j]||)

    Prefer JAX (GPU) if available; fall back to NumPy on CPU otherwise.
    Returns a NumPy array of dtype float32.
    """
    # Quick exits
    if left_mat.size == 0 or right_mat.size == 0:
        return np.zeros((left_mat.shape[0], right_mat.shape[0]), dtype=np.float32)

    t0 = perf_counter()
    if use_jax_if_available and jax_is_available:
        # Import jax.numpy locally to ensure the name is bound in this scope.
        # Doing the import here avoids importing JAX at module import time on
        # systems without GPUs and also satisfies static analyzers that may not
        # be able to determine that `jnp` is assigned at module import time.
        import jax.numpy as jnp  # type: ignore

        # Move arrays to jax (device placement handled by jax)
        L = jnp.asarray(left_mat, dtype=jnp.float32)
        R = jnp.asarray(right_mat, dtype=jnp.float32)

        # Normalize each spectrum (row) first so the dot product is the cosine.
        lnorm = jnp.linalg.norm(L, axis=1, keepdims=True)
        rnorm = jnp.linalg.norm(R, axis=1, keepdims=True)
        lnorm_safe = jnp.where(lnorm > 0.0, lnorm, 1.0)
        rnorm_safe = jnp.where(rnorm > 0.0, rnorm, 1.0)
        Ln = L / lnorm_safe
        Rn = R / rnorm_safe

        sim = (Ln @ Rn.T).astype(jnp.float32)
        duration = perf_counter() - t0
        logger.debug(
            "Proximate similarity (JAX) computed for %dx%d matrices in %.4f s",
            left_mat.shape[0],
            right_mat.shape[0],
            duration,
        )
        return np.asarray(sim)
    else:
        logger.debug("JAX not available; using NumPy for proximate similarity.")
        # NumPy fallback: normalize rows first, then dot-product
        lnorm = np.linalg.norm(left_mat, axis=1, keepdims=True)
        rnorm = np.linalg.norm(right_mat, axis=1, keepdims=True)
        lnorm[lnorm == 0] = 1.0
        rnorm[rnorm == 0] = 1.0
        Ln = left_mat / lnorm
        Rn = right_mat / rnorm
        sim = (Ln @ Rn.T).astype(np.float32)
        duration = perf_counter() - t0
        logger.debug(
            "Proximate similarity (NumPy) computed for %dx%d matrices in %.4f s",
            left_mat.shape[0],
            right_mat.shape[0],
            duration,
        )
        return sim


def _filter_pairs_by_proximate_and_precise(
    batch_df: pl.DataFrame,
    source_df: pl.DataFrame,
    *,
    ms2_tolerance_in_ppm: float,
    threshold: float,
    use_gpu: bool,
    upper_bound: int = 1000,
    intensity_power: float = 0.5,
) -> pl.DataFrame:
    """
    Given an in-memory `batch_df` (left) and the full `source_df` (right), compute a
    proximate (integer-binned) similarity for all left×right spectra grouped by `ion_mode`
    using a matrix multiply (JAX on GPU if available), then only send pairs whose
    proximate similarity >= threshold to the precise (plugin) dot-product calculation.

    Note: this implementation no longer applies a per-spectrum peak-overlap
    prefilter. Instead, it relies on the proximate (integer-binned) similarity
    matrix as the fast, first-pass filter to prune candidate pairs before the
    more expensive precise dot-product calculation.

    Returns a Polars DataFrame with the same column selection and filtering as the
    original pipeline (i.e. it includes `dotprod_similarity` and is filtered by `threshold`).
    """
    candidate_frames: List[pl.DataFrame] = []

    # Iterate ion_modes present in the batch - comparisons only make sense within an ion mode.
    ion_modes = set(batch_df.get_column("ion_mode").to_list())
    for mode in ion_modes:
        left_sub = batch_df.filter(pl.col("ion_mode") == mode)
        right_sub = source_df.filter(pl.col("ion_mode") == mode)
        if len(left_sub) == 0 or len(right_sub) == 0:
            continue

        # Binning will be performed directly from the DataFrames using
        # `_bin_spectra_df_to_matrix` (no explicit flattening in Python).

        # Peak-overlap prefilter removed: rely on the proximate (integer-binned)
        # similarity matrix (approximate search) to prune candidates instead.

        # Build binned matrices (vectorized, directly from DataFrames)
        left_mat = _bin_spectra_df_to_matrix(
            left_sub,
            "cleaned_normalized_mz",
            "cleaned_normalized_intensity",
            upper_bound,
            intensity_power,
        )
        right_mat = _bin_spectra_df_to_matrix(
            right_sub,
            "cleaned_normalized_mz",
            "cleaned_normalized_intensity",
            upper_bound,
            intensity_power,
        )

        if left_mat.size == 0 or right_mat.size == 0:
            continue

        # Compute proximate similarities (matrix multiply; prefer JAX on GPU)
        sim_matrix = _compute_pairwise_proximate_similarity(
            left_mat,
            right_mat,
            use_jax_if_available=use_gpu,
        )

        # Find candidate pairs exceeding proximate threshold
        li, ri = np.where(sim_matrix >= float(threshold))
        if li.size == 0:
            continue

        # Apply base_inchikey filter (avoid self-comparisons)
        left_keys = left_sub.get_column("base_inchikey").to_list()
        right_keys = right_sub.get_column("base_inchikey").to_list()
        keep_mask = [left_keys[i] != right_keys[j] for i, j in zip(li, ri)]
        if not any(keep_mask):
            continue
        li = np.asarray(li)[keep_mask]
        ri = np.asarray(ri)[keep_mask]
        prox_sims = sim_matrix[li, ri].astype(np.float32)

        # Map back to global idx values (these are the idx columns we use downstream)
        left_idxs = np.asarray(left_sub.get_column("idx").to_list(), dtype=np.int64)[li]
        right_idxs = np.asarray(right_sub.get_column("idx").to_list(), dtype=np.int64)[
            ri
        ]

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
        return pl.DataFrame(
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

    candidates_all = pl.concat(candidate_frames, how="vertical").unique()

    # Join candidate index pairs back to the full left/right rows to build structural columns for precise calculation.
    joined = candidates_all.join(batch_df, on="idx").join(
        source_df, left_on="idx_right", right_on="idx", suffix="_right"
    )

    # Compute the precise NIST-like dot-product on only the filtered candidate rows.
    t_precise = perf_counter()
    joined = (
        joined.with_columns(
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
            dotprod_similarity=pl.col("spectra").spectral_similarity.dotprod_similarity(
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
    duration_precise = perf_counter() - t_precise
    logger.debug(
        "Precise dot-product computed on %d candidate rows in %.4f s",
        joined.height,
        duration_precise,
    )

    return joined


def build_and_write_pairs_parquet(
    parquet_paths: List[Path],
    output_path: Union[str, Path],
    threshold: float = 0.8,
    num_spectra: int | None = None,
    ms2_tolerance_ppm: float = 10.0,
    batch_size: int = 1000,
    mass_range: tuple[float, float] | None = None,
    proximate_bin_upper: int = 1000,
    use_gpu: bool = True,
) -> None:
    """
    Build unioned library LF, compute pairwise dot-product similarities (ignoring precursor),
    and write pairs with high similarity to parquet.

    Notes:
      - Adds a GPU-assisted proximate (integer-binned) dot-product prefilter which bins all spectra
        into integer m/z bins (rounded, e.g. 4.78 -> 5) up to `proximate_bin_upper` (default 1000),
        computes the pairwise similarity via a matrix multiply (uses JAX on GPU if available),
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
                        pairs_block = _filter_pairs_by_proximate_and_precise(
                            left_df,
                            right_df,
                            ms2_tolerance_in_ppm=ms2_tolerance_ppm,
                            threshold=threshold,
                            upper_bound=proximate_bin_upper,
                            intensity_power=0.5,
                            use_gpu=use_gpu,
                        )
                    except Exception:
                        _log_message_to_file(
                            f"Exception during tiled build loop (block {i_block},{j_block}):\n{traceback.format_exc()}",
                            log_path,
                            level=logging.ERROR,
                        )
                        raise

                    if len(pairs_block) > 0:
                        table = pairs_block.to_arrow()
                        if writer is None:
                            writer = pq.ParquetWriter(output_path, table.schema)
                        t_write = perf_counter()
                        writer.write_table(table)
                        write_dur = perf_counter() - t_write
                        logger.debug(
                            "Wrote %d rows to %s in %.4f s",
                            len(pairs_block),
                            output_path,
                            write_dur,
                        )
                        total_written += len(pairs_block)
                        del pairs_block
                        del table
                        gc.collect()

                    block_t1 = perf_counter()
                    # Log progress periodically (1 out of every 10 flat blocks)

                    _log_message_to_file(
                        f"Processed block {flat_idx}/{total_blocks} (left_block={i_block}, right_block={j_block}). "
                        f"Written {total_written} pairs so far. "
                        f"Time for block: {block_t1 - block_t0:.3f}s",
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

    except Exception:
        _log_message_to_file(
            f"Exception during build_and_write_pairs_parquet:\n{traceback.format_exc()}",
            log_path,
            level=logging.ERROR,
        )
        raise
    return None


def process_batch_tanimoto(
    df: pl.DataFrame, fp_radius: int = 2, fp_size: int = 2048
) -> pl.DataFrame:
    """Process one batch (a polars DataFrame) of pairs and compute Tanimoto similarity.

    Notes:
      - This implementation canonicalizes SMILES using `sanitize_smiles` and builds the
        union of unique canonical SMILES present in either the left (`smiles`) or right
        (`smiles_right`) columns.
      - We generate fingerprints once per canonical SMILES (in a deterministic `clean_order`),
        then compute the full, memory-constrained N×N Tanimoto similarity matrix across
        the union using `crossTanimotoSimilarityMemoryConstrained`.
      - We avoid indexing into GPU-backed fingerprint objects (they can be asynchronous).
        Instead, we map canonical SMILES to their index in `clean_order` and look up
        per-pair similarity by indexing the computed similarity matrix (matrix[idx1, idx2]).
      - There is no RDKit fallback: if nvmolkit's Tanimoto computation fails we let the
        exception propagate (fail fast).
      - Because we compute the union NxN matrix, this approach fingerprints each SMILES
        once and is most efficient when many rows share SMILES across the dataset; it
        may be more expensive for extremely large unions (tradeoff noted).

    The function expects the DataFrame to contain columns `smiles` and `smiles_right`.

    Args:
      - df: polars DataFrame containing pair rows
      - fp_radius: radius parameter for Morgan fingerprints
      - fp_size: fingerprint size (#bits)
      - chunk_size: (unused) kept for backward-compatibility. Previously used to chunk unique pairs; now nvmolkit's memory-constrained routine is used and the parameter is ignored.

    Returns:
      - A DataFrame equal to the input with an added column `tanimoto_similarity` (Float32)
    """
    # Check for required columns
    if "smiles" not in df.columns or "smiles_right" not in df.columns:
        raise ValueError(
            "Input dataframe must have 'smiles' and 'smiles_right' columns"
        )

    assert isinstance(fp_radius, int) and fp_radius > 0, (
        "fp_radius must be a positive integer"
    )
    assert isinstance(fp_size, int) and fp_size > 0, (
        "fp_size must be a positive integer"
    )

    s1 = df.get_column("smiles")
    s2 = df.get_column("smiles_right")

    # Sanitize and deduplicate the left and right SMILES lists independently (preserve first-seen order).
    # Per your request: do not compute a union; sanitize each side separately and compute
    # left_unique × right_unique similarity matrix.
    s1_list = s1.to_list()
    s2_list = s2.to_list()

    left_originals = [s for s in dict.fromkeys(s1_list) if s]
    right_originals = [s for s in dict.fromkeys(s2_list) if s]

    # Canonicalize each side separately
    left_sanitized = sanitize_smiles(
        left_originals, batch_size=(1 + len(left_originals) // 6)
    )
    right_sanitized = sanitize_smiles(
        right_originals, batch_size=(1 + len(right_originals) // 6)
    )

    # Map original -> canonical (skip invalid/empty canonical forms)
    left_orig_to_clean = {
        orig: clean for orig, clean in zip(left_originals, left_sanitized) if clean
    }
    right_orig_to_clean = {
        orig: clean for orig, clean in zip(right_originals, right_sanitized) if clean
    }

    # Unique canonical SMILES for each side in deterministic (first-seen) order
    left_unique_cleans = [c for c in dict.fromkeys(left_sanitized) if c]
    right_unique_cleans = [c for c in dict.fromkeys(right_sanitized) if c]

    # Convert canonical SMILES to RDKit Mol objects and keep only valid ones (order preserved).
    left_mols: list[Chem.Mol] = []
    left_cleans_valid: list[str] = []
    for clean in left_unique_cleans:
        m = Chem.MolFromSmiles(clean)
        if m is not None:
            left_mols.append(m)
            left_cleans_valid.append(clean)

    right_mols: list[Chem.Mol] = []
    right_cleans_valid: list[str] = []
    for clean in right_unique_cleans:
        m = Chem.MolFromSmiles(clean)
        if m is not None:
            right_mols.append(m)
            right_cleans_valid.append(clean)

    n_rows = len(df)
    # If either side has no valid molecules, return NaNs
    if not left_mols or not right_mols:
        scores = np.full(n_rows, np.nan, dtype=np.float32)
        return df.with_columns(
            pl.Series("tanimoto_similarity", scores, dtype=pl.Float32)
        )

    # Generate fingerprints for the left and right canonical sets in the desired order.
    fpgen = MorganFingerprintGenerator(radius=fp_radius, fpSize=fp_size)
    fps_left = fpgen.GetFingerprints(left_mols)
    fps_right = fpgen.GetFingerprints(right_mols)

    # Compute the full left_unique × right_unique similarity matrix (memory-constrained).
    # If nvmolkit raises here, let it propagate (no fallback).
    sims_matrix = crossTanimotoSimilarityMemoryConstrained(fps_left, fps_right)

    # Index maps for canonical -> matrix index (left/right)
    left_index = {c: i for i, c in enumerate(left_cleans_valid)}
    right_index = {c: i for i, c in enumerate(right_cleans_valid)}

    # Map per-row similarities by looking up indices in the left×right matrix.
    scores = np.full(n_rows, np.nan, dtype=np.float32)
    for i, (a, b) in enumerate(zip(s1_list, s2_list)):
        c1 = left_orig_to_clean.get(a)
        c2 = right_orig_to_clean.get(b)
        if c1 is None or c2 is None:
            continue
        li = left_index.get(c1)
        ri = right_index.get(c2)
        if li is None or ri is None:
            continue
        scores[i] = float(sims_matrix[li, ri])

    return df.with_columns(pl.Series("tanimoto_similarity", scores, dtype=pl.Float32))


def compute_and_save_tanimoto_scores(
    input_parquet_path: Union[str, Path],
    output_path: Union[str, Path],
    batch_size: Union[int, None] = 100_000,
    fp_radius: int = 2,
    fp_size: int = 2048,
) -> None:
    """
    Computes Tanimoto similarity for pairs in the input parquet using nvmolkit.
    Reads 'smiles' and 'smiles_right' columns, computes similarity, and appends 'tanimoto_similarity'.

    If `batch_size` is set to `None`, all matching input parquet files are read into memory
    using Polars and the entire dataset is processed in a single call (no pyarrow iter_batches).
    The processed DataFrame is then written as a single parquet file to `output_path`.
    """

    input_path = Path(input_parquet_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create log early so it exists even if the function fails immediately
    log_path = output_path.with_suffix(".log")
    _log_message_to_file(
        f"Started compute_and_save_tanimoto_scores: input={str(input_parquet_path)} output={str(output_path)} batch_size={batch_size} fp_radius={fp_radius} fp_size={fp_size}",
        log_path,
        overwrite=True,
    )

    writer = None
    total_rows = 0
    start = perf_counter()

    try:
        # Robust input path handling (kept inside try so exceptions are logged)
        parquet_paths = []
        path_str = str(input_parquet_path)
        if input_path.is_dir():
            parquet_paths = sorted(input_path.rglob("*.parquet"))
        elif any(ch in path_str for ch in ("*", "?", "[")):
            parquet_paths = [Path(p) for p in sorted(_glob(path_str, recursive=True))]
        else:
            parquet_paths = [input_path]

        if not parquet_paths:
            raise FileNotFoundError(f"No parquet files found for {input_path}")

        # If batch_size is None, process all input parquet(s) in-memory with Polars
        # and write the final DataFrame in one shot (do not use pyarrow iter_batches).
        if batch_size is None:
            _log_message_to_file(
                f"Processing without batching (batch_size=None). Reading {len(parquet_paths)} file(s) into memory.",
                log_path,
            )
            try:
                dfs = [pl.read_parquet(str(p)) for p in parquet_paths]
            except Exception:
                _log_message_to_file(
                    f"Failed to read parquet files into Polars: {traceback.format_exc()}",
                    log_path,
                    level=logging.ERROR,
                )
                raise

            if len(dfs) == 0:
                raise FileNotFoundError(f"No parquet files found for {input_path}")

            # Concatenate if multiple files were provided
            if len(dfs) == 1:
                df_all = dfs[0]
            else:
                df_all = pl.concat(dfs, how="vertical")

            _log_message_to_file(
                f"Read total rows={df_all.height} from {len(parquet_paths)} files; starting Tanimoto processing",
                log_path,
            )

            # Process the entire DataFrame in one shot
            df_processed = process_batch_tanimoto(
                df_all, fp_radius=fp_radius, fp_size=fp_size
            )

            # Write the processed DataFrame back to storage using Polars (single file)
            try:
                t_write = perf_counter()
                df_processed.write_parquet(str(output_path))
                write_dur = perf_counter() - t_write
                _log_message_to_file(
                    f"Wrote processed DataFrame to {output_path} rows={df_processed.height} in {write_dur:.4f}s",
                    log_path,
                )
                logger.debug(
                    "Wrote processed DataFrame to %s rows=%d in %.4f s",
                    output_path,
                    df_processed.height,
                    write_dur,
                )
            except Exception:
                _log_message_to_file(
                    f"Failed to write processed DataFrame to {output_path}: {traceback.format_exc()}",
                    log_path,
                    level=logging.ERROR,
                )
                raise

            # Also log the total time like the batched path does
            end = perf_counter()
            total_rows = int(df_processed.height)
            _log_message_to_file(
                f"Wrote the pairs with tanimoto to file {str(output_path)} in time {end - start:.3f}s total_rows={total_rows}",
                log_path,
            )

            # Done - skip the per-file batched loop
            return

        for p_path in parquet_paths:
            # Open parquet file and read metadata without loading the full dataset
            try:
                pf = pq.ParquetFile(p_path)
            except Exception:
                _log_message_to_file(
                    f"Processing parquet file: {p_path} (could not open parquet file: {traceback.format_exc()})",
                    log_path,
                    level=logging.ERROR,
                )
                raise

            total_rows_in_file = None
            try:
                if getattr(pf, "metadata", None) is not None:
                    # Prefer the direct num_rows property
                    total_rows_in_file = getattr(pf.metadata, "num_rows", None)
                    # If not present, fall back to summing row-groups
                    if total_rows_in_file is None:
                        total_rows_in_file = sum(
                            int(pf.metadata.row_group(i).num_rows)
                            for i in range(pf.metadata.num_row_groups)
                        )
            except Exception:
                _log_message_to_file(
                    f"Failed to read row count metadata for {p_path}: {traceback.format_exc()}",
                    log_path,
                    level=logging.WARNING,
                )

            # Compute estimated number of batches for progress reporting (if we know total rows)
            if total_rows_in_file is not None:
                try:
                    num_batches = max(
                        1, math.ceil(int(total_rows_in_file) / batch_size)
                    )
                except Exception:
                    num_batches = None
                _log_message_to_file(
                    f"Processing parquet file: {p_path} (total_rows={int(total_rows_in_file)}, estimated_batches={num_batches})",
                    log_path,
                )
            else:
                num_batches = None
                _log_message_to_file(
                    f"Processing parquet file: {p_path} (total_rows=unknown)",
                    log_path,
                    level=logging.WARNING,
                )

            # Iterate over batches
            for batch_idx, batch in enumerate(pf.iter_batches(batch_size=batch_size)):
                # Time the processing and writing of each batch
                batch_t0 = perf_counter()

                # Convert to Polars
                df_batch = pl.from_arrow(batch)
                assert isinstance(df_batch, pl.DataFrame)

                # Process batch
                df_processed = process_batch_tanimoto(
                    df_batch,
                    fp_radius=fp_radius,
                    fp_size=fp_size,
                )

                # Convert back to Arrow
                table = df_processed.to_arrow()

                # Initialize writer if needed
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema)
                    _log_message_to_file("Initialized parquet writer", log_path)

                # Write batch
                t_write = perf_counter()
                writer.write_table(table)
                batch_write_dur = perf_counter() - t_write
                logger.debug(
                    "Wrote a batch to %s in %.4f s", output_path, batch_write_dur
                )

                batch_t1 = perf_counter()
                batch_time = batch_t1 - batch_t0

                # Update counters and log
                batch_rows = getattr(batch, "num_rows", len(batch))
                total_rows += batch_rows
                # Report batch progress with 1-based indexing and total batches if known, include timing
                if num_batches is not None:
                    _log_message_to_file(
                        f"Wrote batch {batch_idx + 1}/{num_batches} from {p_path}: rows={batch_rows}, cumulative_rows={total_rows}, time={batch_time:.3f}s",
                        log_path,
                    )
                else:
                    _log_message_to_file(
                        f"Wrote batch {batch_idx + 1} from {p_path}: rows={batch_rows}, cumulative_rows={total_rows}, time={batch_time:.3f}s",
                        log_path,
                    )

    except Exception:
        _log_message_to_file(
            f"Exception during compute_and_save_tanimoto_scores:\n{traceback.format_exc()}",
            log_path,
            level=logging.ERROR,
        )
        raise
    finally:
        if writer:
            writer.close()

    end = perf_counter()
    _log_message_to_file(
        f"Wrote the pairs with tanimoto to file {str(output_path)} in time {end - start:.3f}s total_rows={total_rows}",
        log_path,
    )
