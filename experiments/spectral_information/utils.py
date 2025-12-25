import gc
import logging
import math
import threading
import traceback
from glob import glob as _glob
from pathlib import Path
from time import perf_counter
from typing import List, Union

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from nvmolkit.fingerprints import MorganFingerprintGenerator
from nvmolkit.similarity import crossTanimotoSimilarityMemoryConstrained
from pyarrow import ChunkedArray, Table
from rdkit import Chem

from hrms_utils.rdkit import sanitize_smiles

# Tracks which log files have been truncated for this process so we only truncate once.
_initialized_log_paths: set[str] = set()
_initialized_log_paths_lock = threading.Lock()


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


def build_and_write_pairs_parquet(
    parquet_paths: List[Path],
    output_path: Union[str, Path],
    threshold: float = 0.8,
    num_spectra: int | None = None,
    ms2_tolerance_ppm: float = 10.0,
    batch_size: int = 1000,
    use_pyarrow_batching: bool = True,
    mass_range: tuple[float, float] | None = None,
) -> None:
    """
    Build unioned library LF, compute pairwise dot-product similarities (ignoring precursor),
    and write pairs with high similarity to parquet.

    Args:
      - parquet_paths: list of Path objects pointing at library parquet files
      - output_path: where to write the pairs with similarities (required)
      - threshold: float (default 0.8). Only pairs with dotprod_similarity >= threshold are saved.
      - num_spectra: Optional[int]. If provided, limit the number of molecules read from the
        unioned input using a lazy .limit(num_spectra) to avoid collecting the full dataset.
      - batched: bool (default True). If True, materialize library and process in batches.
        If False, use streaming mode with pl.Config.set_streaming_chunk_size(batch_size).
      - mass_range: Optional[tuple[float, float]] (default None). If provided, spectra will be
        filtered per-input-parquet by `precursor_mz` such that only spectra with
        `min <= precursor_mz <= max` are retained. This filtering is applied before the union
        of input libraries (i.e. before pairwise computation).

    Returns:
      - None (writes parquet to output_path)
    """
    # Ensure we create the log file early so it exists even on early failures
    output_path = Path(output_path)
    log_path = output_path.with_suffix(".log")
    _log_message_to_file(
        f"Started build_and_write_pairs_parquet: output={str(output_path)} threshold={threshold} num_spectra={num_spectra} parquet_paths={parquet_paths} batched={use_pyarrow_batching} mass_range={mass_range}",
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

        if use_pyarrow_batching:
            _log_message_to_file(
                "Materializing source library for batch processing...", log_path
            )
            df_source = lf.collect()
            n_source = len(df_source)
            _log_message_to_file(f"Source library size: {n_source}", log_path)

            _log_message_to_file(
                f"Starting pairwise similarity computation (batch_size={batch_size})",
                log_path,
            )

            writer = None
            total_written = 0

            try:
                for batch_start in range(0, n_source, batch_size):
                    batch_t0 = perf_counter()
                    # Slice the left side (batch) and use the full right side
                    batch_lf = df_source.slice(batch_start, batch_size).lazy()
                    source_lf = df_source.lazy()

                    # Join and compute similarity
                    pairs_batch = _cross_join_and_score(
                        batch_lf,
                        source_lf,
                        ms2_tolerance_in_ppm=ms2_tolerance_ppm,
                        threshold=threshold,
                        require_peak_overlap=True,
                    ).collect(engine="streaming")

                    if len(pairs_batch) > 0:
                        table = pairs_batch.to_arrow()
                        if writer is None:
                            writer = pq.ParquetWriter(output_path, table.schema)
                        writer.write_table(table)
                        total_written += len(pairs_batch)
                        del pairs_batch
                        del table
                        gc.collect()
                    batch_t1 = perf_counter()
                    if (batch_start // batch_size) % 10 == 0:
                        _log_message_to_file(
                            f"Processed batch {batch_start // batch_size} / {(n_source // batch_size) + 1}. "
                            f"Written {total_written} pairs so far. "
                            f"Time for batch: {batch_t1 - batch_t0:.3f}s",
                            log_path,
                        )

            except Exception:
                _log_message_to_file(
                    f"Exception during build_and_write_pairs_parquet loop:\n{traceback.format_exc()}",
                    log_path,
                    level=logging.ERROR,
                )
                raise
            finally:
                if writer:
                    writer.close()
        else:
            # Streaming mode
            pl.Config.set_streaming_chunk_size(batch_size)
            _log_message_to_file(
                f"Starting pairwise similarity computation (streaming, chunk_size={batch_size})",
                log_path,
            )

            # Join and compute similarity (identical logic, but on full LazyFrame)
            pairs_filtered = _cross_join_and_score(
                lf,
                lf,
                ms2_tolerance_in_ppm=ms2_tolerance_ppm,
                threshold=threshold,
            )

            try:
                pairs_filtered.sink_parquet(str(output_path), engine="streaming")
            except Exception:
                _log_message_to_file(
                    f"Exception during build_and_write_pairs_parquet (streaming):\n{traceback.format_exc()}",
                    log_path,
                    level=logging.ERROR,
                )
                raise

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
                df_processed.write_parquet(str(output_path))
                _log_message_to_file(
                    f"Wrote processed DataFrame to {output_path} rows={df_processed.height}",
                    log_path,
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
                writer.write_table(table)

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
