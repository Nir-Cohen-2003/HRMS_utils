import logging
import math
import tempfile
import threading
import traceback
from pathlib import Path
from time import perf_counter
from typing import List, Optional, Union

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from nvmolkit.fingerprints import MorganFingerprintGenerator
from nvmolkit.similarity import crossTanimotoSimilarityMemoryConstrained
from rdkit import Chem

from parallel_rdkit.mol import sanitize_smiles

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


def process_batch_tanimoto(
    df: pl.DataFrame, fp_radius: int = 2, fp_size: int = 2048
) -> pl.DataFrame:
    """Process one batch (a polars DataFrame) of pairs and compute Tanimoto similarity.

    Notes:
      - Canonicalizes SMILES using `sanitize_smiles` and deduplicates the left and right
        sides independently (preserving first-seen order). No cross-side union is built.
      - Generates fingerprints separately for the left and right canonical sets, then
        computes the left_unique × right_unique memory-constrained Tanimoto matrix via
        `crossTanimotoSimilarityMemoryConstrained`.
      - Avoids indexing into GPU-backed fingerprint objects; maps canonical SMILES to
        their side-specific indices and reads per-pair similarity from the left×right
        matrix.
      - There is no RDKit fallback: if nvmolkit's Tanimoto computation fails we let the
        exception propagate (fail fast).
      - Because fingerprinting is side-specific, each canonical SMILES is fingerprinted
        once per side; this is efficient when many rows reuse molecules within a side
        but can be heavier if both sides are extremely large.

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
    left_library_parquet_path: Union[str, Path],
    right_library_parquet_path: Optional[Union[str, Path]] = None,
    batch_size: Union[int, None] = 100_000,
    fp_radius: int = 2,
    fp_size: int = 2048,
) -> None:
    """
    Join the minimal pairs parquet with saved library snapshot(s) to obtain SMILES,
    compute Tanimoto similarity, and write a minimal output parquet containing only
    idx, idx_right, mol_idx, mol_idx_right, dotprod_similarity, and tanimoto_similarity.

    All joins are executed on LazyFrames with streaming; intermediate data are trimmed
    to the smallest required column set before being written to disk.
    """
    input_path = Path(input_parquet_path)
    output_path = Path(output_path)
    left_lib_path = Path(left_library_parquet_path)
    right_lib_path = (
        Path(right_library_parquet_path)
        if right_library_parquet_path
        else left_lib_path
    )

    assert input_path.exists(), f"Pairs parquet not found: {input_path}"
    assert left_lib_path.exists(), f"Left library parquet not found: {left_lib_path}"
    assert right_lib_path.exists(), f"Right library parquet not found: {right_lib_path}"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    log_path = output_path.with_suffix(".log")
    _log_message_to_file(
        f"Started compute_and_save_tanimoto_scores: input={input_path} output={output_path} left_lib={left_lib_path} right_lib={right_lib_path} batch_size={batch_size} fp_radius={fp_radius} fp_size={fp_size}",
        log_path,
        overwrite=True,
    )

    writer: pq.ParquetWriter | None = None
    total_rows = 0
    start = perf_counter()

    # Temporary parquet that holds only the columns needed for Tanimoto computation
    temp_handle = tempfile.NamedTemporaryFile(
        prefix="tanimoto_join_", suffix=".parquet", delete=False
    )
    temp_join_path = Path(temp_handle.name)
    temp_handle.close()

    try:
        # Build streaming joins to attach SMILES from library snapshots
        pairs_lf = pl.scan_parquet(str(input_path)).select(
            [
                pl.col("idx").cast(pl.Int64),
                pl.col("idx_right").cast(pl.Int64),
                pl.col("mol_idx").cast(pl.Int64),
                pl.col("mol_idx_right").cast(pl.Int64),
                pl.col("dotprod_similarity").cast(pl.Float32),
            ]
        )

        left_lib_lf = (
            pl.scan_parquet(str(left_lib_path))
            .select(
                pl.col("idx").cast(pl.Int64),
                pl.col("smiles"),
            )
            .filter(pl.col("smiles").is_not_null())
        )

        right_lib_lf = (
            pl.scan_parquet(str(right_lib_path))
            .select(
                pl.col("idx").alias("idx_right").cast(pl.Int64),
                pl.col("smiles").alias("smiles_right"),
            )
            .filter(pl.col("smiles_right").is_not_null())
        )

        joined_lf = (
            pairs_lf.join(left_lib_lf, on="idx", how="left")
            .join(right_lib_lf, on="idx_right", how="left")
            .select(
                [
                    "idx",
                    "idx_right",
                    "mol_idx",
                    "mol_idx_right",
                    "dotprod_similarity",
                    "smiles",
                    "smiles_right",
                ]
            )
        )

        _log_message_to_file(
            f"Streaming join of pairs with libraries to {temp_join_path}",
            log_path,
        )
        joined_lf.sink_parquet(str(temp_join_path), maintain_order=False)

        # Fast path: process all rows in one go
        if batch_size is None:
            df_all = pl.read_parquet(str(temp_join_path))
            _log_message_to_file(
                f"Processing without batching (batch_size=None). Rows={df_all.height}",
                log_path,
            )
            df_processed = process_batch_tanimoto(
                df_all, fp_radius=fp_radius, fp_size=fp_size
            ).drop(["smiles", "smiles_right"])
            t_write = perf_counter()
            df_processed.write_parquet(str(output_path))
            write_dur = perf_counter() - t_write
            total_rows = int(df_processed.height)
            _log_message_to_file(
                f"Wrote processed DataFrame to {output_path} rows={total_rows} in {write_dur:.4f}s",
                log_path,
            )
            end = perf_counter()
            _log_message_to_file(
                f"Wrote the pairs with tanimoto to file {str(output_path)} in time {end - start:.3f}s total_rows={total_rows}",
                log_path,
            )
            return

        # Batched processing via PyArrow iterator on the joined parquet
        pf = pq.ParquetFile(temp_join_path)
        total_rows_in_file = getattr(pf.metadata, "num_rows", None)
        num_batches = None
        if total_rows_in_file is not None:
            num_batches = max(1, math.ceil(int(total_rows_in_file) / batch_size))
            _log_message_to_file(
                f"Processing joined parquet {temp_join_path} (total_rows={total_rows_in_file}, estimated_batches={num_batches})",
                log_path,
            )

        for batch_idx, batch in enumerate(pf.iter_batches(batch_size=batch_size)):
            batch_t0 = perf_counter()
            df_batch = pl.from_arrow(batch)
            assert isinstance(df_batch, pl.DataFrame)

            df_processed = process_batch_tanimoto(
                df_batch, fp_radius=fp_radius, fp_size=fp_size
            ).drop(["smiles", "smiles_right"])

            table = df_processed.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
                _log_message_to_file("Initialized parquet writer", log_path)

            t_write = perf_counter()
            writer.write_table(table)
            batch_write_dur = perf_counter() - t_write
            logger.debug("Wrote a batch to %s in %.4f s", output_path, batch_write_dur)

            batch_time = perf_counter() - batch_t0
            batch_rows = df_processed.height
            total_rows += batch_rows
            if num_batches is not None:
                _log_message_to_file(
                    f"Wrote batch {batch_idx + 1}/{num_batches} from {temp_join_path}: rows={batch_rows}, cumulative_rows={total_rows}, time={batch_time:.3f}s",
                    log_path,
                )
            else:
                _log_message_to_file(
                    f"Wrote batch {batch_idx + 1} from {temp_join_path}: rows={batch_rows}, cumulative_rows={total_rows}, time={batch_time:.3f}s",
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
        if temp_join_path.exists():
            try:
                temp_join_path.unlink()
            except Exception:
                _log_message_to_file(
                    f"Failed to delete temporary join parquet {temp_join_path}: {traceback.format_exc()}",
                    log_path,
                    level=logging.WARNING,
                )

    end = perf_counter()
    _log_message_to_file(
        f"Wrote the pairs with tanimoto to file {str(output_path)} in time {end - start:.3f}s total_rows={total_rows}",
        log_path,
    )
