import logging
import math
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
from rdkit import Chem

from hrms_utils.rdkit import sanitize_smiles


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
