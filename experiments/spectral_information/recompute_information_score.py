# HRMS_utils/experiments/spectral_information/recompute_information_score.py
"""
Module for computing alternative spectral information metrics on library dataframes.

Why: Allows testing different information metrics without rerunning expensive similarity
computations. This module provides functions to compute scores from spectral data and
replace the spectral_information_score column in library snapshot dataframes in-memory.

Design:
- All operations are in-memory (no disk I/O)
- Preserves spectral identity via idx-based joins
- Supports both per-spectrum and per-molecule metrics
- Used by plot_similarity_vs_info.py when info_metric != "spectral_information"
"""

import logging
from enum import Enum
from typing import List

import numpy as np
import polars as pl
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class InfoMetric(str, Enum):
    """
    Available information metrics for spectral informativity.

    Why: Using an enum provides type safety and documents available options.
    Add new metrics here as literal values.
    """

    SPECTRAL_INFORMATION = "spectral_information_score"
    SHANNON_ENTROPY = "shannon_entropy"
    SPECTRAL_COMPLEXITY = "spectral_complexity"
    PEAK_COUNT = "peak_count"
    WEIGHTED_PEAK_COUNT = "weighted_peak_count"
    WEIGHTED_SPECTRAL_INFORMATION = "weighted_spectral_information"


def compute_shannon_entropy(intensities: NDArray[np.float64]) -> float:
    """
    Compute Shannon entropy of normalized intensity distribution.

    Args:
        intensities: 1D array of intensity values (shape: (n_peaks,))

    Returns:
        Shannon entropy in nats (natural logarithm)
    """
    assert intensities.ndim == 1, (
        f"intensities must be a 1D array, got {intensities.ndim}D array instead"
    )

    total = np.sum(intensities)
    assert total > 0, "Cannot compute entropy for zero-intensity spectrum"

    probs = intensities / total
    probs_nonzero = probs[probs > 0]

    if len(probs_nonzero) == 0:
        return 0.0

    return float(-np.sum(probs_nonzero * np.log(probs_nonzero)))


def compute_spectral_complexity(
    mz: NDArray[np.float64], intensities: NDArray[np.float64]
) -> float:
    """
    Spectral complexity: entropy weighted by log of peak count.

    Why: Combines information content (entropy) with spectral richness (peak count).
    More peaks with high entropy indicates a more informative spectrum.

    Args:
        mz: 1D array of m/z values (shape: (n_peaks,))
        intensities: 1D array of intensity values (shape: (n_peaks,))

    Returns:
        Spectral complexity score
    """
    assert mz.ndim == 1, f"mz must be a 1D array, got {mz.ndim}D array instead"
    assert intensities.ndim == 1, (
        f"intensities must be a 1D array, got {intensities.ndim}D array instead"
    )
    assert len(mz) == len(intensities), (
        f"mz and intensities must have same length, got {len(mz)} vs {len(intensities)}"
    )

    entropy = compute_shannon_entropy(intensities)
    peak_count_factor = np.log1p(len(intensities))

    return float(entropy * peak_count_factor)


def compute_peak_count(
    mz: NDArray[np.float64], intensities: NDArray[np.float64]
) -> float:
    """
    Simple peak count as information metric.

    Why: More peaks may indicate more structural information.

    Args:
        mz: 1D array of m/z values (shape: (n_peaks,))
        intensities: 1D array of intensity values (shape: (n_peaks,))

    Returns:
        Number of peaks as a float
    """
    assert mz.ndim == 1, f"mz must be a 1D array, got {mz.ndim}D array instead"
    assert intensities.ndim == 1, (
        f"intensities must be a 1D array, got {intensities.ndim}D array instead"
    )
    assert len(mz) == len(intensities), (
        f"mz and intensities must have same length, got {len(mz)} vs {len(intensities)}"
    )

    return float(len(intensities))


def compute_weighted_peak_count(
    mz: NDArray[np.float64], intensities: NDArray[np.float64]
) -> float:
    """
    Peak count weighted by normalized m/z values.

    Why: Higher m/z peaks may carry more structural information.
    Weights each peak by its relative m/z position.

    Args:
        mz: 1D array of m/z values (shape: (n_peaks,))
        intensities: 1D array of intensity values (shape: (n_peaks,))

    Returns:
        Weighted peak count
    """
    assert mz.ndim == 1, f"mz must be a 1D array, got {mz.ndim}D array instead"
    assert intensities.ndim == 1, (
        f"intensities must be a 1D array, got {intensities.ndim}D array instead"
    )
    assert len(mz) == len(intensities), (
        f"mz and intensities must have same length, got {len(mz)} vs {len(intensities)}"
    )

    if len(mz) == 0:
        return 0.0

    max_mz = mz.max()
    if max_mz == 0:
        return float(len(mz))

    weights = mz / max_mz
    normalized_intensities = intensities / intensities.sum()

    return float(np.sum(weights * normalized_intensities) * len(mz))


def compute_information_score_for_spectrum(
    mz: NDArray[np.float64],
    intensities: NDArray[np.float64],
    metric: InfoMetric,
) -> float:
    """
    Compute information score for a single spectrum using the specified metric.

    Why: Centralizes metric dispatch logic. Add new metrics here by extending
    the if/elif chain.

    Args:
        mz: 1D array of m/z values (shape: (n_peaks,))
        intensities: 1D array of intensity values (shape: (n_peaks,))
        metric: Which information metric to compute

    Returns:
        Information score (float)
    """
    assert mz.ndim == 1, f"mz must be a 1D array, got {mz.ndim}D array instead"
    assert intensities.ndim == 1, (
        f"intensities must be a 1D array, got {intensities.ndim}D array instead"
    )
    assert len(mz) == len(intensities), (
        f"mz and intensities must have same length, got {len(mz)} vs {len(intensities)}"
    )

    if metric == InfoMetric.SHANNON_ENTROPY:
        return compute_shannon_entropy(intensities)
    elif metric == InfoMetric.SPECTRAL_COMPLEXITY:
        return compute_spectral_complexity(mz, intensities)
    elif metric == InfoMetric.PEAK_COUNT:
        return compute_peak_count(mz, intensities)
    elif metric == InfoMetric.WEIGHTED_PEAK_COUNT:
        return compute_weighted_peak_count(mz, intensities)
    else:
        raise ValueError(
            f"Unknown metric: {metric}. "
            f"Available metrics: {[m.value for m in InfoMetric]}"
        )


def recompute_information_scores_in_dataframe(
    df_snapshot: pl.DataFrame,
    df_library: pl.DataFrame,
    metric: InfoMetric,
    score_column_name: str = "spectral_information_score",
    mz_column: str = "cleaned_normalized_mz",
    intensity_column: str = "cleaned_normalized_intensity",
) -> pl.DataFrame:
    """
    Compute new information scores from library data and update snapshot dataframe.

    Why: Operates entirely in-memory without disk I/O. Takes a snapshot dataframe
    (with metadata like idx, mol_idx, smiles) and a library dataframe (with full
    spectral data), computes new scores, and returns updated snapshot.

    The join on idx preserves spectral identity - each spectrum gets its own computed
    score even if multiple spectra belong to the same molecule.

    Args:
        df_snapshot: Library snapshot dataframe with columns [idx, mol_idx, ...].
            This dataframe's score column will be replaced.
        df_library: Full library dataframe with spectral data columns.
            Must have idx column matching df_snapshot and contain mz_column/intensity_column.
        metric: Which information metric to compute
        score_column_name: Column name for the information score (replaces if exists)
        mz_column: Column name in df_library containing m/z arrays
        intensity_column: Column name in df_library containing intensity arrays

    Returns:
        Updated snapshot dataframe with new scores (same schema as input snapshot,
        but with score_column_name updated)
    """
    assert "idx" in df_snapshot.columns, (
        f"Snapshot must contain 'idx' column for joining. "
        f"Available columns: {df_snapshot.columns}"
    )
    assert "idx" in df_library.columns, (
        f"Library must contain 'idx' column for joining. "
        f"Available columns: {df_library.columns}"
    )
    assert mz_column in df_library.columns, (
        f"Library must contain '{mz_column}' column. "
        f"Available columns: {df_library.columns}"
    )
    assert intensity_column in df_library.columns, (
        f"Library must contain '{intensity_column}' column. "
        f"Available columns: {df_library.columns}"
    )

    if metric == InfoMetric.WEIGHTED_SPECTRAL_INFORMATION:
        logger.info("Computing weighted spectral information (score / heavy_atoms)")

        # Ensure required columns
        assert "precursor_formula_array" in df_library.columns, (
            "df_library missing 'precursor_formula_array'"
        )

        # We need the original spectral_information_score.
        # Assuming it exists in df_snapshot as 'spectral_information_score'.
        source_score_col = "spectral_information_score"
        assert source_score_col in df_snapshot.columns, (
            f"Snapshot must contain '{source_score_col}' to compute weighted score"
        )

        # Join formula array from library to snapshot
        # We select only necessary columns from library
        lib_subset = df_library.select(["idx", "precursor_formula_array"])

        # Join
        joined = df_snapshot.join(lib_subset, on="idx", how="left")

        # Calculate heavy atoms
        # formula array: index 0 is typically H.
        # heavy atoms = sum(array) - array[0]
        heavy_atoms_expr = pl.col("precursor_formula_array").arr.sum() - pl.col(
            "precursor_formula_array"
        ).arr.get(0)

        # Compute new score
        # Handle potential division by zero if heavy_atoms is 0 (unlikely for valid molecules but possible), and we add scaling to make everything comparable.
        new_score_expr = (
            pl.col(source_score_col) * pl.lit(30) / heavy_atoms_expr
        ).alias(score_column_name)

        df_updated = joined.with_columns(new_score_expr)

        # Drop precursor_formula_array if it wasn't in snapshot
        if "precursor_formula_array" not in df_snapshot.columns:
            df_updated = df_updated.drop("precursor_formula_array")

        # If we are creating a new column name, we might want to keep or drop the old one.
        # But if score_column_name == source_score_col, it is already updated.

        # Log statistics
        score_stats = df_updated[score_column_name].describe()
        logger.info("Score statistics for %s:\n%s", metric.value, score_stats)

        return df_updated

    logger.info(
        "Computing %s scores for %d spectra in library", metric.value, len(df_library)
    )

    # Extract spectral data
    mz_arrays = df_library[mz_column].to_list()
    intensity_arrays = df_library[intensity_column].to_list()
    idx_values = df_library["idx"].to_list()

    new_scores = []
    failed_count = 0

    for i, (mz_arr, int_arr) in enumerate(zip(mz_arrays, intensity_arrays)):
        if i % 10000 == 0 and i > 0:
            logger.debug("  Processed %d / %d spectra", i, len(df_library))

        # Handle None/null values
        if mz_arr is None or int_arr is None:
            new_scores.append(None)
            failed_count += 1
            continue

        # Convert to numpy arrays
        mz_np = np.array(mz_arr, dtype=np.float64)
        int_np = np.array(int_arr, dtype=np.float64)

        # Skip empty spectra
        if len(mz_np) == 0 or len(int_np) == 0:
            new_scores.append(None)
            failed_count += 1
            continue

        try:
            score = compute_information_score_for_spectrum(mz_np, int_np, metric)
            new_scores.append(score)
        except Exception as e:
            logger.warning(
                "Failed to compute %s for spectrum idx=%d: %s",
                metric.value,
                idx_values[i],
                e,
            )
            new_scores.append(None)
            failed_count += 1

    logger.info(
        "Computed %d scores (%d failed/null)",
        len(new_scores) - failed_count,
        failed_count,
    )

    # Create DataFrame with idx and new scores
    df_scores = pl.DataFrame(
        {"idx": idx_values, score_column_name: pl.Series(new_scores, dtype=pl.Float64)}
    )

    # Drop old score column if it exists in snapshot
    if score_column_name in df_snapshot.columns:
        logger.debug("Dropping existing '%s' column from snapshot", score_column_name)
        df_snapshot = df_snapshot.drop(score_column_name)

    # Join on idx to update scores while preserving all other snapshot data
    # Why: Must use idx (spectrum-level) join to preserve spectral identity.
    logger.debug("Joining new scores with snapshot on 'idx'")
    df_updated = df_snapshot.join(df_scores, on="idx", how="left")

    # Verify row count unchanged
    assert len(df_updated) == len(df_snapshot), (
        f"Join changed row count: {len(df_snapshot)} -> {len(df_updated)}. "
        "This indicates idx mismatch between library and snapshot."
    )

    # Check for any missing scores after join
    null_count = df_updated[score_column_name].null_count()
    if null_count > 0:
        logger.warning(
            "Warning: %d spectra have null scores after join (%.1f%%)",
            null_count,
            100.0 * null_count / len(df_updated),
        )

    # Log statistics
    score_stats = df_updated[score_column_name].describe()
    logger.info("Score statistics for %s:\n%s", metric.value, score_stats)

    return df_updated
