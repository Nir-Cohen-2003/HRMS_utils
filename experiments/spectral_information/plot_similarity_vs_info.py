# HRMS_utils/experiments/spectral_information/plot_similarity_vs_info.py
"""
Analysis tools to compute the relationship between spectral informativity and
Tanimoto similarity of matched molecules.

This script:
 - Reads a pairs parquet file that contains pairwise matches (with a
   `tanimoto_similarity` column) produced by `compute_and_save_tanimoto_scores`.
 - Reads the source library parquet (the same library used to build pairs).
 - For each spectrum computes the average Tanimoto similarity to unique matched
   molecules (deduplicating multiple spectra from the same molecule).
   Every spectrum is implicitly given a self-match with Tanimoto 1.0.
 - Produces:
    1) Per-molecule Spearman correlations between informativity and average Tanimoto
       (including all spectra of molecules that had at least one match present in
       the pairs file).
    2) A global Spearman (all spectra from molecules with at least one match),
       plus a binned plot of average Tanimoto vs the chosen information measure.
    3) A global Spearman / binned plot like (2) but using only the single most
       informative spectrum per molecule.

Implementation notes / design decisions:
 - Polars is used for all dataframe work and streaming engine is used where appropriate.
 - Per-spectrum matches to molecules are deduplicated by `mol_idx` (taken as the
   canonical molecule identifier in the pairs parquet). For multiple matches to
   the same molecule we use the maximal Tanimoto value for that (spectrum, molecule).
 - If necessary columns (e.g. 'idx', 'mol_idx', 'tanimoto_similarity', etc.)
   are missing we fail fast with an informative AssertionError.
 - Figures are saved to files (no display / plt.show()).
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from rdkit.Chem.rdchem import PrivateProps
from recompute_information_score import (
    InfoMetric,
    recompute_information_scores_in_dataframe,
)
from scipy.stats import spearmanr

# from scipy.stats.distributions import pairs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class SimilarityVsInfoConfig:
    """
    Configuration for similarity vs information analysis.

    Attributes:
      - pairs_parquet_path: Path to pairs parquet with columns for indices, mol_idx, and tanimoto.
      - left_library_parquet_path: Path to the left library snapshot parquet containing `idx`, `mol_idx`, and the information measure (and `smiles`).
      - right_library_parquet_path: Optional right library snapshot parquet (same schema as left); defaults to left when None.
      - left_library_full_parquet_path: Path to the left library full parquet (with full spectral data and stable `idx`).
        Required if info_metric != InfoMetric.SPECTRAL_INFORMATION or if scores need to be recomputed.
      - right_library_full_parquet_path: Optional path to the right library full parquet.
        Required if info_metric != InfoMetric.SPECTRAL_INFORMATION and right_library_parquet_path is provided.
      - info_metric: Which information metric to use (default: spectral_information, which uses pre-computed scores).
        If not spectral_information, scores will be recomputed from the full library parquet(s).
      - tanimoto_col: name of Tanimoto similarity column in pairs parquet.
      - left_idx_col/right_idx_col: column names for the left/right spectrum global index.
      - left_mol_col/right_mol_col: column names for the left/right molecule id (mol_idx).
      - info_measure: column name in library that holds the informativity measure.
      - x_bin_width/x_range: binning configuration for plotting.
      - min_count_threshold: minimum bin count to include the point in the plot.
      - output_dir: where to write figures and optional CSVs.
      - dotprod_col: name of dot-product similarity column in pairs parquet.
      - dotprod_thresholds: an iterable of dot-product thresholds to use when filtering matches
        for analyses and plotting (defaults to a single value of 0.8).
      - dotprod_bin_size: bin width for the dot-product axis used by the heatmap (default 0.05).
    """

    pairs_parquet_path: Union[str, Path]
    left_library_parquet_path: Union[str, Path]
    right_library_parquet_path: Optional[Union[str, Path]] = None
    left_library_full_parquet_path: Optional[Union[str, Path]] = None
    right_library_full_parquet_path: Optional[Union[str, Path]] = None
    info_metric: InfoMetric = InfoMetric.SPECTRAL_INFORMATION
    tanimoto_col: str = "tanimoto_similarity"
    left_idx_col: str = "idx"
    right_idx_col: str = "idx_right"
    left_mol_col: str = "mol_idx"
    right_mol_col: str = "mol_idx_right"
    info_measure: Optional[str] = None
    x_bin_width: float = 0.5
    x_range: Tuple[float, float] = (0.0, 3.0)
    min_count_threshold: int = 10
    output_dir: Union[str, Path] = Path(".")
    # file names:
    filename_all: str = "avg_tanimoto_vs_{measure}_all.png"
    filename_best: str = "avg_tanimoto_vs_{measure}_best.png"
    per_molecule_summary_name: str = "per_molecule_spearman.parquet"
    per_spectrum_summary_name: str = "per_spectrum_tanimoto.parquet"
    # Filename for the generated heatmap figure. The placeholder `{measure}` will be
    # formatted with `config.info_measure` when constructing the output path.
    filename_heatmap: str = "avg_tanimoto_heatmap_vs_{measure}_dotprod.png"
    filename_heatmap_all_spectra: str = (
        "avg_tanimoto_heatmap_vs_{measure}_dotprod_all_spectra.png"
    )
    filename_correlation_spearman: str = (
        "correlation_spearman_dotprod_tanimoto_vs_{measure}.png"
    )
    filename_correlation_pearson: str = (
        "correlation_pearson_dotprod_tanimoto_vs_{measure}.png"
    )
    # Dot-product (similarity) related options:
    dotprod_col: str = "dotprod_similarity"
    # Default filter(s) to apply when selecting matches by dot-product similarity.
    # Use a tuple so the dataclass field is immutable by default.
    dotprod_thresholds: Tuple[float, ...] = (0.8,)
    # Bin size for the dot-product axis in heatmaps.
    dotprod_bin_size: float = 0.05
    # Optional percentile (0-100) at which to draw a dashed line (same color as the mean).
    # If None, no dashed line is drawn.
    lower_percentile: Optional[float] = None
    # Percentile bounds to display as shaded spans in binned plots.
    # Tuple of (lower_percentile, upper_percentile) in percent (0-100). If None,
    # no percentile spans are computed.
    percentile_bounds: Optional[Tuple[float, float]] = None
    # If True, the heatmap x-axis (and line plot x-axis) will be the average information
    # of the pair of spectra involved in the match, rather than the information of the spectrum itself.
    use_avg_info: bool = True

    def __post_init__(self):
        """
        Post-initialization hook to adjust output_dir based on info_metric.

        Why: When using alternative metrics, we automatically create a subdirectory
        named after the metric to keep outputs organized and avoid overwriting results
        from different metrics. This is invisible to the analysis functions.
        """
        # Automatically determine info_measure if not provided
        if self.info_measure is None:
            if self.info_metric == InfoMetric.SPECTRAL_INFORMATION:
                self.info_measure = "spectral_information_score"
            else:
                self.info_measure = self.info_metric.value

        recompute_from_full = self.left_library_full_parquet_path is not None

        if self.info_metric != InfoMetric.SPECTRAL_INFORMATION:
            # Alternative metrics always require full library data for recomputation
            if not recompute_from_full:
                raise ValueError(
                    f"left_library_full_parquet_path must be provided when "
                    f"info_metric={self.info_metric.value}"
                )

        if recompute_from_full and self.right_library_parquet_path is not None:
            if self.right_library_full_parquet_path is None:
                raise ValueError(
                    f"right_library_full_parquet_path must be provided when "
                    f"using cross-library mode with full library paths"
                )

        if self.info_metric != InfoMetric.SPECTRAL_INFORMATION:
            # Create subdirectory for this metric
            base_output_dir = Path(self.output_dir)
            self.output_dir = base_output_dir / self.info_metric.value
            logger.info(
                "Using alternative metric %s; outputs will be written to %s",
                self.info_metric.value,
                self.output_dir,
            )


def _require_columns_or_fail(
    lf: pl.LazyFrame, cols: List[str], description: str
) -> None:
    """
    Fail fast if any of the provided columns are missing from the lazyframe.
    Uses streaming collect of a single row to validate columns exist.
    """
    try:
        lf.select([pl.col(c) for c in cols]).limit(1).collect(engine="streaming")
    except Exception as exc:
        raise AssertionError(
            f"Required columns {cols} missing from {description}: {exc}"
        ) from exc


def _load_full_library_and_recompute_scores(
    snapshot_path: Path,
    full_library_path: Path,
    metric: InfoMetric,
    info_measure_col: str,
) -> pl.DataFrame:
    """
    Load library snapshot and full library (with stable idx), recompute information scores.

    Why: The full library parquet already contains the correct idx assigned by the
    heavy-compute pipeline. We simply read it and recompute scores, avoiding any
    fragile idx reconstruction from original files.

    Args:
        snapshot_path: Path to library snapshot parquet
        full_library_path: Path to full library parquet with spectral data and stable idx
        metric: Which information metric to compute
        info_measure_col: Column name for the information score

    Returns:
        Updated snapshot dataframe with recomputed scores
    """
    logger.info("Loading library snapshot from %s", snapshot_path)
    df_snapshot = pl.read_parquet(str(snapshot_path))

    logger.info("Loading full library from %s", full_library_path)
    df_library = pl.read_parquet(str(full_library_path))

    assert "idx" in df_snapshot.columns, (
        f"Snapshot missing 'idx'. Columns: {df_snapshot.columns}"
    )
    assert "idx" in df_library.columns, (
        f"Full library missing 'idx'. Columns: {df_library.columns}"
    )

    # Trim to columns needed for score computation to reduce memory
    needed_cols = ["idx", "cleaned_normalized_mz", "cleaned_normalized_intensity"]
    if metric == InfoMetric.SPECTRAL_INFORMATION:
        needed_cols.extend(["precursor_formula_array", "cleaned_fragment_formulas"])
    elif metric == InfoMetric.WEIGHTED_SPECTRAL_INFORMATION:
        needed_cols.extend(["precursor_formula_array", "spectral_information_score"])

    available_needed = [c for c in needed_cols if c in df_library.columns]
    df_library = df_library.select(available_needed)

    logger.info("Recomputing %s scores for %d spectra", metric.value, len(df_library))
    df_snapshot_updated = recompute_information_scores_in_dataframe(
        df_snapshot=df_snapshot,
        df_library=df_library,
        metric=metric,
        score_column_name=info_measure_col,
    )

    return df_snapshot_updated


def compute_per_spectrum_avg_tanimoto(
    config: SimilarityVsInfoConfig, min_dotprod: Optional[float] = None
) -> pl.DataFrame:
    """
    Compute per-spectrum average Tanimoto similarity to unique matched molecules.

    Steps:
      - Read the pairs parquet lazily and validate columns.
      - Optionally filter pairs by a minimum dot-product similarity (config.dotprod_col).
      - Build the matches as (idx, matched_mol_idx, tanimoto) from both directions
        (spectrum appearing on left or right of the pair).
      - Deduplicate matches per (idx, matched_mol_idx) taking the max tanimoto.
      - Add implicit self-matches (tanimoto = 1.0) for every spectrum from the library.
      - Aggregate per spectrum to compute average/median number of matched molecules.

    Parameters:
      - min_dotprod: if provided, only pairs with `config.dotprod_col >= min_dotprod` are considered
        as external matches. If `None`, the first value from `config.dotprod_thresholds` is used if present.

    Returns:
      - A Polars DataFrame (collected with streaming engine) with columns:
        `idx`, `mol_idx` (molecule id), `<info_measure>`, `avg_tanimoto`, `median_tanimoto`, `n_matched_molecules`
    """
    pairs_path = Path(config.pairs_parquet_path)
    left_lib_path = Path(config.left_library_parquet_path)
    right_lib_path = (
        Path(config.right_library_parquet_path)
        if config.right_library_parquet_path is not None
        else left_lib_path
    )
    assert pairs_path.exists(), f"Pairs parquet not found: {pairs_path}"
    assert left_lib_path.exists(), f"Left library parquet not found: {left_lib_path}"
    assert right_lib_path.exists(), f"Right library parquet not found: {right_lib_path}"

    # If full library paths are provided, recompute scores from the full libraries.
    # The full libraries already contain the correct idx from the heavy-compute pipeline.
    if config.left_library_full_parquet_path is not None:
        logger.info(
            "Recomputing information scores using metric: %s", config.info_metric.value
        )
        left_lib_df = _load_full_library_and_recompute_scores(
            snapshot_path=left_lib_path,
            full_library_path=Path(config.left_library_full_parquet_path),
            metric=config.info_metric,
            info_measure_col=config.info_measure,
        )

        if right_lib_path != left_lib_path:
            assert config.right_library_full_parquet_path is not None, (
                "right_library_full_parquet_path required for cross-library recomputation"
            )
            right_lib_df = _load_full_library_and_recompute_scores(
                snapshot_path=right_lib_path,
                full_library_path=Path(config.right_library_full_parquet_path),
                metric=config.info_metric,
                info_measure_col=config.info_measure,
            )
        else:
            right_lib_df = left_lib_df

        left_lib = left_lib_df.lazy().select(
            [
                pl.col("idx").cast(pl.Int64),
                pl.col("mol_idx").cast(pl.Int64),
                "smiles",
                "base_inchikey",
                "ion_mode",
                config.info_measure,
            ]
        )
        libs_left: List[pl.LazyFrame] = [left_lib]
        if right_lib_path != left_lib_path:
            right_lib = right_lib_df.lazy().select(
                [
                    pl.col("idx").cast(pl.Int64),
                    pl.col("mol_idx").cast(pl.Int64),
                    "smiles",
                    "base_inchikey",
                    "ion_mode",
                    config.info_measure,
                ]
            )
            libs_left.append(right_lib)
        lib = pl.concat(libs_left) if len(libs_left) > 1 else left_lib
    else:
        # Use pre-computed scores from snapshot files
        logger.info("Scanning left library parquet: %s", left_lib_path)
        left_lib = (
            pl.scan_parquet(str(left_lib_path))
            .select(
                [
                    pl.col("idx").cast(pl.Int64),
                    pl.col("mol_idx").cast(pl.Int64),
                    "smiles",
                    "base_inchikey",
                    "ion_mode",
                    config.info_measure,
                ]
            )
            .filter(pl.col("smiles").is_not_null())
        )
        libs: List[pl.LazyFrame] = [left_lib]
        if right_lib_path != left_lib_path:
            logger.info("Scanning right library parquet: %s", right_lib_path)
            right_lib = (
                pl.scan_parquet(str(right_lib_path))
                .select(
                    [
                        pl.col("idx").cast(pl.Int64),
                        pl.col("mol_idx").cast(pl.Int64),
                        "smiles",
                        "base_inchikey",
                        "ion_mode",
                        config.info_measure,
                    ]
                )
                .filter(pl.col("smiles").is_not_null())
            )
            libs.append(right_lib)
        lib = pl.concat(libs) if len(libs) > 1 else left_lib

    logger.info("Scanning pairs parquet: %s", pairs_path)
    pairs = pl.scan_parquet(str(pairs_path))

    # Validate pairs contain required columns. Include dotprod column for filtering.
    _require_columns_or_fail(
        pairs,
        [
            config.left_idx_col,
            config.right_idx_col,
            config.left_mol_col,
            config.right_mol_col,
            config.tanimoto_col,
            config.dotprod_col,
        ],
        "pairs parquet",
    )

    # Determine default threshold value from config if not provided by caller.
    if min_dotprod is None:
        if getattr(config, "dotprod_thresholds", None):
            min_dotprod = float(config.dotprod_thresholds[0])
        else:
            min_dotprod = None

    if min_dotprod is not None:
        logger.info(
            "Filtering pairs by %s >= %s for per-spectrum computation",
            config.dotprod_col,
            min_dotprod,
        )

        pairs_filtered = pairs.filter(pl.col(config.dotprod_col) >= float(min_dotprod))

    else:
        pairs_filtered = pairs

    # Filter out self-matches (matches to the same molecule ID)
    # Why: Analysis requires excluding matches of a molecule to itself.
    pairs_filtered = pairs_filtered.filter(
        pl.col(config.left_mol_col) != pl.col(config.right_mol_col)
    )

    _require_columns_or_fail(
        lib,
        ["idx", "mol_idx", "smiles", "ion_mode", config.info_measure],
        "library parquet(s)",
    )

    # Collect the minimal library fields for joining below.
    # If use_avg_info is True, we need this info available during pair processing too.
    # We collect it once here.
    lib_df = lib.select(["idx", "mol_idx", config.info_measure]).collect(
        engine="streaming"
    )

    # Build matches from both directions using the optionally filtered pairs.
    # If using avg_info, we need to join the info to the pairs first.
    if config.use_avg_info:
        # Create lightweight lookup for info (idx -> info)
        # We can use the already collected lib_df
        # We need to join it to the pairs stream.
        # Since lib_df is collected, we can use it to create a lazy lookup or join directly if pairs is lazy.
        # Pairs is lazy. We can join lib_df.lazy().

        info_lookup = lib_df.lazy().select(
            [
                pl.col("idx").cast(pl.Int64),
                pl.col(config.info_measure).cast(pl.Float32).alias("info"),
            ]
        )

        left_lookup = info_lookup.select(
            [pl.col("idx").alias(config.left_idx_col), pl.col("info").alias("info_l")]
        )
        right_lookup = info_lookup.select(
            [pl.col("idx").alias(config.right_idx_col), pl.col("info").alias("info_r")]
        )

        pairs_with_info = (
            pairs_filtered.join(left_lookup, on=config.left_idx_col, how="left")
            .join(right_lookup, on=config.right_idx_col, how="left")
            .with_columns(
                ((pl.col("info_l") + pl.col("info_r")) * 0.5).alias("pair_avg_info")
            )
        )

        # Select including pair_avg_info
        left_matches = pairs_with_info.select(
            [
                pl.col(config.left_idx_col).alias("idx"),
                pl.col(config.right_mol_col).alias("matched_mol_idx"),
                pl.col(config.tanimoto_col).alias("tanimoto"),
                pl.col("pair_avg_info"),
            ]
        )

        right_matches = pairs_with_info.select(
            [
                pl.col(config.right_idx_col).alias("idx"),
                pl.col(config.left_mol_col).alias("matched_mol_idx"),
                pl.col(config.tanimoto_col).alias("tanimoto"),
                pl.col("pair_avg_info"),
            ]
        )
    else:
        left_matches = pairs_filtered.select(
            [
                pl.col(config.left_idx_col).alias("idx"),
                pl.col(config.right_mol_col).alias("matched_mol_idx"),
                pl.col(config.tanimoto_col).alias("tanimoto"),
            ]
        )

        right_matches = pairs_filtered.select(
            [
                pl.col(config.right_idx_col).alias("idx"),
                pl.col(config.left_mol_col).alias("matched_mol_idx"),
                pl.col(config.tanimoto_col).alias("tanimoto"),
            ]
        )

    # Aggregate per-direction in streaming mode and combine. Doing the per-direction
    # grouping first bounds memory use and avoids calling `union` on LazyFrame.

    # Define aggregation expressions
    agg_exprs = [pl.col("tanimoto").max().alias("tanimoto")]
    if config.use_avg_info:
        # For deduplication: pick the pair_avg_info associated with the max tanimoto
        agg_exprs.append(
            pl.col("pair_avg_info").sort_by("tanimoto").last().alias("pair_avg_info")
        )

    left_agg = (
        left_matches.group_by(["idx", "matched_mol_idx"])
        .agg(agg_exprs)
        .collect(engine="streaming")
    )

    right_agg = (
        right_matches.group_by(["idx", "matched_mol_idx"])
        .agg(agg_exprs)
        .collect(engine="streaming")
    )

    if left_agg.height == 0 and right_agg.height == 0:
        # No external cross-molecule matches that passed the filtering; dedup_matches stays empty.
        dedup_matches = left_agg
    else:
        union_df = pl.concat([left_agg, right_agg], how="vertical")
        dedup_matches = union_df.group_by(["idx", "matched_mol_idx"]).agg(agg_exprs)

    # Combine external matches (deduped).
    all_matches = dedup_matches.cast(
        {"matched_mol_idx": pl.Int64, "tanimoto": pl.Float32}
    )

    # Aggregate per-spectrum stats
    # Only spectra with at least one cross-match will appear here.
    per_stats_exprs = [
        pl.col("tanimoto").mean().alias("avg_tanimoto"),
        pl.col("tanimoto").median().alias("median_tanimoto"),
        pl.len().alias("n_matched_molecules"),
    ]
    if config.use_avg_info:
        per_stats_exprs.append(
            pl.col("pair_avg_info").mean().alias("avg_info_for_plot")
        )

    per_stats = all_matches.group_by("idx").agg(per_stats_exprs)

    # Join stats back to library (inner join to keep only matched spectra)
    # Why: We only want to analyze spectra that had at least one match to a DIFFERENT molecule.
    # lib_df already collected above
    select_cols = [
        "idx",
        "mol_idx",
        config.info_measure,
        "avg_tanimoto",
        "median_tanimoto",
        "n_matched_molecules",
    ]
    if config.use_avg_info:
        select_cols.append("avg_info_for_plot")

    per_spectrum = lib_df.join(per_stats, on="idx", how="inner").select(select_cols)

    # Ensure expected dtypes
    per_spectrum = per_spectrum.with_columns(
        pl.col("avg_tanimoto").cast(pl.Float64),
        pl.col("median_tanimoto").cast(pl.Float64),
        pl.col("n_matched_molecules").cast(pl.Int64),
    )

    # Fail fast if no per-spectrum entries were produced. The rest of the
    # analysis expects real per-spectrum statistics; failing early makes issues
    # visible to the user instead of silently continuing.
    if per_spectrum.height == 0:
        logger.error(
            "No per-spectrum matches found after filtering. Check pairs parquet and dotprod thresholds."
        )
        raise AssertionError(
            "No per-spectrum matches found after filtering. Check pairs parquet and dotprod thresholds."
        )

    return per_spectrum


def _molecules_with_any_matches(
    pairs_parquet_path: Union[str, Path],
    config: SimilarityVsInfoConfig,
    min_dotprod: Optional[float] = None,
) -> set:
    """
    Return a set of molecule ids (mol_idx) that appear anywhere in the pairs file
    (either on left or right). If `min_dotprod` is provided, only pairs whose
    `config.dotprod_col` value is >= `min_dotprod` are considered.

    Used to determine which molecules had matches that pass a dot-product filter.
    """
    pairs = pl.scan_parquet(str(pairs_parquet_path))

    # Validate columns exist, including the dotprod column used for optional filtering.
    _require_columns_or_fail(
        pairs,
        [config.left_mol_col, config.right_mol_col, config.dotprod_col],
        "pairs parquet (for molecule presence check)",
    )

    if min_dotprod is not None:
        pairs = pairs.filter(pl.col(config.dotprod_col) >= float(min_dotprod))

    # Filter out self-matches (matches to the same molecule)
    pairs = pairs.filter(pl.col(config.left_mol_col) != pl.col(config.right_mol_col))

    left = (
        pairs.select(pl.col(config.left_mol_col)).unique().collect(engine="streaming")
    )
    right = (
        pairs.select(pl.col(config.right_mol_col)).unique().collect(engine="streaming")
    )

    left_list = (
        left.get_column(config.left_mol_col).to_list()
        if config.left_mol_col in left.columns
        else []
    )
    right_list = (
        right.get_column(config.right_mol_col).to_list()
        if config.right_mol_col in right.columns
        else []
    )

    # Remove potential nulls
    left_set = set([int(x) for x in left_list if x is not None])
    right_set = set([int(x) for x in right_list if x is not None])
    return left_set.union(right_set)


def compute_per_molecule_spearman(
    per_spectrum_df: pl.DataFrame,
    molecules_with_matches: set,
    config: SimilarityVsInfoConfig,
) -> pl.DataFrame:
    """
    For each molecule present in `molecules_with_matches`, compute the Spearman
    correlation between `info_measure` and `avg_tanimoto` across all spectra
    belonging to that molecule (including spectra of that molecule that had no
    external matches).
    """
    # Filter to molecules we should include (those that had any matches in pairs parquet).
    included = per_spectrum_df.filter(
        pl.col("mol_idx").is_in(list(molecules_with_matches))
    )

    # Fail fast: the analysis is defined only for molecules that had at least one
    # match in the pairs file. If there are no such spectra the caller should
    # correct the input rather than silently continuing.
    assert included.height > 0, (
        "No spectra for molecules with matches; ensure the pairs parquet contains matches "
        "and that the library index columns are correct."
    )

    # When using average pair info, use the pre-computed avg_info_for_plot column
    info_col = config.info_measure
    if config.use_avg_info and "avg_info_for_plot" in per_spectrum_df.columns:
        info_col = "avg_info_for_plot"

    # Group and collect lists for each molecule (DataFrame APIs: group_by on an already-collected DataFrame).
    grouped = included.group_by("mol_idx").agg(
        [
            pl.col(info_col).alias("info_list"),
            pl.col("avg_tanimoto").alias("avg_tani_list"),
            pl.len().alias("n_points"),
        ]
    )

    results = []
    # Iterate using Python dicts to avoid static type-checker confusion with Polars' ExprList namespace.
    for d in grouped.to_dicts():
        mol_idx = d["mol_idx"]
        info_list = d.get("info_list", []) or []
        tani_list = d.get("avg_tani_list", []) or []
        # Convert to numpy and drop NaNs pairwise
        x = (
            np.asarray(info_list, dtype=float)
            if len(info_list) > 0
            else np.array([], dtype=float)
        )
        y = (
            np.asarray(tani_list, dtype=float)
            if len(tani_list) > 0
            else np.array([], dtype=float)
        )
        mask = ~np.isnan(x) & ~np.isnan(y)
        x = x[mask]
        y = y[mask]
        # If there are fewer than 2 data points, we cannot compute a correlation.
        if x.size < 2:
            rho, pval = float("nan"), float("nan")
            tanimoto_constant = False
        else:
            # If the avg_tanimoto (y) is constant across spectra for this molecule,
            # define the Spearman correlation to be 0 (user-requested behavior).
            # Use `np.allclose` to avoid floating-point equality issues on numeric arrays.
            if np.allclose(y, y[0]):
                rho, pval = 0.0, float("nan")
                tanimoto_constant = True
            else:
                rho, pval = spearmanr(x, y)
                # spearmanr can return nan in degenerate cases
                if np.isnan(rho):
                    rho, pval = float("nan"), float("nan")
                tanimoto_constant = False
        results.append(
            {
                "mol_idx": int(mol_idx),
                "spearman_rho": float(rho) if not np.isnan(rho) else float("nan"),
                "p_value": float(pval) if not np.isnan(pval) else float("nan"),
                "n_points": int(x.size),
                "tanimoto_constant": bool(tanimoto_constant),
            }
        )

    return pl.DataFrame(results)


def _compute_binned_stats(
    df: pl.DataFrame, config: SimilarityVsInfoConfig
) -> Dict[str, object]:
    """
    Compute Spearman correlation and aggregated binned statistics for a dataframe
    that contains `config.info_measure` and `avg_tanimoto`. This helper does not
    produce any plots; it simply returns the computed stats for further use.
    """
    # Prepare arrays for Spearman (ensure numpy float dtype for numeric operations)
    x = np.asarray(df.get_column(config.info_measure).to_numpy(), dtype=float)
    y = np.asarray(df.get_column("avg_tanimoto").to_numpy(), dtype=float)
    mask = (~np.isnan(x)) & (~np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]

    if x_clean.size < 2:
        logger.error(
            "Not enough valid pairs to compute Spearman correlation (need >= 2). Found: %s",
            x_clean.size,
        )
        raise AssertionError(
            "Not enough valid pairs to compute Spearman correlation (need >= 2)."
        )
    # Compute Spearman and fail if the result is not finite (degenerate inputs).
    rho, pval = spearmanr(x_clean, y_clean)
    if np.isnan(rho):
        logger.error("Spearman computation returned NaN (degenerate inputs).")
        raise AssertionError("Spearman computation resulted in NaN; check inputs.")

    logger.info("Raw Spearman: rho=%s p=%s (n=%s)", rho, pval, x_clean.size)

    # Binning (numpy-based)
    x_min = float(config.x_range[0])
    x_max = float(config.x_range[1])

    x_arr = np.asarray(df.get_column(config.info_measure).to_numpy(), dtype=float)
    y_arr = np.asarray(df.get_column("avg_tanimoto").to_numpy(), dtype=float)
    valid = (~np.isnan(x_arr)) & (~np.isnan(y_arr))

    if not np.any(valid):
        logger.error(
            "No valid (non-NaN) pairs of information and avg_tanimoto found; cannot compute binned statistics."
        )
        raise AssertionError(
            "No valid (non-NaN) pairs of information and avg_tanimoto found; check inputs and filters."
        )
    else:
        x_arr = x_arr[valid]
        y_arr = y_arr[valid]

        bin_vals = np.floor(x_arr / float(config.x_bin_width)) * float(
            config.x_bin_width
        )

        in_range = (bin_vals >= x_min) & (bin_vals <= x_max)
        if not np.any(in_range):
            logger.error(
                "No data points fall within the configured x_range (%s, %s).",
                x_min,
                x_max,
            )
            raise AssertionError(
                f"No data points in info measure range {x_min}..{x_max}; adjust x_range or check inputs."
            )
        else:
            bin_vals = bin_vals[in_range]
            y_vals = y_arr[in_range]

            # Temporary dataframe not required here; we'll aggregate directly from numpy arrays.
            # Compute per-bin aggregates using numpy grouping so we don't rely on
            # Polars list-aggregation APIs (which can differ across Polars versions).
            pct_bounds = getattr(config, "percentile_bounds", None)
            low_pct = high_pct = None
            if pct_bounds is not None:
                # Validate percentile_bounds: expect a (low, high) tuple in 0..100 with low < high.
                assert isinstance(pct_bounds, (tuple, list)) and len(pct_bounds) == 2, (
                    "percentile_bounds must be a tuple (low_pct, high_pct) in 0..100 or None"
                )
                low_pct, high_pct = pct_bounds
                # Cast to float for downstream usage and static-type compatibility.
                low_pct = float(low_pct)
                high_pct = float(high_pct)
                assert 0.0 <= low_pct < high_pct <= 100.0, (
                    "percentile_bounds must satisfy 0 <= low < high <= 100"
                )
            # Optional single percentile (0-100) to draw as a dashed line (same color as mean)
            single_lower_pct = getattr(config, "lower_percentile", None)
            if single_lower_pct is not None:
                # Cast to float to satisfy static type checks and guarantee a proper numeric
                # value is supplied to `np.percentile`.
                single_lower_pct = float(single_lower_pct)
                assert 0.0 <= single_lower_pct <= 100.0, (
                    "lower_percentile must be between 0 and 100"
                )

            rows = []
            unique_bins = np.unique(bin_vals)
            for info_val in np.sort(unique_bins):
                mask = bin_vals == info_val
                vals = y_vals[mask]
                if vals.size == 0:
                    rows.append(
                        {
                            "info_bin_val": float(info_val),
                            "avg_tanimoto": float("nan"),
                            "median_tanimoto": float("nan"),
                            "lower_percentile": float("nan"),
                            "upper_percentile": float("nan"),
                            "lower_percentile_line": float("nan"),
                            "count": 0,
                        }
                    )
                else:
                    lower = (
                        float(np.percentile(vals, low_pct))
                        if low_pct is not None
                        else float("nan")
                    )
                    upper = (
                        float(np.percentile(vals, high_pct))
                        if high_pct is not None
                        else float("nan")
                    )
                    single_lower = (
                        float(np.percentile(vals, single_lower_pct))
                        if single_lower_pct is not None
                        else float("nan")
                    )
                    rows.append(
                        {
                            "info_bin_val": float(info_val),
                            "avg_tanimoto": float(np.nanmean(vals)),
                            "median_tanimoto": float(np.nanmedian(vals)),
                            "lower_percentile": lower,
                            "upper_percentile": upper,
                            "lower_percentile_line": single_lower,
                            "count": int(vals.size),
                        }
                    )

            binned = pl.DataFrame(rows).sort("info_bin_val")

    return {
        "rho": float(rho) if not np.isnan(rho) else float("nan"),
        "p_value": float(pval) if not np.isnan(pval) else float("nan"),
        "n_points": int(x_clean.size),
        "binned_stats": binned,
    }


def _plot_binned_stats(
    primary_stats: Dict[str, object],
    overlay_stats: Dict[str, Dict[str, object]],
    out_path: Path,
    label: str,
    config: SimilarityVsInfoConfig,
) -> None:
    """
    Produce a binned plot from pre-computed statistics and save to `out_path`.

    Args:
        primary_stats: dict containing 'binned_stats' DataFrame for the primary series.
        overlay_stats: dict mapping label -> stats dict (with 'binned_stats') for overlays.
        out_path: where to save the figure.
        label: legend label for the primary series.
        config: configuration object.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")

    # Plot primary series. Require that `_compute_binned_stats` returned a valid,
    # non-empty `binned_stats` DataFrame; do not silently fall back to an empty DF.
    assert "binned_stats" in primary_stats, (
        "primary_stats must contain 'binned_stats' produced by _compute_binned_stats"
    )
    plot_df = primary_stats["binned_stats"]
    assert isinstance(plot_df, pl.DataFrame), (
        "primary_stats['binned_stats'] must be a Polars DataFrame"
    )
    # Ensure there is at least one binned row to consider before applying the count filter.
    assert plot_df.height > 0, (
        f"primary binned_stats is empty for {label}; no valid binned points"
    )
    plot_df = plot_df.filter(pl.col("count") > config.min_count_threshold)
    plotted_any = False
    if plot_df.height > 0:
        xvals = plot_df.get_column("info_bin_val").to_numpy()
        yvals = plot_df.get_column("avg_tanimoto").to_numpy()
        line = ax.plot(
            xvals, yvals, marker="o", linestyle="-", linewidth=2.0, label=str(label)
        )[0]
        # Draw percentile span (if available) using the same color as the line.
        if (
            "lower_percentile" in plot_df.columns
            and "upper_percentile" in plot_df.columns
        ):
            lower = plot_df.get_column("lower_percentile").to_numpy()
            upper = plot_df.get_column("upper_percentile").to_numpy()
            mask = (~np.isnan(lower)) & (~np.isnan(upper))
            if np.any(mask):
                ax.fill_between(
                    xvals[mask],
                    lower[mask],
                    upper[mask],
                    color=line.get_color(),
                    alpha=0.5,
                    linewidth=0,
                    zorder=line.get_zorder() - 1,
                )
        # Draw single dashed percentile line (if requested) using same color as the line.
        single_pct = getattr(config, "lower_percentile", None)
        if single_pct is not None and "lower_percentile_line" in plot_df.columns:
            single_vals = plot_df.get_column("lower_percentile_line").to_numpy()
            mask_single = ~np.isnan(single_vals)
            if np.any(mask_single):
                ax.plot(
                    xvals[mask_single],
                    single_vals[mask_single],
                    linestyle="--",
                    color=line.get_color(),
                    alpha=0.7,
                )
        plotted_any = True
    else:
        logger.error("No binned points above min_count_threshold for %s", label)
        # Fail fast when the primary series has no valid binned points above the
        # configured threshold. The caller should correct input data or thresholds.
        raise AssertionError(f"No binned points above min_count_threshold for {label}")

    # Plot overlay series (if any)
    # Use a color cycle to ensure distinct colors
    for i, (lab, stats) in enumerate(overlay_stats.items()):
        # Require overlay stats to contain a valid, non-empty `binned_stats` DataFrame.
        assert "binned_stats" in stats, f"Overlay series '{lab}' missing 'binned_stats'"
        sdf = stats["binned_stats"]
        assert isinstance(sdf, pl.DataFrame), (
            f"Overlay series '{lab}' binned_stats must be a Polars DataFrame"
        )
        # Ensure overlay has some binned rows (before applying the count filter).
        assert sdf.height > 0, (
            f"Overlay series '{lab}' binned_stats is empty; no valid binned points"
        )
        sdf = sdf.filter(pl.col("count") > config.min_count_threshold)
        if sdf.height > 0:
            xvals = sdf.get_column("info_bin_val").to_numpy()
            yvals = sdf.get_column("avg_tanimoto").to_numpy()
            line = ax.plot(
                xvals,
                yvals,
                marker="o",
                linestyle="-",
                linewidth=2.0,
                alpha=1.0,
                label=str(lab),
            )[0]
            # Draw percentile span (if available) in the same color as the line.
            if "lower_percentile" in sdf.columns and "upper_percentile" in sdf.columns:
                lower = sdf.get_column("lower_percentile").to_numpy()
                upper = sdf.get_column("upper_percentile").to_numpy()
                mask = (~np.isnan(lower)) & (~np.isnan(upper))
                if np.any(mask):
                    ax.fill_between(
                        xvals[mask],
                        lower[mask],
                        upper[mask],
                        color=line.get_color(),
                        alpha=0.5,
                        linewidth=0,
                        zorder=line.get_zorder() - 1,
                    )
            # Draw single dashed percentile line for overlays (if requested).
            single_pct = getattr(config, "lower_percentile", None)
            if single_pct is not None and "lower_percentile_line" in sdf.columns:
                single_vals = sdf.get_column("lower_percentile_line").to_numpy()
                mask_single = ~np.isnan(single_vals)
                if np.any(mask_single):
                    ax.plot(
                        xvals[mask_single],
                        single_vals[mask_single],
                        linestyle="--",
                        color=line.get_color(),
                        alpha=0.7,
                    )
            plotted_any = True
        else:
            logger.error(
                "Overlay series %s had no binned points above min_count_threshold", lab
            )
            raise AssertionError(
                f"Overlay series '{lab}' had no binned points above min_count_threshold"
            )

    if not plotted_any:
        logger.error("No data to plot for %s (primary and overlays empty)", label)
        # Fail fast when nothing is available to plot.
        raise AssertionError(
            f"No data to plot for {label} (primary and overlays empty)"
        )

    ax.set_xlabel("Spectral Information Score")
    ax.set_ylabel("Average Tanimoto Similarity of Matches")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white", transparent=False)
    plt.close(fig)


def plot_heatmap_avg_tanimoto_vs_info_and_dotprod(
    config: SimilarityVsInfoConfig,
    library_scores_df: Optional[pl.DataFrame] = None,
) -> None:
    """
    Create a 2D heatmap where:
      - x axis: informativity (binned by `config.x_bin_width`)
      - y axis: dot-prod similarity (binned by `config.dotprod_bin_size`)
      - color: average Tanimoto similarity of matches inside each (x_bin, y_bin)

    If `min_dotprod` is provided, only pairs with dot-prod >= that value are included.
    """
    pairs_path = Path(config.pairs_parquet_path)
    left_lib_path = Path(config.left_library_parquet_path)
    assert pairs_path.exists(), f"Pairs parquet not found: {pairs_path}"
    assert left_lib_path.exists(), f"Left library parquet not found: {left_lib_path}"

    # Validate columns exist
    pairs_scan = pl.scan_parquet(str(pairs_path))
    _require_columns_or_fail(
        pairs_scan,
        [
            config.left_idx_col,
            config.right_idx_col,
            config.left_mol_col,
            config.right_mol_col,
            config.dotprod_col,
            config.tanimoto_col,
        ],
        "pairs parquet (for heatmap)",
    )

    # Create bidirectional pairs stream (A->B and B->A)
    # Why: Analysis cares about matches in both directions (e.g. 1000->990 and 990->1000).
    # Filter self-matches (matches to the same molecule).
    pairs_scan = pairs_scan.filter(
        pl.col(config.left_mol_col) != pl.col(config.right_mol_col)
    )

    # Create bidirectional pairs stream (A->B and B->A)
    # Why: Analysis cares about matches in both directions (e.g. 1000->990 and 990->1000).
    # Filter self-matches (matches to the same molecule).
    pairs_scan = pairs_scan.filter(
        pl.col(config.left_mol_col) != pl.col(config.right_mol_col)
    )

    if library_scores_df is not None:
        logger.info("Using provided library_scores_df for heatmap.")
        left_lib_lf = library_scores_df.lazy().select(
            [pl.col("idx").cast(pl.Int64), pl.col(config.info_measure)]
        )
        right_lib_lf = left_lib_lf
    elif config.left_library_full_parquet_path is not None:
        logger.info(
            "Recomputing scores for heatmap because full library path provided.",
        )
        left_lib_df = _load_full_library_and_recompute_scores(
            snapshot_path=left_lib_path,
            full_library_path=Path(config.left_library_full_parquet_path),
            metric=config.info_metric,
            info_measure_col=config.info_measure,
        )
        left_lib_lf = left_lib_df.lazy().select(
            [pl.col("idx").cast(pl.Int64), pl.col(config.info_measure)]
        )
        if right_lib_path != left_lib_path:
            assert config.right_library_full_parquet_path is not None, (
                "right_library_full_parquet_path required for cross-library heatmap"
            )
            right_lib_df = _load_full_library_and_recompute_scores(
                snapshot_path=right_lib_path,
                full_library_path=Path(config.right_library_full_parquet_path),
                metric=config.info_metric,
                info_measure_col=config.info_measure,
            )
            right_lib_lf = right_lib_df.lazy().select(
                [pl.col("idx").cast(pl.Int64), pl.col(config.info_measure)]
            )
        else:
            right_lib_lf = left_lib_lf
    else:
        left_lib_lf = pl.scan_parquet(str(left_lib_path)).select(
            [pl.col("idx").cast(pl.Int64), pl.col(config.info_measure)]
        )
        right_lib_lf = left_lib_lf

    # Prepare library info for joining (aliased) to get info for both sides of the match
    # Cast info to Float32 early to save memory
    lib_lf_left = left_lib_lf.select(
        [
            pl.col("idx").cast(pl.Int64),
            pl.col(config.info_measure).cast(pl.Float32),
        ]
    )
    lib_lf_right = right_lib_lf.select(
        [
            pl.col("idx").cast(pl.Int64),
            pl.col(config.info_measure).cast(pl.Float32),
        ]
    )

    if config.use_avg_info:
        # Case 1: Use average info of the pair.
        # We process the pairs directly (no explosion needed if we consider the match as the unit).
        # We join info for both left and right, compute average, and project.

        # Aliases for joining
        lib_left = lib_lf_left.select(
            [
                pl.col("idx").alias(config.left_idx_col),
                pl.col(config.info_measure).alias("info_left"),
            ]
        )
        lib_right = lib_lf_right.select(
            [
                pl.col("idx").alias(config.right_idx_col),
                pl.col(config.info_measure).alias("info_right"),
            ]
        )

        agg_input = (
            pairs_scan.select(
                [
                    pl.col(config.left_idx_col).cast(pl.Int64),
                    pl.col(config.right_idx_col).cast(pl.Int64),
                    pl.col(config.dotprod_col).cast(pl.Float32),
                    pl.col(config.tanimoto_col).cast(pl.Float32),
                ]
            )
            .join(lib_left, on=config.left_idx_col, how="left")
            .join(lib_right, on=config.right_idx_col, how="left")
            .select(
                [
                    pl.col(config.dotprod_col),
                    pl.col(config.tanimoto_col),
                    ((pl.col("info_left") + pl.col("info_right")) * 0.5)
                    .cast(pl.Float32)
                    .alias(config.info_measure),
                ]
            )
        )

    else:
        # Case 2: Use individual spectrum info.
        # We need to explode the pairs (Left->Right and Right->Left) so each spectrum in the pair
        # gets plotted against its own info score.
        # To save memory, we project minimal columns first, concat, AND THEN join info.

        p1 = pairs_scan.select(
            [
                pl.col(config.left_idx_col).cast(pl.Int64).alias("idx"),
                pl.col(config.dotprod_col).cast(pl.Float32),
                pl.col(config.tanimoto_col).cast(pl.Float32),
            ]
        )
        p2 = pairs_scan.select(
            [
                pl.col(config.right_idx_col).cast(pl.Int64).alias("idx"),
                pl.col(config.dotprod_col).cast(pl.Float32),
                pl.col(config.tanimoto_col).cast(pl.Float32),
            ]
        )

        agg_input_left = (
            p1.join(lib_lf_left, on="idx", how="left")
            .select(
                [
                    pl.col(config.dotprod_col),
                    pl.col(config.tanimoto_col),
                    pl.col(config.info_measure),
                ]
            )
        )
        agg_input_right = (
            p2.join(lib_lf_right, on="idx", how="left")
            .select(
                [
                    pl.col(config.dotprod_col),
                    pl.col(config.tanimoto_col),
                    pl.col(config.info_measure),
                ]
            )
        )
        agg_input = pl.concat([agg_input_left, agg_input_right])

    # Common filtering and aggregation logic
    joined = (
        agg_input.drop_nans()
        .drop_nulls()
        .filter(
            pl.col(config.info_measure).is_between(config.x_range[0], config.x_range[1])
        )
    )

    all_agg = (
        joined.with_columns(
            pl.col(config.info_measure)
            .truediv(config.x_bin_width)
            .round(decimals=0)
            .mul(config.x_bin_width)
            .cast(pl.Float32)
            .alias(config.info_measure),
            pl.col(config.dotprod_col)
            .truediv(config.dotprod_bin_size)
            .round(decimals=0)
            .mul(config.dotprod_bin_size)
            .cast(pl.Float32)
            .alias(config.dotprod_col),
        )
        .group_by([config.info_measure, config.dotprod_col])
        .mean()
    )

    all_df = all_agg.collect(engine="streaming")

    _plot_heatmap(
        all_df,
        config,
        filename=config.filename_heatmap_all_spectra.format(
            measure=config.info_measure
        ),
    )
    # now we plot a heatmap for each specific case
    # for the mos tinformative spectrum per molecule


def _plot_heatmap(
    frame: pl.DataFrame,
    config: SimilarityVsInfoConfig,
    cmap: str = "viridis",
    colorbar: bool = True,
    xlabel: str = "Spectral Information Score",
    ylabel: str = "Dot-Product Similarity",
    title: str = "",
    width: int = 600,
    height: int = 400,
    filename: Optional[str] = None,
) -> None:
    """
    Render a 2D heatmap using matplotlib from a binned dataframe with columns:
      - x: `config.info_measure` (binned centers)
      - y: `config.dotprod_col` (binned centers)
      - z: `config.tanimoto_col` (aggregated value to color with)

    The plot respects your original arguments: `cmap`, `colorbar`, `xlabel`,
    `ylabel`, `title` (label), `width`, and `height`. The output file will be
    written to `config.output_dir / filename` where `filename` defaults to
    `config.filename_heatmap`.
    """
    if filename is None:
        filename = config.filename_heatmap.format(measure=config.info_measure)

    df = frame.to_pandas()
    if df.empty:
        logger.error("No data for heatmap %s; aborting.", filename)
        # Fail fast: no data to visualize for this heatmap.
        raise AssertionError(f"No data for heatmap {filename}")

    # Pivot so rows -> y (dotprod), columns -> x (info measure), values -> z (tanimoto)
    pivot = df.pivot(
        index=config.dotprod_col,
        columns=config.info_measure,
        values=config.tanimoto_col,
    )
    # Ensure consistent ordering
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)

    # Extract centers and the Z matrix
    x_centers = pivot.columns.to_numpy(dtype=float)
    y_centers = pivot.index.to_numpy(dtype=float)
    Z = pivot.to_numpy()

    # Compute bin edges from centers using config bin widths
    x_half = float(config.x_bin_width) / 2.0
    y_half = float(config.dotprod_bin_size) / 2.0
    x_edges = np.concatenate([x_centers - x_half, [x_centers[-1] + x_half]])
    y_edges = np.concatenate([y_centers - y_half, [y_centers[-1] + y_half]])

    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    mesh = ax.pcolormesh(x_edges, y_edges, Z, shading="auto", cmap=cmap)
    if colorbar:
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label("Average Tanimoto Similarity")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    # place ticks at bin centers for clarity
    ax.set_xticks(x_centers)
    ax.set_yticks(y_centers)
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])
    out_path = Path(config.output_dir) / filename
    fig.tight_layout()
    fig.savefig(str(out_path), facecolor="white", transparent=False)
    plt.close(fig)


def plot_correlation_dotprod_tanimoto_vs_info(
    config: SimilarityVsInfoConfig,
    library_scores_df: Optional[pl.DataFrame] = None,
) -> None:
    """
    Compute and plot the rolling correlation (Spearman and Pearson) between dot-product and
    Tanimoto similarity, sorted by the average spectral information of the pair.

    The x-axis represents the percentile of the information score.
    """
    pairs_path = Path(config.pairs_parquet_path)
    left_lib_path = Path(config.left_library_parquet_path)
    assert pairs_path.exists(), f"Pairs parquet not found: {pairs_path}"
    assert left_lib_path.exists(), f"Left library parquet not found: {left_lib_path}"

    # Validate columns exist
    pairs_scan = pl.scan_parquet(str(pairs_path))
    _require_columns_or_fail(
        pairs_scan,
        [
            config.left_idx_col,
            config.right_idx_col,
            config.left_mol_col,
            config.right_mol_col,
            config.dotprod_col,
            config.tanimoto_col,
        ],
        "pairs parquet (for correlation plot)",
    )

    # Filter self-matches
    pairs_scan = (
        pairs_scan.filter(pl.col(config.left_mol_col) != pl.col(config.right_mol_col))
        .drop([config.left_mol_col, config.right_mol_col])
        # .cast({config.dotprod_col: pl.Float32, config.tanimoto_col: pl.Float32})
        .cast({config.dotprod_col: pl.Float64, config.tanimoto_col: pl.Float64})
    )

    # Prepare library info lazyframes for left and right sides
    if library_scores_df is not None:
        left_lib_lf = library_scores_df.lazy().select(
            [pl.col("idx").cast(pl.Int64), pl.col(config.info_measure)]
        )
        right_lib_lf = left_lib_lf
    elif config.left_library_full_parquet_path is not None:
        left_lib_df = _load_full_library_and_recompute_scores(
            snapshot_path=left_lib_path,
            full_library_path=Path(config.left_library_full_parquet_path),
            metric=config.info_metric,
            info_measure_col=config.info_measure,
        )
        left_lib_lf = left_lib_df.lazy().select(
            [pl.col("idx").cast(pl.Int64), pl.col(config.info_measure)]
        )
        if right_lib_path != left_lib_path:
            assert config.right_library_full_parquet_path is not None, (
                "right_library_full_parquet_path required for cross-library correlation plot"
            )
            right_lib_df = _load_full_library_and_recompute_scores(
                snapshot_path=right_lib_path,
                full_library_path=Path(config.right_library_full_parquet_path),
                metric=config.info_metric,
                info_measure_col=config.info_measure,
            )
            right_lib_lf = right_lib_df.lazy().select(
                [pl.col("idx").cast(pl.Int64), pl.col(config.info_measure)]
            )
        else:
            right_lib_lf = left_lib_lf
    else:
        left_lib_lf = pl.scan_parquet(str(left_lib_path)).select(
            [pl.col("idx").cast(pl.Int64), pl.col(config.info_measure)]
        )
        right_lib_lf = left_lib_lf

    # Create distinct lazyframes for left and right joins with explicit aliasing
    lib_lf_left = left_lib_lf.select(
        [
            pl.col("idx"),
            pl.col(config.info_measure).alias("info_left"),
        ]
    )
    lib_lf_right = right_lib_lf.select(
        [
            pl.col("idx"),
            pl.col(config.info_measure).alias("info_right"),
        ]
    )

    # Join to get info for LEFT side
    lf_joined = pairs_scan.join(
        lib_lf_left,
        left_on=config.left_idx_col,
        right_on="idx",
        how="left",
    )

    # Join to get info for RIGHT side
    # Note: 'idx' column from lib_lf_left might be present, but we join on right_idx_col vs idx from lib_lf_right
    lf_joined = lf_joined.join(
        lib_lf_right,
        left_on=config.right_idx_col,
        right_on="idx",
        how="left",
    )
    print(lf_joined.collect_schema())

    df_stats = (
        lf_joined.select(
            [
                config.dotprod_col,
                config.tanimoto_col,
                ((pl.col("info_left") + pl.col("info_right")) / 2.0).alias("avg_info"),
            ]
        )
        .drop_nulls()
        .filter(pl.col("avg_info").is_between(config.x_range[0], config.x_range[1]))
        .with_columns(
            pl.col("avg_info")
            .truediv(config.x_bin_width)
            .round(decimals=0)
            .mul(config.x_bin_width)
            .alias("binned_info")
        )
        .drop("avg_info")
    )
    print(df_stats.collect_schema())
    df_stats = (
        df_stats.group_by("binned_info")
        .agg(
            pl.corr(
                config.dotprod_col,
                config.tanimoto_col,
                method="spearman",
            ).alias("spearman_correlation"),
            pl.corr(
                config.dotprod_col,
                config.tanimoto_col,
                method="pearson",
            ).alias("pearson_correlation"),
        )
        .collect(engine="streaming")
    )

    def _plot_and_save(
        df: pl.DataFrame, y_col: str, filename_template: str, method_label: str
    ) -> None:
        if df.height == 0:
            logger.warning("No data for %s correlation plot.", method_label)
            return

        if y_col not in df.columns:
            logger.error(
                "Correlation column '%s' missing for %s. Columns: %s",
                y_col,
                method_label,
                df.columns,
            )
            return

        # Sort by bin
        df_sorted = df.sort("binned_info")
        x_vals = df_sorted.get_column("binned_info").to_numpy()
        y_vals = df_sorted.get_column(y_col).to_numpy()

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
        ax.plot(
            x_vals,
            y_vals,
            marker="o",
            linestyle="-",
            label=f"{method_label} Correlation",
        )
        ax.set_xlabel("Spectral Information Score")
        ax.set_ylabel(f"{method_label} Correlation (Dot-Product vs Tanimoto)")
        ax.set_title(f"{method_label} Correlation vs Information Score")
        ax.grid(True, alpha=0.3)
        ax.legend()

        out_filename = filename_template.format(measure=config.info_measure)
        out_path = Path(config.output_dir) / out_filename
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fig.tight_layout()
        fig.savefig(str(out_path), facecolor="white")
        plt.close(fig)
        logger.info("Saved %s correlation plot to %s", method_label, out_path)

    _plot_and_save(
        df_stats,
        "spearman_correlation",
        config.filename_correlation_spearman,
        "Spearman",
    )
    _plot_and_save(
        df_stats,
        "pearson_correlation",
        config.filename_correlation_pearson,
        "Pearson",
    )


def run_per_molecule_analysis(config: SimilarityVsInfoConfig) -> None:
    """
    Perform per-molecule Spearman correlation analysis.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Per-Molecule Analysis...")

    # 1) Per-spectrum average tanimoto (includes self-match)
    logger.info("Computing per-spectrum average Tanimoto similarity...")
    per_spectrum_df = compute_per_spectrum_avg_tanimoto(config)

    # Optionally persist per-spectrum summaries
    per_spec_out = output_dir / config.per_spectrum_summary_name
    per_spectrum_df.write_parquet(str(per_spec_out))
    logger.info("Wrote per-spectrum summary to %s", per_spec_out)

    # 2) Determine which molecules had any matches
    mols_with_matches = _molecules_with_any_matches(config.pairs_parquet_path, config)
    logger.info(
        "Found %d molecules with at least one external match", len(mols_with_matches)
    )
    if len(mols_with_matches) == 0:
        logger.error(
            "No molecules with external matches found. Aborting per-molecule analysis."
        )
        return

    # 3) Per-molecule Spearman
    logger.info("Computing per-molecule Spearman correlations...")
    per_molecule_spearman_df = compute_per_molecule_spearman(
        per_spectrum_df, mols_with_matches, config
    )

    per_mol_out = output_dir / config.per_molecule_summary_name
    per_molecule_spearman_df.write_parquet(str(per_mol_out))
    logger.info("Wrote per-molecule spearman summary to %s", per_mol_out)

    # Generate Text Report
    rho_arr = (
        per_molecule_spearman_df.get_column("spearman_rho").to_numpy()
        if per_molecule_spearman_df.height > 0
        else np.array([], dtype=float)
    )
    tan_const_arr = (
        per_molecule_spearman_df.get_column("tanimoto_constant").to_numpy()
        if "tanimoto_constant" in per_molecule_spearman_df.columns
        else np.zeros_like(rho_arr, dtype=bool)
    )

    valid_mask = ~np.isnan(rho_arr)
    n_valid = int(np.sum(valid_mask))

    if n_valid > 0:
        mean_with = float(np.nanmean(rho_arr))
        n_with = int(np.sum(valid_mask))
        exclude_mask = valid_mask & (~tan_const_arr)
        n_excl = int(np.sum(exclude_mask))
        mean_without = (
            float(np.nanmean(rho_arr[exclude_mask])) if n_excl > 0 else float("nan")
        )

        summary_path = output_dir / "per_molecule_spearman_summary.txt"
        with open(summary_path, "w") as fh:
            fh.write("Per-molecule Spearman summary\n\n")
            fh.write(
                f"Mean (incl constant avg_tanimoto): {mean_with:.4f} (n={n_with})\n"
            )
            fh.write(
                f"Mean (excl constant avg_tanimoto): {mean_without:.4f} (n={n_excl})\n"
            )
        logger.info("Wrote summary text to %s", summary_path)


def run_global_line_plots(config: SimilarityVsInfoConfig) -> None:
    """
    Compute and plot global and best-per-molecule binned statistics across thresholds.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Global Line Plots Analysis...")

    thresholds: List[Optional[float]] = (
        [float(t) for t in config.dotprod_thresholds]
        if getattr(config, "dotprod_thresholds", None)
        else [None]
    )

    per_spectrum_by_threshold: Dict[Optional[float], pl.DataFrame] = {}
    for t in thresholds:
        logger.info("Computing per-spectrum stats for threshold %s...", t)
        per_spectrum_by_threshold[t] = compute_per_spectrum_avg_tanimoto(
            config, min_dotprod=t
        )

    # --- 1. Global (All Spectra) ---
    global_by_threshold: Dict[str, Dict[str, object]] = {}
    for t, ps in per_spectrum_by_threshold.items():
        mols_with_matches_t = _molecules_with_any_matches(
            config.pairs_parquet_path, config, min_dotprod=t
        )
        included_spectra_t = ps.filter(
            pl.col("mol_idx").is_in(list(mols_with_matches_t))
        )
        if config.use_avg_info and "avg_info_for_plot" in included_spectra_t.columns:
            included_spectra_t = included_spectra_t.with_columns(
                pl.col("avg_info_for_plot").alias(config.info_measure)
            )
        label = (
            f"Dot-Product above {t:.2f}" if t is not None else "Dot-Product (no filter)"
        )
        global_by_threshold[label] = _compute_binned_stats(included_spectra_t, config)

    primary_t = thresholds[0]
    primary_label = (
        f"Dot-Product above {primary_t:.2f}"
        if primary_t is not None
        else "Dot-Product (no filter)"
    )

    if primary_label in global_by_threshold:
        plot_all_path = output_dir / config.filename_all.format(
            measure=config.info_measure
        )
        _plot_binned_stats(
            primary_stats=global_by_threshold[primary_label],
            overlay_stats={
                k: v for k, v in global_by_threshold.items() if k != primary_label
            },
            out_path=plot_all_path,
            label=primary_label,
            config=config,
        )

    # --- 2. Best Spectrum per Molecule ---
    best_by_threshold: Dict[str, Dict[str, object]] = {}
    for t, ps in per_spectrum_by_threshold.items():
        mols_with_matches_t = _molecules_with_any_matches(
            config.pairs_parquet_path, config, min_dotprod=t
        )
        included_spectra_t = ps.filter(
            pl.col("mol_idx").is_in(list(mols_with_matches_t))
        )
        best_info = included_spectra_t.group_by("mol_idx").agg(
            pl.col(config.info_measure).max().alias("max_info")
        )
        candidates = included_spectra_t.join(
            best_info, on="mol_idx", how="inner"
        ).filter(pl.col(config.info_measure) == pl.col("max_info"))
        best_per_molecule_t = (
            candidates.sort(["mol_idx", "idx"])
            .group_by("mol_idx")
            .agg(pl.col("idx").first().alias("idx"))
            .join(included_spectra_t, on=["mol_idx", "idx"], how="left")
        )

        if config.use_avg_info and "avg_info_for_plot" in best_per_molecule_t.columns:
            best_per_molecule_t = best_per_molecule_t.with_columns(
                pl.col("avg_info_for_plot").alias(config.info_measure)
            )

        label = (
            f"Dot-Product above {t:.2f}" if t is not None else "Dot-Product (no filter)"
        )
        best_by_threshold[label] = _compute_binned_stats(best_per_molecule_t, config)

    if primary_label in best_by_threshold:
        plot_best_path = output_dir / config.filename_best.format(
            measure=config.info_measure
        )
        _plot_binned_stats(
            primary_stats=best_by_threshold[primary_label],
            overlay_stats={
                k: v for k, v in best_by_threshold.items() if k != primary_label
            },
            out_path=plot_best_path,
            label=primary_label,
            config=config,
        )


if __name__ == "__main__":
    # Workflow A: use pre-computed scores from the snapshot.
    # left_library_full_parquet_path=None means "use the scores already present
    # in left_library_parquet_path"; no in-memory recomputation is performed.
    cfg = SimilarityVsInfoConfig(
        pairs_parquet_path=Path(
            "/home/analytit_admin/Data/spectral_libs/info_score/combined_library_pairs_with_tanimoto_260104.parquet"
        ),
        left_library_parquet_path=Path(
            "/home/analytit_admin/Data/spectral_libs/info_score/combined_library_recomputed_info.parquet"
        ),
        left_library_full_parquet_path=None,
        right_library_parquet_path=None,
        right_library_full_parquet_path=None,
        tanimoto_col="tanimoto_similarity",
        left_idx_col="idx",
        right_idx_col="idx_right",
        left_mol_col="mol_idx",
        right_mol_col="mol_idx_right",
        info_metric=InfoMetric.SPECTRAL_INFORMATION,
        x_bin_width=1.0,
        x_range=(0.0, 10.0),
        min_count_threshold=10,
        output_dir=Path(
            "/home/analytit_admin/Data/spectral_libs/info_score/sim_vs_info_analysis_260518"
        ),
        dotprod_thresholds=(0.8, 0.9),
        dotprod_bin_size=0.1,
        use_avg_info=True,
    )

    # 1. Per-Molecule Spearman Analysis
    run_per_molecule_analysis(cfg)

    # 2. Global Line Plots (All & Best)
    run_global_line_plots(cfg)

    # 3. Heatmap
    heatmap_fname = Path(cfg.filename_heatmap.format(measure=cfg.info_measure)).name
    heatmap_path = Path(cfg.output_dir) / heatmap_fname
    logger.info("Producing overall heatmap at %s", heatmap_path)

    plot_heatmap_avg_tanimoto_vs_info_and_dotprod(cfg)
