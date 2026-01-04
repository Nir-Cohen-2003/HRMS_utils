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
from scipy.stats import spearmanr

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
      - tanimoto_col: name of Tanimoto similarity column in pairs parquet.
      - left_idx_col/right_idx_col: column names for the left/right spectrum global index.
      - left_mol_col/right_mol_col: column names for the left/right molecule id (mol_idx).
      - info_measure: column name in library that holds the informativity measure.
      - x_bin_width/x_range: binning configuration for plotting.
      - min_count_threshold: minimum bin count to include the point in the plot.
      - output_dir: where to write figures and optional CSVs.
    """

    pairs_parquet_path: Union[str, Path]
    left_library_parquet_path: Union[str, Path]
    right_library_parquet_path: Optional[Union[str, Path]] = None
    tanimoto_col: str = "tanimoto_similarity"
    left_idx_col: str = "idx"
    right_idx_col: str = "idx_right"
    left_mol_col: str = "mol_idx"
    right_mol_col: str = "mol_idx_right"
    info_measure: str = "spectral_information_score"
    x_bin_width: float = 0.5
    x_range: Tuple[float, float] = (0.0, 3.0)
    min_count_threshold: int = 10
    output_dir: Union[str, Path] = Path(".")
    # file names:
    filename_all: str = "avg_tanimoto_vs_{measure}_all.png"
    filename_best: str = "avg_tanimoto_vs_{measure}_best.png"
    filename_all_including_unmatched: str = (
        "avg_tanimoto_vs_{measure}_all_including_unmatched.png"
    )
    filename_best_including_unmatched: str = (
        "avg_tanimoto_vs_{measure}_best_including_unmatched.png"
    )
    per_molecule_summary_name: str = "per_molecule_spearman.parquet"
    per_spectrum_summary_name: str = "per_spectrum_tanimoto.parquet"


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


def compute_per_spectrum_avg_tanimoto(
    config: SimilarityVsInfoConfig,
) -> pl.DataFrame:
    """
    Compute per-spectrum average Tanimoto similarity to unique matched molecules.

    Steps:
      - Read the pairs parquet lazily and validate columns.
      - Build the matches as (idx, matched_mol_idx, tanimoto) from both directions
        (spectrum appearing on left or right of the pair).
      - Deduplicate matches per (idx, matched_mol_idx) taking the max tanimoto.
      - Add implicit self-matches (tanimoto = 1.0) for every spectrum from the library.
      - Aggregate per spectrum to compute average/median number of matched molecules.

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

    logger.info("Scanning pairs parquet: %s", pairs_path)
    pairs = pl.scan_parquet(str(pairs_path))

    # Validate pairs contain required columns.
    _require_columns_or_fail(
        pairs,
        [
            config.left_idx_col,
            config.right_idx_col,
            config.left_mol_col,
            config.right_mol_col,
            config.tanimoto_col,
        ],
        "pairs parquet",
    )

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
    _require_columns_or_fail(
        lib,
        ["idx", "mol_idx", "smiles", "ion_mode", config.info_measure],
        "library parquet(s)",
    )

    # Build matches from both directions. Use 'matched_mol_idx' as the canonical other-molecule id.
    left_matches = pairs.select(
        [
            pl.col(config.left_idx_col).alias("idx"),
            pl.col(config.right_mol_col).alias("matched_mol_idx"),
            pl.col(config.tanimoto_col).alias("tanimoto"),
        ]
    )
    right_matches = pairs.select(
        [
            pl.col(config.right_idx_col).alias("idx"),
            pl.col(config.left_mol_col).alias("matched_mol_idx"),
            pl.col(config.tanimoto_col).alias("tanimoto"),
        ]
    )

    # Aggregate per-direction in streaming mode and combine. Doing the per-direction
    # grouping first bounds memory use and avoids calling `union` on LazyFrame.
    left_agg = (
        left_matches.group_by(["idx", "matched_mol_idx"])
        .agg(pl.col("tanimoto").max().alias("tanimoto"))
        .collect(engine="streaming")
    )
    right_agg = (
        right_matches.group_by(["idx", "matched_mol_idx"])
        .agg(pl.col("tanimoto").max().alias("tanimoto"))
        .collect(engine="streaming")
    )
    if left_agg.height == 0 and right_agg.height == 0:
        # No cross-molecule matches; dedup_matches stays empty DataFrame.
        dedup_matches = left_agg
    else:
        union_df = pl.concat([left_agg, right_agg], how="vertical")
        dedup_matches = union_df.group_by(["idx", "matched_mol_idx"]).agg(
            pl.col("tanimoto").max().alias("tanimoto")
        )

    # Self-matches (every spectrum matched to its own molecule with similarity 1.0).
    self_matches = (
        lib.select([pl.col("idx"), pl.col("mol_idx").alias("matched_mol_idx")])
        .with_columns(pl.lit(1.0, dtype=pl.Float32).alias("tanimoto"))
        .collect(engine="streaming")
    )
    # print(self_matches.schema)
    # print(dedup_matches.schema)
    # Combine external matches (deduped) with self matches.
    all_matches = pl.concat(
        [dedup_matches.cast({"matched_mol_idx": pl.Int64}), self_matches],
        how="vertical",
    )

    # Collect the minimal library fields for joining below.
    lib_df = lib.select(["idx", "mol_idx", config.info_measure]).collect(
        engine="streaming"
    )

    # Aggregate per-spectrum stats
    per_stats = all_matches.group_by("idx").agg(
        pl.col("tanimoto").mean().alias("avg_tanimoto"),
        pl.col("tanimoto").median().alias("median_tanimoto"),
        pl.len().alias("n_matched_molecules"),
    )
    per_spectrum = per_stats.join(lib_df, on="idx", how="left").select(
        [
            "idx",
            "mol_idx",
            config.info_measure,
            "avg_tanimoto",
            "median_tanimoto",
            "n_matched_molecules",
        ]
    )

    # Ensure expected dtypes
    per_spectrum = per_spectrum.with_columns(
        pl.col("avg_tanimoto").cast(pl.Float64),
        pl.col("median_tanimoto").cast(pl.Float64),
        pl.col("n_matched_molecules").cast(pl.Int64),
    )

    return per_spectrum


def _molecules_with_any_matches(
    pairs_parquet_path: Union[str, Path], config: SimilarityVsInfoConfig
) -> set:
    """
    Return a set of molecule ids (mol_idx) that appear anywhere in the pairs file
    (either on left or right). This is used to define which molecules had any
    matches and therefore should be included in the cross-spectrum analyses.
    """
    pairs = pl.scan_parquet(str(pairs_parquet_path))

    _require_columns_or_fail(
        pairs,
        [config.left_mol_col, config.right_mol_col],
        "pairs parquet (for molecule presence check)",
    )

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

    if included.height == 0:
        logger.warning("No spectra for molecules with matches; returning empty result")
        return pl.DataFrame(
            {
                "mol_idx": pl.Series([], dtype=pl.Int64),
                "spearman_rho": pl.Series([], dtype=pl.Float64),
                "p_value": pl.Series([], dtype=pl.Float64),
                "n_points": pl.Series([], dtype=pl.Int64),
                # Whether the avg_tanimoto (y) was constant across spectra for this molecule
                "tanimoto_constant": pl.Series([], dtype=pl.Boolean),
            }
        )

    # Group and collect lists for each molecule (DataFrame APIs: group_by on an already-collected DataFrame).
    grouped = included.group_by("mol_idx").agg(
        [
            pl.col(config.info_measure).alias("info_list"),
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


def _compute_global_spearman_and_binned_plot(
    df: pl.DataFrame, config: SimilarityVsInfoConfig, out_path: Path, label: str
) -> Dict[str, object]:
    """
    Compute raw Spearman correlation on df (expects df contains columns
    config.info_measure and 'avg_tanimoto'), and produce a binned plot saved to `out_path`.

    Returns a dict with keys:
      - 'rho', 'p_value', 'n_points', 'binned_stats' (Polars DataFrame)
    """
    # Prepare arrays for Spearman
    x = df.get_column(config.info_measure).to_numpy()
    y = df.get_column("avg_tanimoto").to_numpy()
    mask = (~np.isnan(x)) & (~np.isnan(y))
    x = x[mask]
    y = y[mask]

    if x.size < 2:
        rho, pval = float("nan"), float("nan")
    else:
        rho, pval = spearmanr(x, y)

    logger.info("%s raw Spearman: rho=%s p=%s (n=%s)", label, rho, pval, x.size)

    # Binning (numpy-based) to avoid expression-type issues and to keep the
    # pipeline explicit and easy to reason about. `df` is a collected Polars DataFrame.
    x_min = float(config.x_range[0])
    x_max = float(config.x_range[1])

    # Extract arrays and drop NaNs
    x_arr = df.get_column(config.info_measure).to_numpy()
    y_arr = df.get_column("avg_tanimoto").to_numpy()
    valid = (~np.isnan(x_arr)) & (~np.isnan(y_arr))

    if not np.any(valid):
        # Empty binned frame
        binned = pl.DataFrame(
            {"info_bin_val": [], "avg_tanimoto": [], "median_tanimoto": [], "count": []}
        )
    else:
        x_arr = x_arr[valid]
        y_arr = y_arr[valid]

        # Compute bin values (floor to nearest bin width)
        bin_vals = np.floor(x_arr / float(config.x_bin_width)) * float(
            config.x_bin_width
        )

        # Filter bins to requested range
        in_range = (bin_vals >= x_min) & (bin_vals <= x_max)
        if not np.any(in_range):
            binned = pl.DataFrame(
                {
                    "info_bin_val": [],
                    "avg_tanimoto": [],
                    "median_tanimoto": [],
                    "count": [],
                }
            )
        else:
            bin_vals = bin_vals[in_range]
            y_vals = y_arr[in_range]

            # Build a small Polars DF and group to compute bin statistics
            bdf = pl.DataFrame({"info_bin_val": bin_vals, "avg_tanimoto": y_vals})
            binned = (
                bdf.group_by("info_bin_val")
                .agg(
                    [
                        pl.col("avg_tanimoto").mean().alias("avg_tanimoto"),
                        pl.col("avg_tanimoto").median().alias("median_tanimoto"),
                        pl.len().alias("count"),
                    ]
                )
                .sort("info_bin_val")
            )

    # Plot binned average tanimoto vs info
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = binned.filter(pl.col("count") > config.min_count_threshold)
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    if plot_df.height > 0:
        ax.plot(
            plot_df.get_column("info_bin_val").to_numpy(),
            plot_df.get_column("avg_tanimoto").to_numpy(),
            marker="o",
            linestyle="-",
        )
    else:
        logger.warning("No binned points above min_count_threshold for %s", label)

    ax.set_xlabel(config.info_measure)
    ax.set_ylabel("Average Tanimoto Similarity of Matches")
    ax.set_title(label)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white", transparent=False)
    plt.close(fig)

    return {
        "rho": float(rho) if not np.isnan(rho) else float("nan"),
        "p_value": float(pval) if not np.isnan(pval) else float("nan"),
        "n_points": int(x.size),
        "binned_stats": binned,
        "plot_path": out_path,
    }


def analyze_similarity_vs_info(
    config: SimilarityVsInfoConfig,
) -> Dict[str, Union[pl.DataFrame, Dict]]:
    """
    Perform the full analysis as requested:
     1) Per-molecule Spearman correlations (across spectra in that molecule).
     2) Global Spearman (all spectra belonging to molecules that had any match),
        plus binned plot saved to file (average tanimoto vs the info measure).
     3) Same as (2) but restricting to the single most informative spectrum per molecule.

    Returned dict contains:
      - 'per_spectrum': polars DataFrame with per-spectrum stats (avg_tanimoto etc.)
      - 'per_molecule_spearman': polars DataFrame with per-molecule Spearman rho/p-values
      - 'global': dict with keys from _compute_global_spearman_and_binned_plot
      - 'best_per_molecule': dict similar to 'global' but computed on best-spectrum-per-molecule
      - 'paths': dict with file paths written (plots and summary parquet names)
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Per-spectrum average tanimoto (includes self-match)
    logger.info("Computing per-spectrum average Tanimoto similarity...")
    per_spectrum_df = compute_per_spectrum_avg_tanimoto(config)

    # Optionally persist per-spectrum summaries for inspection
    per_spec_out = output_dir / config.per_spectrum_summary_name
    try:
        per_spectrum_df.write_parquet(str(per_spec_out))
        logger.info("Wrote per-spectrum summary to %s", per_spec_out)
    except Exception:
        logger.exception("Failed to write per-spectrum summary (continuing)")

    # 2) Determine which molecules had any matches in the pairs file.
    mols_with_matches = _molecules_with_any_matches(config.pairs_parquet_path, config)
    logger.info(
        "Found %d molecules with at least one external match", len(mols_with_matches)
    )

    # 3) Per-molecule Spearman (inside each molecule)
    logger.info("Computing per-molecule Spearman correlations...")
    per_molecule_spearman_df = compute_per_molecule_spearman(
        per_spectrum_df, mols_with_matches, config
    )

    # Save per-molecule results
    per_mol_out = output_dir / config.per_molecule_summary_name
    try:
        per_molecule_spearman_df.write_parquet(str(per_mol_out))
        logger.info("Wrote per-molecule spearman summary to %s", per_mol_out)
    except Exception:
        logger.exception("Failed to write per-molecule summary (continuing)")

    # Compute and print summary statistics (mean and std) of per-molecule Spearman rho.
    # If avg_tanimoto was constant for a molecule we set its Spearman rho to 0 and
    # mark it with `tanimoto_constant`. We report statistics both including these
    # forced zeros and excluding them, and also write the summary (including the
    # example dataframe) to a text file in the output directory.
    try:
        rho_arr = (
            per_molecule_spearman_df.get_column("spearman_rho").to_numpy()
            if per_molecule_spearman_df.height > 0
            else np.array([], dtype=float)
        )

        # Boolean mask for molecules where avg_tanimoto was constant (if column present)
        if "tanimoto_constant" in per_molecule_spearman_df.columns:
            tan_const_arr = per_molecule_spearman_df.get_column(
                "tanimoto_constant"
            ).to_numpy()
        else:
            tan_const_arr = np.zeros_like(rho_arr, dtype=bool)

        valid_mask = ~np.isnan(rho_arr)
        n_valid = int(np.sum(valid_mask))
        if n_valid == 0:
            logger.warning(
                "Per-molecule Spearman: no valid rho values present (all NaN)"
            )
            # Still write a small summary file to indicate nothing was available.
            summary_path = output_dir / "per_molecule_spearman_summary.txt"
            try:
                with open(summary_path, "w") as fh:
                    fh.write("Per-molecule Spearman summary\n\n")
                    fh.write("No valid Spearman rho values available (all NaN)\n")
                logger.info("Wrote per-molecule Spearman summary to %s", summary_path)
            except Exception:
                logger.exception(
                    "Failed to write per-molecule Spearman summary to file (continuing)"
                )
        else:
            # Stats including zeros (zeros were written for constant-tanimoto molecules)
            mean_with = float(np.nanmean(rho_arr))
            std_with = float(np.nanstd(rho_arr))
            n_with = int(np.sum(valid_mask))

            # Stats excluding molecules that had constant avg_tanimoto
            exclude_mask = valid_mask & (~tan_const_arr)
            n_excl = int(np.sum(exclude_mask))
            if n_excl > 0:
                mean_without = float(np.nanmean(rho_arr[exclude_mask]))
                std_without = float(np.nanstd(rho_arr[exclude_mask]))
            else:
                mean_without = float("nan")
                std_without = float("nan")

            logger.info(
                "Per-molecule Spearman rho (including zeros for constant avg_tanimoto): mean=%s std=%s (n=%s)",
                mean_with,
                std_with,
                n_with,
            )
            logger.info(
                "Per-molecule Spearman rho (excluding constant avg_tanimoto molecules): mean=%s std=%s (n=%s)",
                mean_without,
                std_without,
                n_excl,
            )

            # Examples are selected by sorting the valid per-molecule Spearman values.
            # Exact percentiles are not used here (we sample first, last and intermediate rows).

            # Reconstruct a minimal library mapping (mol_idx -> smiles, ion_mode)
            left_lib_path = Path(config.left_library_parquet_path)
            right_lib_path = (
                Path(config.right_library_parquet_path)
                if config.right_library_parquet_path is not None
                else left_lib_path
            )
            assert left_lib_path.exists(), (
                f"Left library parquet not found: {left_lib_path}"
            )
            assert right_lib_path.exists(), (
                f"Right library parquet not found: {right_lib_path}"
            )
            libs: List[pl.LazyFrame] = [
                pl.scan_parquet(str(left_lib_path)).filter(
                    pl.col("smiles").is_not_null()
                )
            ]
            if right_lib_path != left_lib_path:
                libs.append(
                    pl.scan_parquet(str(right_lib_path)).filter(
                        pl.col("smiles").is_not_null()
                    )
                )
            lib_df = (
                (pl.concat(libs) if len(libs) > 1 else libs[0])
                .select(["mol_idx", "smiles", "ion_mode"])
                .collect(engine="streaming")
            )

            # Reduce to one representative SMILES/ion_mode per molecule
            mol_map = lib_df.group_by("mol_idx").agg(
                [
                    pl.col("smiles").first().alias("smiles"),
                    pl.col("ion_mode").first().alias("ion_mode"),
                ]
            )

            # Work only with molecules that have a valid Spearman rho
            valid_df = per_molecule_spearman_df.filter(
                pl.col("spearman_rho").is_not_null()
            )
            n_valid_df = valid_df.height

            # Select examples by sorting the valid DF by Spearman rho and taking
            # the first, last, and several in-between positions determined by
            # the dataframe length. Exact percentiles aren't required, and this
            # guarantees distinct molecules for each selection.
            examples = []
            if n_valid_df > 0:
                sorted_df = valid_df.sort("spearman_rho")
                if n_valid_df == 1:
                    positions = [0]
                else:
                    max_i = n_valid_df - 1
                    pos_candidates = [
                        0,
                        int(max_i * 0.1),
                        int(max_i * 0.25),
                        int(max_i * 0.5),
                        int(max_i * 0.75),
                        int(max_i * 0.9),
                        max_i,
                    ]
                    # Deduplicate and clamp to valid indices
                    positions = sorted(
                        set([min(max_i, max(0, p)) for p in pos_candidates])
                    )

                # Convert to list-of-dicts for simple index access and build a small lookup
                sorted_dicts = sorted_df.to_dicts()
                mol_map_dict = {
                    d["mol_idx"]: (d["smiles"], d["ion_mode"])
                    for d in mol_map.to_dicts()
                }

                for pos in positions:
                    row = sorted_dicts[pos]
                    sel_mol = int(row["mol_idx"])
                    sel_rho = float(row["spearman_rho"])
                    approx_pct = (
                        0
                        if n_valid_df == 1
                        else int(round(100.0 * pos / (n_valid_df - 1)))
                    )
                    smiles, ion_mode = mol_map_dict.get(sel_mol, (None, None))
                    examples.append(
                        {
                            "percentile": int(approx_pct),
                            "percentile_value": float(sel_rho),
                            "mol_idx": sel_mol,
                            "spearman_rho": sel_rho,
                            "smiles": smiles,
                            "ion_mode": ion_mode,
                            "position": int(pos),
                        }
                    )

            # Count NaNs excluding molecules whose avg_tanimoto was constant
            rho_all = per_molecule_spearman_df.get_column("spearman_rho").to_numpy()
            tan_const_all = (
                per_molecule_spearman_df.get_column("tanimoto_constant").to_numpy()
                if "tanimoto_constant" in per_molecule_spearman_df.columns
                else np.zeros_like(rho_all, dtype=bool)
            )
            n_nans_excl_const = int(np.sum(np.isnan(rho_all) & (~tan_const_all)))
            logger.info(
                "Number of molecules with NaN Spearman (excluding constant avg_tanimoto): %s",
                n_nans_excl_const,
            )

            examples_df = (
                pl.DataFrame(examples) if len(examples) > 0 else pl.DataFrame()
            )

            # Write summary and examples to a text file in the output directory
            summary_path = output_dir / "per_molecule_spearman_summary.txt"
            try:
                with open(summary_path, "w") as fh:
                    fh.write("Per-molecule Spearman summary\n\n")
                    fh.write(
                        "Including zeros for molecules where avg_tanimoto was constant:\n"
                    )
                    fh.write(
                        f" mean = {mean_with:.6f}\n std = {std_with:.6f}\n n = {n_with}\n\n"
                    )
                    fh.write("Excluding molecules where avg_tanimoto was constant:\n")
                    if not np.isnan(mean_without):
                        fh.write(
                            f" mean = {mean_without:.6f}\n std = {std_without:.6f}\n n = {n_excl}\n\n"
                        )
                    else:
                        fh.write(
                            " (no molecules remain after excluding constant avg_tanimoto entries)\n\n"
                        )
                    fh.write(
                        "Selection method: examples selected by sorting molecules by Spearman rho and sampling first, last, and intermediate positions (exact percentiles are approximate).\n"
                    )
                    fh.write(
                        f"Number of valid (non-NaN) molecules used for example selection: {n_valid_df}\n\n"
                    )
                    fh.write("Example molecules (sorted Spearman selections):\n")
                    fh.write(examples_df.to_init_repr() + "\n")

                    fh.write(
                        f"\nNumber of NaN Spearman entries (excluding constant avg_tanimoto): {n_nans_excl_const}\n"
                    )
                logger.info("Wrote per-molecule Spearman summary to %s", summary_path)
            except Exception:
                logger.exception(
                    "Failed to write per-molecule Spearman summary to file (continuing)"
                )
    except Exception:
        logger.exception("Failed to compute per-molecule Spearman summary (continuing)")

    # 4) Global Spearman across all spectra of molecules that had any match + binned plot
    included_spectra = per_spectrum_df.filter(
        pl.col("mol_idx").is_in(list(mols_with_matches))
    )
    plot_all_path = output_dir / config.filename_all.format(measure=config.info_measure)
    logger.info(
        "Computing global Spearman and producing binned plot for ALL spectra (selected molecules)..."
    )
    global_stats = _compute_global_spearman_and_binned_plot(
        included_spectra,
        config,
        plot_all_path,
        label="All spectra (molecules with any match)",
    )

    # 5) Best-spectrum-per-molecule: select one spectrum per molecule (highest info)
    logger.info(
        "Selecting most informative spectrum per molecule and computing stats..."
    )
    # Compute the max info per molecule
    best_info = included_spectra.group_by("mol_idx").agg(
        pl.col(config.info_measure).max().alias("max_info")
    )
    # Join and select the spectrum(s) that match the max; resolve ties by taking the smallest idx
    joined = included_spectra.join(best_info, on="mol_idx", how="inner")
    candidates = joined.filter(pl.col(config.info_measure) == pl.col("max_info"))
    # If multiple candidates per molecule (tie), pick the one with smallest idx
    best_per_molecule = (
        candidates.sort(["mol_idx", "idx"])
        .group_by("mol_idx")
        .agg(pl.col("idx").first().alias("idx"))
        .join(included_spectra, on=["mol_idx", "idx"], how="left")
        .select(["idx", "mol_idx", config.info_measure, "avg_tanimoto"])
    )

    plot_best_path = output_dir / config.filename_best.format(
        measure=config.info_measure
    )
    best_stats = _compute_global_spearman_and_binned_plot(
        best_per_molecule,
        config,
        plot_best_path,
        label="Most informative spectrum per molecule (molecules with any match)",
    )

    # 6) Also produce plots including molecules that lacked matches (their avg_tanimoto will be 1.0)
    plot_all_incl_path = output_dir / config.filename_all_including_unmatched.format(
        measure=config.info_measure
    )
    logger.info(
        "Computing global Spearman and producing binned plot for ALL spectra (including molecules without match)..."
    )
    global_incl_stats = _compute_global_spearman_and_binned_plot(
        per_spectrum_df,
        config,
        plot_all_incl_path,
        label="All spectra (including molecules without any match)",
    )

    # Best spectrum per molecule including unmatched molecules
    logger.info(
        "Selecting most informative spectrum per molecule across ALL molecules and computing stats..."
    )
    best_info_all = per_spectrum_df.group_by("mol_idx").agg(
        pl.col(config.info_measure).max().alias("max_info")
    )
    joined_all = per_spectrum_df.join(best_info_all, on="mol_idx", how="inner")
    candidates_all = joined_all.filter(
        pl.col(config.info_measure) == pl.col("max_info")
    )
    best_per_molecule_all = (
        candidates_all.sort(["mol_idx", "idx"])
        .group_by("mol_idx")
        .agg(pl.col("idx").first().alias("idx"))
        .join(per_spectrum_df, on=["mol_idx", "idx"], how="left")
        .select(["idx", "mol_idx", config.info_measure, "avg_tanimoto"])
    )

    plot_best_incl_path = output_dir / config.filename_best_including_unmatched.format(
        measure=config.info_measure
    )
    best_incl_stats = _compute_global_spearman_and_binned_plot(
        best_per_molecule_all,
        config,
        plot_best_incl_path,
        label="Most informative spectrum per molecule (including molecules without any match)",
    )

    # Append global Spearman results for the plots to the per-molecule summary file
    summary_path = output_dir / "per_molecule_spearman_summary.txt"
    try:
        with open(summary_path, "a") as fh:
            fh.write(
                "\nGlobal/binned plot Spearman summary (these accompany the generated plots):\n\n"
            )

            def _write_plot_stats(
                name: str, stats: Dict[str, object], path: Path
            ) -> None:
                rho = stats.get("rho", float("nan"))
                pval = stats.get("p_value", float("nan"))
                n = stats.get("n_points", 0)
                fh.write(f"{name}:\n")
                # Use safe formatting; NaNs will be written as 'nan'
                fh.write(f" rho = {rho:.6f}  p = {pval:.6g}  n = {n}\n")
                fh.write(f" plot = {path}\n\n")

            _write_plot_stats(
                "All spectra (molecules with any match)", global_stats, plot_all_path
            )
            _write_plot_stats(
                "Most informative spectrum per molecule (molecules with any match)",
                best_stats,
                plot_best_path,
            )
            _write_plot_stats(
                "All spectra (including molecules without any match)",
                global_incl_stats,
                plot_all_incl_path,
            )
            _write_plot_stats(
                "Most informative spectrum per molecule (including molecules without any match)",
                best_incl_stats,
                plot_best_incl_path,
            )
        logger.info(
            "Appended global Spearman stats to per-molecule summary at %s", summary_path
        )
    except Exception:
        logger.exception(
            "Failed to append global Spearman stats to per-molecule summary (continuing)"
        )

    return {
        "per_spectrum": per_spectrum_df,
        "per_molecule_spearman": per_molecule_spearman_df,
        "global": global_stats,
        "best": best_stats,
        "global_including_unmatched": global_incl_stats,
        "best_including_unmatched": best_incl_stats,
        "paths": {
            "per_spectrum_parquet": str(per_spec_out),
            "per_molecule_parquet": str(per_mol_out),
            "plot_all": str(plot_all_path),
            "plot_best": str(plot_best_path),
            "plot_all_including_unmatched": str(plot_all_incl_path),
            "plot_best_including_unmatched": str(plot_best_incl_path),
        },
    }


if __name__ == "__main__":
    # Example usage with the same paths used elsewhere in the repo.
    LIBRARY_PATH = Path(
        "/home/analytit_admin/Data/spectral_libs/info_score/combined_library.parquet"
    )
    PAIRS_WITH_TANIMOTO_PATH = Path(
        "/home/analytit_admin/Data/spectral_libs/info_score/combined_library_pairs_with_tanimoto_251231.parquet"
    )
    OUTPUT_DIR = Path(
        "/home/analytit_admin/Data/spectral_libs/info_score/sim_vs_info_analysis_251231"
    )

    cfg = SimilarityVsInfoConfig(
        pairs_parquet_path=PAIRS_WITH_TANIMOTO_PATH,
        left_library_parquet_path=LIBRARY_PATH,
        right_library_parquet_path=None,
        tanimoto_col="tanimoto_similarity",
        left_idx_col="idx",
        right_idx_col="idx_right",
        left_mol_col="mol_idx",
        right_mol_col="mol_idx_right",
        info_measure="spectral_information_score",
        x_bin_width=0.3,
        x_range=(0.0, 3.0),
        min_count_threshold=10,
        output_dir=OUTPUT_DIR,
    )

    results = analyze_similarity_vs_info(cfg)
    logger.info("Analysis complete. Results summary:")
    logger.info("Per-molecule spearman (sample):")
    try:
        per_mol_df = results.get("per_molecule_spearman")
        if isinstance(per_mol_df, pl.DataFrame):
            print(per_mol_df.head(10))
        else:
            print(per_mol_df)
    except Exception:
        logger.exception("Failed to show per-molecule spearman sample")

    logger.info("Global stats (all spectra): %s", results["global"])
    logger.info("Global stats (best per molecule): %s", results["best"])
    logger.info("Generated files: %s", results["paths"])
