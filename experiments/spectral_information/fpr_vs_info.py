import marimo

__generated_with = "0.18.1"
app = marimo.App(width="full")


@app.cell
def _():
    import polars as pl
    import numpy as np
    import matplotlib.pyplot as plt
    import hrms_utils
    from pathlib import Path
    from dataclasses import dataclass, field
    from typing import List, Tuple, Union, Optional
    from rdkit import Chem
    from nvmolkit.fingerprints import MorganFingerprintGenerator
    from nvmolkit.similarity import crossTanimotoSimilarityMemoryConstrained, crossCosineSimilarityMemoryConstrained
    import torch
    return (
        Chem,
        List,
        MorganFingerprintGenerator,
        Optional,
        Path,
        Tuple,
        Union,
        crossTanimotoSimilarityMemoryConstrained,
        dataclass,
        field,
        np,
        pl,
        plt,
    )


@app.cell
def _(List, Path, Union, pl):
    def build_and_write_pairs_parquet(
        parquet_paths: List[Path],
        output_path: Union[str, Path] = "/home/analytit_admin/Data/spectral_libs/all_pairs_with_similarities.parquet",
        min_isomers: int = 10,
        ms2_tolerance_ppm: float = 10.0,
        sink_engine: str = "streaming",
    ) -> None:
        """
        Build unioned library LF, compute pairwise spectral similarities, and write pairs parquet.

        Args:
          - parquet_paths: list of Path objects pointing at library parquet files
          - output_path: where to write the pairs with similarities
          - min_isomers: minimum number of isomers per formula used to filter the library
          - ms2_tolerance_ppm: used by the spectral similarity functions
          - sink_engine: polars sink engine (streaming preferred for large data)

        Returns:
          - None (writes parquet to output_path)
        """
        assert len(parquet_paths) > 0, "parquet_paths must contain at least one path"
        # Load and union into a single lazyframe
        lf_list = []
        for PARQUET_PATH in parquet_paths:
            assert Path(PARQUET_PATH).exists(), f"Requested parquet does not exist: {PARQUET_PATH}"
            lf_list.append(pl.scan_parquet(PARQUET_PATH))

        lf = pl.union(lf_list).filter(
            pl.col("clean_precursor"),
            pl.len().over("precursor_formula_array").ge(min_isomers),
        )

        # Keep only necessary columns; add idx and nominal_mass to join on
        lf = lf.select(
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
        ).filter(
            pl.col("smiles").is_not_null()
        ).with_row_index("idx").with_columns(
            nominal_mass=pl.col("precursor_mz").round(0)
        )

        pairs_filtered = lf.join(
            other=lf, on=["nominal_mass", "ion_mode"], suffix="_right"
        ).filter(
            pl.col("precursor_mz").is_close(pl.col("precursor_mz_right"), rel_tol=5e-6),
            pl.col("base_inchikey") != pl.col("base_inchikey_right"),
        ).with_columns(
            spectra=pl.struct(
                mz1=pl.col("cleaned_normalized_mz").alias("mz1"),
                intensities1=pl.col("cleaned_normalized_intensity").alias("intensities1"),
                mz2=pl.col("cleaned_normalized_mz_right").alias("mz2"),
                intensities2=pl.col("cleaned_normalized_intensity_right").alias("intensities2"),
                precursor_mz1=pl.col("precursor_mz").alias("precursor_mz1"),
                precursor_mz2=pl.col("precursor_mz_right").alias("precursor_mz2"),
            )
        ).drop(
            [
                "cleaned_normalized_mz",
                "cleaned_normalized_intensity",
                "cleaned_normalized_mz_right",
                "cleaned_normalized_intensity_right",
                "precursor_mz",
                "precursor_mz_right",
            ]
        ).with_columns(
            dotprod_similarity=pl.col("spectra").spectral_similarity.dotprod_similarity(
                ms2_tolerance_in_ppm=ms2_tolerance_ppm,
                clean_spectra_first=False,
                ignore_precursor=True,
            ),
            dotprod_similarity_with_precursor=pl.col("spectra").spectral_similarity.dotprod_similarity(
                ms2_tolerance_in_ppm=ms2_tolerance_ppm,
                clean_spectra_first=False,
                ignore_precursor=False,
            ),
            entropy_similarity=pl.col("spectra").spectral_similarity.entropy_similarity(
                ms2_tolerance_in_ppm=ms2_tolerance_ppm,
                clean_spectra_first=False,
                ignore_precursor=True,
            ),
            mass_sqrt_cosine_similarity=pl.col("spectra").spectral_similarity.general_cosine_similarity(
                ms2_tolerance_in_ppm=ms2_tolerance_ppm,
                clean_spectra_first=False,
                ignore_precursor=True,
                mass_power=0.5,
                intensity_power=0.5,
            ),
        )

        # Sink to parquet for downstream analysis and return the lazyframe
        pairs_filtered.sink_parquet(str(output_path), engine=sink_engine)
    return (build_and_write_pairs_parquet,)


@app.cell
def _(List, Optional, Path, Tuple, Union, dataclass, field, np, pl, plt):
    @dataclass
    class FprVsInfoConfig:
        # Path (or lazy/dataframe) to the pairs parquet produced earlier.
        pairs_parquet_path: Union[str, Path] = "/home/analytit_admin/Data/spectral_libs/all_pairs_with_similarities.parquet"
        max_info: float = 3.0
        bin_width: float = 0.1
        # metric_key, threshold, human_label, color, marker
        metrics_config: List[Tuple[str, float, str, str, str]] = field(
            default_factory=lambda: [
                ("dotprod_similarity", 0.8, "Dot Product", "C0", "o"),
                ("dotprod_similarity", 0.95, "Dot Product", "C1", "o"),
                ("entropy_similarity", 0.75, "Entropy", "C2", "^"),
            ]
        )
        min_count_threshold: int = 5
        # Why: Allow overriding names/targets for output files from callers/tests.
        metrics_output_path: Union[str, Path] = "fpr_vs_info_metrics.png"
        matched_info_output_path: Union[str, Path] = "fpr_vs_info_avg_matched_info_diff.png"

    def compute_fpr_vs_info_stats(
        config: FprVsInfoConfig,
        pairs_input: Optional[Union[str, Path, pl.DataFrame, pl.LazyFrame]] = None,
    ) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
        """
        Compute aggregated statistics for FPR vs spectral information and compute molecule-level CDF.
        Returns:
          - all_stats: pl.DataFrame (collected) with aggregated bin stats for each metric and threshold
          - cdf_x: np.ndarray (cdf bin edges for max_info)
          - cdf_y: np.ndarray (reverse CDF / survival function for max_info)
        """
        # Accept either an explicit pairs_input or default to config.pairs_parquet_path.
        if pairs_input is None:
            pairs_input = config.pairs_parquet_path

        # Accept either Path/str (load lazy), or an existing polars DataFrame/LazyFrame
        if isinstance(pairs_input, (str, Path)):
            assert Path(pairs_input).exists(), (
                f"Parquet path does not exist: {pairs_input}; "
                "ensure compute pipeline or prior cells produced the pairs file."
            )
            pairs_sim = pl.scan_parquet(str(pairs_input))
        elif isinstance(pairs_input, pl.DataFrame):
            pairs_sim = pairs_input.lazy()
        elif isinstance(pairs_input, pl.LazyFrame):
            pairs_sim = pairs_input
        else:
            raise AssertionError("pairs_input must be str/Path/pl.DataFrame/pl.LazyFrame")

        stats = []
        # Why: Compute metric-specific statistics per left-spectrum (idx), grouped into info bins.
        for sim_col, thresh, label, color, marker in config.metrics_config:
            print(f"Processing metric: {label} with threshold {thresh}")
            unique_spectra_stats = pairs_sim.with_columns(
                # Flag which pair is a false match by metric threshold
                is_false_match=(pl.col(sim_col).ge(thresh)).cast(pl.Int64)
            ).with_columns(
                # Count of false matches per compound (base_inchikey_right) and per spectrum (idx)
                false_compound_count=pl.col("is_false_match").sum().over("base_inchikey_right", "idx"),
                false_spectra_count=pl.col("is_false_match").sum().over("idx"),
                # Compute average spectral info difference across matches above threshold (per idx)
                avg_matched_info_diff=(pl.col("spectral_information_score_right") - pl.col("spectral_information_score"))
                .filter(pl.col("is_false_match").eq(1))
                .mean()
                .over("idx"),
            ).unique(subset="idx", keep="any").with_columns(
                info_bin_val=(pl.col("spectral_information_score") / config.bin_width).floor() * config.bin_width
            ).filter(
                pl.col("info_bin_val").ge(0.0),
                pl.col("info_bin_val").le(config.max_info),
            )

            # Aggregate per info bin to compute averages/counts
            res = unique_spectra_stats.group_by("info_bin_val").agg(
                total_count=pl.len(),
                avg_false_matches=pl.col("false_compound_count").mean(),
                avg_matched_info_diff=pl.col("avg_matched_info_diff").mean(),
            ).with_columns(
                metric_label=pl.lit(label),
                metric_name=pl.lit(sim_col),
                threshold_used=pl.lit(thresh),
                plot_color=pl.lit(color),
                plot_marker=pl.lit(marker),
            ).sort("info_bin_val")

            stats.append(res)

        # Collect into memory for plotting (fail fast if aggregation didn't compute)
        all_stats = pl.concat(stats).collect(engine="streaming")
        pl.Config.set_tbl_rows(100)
        print(all_stats)

        # Compute molecule-level max info distribution (CDF / survival function)
        molecule_max_info = pairs_sim.group_by("base_inchikey").agg(
            max_info=pl.col("spectral_information_score").max()
        ).collect(engine="streaming")

        cdf_x = np.arange(0, config.max_info + config.bin_width, config.bin_width)
        total_molecules = molecule_max_info.height
        cdf_y = np.array(
            [molecule_max_info.filter(pl.col("max_info") >= float(x)).height / total_molecules for x in cdf_x]
        )

        return all_stats, cdf_x, cdf_y

    def plot_fpr_vs_info_metrics(
        all_stats: pl.DataFrame,
        cdf_x: np.ndarray,
        cdf_y: np.ndarray,
        config: FprVsInfoConfig,
        show_plot: bool = False
    ) -> None:
        """
        Plot average false matches vs Spectral Information score and optionally save as config.metrics_output_path.
        """
        assert "metric_name" in all_stats.columns, "all_stats missing 'metric_name' column; compute stats first."
        fig, ax = plt.subplots(figsize=(10, 6))

        for metric_key, thresh, label, color, marker in config.metrics_config:
            subset = all_stats.filter(
                (pl.col("metric_name") == metric_key) &
                (pl.col("threshold_used") == thresh) &
                (pl.col("total_count") > config.min_count_threshold)
            )

            if subset.height > 0:
                info_x = subset.select("info_bin_val").to_series().to_numpy()
                avg_false_y = subset.select("avg_false_matches").to_series().to_numpy()

                ax.plot(
                    info_x,
                    avg_false_y,
                    marker=marker,
                    label=f"{label} (Thresh: {thresh})",
                    color=color,
                )
            else:
                print(f"Warning: No data for {label} at threshold {thresh}")

        # Secondary axis for Molecule Coverage CDF
        ax2 = ax.twinx()
        ax2.plot(
            cdf_x,
            cdf_y,
            color="black",
            linestyle="--",
            linewidth=2,
            alpha=0.6,
            label="Molecule Coverage (Max Info ≥ X)",
        )
        ax2.set_ylabel("Fraction of Molecules")
        ax2.set_ylim(0, 1.05)

        ax.set_xlabel("Spectral Information Score")
        ax.set_ylabel("Average Number of False Matches")
        ax.set_title("Average Number of False Matches vs Spectral Information Score")

        lines_1, labels_1 = ax.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

        ax.grid(True, alpha=0.3)
        fig.savefig(str(config.metrics_output_path))
        if show_plot:
            plt.show()
        plt.close(fig)

    def plot_matched_avg_info_diff(
        all_stats: pl.DataFrame,
        config: FprVsInfoConfig,
        show_plot: bool = False
    ) -> None:
        """
        Plot average matched information difference vs Spectral Information score (filtered) and save to file.
        """
        assert "avg_matched_info_diff" in all_stats.columns, (
            "avg_matched_info_diff missing from aggregated results; ensure compute pipeline computed this column"
        )

        fig_matched, ax_matched = plt.subplots(figsize=(10, 6))

        for metric_key, thresh, label, color, marker in config.metrics_config:
            subset = all_stats.filter(
                (pl.col("metric_name") == metric_key) &
                (pl.col("threshold_used") == thresh) &
                (pl.col("total_count") > config.min_count_threshold)
            )
            matched_subset = subset.filter(pl.col("avg_matched_info_diff").is_not_null())

            if matched_subset.height > 0:
                info_x_matched = matched_subset.select("info_bin_val").to_series().to_numpy()
                avg_matched_y = matched_subset.select("avg_matched_info_diff").to_series().to_numpy()
                ax_matched.plot(
                    info_x_matched,
                    avg_matched_y,
                    linestyle="--",
                    marker="x",
                    label=f"{label} (Thresh: {thresh})",
                    color=color,
                    alpha=0.9,
                )
            else:
                print(f"Warning: No matched-info data for {label} at threshold {thresh}")

        ax_matched.set_xlabel("Spectral Information Score")
        ax_matched.set_ylabel("Average Matched Information Difference")
        ax_matched.set_title("Average Matched Information Difference vs Spectral Information Score")
        ax_matched.legend(loc="upper right")
        ax_matched.grid(True, alpha=0.3)

        fig_matched.savefig(str(config.matched_info_output_path))
        if show_plot:
            plt.show()
        plt.close(fig_matched)
    return (
        FprVsInfoConfig,
        compute_fpr_vs_info_stats,
        plot_fpr_vs_info_metrics,
        plot_matched_avg_info_diff,
    )


@app.cell
def _(
    Chem,
    FprVsInfoConfig,
    List,
    MorganFingerprintGenerator,
    Optional,
    Path,
    Tuple,
    Union,
    crossTanimotoSimilarityMemoryConstrained,
    pl,
    plt,
):
    # ...existing code...
    def compute_avg_tanimoto_between_matched_pairs(
        pairs_input: Optional[Union[str, Path, pl.DataFrame, pl.LazyFrame]] = None,
        left_smiles_col: str = "smiles",
        right_smiles_col: str = "smiles_right",
        metrics_config: Optional[List[Tuple[str, float, str, str, str]]] = None,
        group_by_cols: Optional[List[str]] = None,
        fp_radius: int = 2,
        fp_size: int = 1024,
        fp_num_threads: int = 0,
        bin_width: float = 0.1,
        max_info: float = 3.0,
        plot_output_path: Optional[Union[str, Path]] = None,
        min_count_threshold: int = 5,
    ) -> pl.DataFrame:
        """
        Compute average Tanimoto similarity for matched pairs, optionally binned by spectral information.

        This function:
          - Loads a pairs dataset (lazy or scanned parquet).
          - Extracts unique smiles on the left and right.
          - Builds Morgan fingerprints with nvmolkit (once per unique SMILES).
          - Computes cross-group Tanimoto similarities and looks up values only for
            pairs present in the dataset (unique combos).
          - Joins back similarity values and aggregates.

        Args:
          - pairs_input: Path/lazyframe/dataframe for the pairwise dataset
          - left_smiles_col/right_smiles_col: column names for SMILES in left and right sides
          - metrics_config: List of (metric_key, threshold, label, color, marker). If provided,
            stats are computed per metric/threshold configuration and binned by spectral info.
          - group_by_cols: columns to aggregate by (e.g., ["ion_mode"]). Used only if metrics_config is None.
          - fp_radius/fp_size/fp_num_threads: parameters for fingerprint generation.
          - bin_width: width of spectral information bins (used if metrics_config is provided).
          - max_info: maximum spectral information score to consider (used if metrics_config is provided).
          - plot_output_path: if provided, saves a plot of Avg Tanimoto vs Spectral Info.
          - min_count_threshold: minimum number of pairs in a bin to include in the plot.

        Returns:
          - pl.DataFrame with columns:
            * avg_tanimoto: mean Tanimoto similarity for paired matches
            * median_tanimoto: median
            * count: number of pairs used in the aggregate
            * (plus metric metadata and info_bin_val if metrics_config is used)
        """
        # Accept either an explicit pairs_input or default to the default pairs file
        # We use the FprVsInfoConfig default parquet if none is passed to be consistent.
        if pairs_input is None:
            pairs_input = FprVsInfoConfig().pairs_parquet_path

        # Accept either Path/str (load lazy), or an existing polars DataFrame/LazyFrame
        if isinstance(pairs_input, (str, Path)):
            assert Path(pairs_input).exists(), (
                f"Parquet path does not exist: {pairs_input}; "
                "ensure compute pipeline or prior cells produced the pairs file."
            )
            pairs_sim = pl.scan_parquet(str(pairs_input))
        elif isinstance(pairs_input, pl.DataFrame):
            pairs_sim = pairs_input.lazy()
        elif isinstance(pairs_input, pl.LazyFrame):
            pairs_sim = pairs_input
        else:
            raise AssertionError("pairs_input must be str/Path/pl.DataFrame/pl.LazyFrame")

        # Why: Collect small sample to verify columns exist before expensive ops
        sample = pairs_sim.select([left_smiles_col, right_smiles_col]).limit(1).collect(engine="streaming")
        assert left_smiles_col in sample.columns and right_smiles_col in sample.columns, (
            f"Expected columns '{left_smiles_col}' and '{right_smiles_col}' in pairs dataset"
        )

        # Compute unique SMILES lists for left and right
        # Why: Add row index to map back to the similarity matrix coordinates
        # Use streaming to handle large datasets without OOM
        left_unique = pairs_sim.select(left_smiles_col).unique().with_row_index("l_idx").collect(engine="streaming")
        right_unique = pairs_sim.select(right_smiles_col).unique().with_row_index("r_idx").collect(engine="streaming")

        left_smiles_list = left_unique.get_column(left_smiles_col).to_list()
        right_smiles_list = right_unique.get_column(right_smiles_col).to_list()

        assert len(left_smiles_list) > 0 and len(right_smiles_list) > 0, (
            "No smiles found on left or right side. Please verify SMILES columns."
        )

        # Convert smiles strings to RDKit mols; fail fast if any invalid SMILES encountered.
        left_mols = [Chem.MolFromSmiles(s) for s in left_smiles_list]
        right_mols = [Chem.MolFromSmiles(s) for s in right_smiles_list]
        invalid_left = [s for s, m in zip(left_smiles_list, left_mols) if m is None]
        invalid_right = [s for s, m in zip(right_smiles_list, right_mols) if m is None]
        assert not invalid_left and not invalid_right, (
            f"Invalid SMILES encountered. Left invalid examples: {invalid_left[:3]}, "
            f"Right invalid examples: {invalid_right[:3]}"
        )

        # Generate fingerprints once per unique SMILES using nvmolkit
        fpgen = MorganFingerprintGenerator(radius=fp_radius, fpSize=fp_size)
        fps_left = fpgen.GetFingerprints(left_mols, num_threads=fp_num_threads)
        fps_right = fpgen.GetFingerprints(right_mols, num_threads=fp_num_threads)

        # Compute pairwise Tanimoto only between the unique left/right groups
        # Why: crossTanimotoSimilarityMemoryConstrained returns a numpy array directly
        # Note: This matrix can be large (N_left * N_right * 4 bytes).
        sims_matrix = crossTanimotoSimilarityMemoryConstrained(fps_left, fps_right)  # shape [len(left), len(right)]

        # Build mapping of unique pairs -> tanimoto
        # Why: Use streaming to avoid OOM on large pair sets
        unique_pairs = pairs_sim.select([left_smiles_col, right_smiles_col]).unique().collect(engine="streaming")
        if unique_pairs.height == 0:
            # No pairs to compute.
            print("No unique pairs found for Tanimoto computation; returning empty DataFrame.")
            return pl.DataFrame(
                {"avg_tanimoto": [], "median_tanimoto": [], "count": []}
            )

        # Join unique pairs with the indexed unique lists to get matrix coordinates
        # Why: Use Polars join for efficiency instead of Python dictionary lookups
        pairs_indices = unique_pairs.join(
            left_unique, on=left_smiles_col
        ).join(
            right_unique, on=right_smiles_col
        )

        l_idxs = pairs_indices.get_column("l_idx").to_numpy()
        r_idxs = pairs_indices.get_column("r_idx").to_numpy()

        # Extract values using numpy fancy indexing
        # Why: Vectorized lookup is much faster than iterating in Python
        tanimoto_values = sims_matrix[l_idxs, r_idxs]

        mapping_df = pairs_indices.select([left_smiles_col, right_smiles_col]).with_columns(
            tanimoto_similarity=pl.Series(tanimoto_values)
        )

        # Join mapping back to the pairs lazyframe to assign similarity to each pair
        pairs_with_sim = pairs_sim.join(mapping_df.lazy(), left_on=[left_smiles_col, right_smiles_col], right_on=[left_smiles_col, right_smiles_col])

        # Case 1: Metrics Config provided (Iterate, Filter, Aggregate, Plot)
        if metrics_config is not None:
            stats = []
            for metric_key, thresh, label, color, marker in metrics_config:
                # Filter pairs that meet the threshold for this specific metric
                # And bin by spectral information score
                res = pairs_with_sim.filter(
                    pl.col(metric_key).ge(thresh)
                ).with_columns(
                    info_bin_val=(pl.col("spectral_information_score") / bin_width).floor() * bin_width
                ).filter(
                    pl.col("info_bin_val").ge(0.0),
                    pl.col("info_bin_val").le(max_info),
                ).group_by("info_bin_val").agg(
                    avg_tanimoto=pl.col("tanimoto_similarity").mean(),
                    median_tanimoto=pl.col("tanimoto_similarity").median(),
                    count=pl.len(),
                ).with_columns(
                    metric_name=pl.lit(metric_key),
                    threshold_used=pl.lit(thresh),
                    metric_label=pl.lit(label),
                    plot_color=pl.lit(color),
                    plot_marker=pl.lit(marker),
                ).sort("info_bin_val")
                stats.append(res)

            all_stats = pl.concat(stats).collect(engine="streaming")

            if plot_output_path:
                fig, ax = plt.subplots(figsize=(10, 6))
                for metric_key, thresh, label, color, marker in metrics_config:
                    subset = all_stats.filter(
                        (pl.col("metric_name") == metric_key) &
                        (pl.col("threshold_used") == thresh) &
                        (pl.col("count") > min_count_threshold)
                    )
                    if subset.height > 0:
                        ax.plot(
                            subset.get_column("info_bin_val").to_numpy(),
                            subset.get_column("avg_tanimoto").to_numpy(),
                            marker=marker,
                            label=f"{label} (Thresh: {thresh})",
                            color=color,
                        )
            
                ax.set_xlabel("Spectral Information Score")
                ax.set_ylabel("Average Tanimoto Similarity of Matches")
                ax.set_title("Average Tanimoto Similarity of Matches vs Spectral Information Score")
                ax.legend(loc="lower right")
                ax.grid(True, alpha=0.3)
                fig.savefig(str(plot_output_path))
                plt.close(fig)

            return all_stats

        # Case 2: Standard Group By (if columns exist in dataframe)
        elif group_by_cols:
            assert isinstance(group_by_cols, list) and len(group_by_cols) > 0, "group_by_cols must be a non-empty list"
            agg = pairs_with_sim.group_by(*group_by_cols).agg(
                avg_tanimoto=pl.col("tanimoto_similarity").mean(),
                median_tanimoto=pl.col("tanimoto_similarity").median(),
                count=pl.col("tanimoto_similarity").count(),
            ).collect(engine="streaming")
            return agg

        # Case 3: Global Aggregate
        else:
            agg = pairs_with_sim.select(
                [
                    pl.col("tanimoto_similarity").mean().alias("avg_tanimoto"),
                    pl.col("tanimoto_similarity").median().alias("median_tanimoto"),
                    pl.col("tanimoto_similarity").count().alias("count"),
                ]
            ).collect(engine="streaming")
            return agg
    # ...existing code...
    return (compute_avg_tanimoto_between_matched_pairs,)


@app.cell
def _(Path):
    # Configuration
    # Why: Define constants for reproducibility and easy adjustment.
    # Update PARQUET_PATH to point to your actual data file.
    PARQUET_PATHS =[
        Path("/home/analytit_admin/Data/spectral_libs/msp_for_Yonathan/NIST.parquet"),
        Path("/home/analytit_admin/Data/spectral_libs/msp_for_Yonathan/fraghub.parquet")
    ]
    MIN_ISOMERS = 10
    OUTPUT_PAIRS_PATH = Path("/home/analytit_admin/Data/spectral_libs/all_pairs_with_similarities.parquet")
    return MIN_ISOMERS, OUTPUT_PAIRS_PATH, PARQUET_PATHS


@app.cell
def _(
    MIN_ISOMERS,
    OUTPUT_PAIRS_PATH,
    PARQUET_PATHS,
    build_and_write_pairs_parquet,
):
    build_and_write_pairs_parquet(
        parquet_paths=PARQUET_PATHS,
        output_path=OUTPUT_PAIRS_PATH,
        min_isomers=MIN_ISOMERS,
    )
    return


@app.cell
def _(
    FprVsInfoConfig,
    OUTPUT_PAIRS_PATH,
    compute_fpr_vs_info_stats,
    plot_fpr_vs_info_metrics,
    plot_matched_avg_info_diff,
):
    # Analysis and Plotting
    # Why: Bin spectral info score and calculate average number of false matches for different similarity metrics.
    # We aggregate by the left spectrum ('idx') to count how many false positive matches exist above the threshold.

    # Use the new dataclass/config-driven analysis and plotting
    config = FprVsInfoConfig(
        pairs_parquet_path=OUTPUT_PAIRS_PATH,
        max_info=3.0,
        bin_width=0.1,
        metrics_config=[
            ("dotprod_similarity", 0.8, "Dot Product", "C0", "o"),
            ("dotprod_similarity", 0.95, "Dot Product", "C1", "o"),
            ("entropy_similarity", 0.75, "Entropy", "C2", "^"),
        ],
        metrics_output_path="fpr_vs_info_metrics.png",
        matched_info_output_path="fpr_vs_info_avg_matched_info_diff.png",
    )
    all_stats, cdf_x, cdf_y = compute_fpr_vs_info_stats(config)

    plot_fpr_vs_info_metrics(all_stats, cdf_x, cdf_y, config)
    plot_matched_avg_info_diff(all_stats, config)
    return (config,)


@app.cell
def _(compute_avg_tanimoto_between_matched_pairs, config):

    avg_tanimoto_df = compute_avg_tanimoto_between_matched_pairs(
        pairs_input=config.pairs_parquet_path,
        left_smiles_col="smiles",
        right_smiles_col="smiles_right",
        metrics_config=config.metrics_config,
        fp_radius=2,
        fp_size=2048,
        fp_num_threads=0,
    )
    print(avg_tanimoto_df)
    return


if __name__ == "__main__":
    app.run()
