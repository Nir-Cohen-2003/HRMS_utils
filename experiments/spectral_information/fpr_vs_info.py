import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import polars as pl
    import numpy as np
    import matplotlib.pyplot as plt
    plt.style.use('default')
    import hrms_utils
    from pathlib import Path
    from dataclasses import dataclass, field
    from typing import List, Tuple, Union, Optional, Any
    import numpy.typing as npt
    from rdkit import Chem
    from nvmolkit.fingerprints import MorganFingerprintGenerator
    from nvmolkit.similarity import crossTanimotoSimilarityMemoryConstrained, crossCosineSimilarityMemoryConstrained
    import torch
    import math
    import os
    os.environ["RUST_BACKTRACE"] = "full"
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
        math,
        np,
        pl,
        plt,
    )


@app.cell
def _(List, Path, Union, math, pl):
    def build_and_write_pairs_parquet(
        parquet_paths: List[Path],
        output_path: Union[str, Path] = "/home/analytit_admin/Data/spectral_libs/all_pairs_with_similarities.parquet",
        min_isomers: int = 10,
        ms2_tolerance_ppm: float = 10.0,
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
        lf = lf.collect().select(
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
        ).with_row_index(
            "idx"
        ).with_columns(
            nominal_mass=pl.col("precursor_mz").round(0),
            weighted_spectral_information_score=pl.lit(100.0).mul(pl.col("spectral_information_score")).truediv(
                pl.col("precursor_formula_array").arr.sum()-pl.col("precursor_formula_array").arr.get(0)
                ),
            spectral_entropy=(
                pl.col("cleaned_normalized_intensity") / pl.col("cleaned_normalized_intensity").list.sum()
                ).list.eval(pl.element().log(base=math.e).mul(pl.element())).list.sum().neg(),
            num_clean_peaks=pl.col("cleaned_normalized_mz").list.len()
        ).lazy()

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
        pairs_filtered.sink_parquet(str(output_path), engine="streaming")
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
        use_weighted_information_score: bool = False

        # Tanimoto similarity function arguments
        left_smiles_col: str = "smiles"
        right_smiles_col: str = "smiles_right"
        fp_radius: int = 2
        fp_size: int = 1024
        fp_num_threads: int = 0
        plot_avg_tanimoto_output_path: Optional[Union[str, Path]] = "avg_tanimoto_vs_info.png"

        show_molecule_cdf: bool = False

        # Additional bin widths for entropy & peak-count measures; defaults to bin_width if None.
        entropy_bin_width: Optional[float] = None
        peaks_bin_width: Optional[float] = None
        # Optional maximum x-value for entropy or peak number plots (None = no clipping).
        # Why: small/large tails often create noisy correlations; allow caller to bound the plot/CORR range.
        max_spectral_entropy: Optional[float] = None
        max_num_clean_peaks: Optional[float] = None

        # Output paths for entropy/peak plots with Spearman annotations
        plot_entropy_output_path: Optional[Union[str, Path]] = "fpr_vs_spectral_entropy.png"
        plot_peaks_output_path: Optional[Union[str, Path]] = "fpr_vs_num_clean_peaks.png"

    def compute_fpr_vs_info_stats(
        config: FprVsInfoConfig,
        pairs_input: Optional[Union[str, Path, pl.DataFrame, pl.LazyFrame]] = None,
    ) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
        """
        Compute aggregated statistics for FPR vs spectral information and compute molecule-level CDF.

        Returns:
          - all_stats: pl.DataFrame (collected) with aggregated bin stats for each metric and threshold
          - cdf_x: np.ndarray (cdf bin edges for max_info) - empty array if config.show_molecule_cdf == False
          - cdf_y: np.ndarray (reverse CDF / survival function for max_info) - empty array if config.show_molecule_cdf == False
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
        info_col = "weighted_spectral_information_score" if config.use_weighted_information_score else "spectral_information_score"
        info_col_right = f"{info_col}_right"

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
                avg_matched_info_diff=(pl.col(info_col_right) - pl.col(info_col))
                .filter(pl.col("is_false_match").eq(1))
                .mean()
                .over("idx"),
            ).unique(subset="idx", keep="any").with_columns(
                info_bin_val=(pl.col(info_col) / config.bin_width).floor() * config.bin_width
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

        # Compute molecule-level max info distribution (CDF / survival function) only if requested
        if config.show_molecule_cdf:
            molecule_max_info = pairs_sim.group_by("base_inchikey").agg(
                max_info=pl.col(info_col).max()
            ).collect(engine="streaming")

            cdf_x = np.arange(0, config.max_info + config.bin_width, config.bin_width)
            total_molecules = molecule_max_info.height
            cdf_y = np.array(
                [molecule_max_info.filter(pl.col("max_info") >= float(x)).height / total_molecules for x in cdf_x]
            )
        else:
            # Keep type consistent: return empty arrays if caller doesn't want the CDF computed/used.
            cdf_x = np.array([], dtype=float)
            cdf_y = np.array([], dtype=float)

        return all_stats, cdf_x, cdf_y

    def plot_fpr_vs_info_metrics(
        all_stats: pl.DataFrame,
        cdf_x: np.ndarray,
        cdf_y: np.ndarray,
        config: FprVsInfoConfig,
    ) -> None:
        """
        Plot average false matches vs Spectral Information score and optionally save as config.metrics_output_path.
        If config.show_molecule_cdf == False, the Molecule Coverage CDF isn't plotted.
        """
        assert "metric_name" in all_stats.columns, "all_stats missing 'metric_name' column; compute stats first."

        fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")

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
                    label=f"{label} (Threshold: {thresh})",
                    color=color,
                )
            else:
                print(f"Warning: No data for {label} at threshold {thresh}")

        # Secondary axis for Molecule Coverage CDF if requested
        if config.show_molecule_cdf and cdf_x.size and cdf_y.size:
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

            lines_1, labels_1 = ax.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
        else:
            # If we didn't plot the CDF, just show the main legend
            ax.legend(loc="upper right")

        xlabel = "Weighted Spectral Information Score" if config.use_weighted_information_score else "Spectral Information Score"
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Average Number of False Matches")
        # ax.set_title(f"Average Number of False Matches vs {xlabel}")

        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(config.metrics_output_path), facecolor="white", transparent=False)
        plt.close(fig)

    def plot_matched_avg_info_diff(
        all_stats: pl.DataFrame,
        config: FprVsInfoConfig,
        show_plot: bool = False,
        relative: bool = False,
    ) -> None:
        """
        Plot average matched information difference vs Spectral Information score (filtered) and save to file.
        """
        assert "avg_matched_info_diff" in all_stats.columns, (
            "avg_matched_info_diff missing from aggregated results; ensure compute pipeline computed this column"
        )

        fig_matched, ax_matched = plt.subplots(figsize=(10, 6), facecolor="white")

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
                    avg_matched_y if not relative else avg_matched_y / info_x_matched,
                    linestyle="--",
                    marker="x",
                    label=f"{label} (Threshold: {thresh})",
                    color=color,
                    alpha=0.9,
                )
            else:
                print(f"Warning: No matched-info data for {label} at threshold {thresh}")

        xlabel = "Weighted Spectral Information Score" if config.use_weighted_information_score else "Spectral Information Score"
        ylabel = "Average Matched Weighted Information Difference" if config.use_weighted_information_score else "Average Matched Information Difference"

        ax_matched.set_xlabel(xlabel)
        ax_matched.set_ylabel(ylabel)
        # ax_matched.set_title(f"{ylabel} vs {xlabel}")
        ax_matched.legend(loc="upper right")
        ax_matched.grid(True, alpha=0.3)
        fig_matched.tight_layout()
        fig_matched.savefig(str(config.matched_info_output_path), facecolor="white", transparent=False)
        if show_plot:
            plt.show()
        plt.close(fig_matched)

    def _rankdata(arr: np.ndarray) -> np.ndarray:
        """
        Compute average ranks for ties (equivalent to scipy.stats.rankdata(method='average')).
        Why: We want a SciPy-free implementation for Spearman correlation.
        """
        assert arr.ndim == 1, "rankdata requires a 1D array"
        n = arr.size
        if n == 0:
            return np.array([], dtype=float)
        # Compute initial ranks (1..n)
        order = np.argsort(arr, kind='mergesort')
        ranks = np.empty(n, dtype=float)
        ranks[order] = np.arange(1, n + 1)
        # Handle ties: average the ranks of tied values
        unique_vals, inverse_indices, counts = np.unique(arr, return_inverse=True, return_counts=True)
        if unique_vals.size == n:
            return ranks  # all unique
        # For each unique value, set tied ranks to their mean rank
        for val_idx, count in enumerate(counts):
            if count == 1:
                continue
            mask = inverse_indices == val_idx
            ranks[mask] = ranks[mask].mean()
        return ranks

    def _spearmanrho(x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute Spearman rho between two vectors using rank-transformed Pearson correlation.
        Why: Fast, dependency-free, returns NaN when undefined (e.g., insufficient variability).
        """
        assert x.ndim == 1 and y.ndim == 1 and x.size == y.size, "Inputs to spearman must be 1D arrays of equal length."
        if x.size < 2:
            return float("nan")
        rx = _rankdata(x.astype(float))
        ry = _rankdata(y.astype(float))
        # Handle constant arrays: Pearson correlation undefined
        if np.all(rx == rx[0]) or np.all(ry == ry[0]):
            return float("nan")
        rho = np.corrcoef(rx, ry)[0, 1]
        return float(rho)
    # ...existing code...
    def compute_spearman_correlations(
        config: FprVsInfoConfig,
        pairs_input: Optional[Union[str, Path, pl.DataFrame, pl.LazyFrame]] = None,
        compute_on_binned: bool = True,
    ) -> pl.DataFrame:
        """
        Compute per-metric Spearman correlations between binned FPR (avg_false_matches) and measures:
          * Spectral Information Score (config.bin_width)
          * Spectral Entropy (config.entropy_bin_width or config.bin_width)
          * Number of Clean Peaks (peaks_bin_width or 1)

        Returns:
            pl.DataFrame with columns:
              - metric_name, threshold_used, measure_name, spearman_rho, n_bins_used
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

        info_col = "weighted_spectral_information_score" if config.use_weighted_information_score else "spectral_information_score"

        # Measures (col, label, bin_width)
        ent_bw = config.entropy_bin_width if config.entropy_bin_width is not None else config.bin_width
        peaks_bw = config.peaks_bin_width if config.peaks_bin_width is not None else 1.0

        measures = [
            (info_col, "Spectral Information Score", config.bin_width),
            ("spectral_entropy", "Spectral Entropy", ent_bw),
            ("num_clean_peaks", "Number of Clean Peaks", peaks_bw),
        ]

        results = []
        for sim_col, thresh, label, color, marker in config.metrics_config:
            print(f"Computing Spearman correlations for metric {label} (threshold {thresh})")
            # Create a per-spectrum row with is_false_match flags and measure values
            base = (
                pairs_sim.with_columns(is_false_match=(pl.col(sim_col).ge(thresh)).cast(pl.Int64))
                .with_columns(false_compound_count=pl.col("is_false_match").sum().over("base_inchikey_right", "idx"))
                .with_columns(false_spectra_count=pl.col("is_false_match").sum().over("idx"))
                .unique(subset="idx", keep="any")
            )

            for col_name, measure_label, bw in measures:
                if compute_on_binned:
                    # existing binned logic (unchanged)
                    binned = base.with_columns(
                        measure_val=pl.col(col_name),
                        measure_bin_val=(pl.col(col_name) / bw).floor() * bw
                    ).filter(
                        pl.col("measure_bin_val").is_not_null()
                    ).group_by("measure_bin_val").agg(
                        total_count=pl.len(),
                        avg_false_matches=pl.col("false_compound_count").mean()
                    ).sort("measure_bin_val").collect(engine="streaming")

                    # Filter bins with insufficient counts
                    binned_filtered = binned.filter(pl.col("total_count") > config.min_count_threshold)
                    if binned_filtered.height < 2:
                        # not enough bins to compute Spearman
                        results.append({
                            "metric_name": sim_col,
                            "threshold_used": thresh,
                            "measure_name": measure_label,
                            "spearman_rho": float("nan"),
                            "n_bins_used": int(binned_filtered.height),
                        })
                        continue

                    bins_x = binned_filtered.get_column("measure_bin_val").to_numpy().astype(float)
                    avg_y = binned_filtered.get_column("avg_false_matches").to_numpy().astype(float)
                    rho = _spearmanrho(bins_x, avg_y)
                    results.append({
                        "metric_name": sim_col,
                        "threshold_used": thresh,
                        "measure_name": measure_label,
                        "spearman_rho": float(rho),
                        "n_bins_used": int(binned_filtered.height),
                    })
                else:
                    # Raw (per-spectrum) Spearman computation:
                    # - Build per-spectrum measure_val and false_compound_count
                    # - Optionally clip measure range (entropy/peaks)
                    # - Collect the per-spectrum frame (reduced size), and compute spearman
                    df_raw = base.with_columns(
                        measure_val=pl.col(col_name)
                    ).select(
                        ["idx", "measure_val", "false_compound_count"]
                    ).filter(
                        pl.col("measure_val").is_not_null()
                    )

                    # Clip to optional max values (same semantics as plotting)
                    if col_name == "spectral_entropy" and config.max_spectral_entropy is not None:
                        df_raw = df_raw.filter(pl.col("measure_val") <= float(config.max_spectral_entropy))
                    if col_name == "num_clean_peaks" and config.max_num_clean_peaks is not None:
                        df_raw = df_raw.filter(pl.col("measure_val") <= float(config.max_num_clean_peaks))

                    # Collect reduced per-spectrum table for rank calculation:
                    df_collected = df_raw.collect(engine="streaming")
                    n_rows = df_collected.height
                    if n_rows < max(2, config.min_count_threshold):
                        results.append({
                            "metric_name": sim_col,
                            "threshold_used": thresh,
                            "measure_name": measure_label,
                            "spearman_rho": float("nan"),
                            "n_bins_used": int(n_rows),  # not bins; keep concisely named for compatibility
                        })
                        continue

                    x = df_collected.get_column("measure_val").to_numpy().astype(float)
                    y = df_collected.get_column("false_compound_count").to_numpy().astype(float)
                    rho = _spearmanrho(x, y)
                    results.append({
                        "metric_name": sim_col,
                        "threshold_used": thresh,
                        "measure_name": measure_label,
                        "spearman_rho": float(rho),
                        "n_bins_used": int(n_rows),
                    })

        return pl.DataFrame(results)
    # ...existing code...
    def plot_fpr_vs_measure_with_spearman(
        pairs_input: Optional[Union[str, Path, pl.DataFrame, pl.LazyFrame]],
        config: FprVsInfoConfig,
        measure_col: str,
        measure_label: str,
        measure_bin_width: float,
        spearman_df: Optional[pl.DataFrame] = None,
        output_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Generic plotter for FPR (average false matches) vs a given per-spectrum measure.
        Annotates Spearman rho for each metric/threshold if provided via spearman_df.

        Args:
          - pairs_input: dataset (str/path/DF/lazy)
          - config: FprVsInfoConfig
          - measure_col: column name in pairs dataset (e.g., 'spectral_entropy')
          - measure_label: human-readable label for x axis
          - measure_bin_width: bin width to group by for plotting
          - spearman_df: optional precomputed spearman DataFrame (pl.DataFrame)
          - output_path: optional output figure file path (falls back to config fields if None)
        """
        if pairs_input is None:
            pairs_input = config.pairs_parquet_path

        if isinstance(pairs_input, (str, Path)):
            assert Path(pairs_input).exists(), f"Parquet path does not exist: {pairs_input}"
            pairs_sim = pl.scan_parquet(str(pairs_input))
        elif isinstance(pairs_input, pl.DataFrame):
            pairs_sim = pairs_input.lazy()
        elif isinstance(pairs_input, pl.LazyFrame):
            pairs_sim = pairs_input
        else:
            raise AssertionError("pairs_input must be str/Path/pl.DataFrame/pl.LazyFrame")

        # compute aggregated stats for plotting per metric like earlier but for a generic measure
        fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
        for metric_key, thresh, label, color, marker in config.metrics_config:
            df = (pairs_sim.with_columns(is_false_match=(pl.col(metric_key).ge(thresh)).cast(pl.Int64))
                  .with_columns(false_compound_count=pl.col("is_false_match").sum().over("base_inchikey_right", "idx"))
                  .unique(subset="idx", keep="any")
                  .with_columns(measure_bin_val=(pl.col(measure_col) / measure_bin_width).floor() * measure_bin_width)
                  .filter(pl.col("measure_bin_val").is_not_null())
                  .group_by("measure_bin_val").agg(
                        avg_false_matches=pl.col("false_compound_count").mean(),
                        total_count=pl.len()
                    ).sort("measure_bin_val").collect(engine="streaming"))

            # Clip to optional max values for visual consistency and consistent rho calculation.
            if measure_col == "spectral_entropy" and config.max_spectral_entropy is not None:
                df = df.filter(pl.col("measure_bin_val") <= float(config.max_spectral_entropy))
            if measure_col == "num_clean_peaks" and config.max_num_clean_peaks is not None:
                df = df.filter(pl.col("measure_bin_val") <= float(config.max_num_clean_peaks))

            subset = df.filter(pl.col("total_count") > config.min_count_threshold)
            if subset.height == 0:
                print(f"Warning: No data (satisfies count threshold) for {measure_label} with {label} at threshold {thresh}")
                continue

            x = subset.get_column("measure_bin_val").to_numpy()
            y = subset.get_column("avg_false_matches").to_numpy()
            ax.plot(x, y, marker=marker, label=f"{label} (threshold {thresh})", color=color)

        ax.set_xlabel(measure_label)
        ax.set_ylabel("Average Number of False Matches")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
        fig.tight_layout()

        if output_path is None:
            if measure_col == "spectral_entropy":
                out = config.plot_entropy_output_path
            elif measure_col == "num_clean_peaks":
                out = config.plot_peaks_output_path
            elif measure_col in ("spectral_information_score", "weighted_spectral_information_score"):
                out = config.metrics_output_path
            else:
                out = f"fpr_vs_{measure_col}.png"
        else:
            out = output_path
        fig.savefig(str(out), facecolor="white", transparent=False)
        plt.close(fig)
    return (
        FprVsInfoConfig,
        compute_fpr_vs_info_stats,
        compute_spearman_correlations,
        plot_fpr_vs_info_metrics,
        plot_fpr_vs_measure_with_spearman,
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
    Union,
    crossTanimotoSimilarityMemoryConstrained,
    pl,
    plt,
):
    def compute_avg_tanimoto_between_matched_pairs(
        config: FprVsInfoConfig,
        pairs_input: Optional[Union[str, Path, pl.DataFrame, pl.LazyFrame]] = None,
        group_by_cols: Optional[List[str]] = None,
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
          - config: FprVsInfoConfig object containing various parameters for the computation and plotting.
          - pairs_input: Path/lazyframe/dataframe for the pairwise dataset. If None, uses config.pairs_parquet_path.
          - group_by_cols: columns to aggregate by (e.g., ["ion_mode"]). Used only if metrics_config is None.

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
        sample = pairs_sim.select([config.left_smiles_col, config.right_smiles_col]).limit(1).collect(engine="streaming")
        assert config.left_smiles_col in sample.columns and config.right_smiles_col in sample.columns, (
            f"Expected columns '{config.left_smiles_col}' and '{config.right_smiles_col}' in pairs dataset"
        )

        # Compute unique SMILES lists for left and right
        # Why: Add row index to map back to the similarity matrix coordinates
        # Use streaming to handle large datasets without OOM
        left_unique = pairs_sim.select(config.left_smiles_col).unique().with_row_index("l_idx").collect(engine="streaming")
        right_unique = pairs_sim.select(config.right_smiles_col).unique().with_row_index("r_idx").collect(engine="streaming")

        left_smiles_list = left_unique.get_column(config.left_smiles_col).to_list()
        right_smiles_list = right_unique.get_column(config.right_smiles_col).to_list()

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
        fpgen = MorganFingerprintGenerator(radius=config.fp_radius, fpSize=config.fp_size)
        fps_left = fpgen.GetFingerprints(left_mols, num_threads=config.fp_num_threads)
        fps_right = fpgen.GetFingerprints(right_mols, num_threads=config.fp_num_threads)

        # Compute pairwise Tanimoto only between the unique left/right groups
        # Why: crossTanimotoSimilarityMemoryConstrained returns a numpy array directly
        # Note: This matrix can be large (N_left * N_right * 4 bytes).
        sims_matrix = crossTanimotoSimilarityMemoryConstrained(fps_left, fps_right)  # shape [len(left), len(right)]

        # Build mapping of unique pairs -> tanimoto
        # Why: Use streaming to avoid OOM on large pair sets
        unique_pairs = pairs_sim.select([config.left_smiles_col, config.right_smiles_col]).unique().collect(engine="streaming")
        if unique_pairs.height == 0:
            # No pairs to compute.
            print("No unique pairs found for Tanimoto computation; returning empty DataFrame.")
            return pl.DataFrame(
                {"avg_tanimoto": [], "median_tanimoto": [], "count": []}
            )

        # Join unique pairs with the indexed unique lists to get matrix coordinates
        # Why: Use Polars join for efficiency instead of Python dictionary lookups
        pairs_indices = unique_pairs.join(
            left_unique, on=config.left_smiles_col
        ).join(
            right_unique, on=config.right_smiles_col
        )

        l_idxs = pairs_indices.get_column("l_idx").to_numpy()
        r_idxs = pairs_indices.get_column("r_idx").to_numpy()

        # Extract values using numpy fancy indexing
        # Why: Vectorized lookup is much faster than iterating in Python
        tanimoto_values = sims_matrix[l_idxs, r_idxs]

        mapping_df = pairs_indices.select([config.left_smiles_col, config.right_smiles_col]).with_columns(
            tanimoto_similarity=pl.Series(tanimoto_values)
        )

        # Join mapping back to the pairs lazyframe to assign similarity to each pair
        pairs_with_sim = pairs_sim.join(mapping_df.lazy(), left_on=[config.left_smiles_col, config.right_smiles_col], right_on=[config.left_smiles_col, config.right_smiles_col])

        info_col = "weighted_spectral_information_score" if config.use_weighted_information_score else "spectral_information_score"

        # Case 1: Metrics Config provided (Iterate, Filter, Aggregate, Plot)
        if config.metrics_config is not None:
            stats = []
            for metric_key, thresh, label, color, marker in config.metrics_config:
                # Filter pairs that meet the threshold for this specific metric
                # And bin by spectral information score
                res = pairs_with_sim.filter(
                    pl.col(metric_key).ge(thresh)
                ).with_columns(
                    info_bin_val=(pl.col(info_col) / config.bin_width).floor() * config.bin_width
                ).filter(
                    pl.col("info_bin_val").ge(0.0),
                    pl.col("info_bin_val").le(config.max_info),
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

            if config.plot_avg_tanimoto_output_path is not None:
                fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
                for metric_key, thresh, label, color, marker in config.metrics_config:
                    subset = all_stats.filter(
                        (pl.col("metric_name") == metric_key) &
                        (pl.col("threshold_used") == thresh) &
                        (pl.col("count") > config.min_count_threshold)
                    )
                    if subset.height > 0:
                        ax.plot(
                            subset.get_column("info_bin_val").to_numpy(),
                            subset.get_column("avg_tanimoto").to_numpy(),
                            marker=marker,
                            label=f"{label} (Threshold: {thresh})",
                            color=color,
                        )

                xlabel = "Weighted Spectral Information Score" if config.use_weighted_information_score else "Spectral Information Score"
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Average Tanimoto Similarity of Matches")
                ax.set_title(f"Average Tanimoto Similarity of Matches vs {xlabel}")
                ax.legend(loc="lower right")
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(str(config.plot_avg_tanimoto_output_path), facecolor="white", transparent=False)
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
def _(FprVsInfoConfig, OUTPUT_PAIRS_PATH):
    # Analysis and Plotting
    # Why: Bin spectral info score and calculate average number of false matches for different similarity metrics.
    # We aggregate by the left spectrum ('idx') to count how many false positive matches exist above the threshold.

    # Use the new dataclass/config-driven analysis and plotting
    config = FprVsInfoConfig(
        pairs_parquet_path=OUTPUT_PAIRS_PATH,
        max_info=3,
        bin_width=0.2,
        entropy_bin_width=0.5,
        max_spectral_entropy=5.0,
        peaks_bin_width=3,
        max_num_clean_peaks=200,
        metrics_config=[
            ("dotprod_similarity", 0.80, "Dot Product", "C0", "o"),
            ("dotprod_similarity", 0.90, "Dot Product", "C1", "o"),
            # ("entropy_similarity", 0.75, "Entropy", "C2", "^"),
        ],
        metrics_output_path="fpr_vs_info_metrics.png",
        matched_info_output_path="avg_matched_info_diff.png",
        left_smiles_col="smiles",
        right_smiles_col="smiles_right",
        fp_radius=2,
        fp_size=2048,
        fp_num_threads=0,
        plot_avg_tanimoto_output_path="avg_tanimoto_vs_info.png",
        use_weighted_information_score=False,
        show_molecule_cdf=False,
    )
    return (config,)


@app.cell
def _(
    compute_fpr_vs_info_stats,
    config,
    plot_fpr_vs_info_metrics,
    plot_matched_avg_info_diff,
):

    all_stats, cdf_x, cdf_y = compute_fpr_vs_info_stats(config)

    plot_fpr_vs_info_metrics(all_stats, cdf_x, cdf_y, config)
    plot_matched_avg_info_diff(all_stats, config,relative=True)
    return


@app.cell
def _(FprVsInfoConfig, OUTPUT_PAIRS_PATH):
    tanimoto_config = FprVsInfoConfig(
        pairs_parquet_path=OUTPUT_PAIRS_PATH,
        max_info=3,
        bin_width=0.5,
        metrics_config=[
            # ("dotprod_similarity", 0.80, "Dot Product", "C0", "o"),
            ("dotprod_similarity", 0.90, "Dot Product", "C1", "o"),
            # ("entropy_similarity", 0.75, "Entropy", "C2", "^"),
        ],
        metrics_output_path="fpr_vs_info_metrics.png",
        matched_info_output_path="avg_matched_info_diff.png",
        left_smiles_col="smiles",
        right_smiles_col="smiles_right",
        fp_radius=2,
        fp_size=2048,
        fp_num_threads=0,
        plot_avg_tanimoto_output_path="avg_tanimoto_vs_info.png",
        use_weighted_information_score=False,
        show_molecule_cdf=False,
    )
    return (tanimoto_config,)


@app.cell
def _(compute_avg_tanimoto_between_matched_pairs, tanimoto_config):

    avg_tanimoto_df = compute_avg_tanimoto_between_matched_pairs(
        config=tanimoto_config,
        pairs_input=tanimoto_config.pairs_parquet_path,
    )
    avg_tanimoto_df
    return


@app.cell
def _(mo):
    mo.md(r"""
    Computing Spearman correlations for metric Dot Product (threshold 0.8)
    Computing Spearman correlations for metric Dot Product (threshold 0.9)
    Spearman correlations:
    shape: (6, 5)
    ┌────────────────────┬────────────────┬────────────────────────────┬──────────────┬─────────────┐
    │ metric_name        ┆ threshold_used ┆ measure_name               ┆ spearman_rho ┆ n_bins_used │
    │ ---                ┆ ---            ┆ ---                        ┆ ---          ┆ ---         │
    │ str                ┆ f64            ┆ str                        ┆ f64          ┆ i64         │
    ╞════════════════════╪════════════════╪════════════════════════════╪══════════════╪═════════════╡
    │ dotprod_similarity ┆ 0.8            ┆ Spectral Information Score ┆ -0.151778    ┆ 1193386     │
    │ dotprod_similarity ┆ 0.8            ┆ Spectral Entropy           ┆ -0.14195     ┆ 1192688     │
    │ dotprod_similarity ┆ 0.8            ┆ Number of Clean Peaks      ┆ -0.133371    ┆ 1182361     │
    │ dotprod_similarity ┆ 0.9            ┆ Spectral Information Score ┆ -0.149285    ┆ 1193386     │
    │ dotprod_similarity ┆ 0.9            ┆ Spectral Entropy           ┆ -0.146958    ┆ 1192688     │
    │ dotprod_similarity ┆ 0.9            ┆ Number of Clean Peaks      ┆ -0.137213    ┆ 1182361     │
    └────────────────────┴────────────────┴────────────────────────────┴──────────────┴─────────────┘
    those are the non binned results, which take 10 minuted to calculate
    """)
    return


@app.cell
def _(FprVsInfoConfig, OUTPUT_PAIRS_PATH, compute_spearman_correlations):
    spearman_config = FprVsInfoConfig(
        pairs_parquet_path=OUTPUT_PAIRS_PATH,
        max_info=3,
        bin_width=0.01,
        entropy_bin_width=0.01,
        max_spectral_entropy=5.0,
        peaks_bin_width=1.0,
        max_num_clean_peaks=200,
        metrics_config=[
            ("dotprod_similarity", 0.90, "Dot Product", "C1", "o"),
        ],
        metrics_output_path="fpr_vs_info_metrics.png",
        matched_info_output_path="avg_matched_info_diff.png",
  
    )
    print(compute_spearman_correlations(spearman_config, pairs_input=spearman_config.pairs_parquet_path,compute_on_binned=True))
    return


@app.cell
def _(
    compute_spearman_correlations,
    config,
    plot_fpr_vs_measure_with_spearman,
):
    spearman_df = compute_spearman_correlations(
        config,
        pairs_input=config.pairs_parquet_path,
        compute_on_binned=True,
    )
    plot_fpr_vs_measure_with_spearman(config.pairs_parquet_path, config, "spectral_entropy", "Spectral Entropy", config.entropy_bin_width or config.bin_width, spearman_df)
    # Plot for number of clean peaks
    plot_fpr_vs_measure_with_spearman(config.pairs_parquet_path, config, "num_clean_peaks", "Number of Clean Peaks", config.peaks_bin_width or 1.0, spearman_df)
    # Plot for spectral information score (existing behavior annotated with rho)
    plot_fpr_vs_measure_with_spearman(config.pairs_parquet_path, config, ("weighted_spectral_information_score" if config.use_weighted_information_score else "spectral_information_score"), \
        ("Weighted Spectral Information Score" if config.use_weighted_information_score else "Spectral Information Score"), config.bin_width, spearman_df, output_path=config.metrics_output_path)
    return


if __name__ == "__main__":
    app.run()
