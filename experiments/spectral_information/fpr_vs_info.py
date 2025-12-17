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
        lf = lf.collect().lazy().select(
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
            num_clean_peaks=pl.col("cleaned_normalized_mz").list.len(),
            normalized_spectral_information_score=( 
                # here we normalize the SIS per molecule+Ion mode, so its a fraction of the max possible SIS for that molecule
                pl.col("spectral_information_score").truediv(
                    pl.col("spectral_information_score").max().over(["base_inchikey", "ion_mode"])
                    )
            )
        ).with_columns(
            most_informative=pl.col("normalized_spectral_information_score").eq(1.0),
        ).collect().lazy()

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
def _(
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
):
    from dataclasses import replace
    from typing import Literal, Dict

    MeasureName = Literal[
        "spectral_information_score",
        "weighted_spectral_information_score",
        "normalized_spectral_information_score",
        "spectral_entropy",
        "num_clean_peaks",
    ]

    InfoMeasureName = Literal[
        "spectral_information_score",
        "weighted_spectral_information_score",
        "normalized_spectral_information_score",
    ]

    def _measure_label(measure: MeasureName) -> str:
        # Why: Keep all axis naming in one place so config changes propagate everywhere.
        match measure:
            case "spectral_information_score":
                return "Spectral Information Score"
            case "weighted_spectral_information_score":
                return "Weighted Spectral Information Score"
            case "normalized_spectral_information_score":
                return "Normalized Spectral Information Score"
            case "spectral_entropy":
                return "Spectral Entropy"
            case "num_clean_peaks":
                return "Number of Clean Peaks"
            case _:
                raise AssertionError(
                    f"Unknown measure '{measure}'. Supported measures are: "
                    "spectral_information_score, weighted_spectral_information_score, normalized_spectral_information_score, "
                    "spectral_entropy, num_clean_peaks."
                )

    def _default_x_range(measure: MeasureName) -> tuple[float, float]:
        # Why: Provide deterministic defaults while still letting callers override with x_range.
        match measure:
            case "spectral_information_score" | "weighted_spectral_information_score" | "normalized_spectral_information_score":
                return (0.0, 3.0)
            case "spectral_entropy":
                return (0.0, 5.0)
            case "num_clean_peaks":
                return (0.0, 200.0)
            case _:
                raise AssertionError(f"No default x_range for measure '{measure}'.")

    def _resolve_x_range(config: "FprVsInfoConfig") -> tuple[float, float]:
        if config.x_range is not None:
            x_min, x_max = config.x_range
            assert x_max >= x_min, f"x_range must satisfy max>=min, got {config.x_range}"
            return (float(x_min), float(x_max))
        return _default_x_range(config.x_measure)

    def _resolve_info_difference_measure(config: "FprVsInfoConfig") -> InfoMeasureName:
        # Why: Matched-info-diff is only meaningful for info-based measures; allow override but keep sane default.
        if config.info_difference_measure is not None:
            return config.info_difference_measure
        match config.x_measure:
            case "spectral_information_score" | "weighted_spectral_information_score" | "normalized_spectral_information_score":
                return config.x_measure  # type: ignore[return-value]
            case _:
                return "spectral_information_score"

    @dataclass
    class FprVsInfoConfig:
        pairs_parquet_path: Union[str, Path] = "/home/analytit_admin/Data/spectral_libs/all_pairs_with_similarities.parquet"

        # Unified x-axis configuration used by FPR, Tanimoto, and Spearman computations/plots.
        x_measure: MeasureName = "spectral_information_score"
        x_bin_width: float = 0.1
        x_range: Optional[Tuple[float, float]] = None

        # Matched-info difference configuration (info-only; defaults to x_measure if it is an info measure).
        info_difference_measure: Optional[InfoMeasureName] = None

        # metric_key, threshold, human_label, color, marker
        metrics_config: List[Tuple[str, float, str, str, str]] = field(
            default_factory=lambda: [
                ("dotprod_similarity", 0.8, "Dot Product", "C0", "o"),
                ("dotprod_similarity", 0.95, "Dot Product", "C1", "o"),
                ("entropy_similarity", 0.75, "Entropy", "C2", "^"),
            ]
        )
        min_count_threshold: int = 5

        # Unified output paths (used by all plots).
        fpr_output_path: Union[str, Path] = "fpr_vs_measure.png"
        matched_info_output_path: Union[str, Path] = "avg_matched_info_diff.png"
        tanimoto_output_path: Optional[Union[str, Path]] = "avg_tanimoto_vs_measure.png"

        # Optional plot features
        plot_show_std: bool = True
        plot_only_most_informative: bool = False
        show_molecule_cdf: bool = False

        # Tanimoto similarity function arguments
        left_smiles_col: str = "smiles"
        right_smiles_col: str = "smiles"
        fp_radius: int = 2
        fp_size: int = 1024
        fp_num_threads: int = 0

        # Why: convenience helper to create a modified copy of the config with fail-fast validation.
        def copy(self, **changes) -> "FprVsInfoConfig":
            valid_fields = set(self.__dataclass_fields__.keys())
            unknown = set(changes) - valid_fields
            assert not unknown, f"Unknown fields for FprVsInfoConfig.copy(): {sorted(list(unknown))}"
            return replace(self, **changes)

    def compute_fpr_vs_info_stats(
        config: FprVsInfoConfig,
        pairs_input: Optional[Union[str, Path, pl.DataFrame, pl.LazyFrame]] = None,
    ) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
        if pairs_input is None:
            pairs_input = config.pairs_parquet_path

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

        assert config.x_bin_width > 0, f"x_bin_width must be > 0, got {config.x_bin_width}"
        x_min, x_max = _resolve_x_range(config)

        # Fail fast on required columns before any heavy computation.
        required_cols = [
            "idx",
            "base_inchikey",
            "base_inchikey_right",
            config.x_measure,
            "most_informative",
        ]
        info_diff_col = _resolve_info_difference_measure(config)
        required_cols.extend([info_diff_col, f"{info_diff_col}_right"])
        required_cols = list(dict.fromkeys(required_cols))  # Deduplicate while preserving order
        pairs_sim.select(required_cols).limit(1).collect(engine="streaming")

        if config.plot_only_most_informative:
            pairs_sim = pairs_sim.filter(pl.col("most_informative"))

        stats = []
        for sim_col, thresh, label, color, marker in config.metrics_config:
            print(f"Processing metric: {label} with threshold {thresh}")

            unique_spectra_stats = (
                pairs_sim.with_columns(is_false_match=(pl.col(sim_col).ge(thresh)).cast(pl.Int64))
                .with_columns(
                    false_compound_count=pl.col("is_false_match").sum().over("base_inchikey_right", "idx"),
                    false_spectra_count=pl.col("is_false_match").sum().over("idx"),
                    avg_matched_info_diff=(
                        (pl.col(f"{info_diff_col}_right") - pl.col(f"{info_diff_col}"))
                        .filter(pl.col("is_false_match").eq(1))
                        .mean()
                        .over("idx")
                    ),
                )
                .unique(subset="idx", keep="any")
                .with_columns(info_bin_val=(pl.col(config.x_measure) / config.x_bin_width).floor() * config.x_bin_width)
                .filter(
                    pl.col("info_bin_val").is_not_null(),
                    pl.col("info_bin_val").ge(float(x_min)),
                    pl.col("info_bin_val").le(float(x_max)),
                )
            )

            res = (
                unique_spectra_stats.group_by("info_bin_val")
                .agg(
                    total_count=pl.len(),
                    avg_false_matches=pl.col("false_compound_count").mean(),
                    # Replace std with 10th and 90th percentiles so plots can show distribution tails.
                    lower_percentile=pl.col("false_compound_count").quantile(0.1).alias("lower_percentile"),
                    upper_percentile=pl.col("false_compound_count").quantile(0.9).alias("upper_percentile"),
                    avg_matched_info_diff=pl.col("avg_matched_info_diff").mean(),
                )
                .with_columns(
                    metric_label=pl.lit(label),
                    metric_name=pl.lit(sim_col),
                    threshold_used=pl.lit(thresh),
                    plot_color=pl.lit(color),
                    plot_marker=pl.lit(marker),
                )
                .sort("info_bin_val")
            )
            stats.append(res)

        all_stats = pl.concat(stats).collect(engine="streaming")

        if config.show_molecule_cdf:
            assert config.x_measure in (
                "spectral_information_score",
                "weighted_spectral_information_score",
                "normalized_spectral_information_score",
            ), (
                "show_molecule_cdf is only supported for info-based x_measure values; "
                f"got x_measure={config.x_measure}."
            )
            molecule_max_x = pairs_sim.group_by("base_inchikey").agg(max_x=pl.col(config.x_measure).max()).collect(engine="streaming")
            cdf_x = np.arange(float(x_min), float(x_max) + float(config.x_bin_width), float(config.x_bin_width))
            total_molecules = molecule_max_x.height
            cdf_y = np.array([molecule_max_x.filter(pl.col("max_x") >= float(x)).height / total_molecules for x in cdf_x])
        else:
            cdf_x = np.array([], dtype=float)
            cdf_y = np.array([], dtype=float)

        return all_stats, cdf_x, cdf_y

    def plot_fpr_vs_info_metrics(
        all_stats: pl.DataFrame,
        cdf_x: np.ndarray,
        cdf_y: np.ndarray,
        config: FprVsInfoConfig,
    ) -> None:
        assert "metric_name" in all_stats.columns, "all_stats missing 'metric_name' column; compute stats first."

        fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")

        for metric_key, thresh, label, color, marker in config.metrics_config:
            subset = all_stats.filter(
                (pl.col("metric_name") == metric_key) &
                (pl.col("threshold_used") == thresh) &
                (pl.col("total_count") > config.min_count_threshold)
            )
            if subset.height == 0:
                print(f"Warning: No data for {label} at threshold {thresh}")
                continue

            info_x = subset.get_column("info_bin_val").to_numpy()
            avg_false_y = subset.get_column("avg_false_matches").to_numpy()

            # Spearman computed on binned x then binned y (Polars).
            rho_val = subset.select(pl.corr("info_bin_val", "avg_false_matches", method="spearman")).item()
            # Print the correlation for visibility in logs and keep it in the legend label.
            print(f"Spearman rho for {label} (threshold {thresh}): {np.round(rho_val, 3)}")
            series_label = f"{label} (threshold {thresh})"

            ax.plot(info_x, avg_false_y, marker=marker, label=series_label, color=color)

            # Show percentile boundaries and fill between them (replace earlier std shading).
            if config.plot_show_std and "lower_percentile" in subset.columns and "upper_percentile" in subset.columns:
                lower_y = subset.get_column("lower_percentile").to_numpy()
                upper_y = subset.get_column("upper_percentile").to_numpy()
                # Fill between 10th and 90th percentiles (similar visual cue to previous std fill).
                ax.fill_between(info_x, lower_y, upper_y, color=color, alpha=0.15)
                # Also draw dashed lines at the percentile boundaries for clarity.
                ax.plot(info_x, lower_y, linestyle="--", color=color, alpha=0.7)
                ax.plot(info_x, upper_y, linestyle="--", color=color, alpha=0.7)

        if config.show_molecule_cdf and cdf_x.size and cdf_y.size:
            ax2 = ax.twinx()
            ax2.plot(
                cdf_x,
                cdf_y,
                color="black",
                linestyle="--",
                linewidth=2,
                alpha=0.6,
                label=f"Molecule Coverage (max {_measure_label(config.x_measure)} ≥ X)",
            )
            ax2.set_ylabel("Fraction of Molecules")
            ax2.set_ylim(0, 1.05)
            lines_1, labels_1 = ax.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
        else:
            ax.legend(loc="upper right")

        ax.set_xlabel(_measure_label(config.x_measure))
        ax.set_ylabel("Average Number of False Matches")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(config.fpr_output_path), facecolor="white", transparent=False)
        plt.close(fig)

    def plot_matched_avg_info_diff(
        all_stats: pl.DataFrame,
        config: FprVsInfoConfig,
        show_plot: bool = False,
        relative: bool = False,
    ) -> None:
        assert "avg_matched_info_diff" in all_stats.columns, (
            "avg_matched_info_diff missing from aggregated results; ensure compute pipeline computed this column"
        )

        info_diff_col = _resolve_info_difference_measure(config)
        fig_matched, ax_matched = plt.subplots(figsize=(10, 6), facecolor="white")

        for metric_key, thresh, label, color, marker in config.metrics_config:
            subset = all_stats.filter(
                (pl.col("metric_name") == metric_key) &
                (pl.col("threshold_used") == thresh) &
                (pl.col("total_count") > config.min_count_threshold)
            )
            matched_subset = subset.filter(pl.col("avg_matched_info_diff").is_not_null())
            if matched_subset.height == 0:
                print(f"Warning: No matched-info data for {label} at threshold {thresh}")
                continue

            x_vals = matched_subset.get_column("info_bin_val").to_numpy()
            y_vals = matched_subset.get_column("avg_matched_info_diff").to_numpy()
            ax_matched.plot(
                x_vals,
                y_vals if not relative else (y_vals / x_vals),
                linestyle="--",
                marker="x",
                label=f"{label} (threshold {thresh})",
                color=color,
                alpha=0.9,
            )

        ax_matched.set_xlabel(_measure_label(config.x_measure))
        ylabel = f"Average Matched Δ ({_measure_label(info_diff_col)})"
        ax_matched.set_ylabel(ylabel)
        ax_matched.legend(loc="upper right")
        ax_matched.grid(True, alpha=0.3)
        fig_matched.tight_layout()
        fig_matched.savefig(str(config.matched_info_output_path), facecolor="white", transparent=False)
        if show_plot:
            plt.show()
        plt.close(fig_matched)

    def compute_spearman_correlations(
        config: FprVsInfoConfig,
        pairs_input: Optional[Union[str, Path, pl.DataFrame, pl.LazyFrame]] = None,
        compute_on_binned: bool = True,
    ) -> pl.DataFrame:
        if pairs_input is None:
            pairs_input = config.pairs_parquet_path

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

        assert config.x_bin_width > 0, f"x_bin_width must be > 0, got {config.x_bin_width}"
        x_min, x_max = _resolve_x_range(config)

        if config.plot_only_most_informative:
            pairs_sim = pairs_sim.filter(pl.col("most_informative"))

        measures: list[tuple[str, str, float]] = [
            (config.x_measure, _measure_label(config.x_measure), config.x_bin_width),
        ]

        results: list[dict[str, object]] = []
        for sim_col, thresh, label, color, marker in config.metrics_config:
            base = (
                pairs_sim.with_columns(is_false_match=(pl.col(sim_col).ge(thresh)).cast(pl.Int64))
                .with_columns(false_compound_count=pl.col("is_false_match").sum().over("base_inchikey_right", "idx"))
                .unique(subset="idx", keep="any")
            )

            for measure_col, measure_label, bw in measures:
                if compute_on_binned:
                    binned = (
                        base.with_columns(measure_bin_val=(pl.col(measure_col) / bw).floor() * bw)
                        .filter(pl.col("measure_bin_val").is_not_null())
                        .filter(pl.col("measure_bin_val").ge(float(x_min)), pl.col("measure_bin_val").le(float(x_max)))
                        .group_by("measure_bin_val")
                        .agg(total_count=pl.len(), avg_false_matches=pl.col("false_compound_count").mean())
                        .filter(pl.col("total_count") > config.min_count_threshold)
                        .sort("measure_bin_val")
                        .collect(engine="streaming")
                    )

                    if binned.height < 2:
                        results.append({
                            "metric_name": sim_col,
                            "threshold_used": float(thresh),
                            "measure_name": measure_label,
                            "spearman_rho": float("nan"),
                            "n_points_used": int(binned.height),
                        })
                        continue

                    rho = binned.select(pl.corr("measure_bin_val", "avg_false_matches", method="spearman")).item()
                    results.append({
                        "metric_name": sim_col,
                        "threshold_used": float(thresh),
                        "measure_name": measure_label,
                        "spearman_rho": float(rho),
                        "n_points_used": int(binned.height),
                    })
                else:
                    df_raw = (
                        base.select(
                            pl.col(measure_col).alias("x_val"),
                            pl.col("false_compound_count").alias("y_val"),
                        )
                        .filter(pl.col("x_val").is_not_null())
                        .filter(pl.col("x_val").ge(float(x_min)), pl.col("x_val").le(float(x_max)))
                        .collect(engine="streaming")
                    )

                    if df_raw.height < 2:
                        results.append({
                            "metric_name": sim_col,
                            "threshold_used": float(thresh),
                            "measure_name": measure_label,
                            "spearman_rho": float("nan"),
                            "n_points_used": int(df_raw.height),
                        })
                        continue

                    rho = df_raw.select(pl.corr("x_val", "y_val", method="spearman")).item()
                    results.append({
                        "metric_name": sim_col,
                        "threshold_used": float(thresh),
                        "measure_name": measure_label,
                        "spearman_rho": float(rho),
                        "n_points_used": int(df_raw.height),
                    })

        return pl.DataFrame(results)

    def compute_avg_tanimoto_between_matched_pairs(
        config: FprVsInfoConfig,
        pairs_input: Optional[Union[str, Path, pl.DataFrame, pl.LazyFrame]] = None,
        group_by_cols: Optional[List[str]] = None,
    ) -> pl.DataFrame:
        if pairs_input is None:
            pairs_input = config.pairs_parquet_path

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

        assert config.x_bin_width > 0, f"x_bin_width must be > 0, got {config.x_bin_width}"
        x_min, x_max = _resolve_x_range(config)

        # Fail fast on required columns for tanimoto + measure binning.
        pairs_sim.select([config.left_smiles_col, config.right_smiles_col, config.x_measure, "most_informative"]).limit(1).collect(engine="streaming")

        if config.plot_only_most_informative:
            pairs_sim = pairs_sim.filter(pl.col("most_informative"))

        left_unique = pairs_sim.select(config.left_smiles_col).unique().with_row_index("l_idx").collect(engine="streaming")
        right_unique = pairs_sim.select(config.right_smiles_col).unique().with_row_index("r_idx").collect(engine="streaming")

        left_smiles_list = left_unique.get_column(config.left_smiles_col).to_list()
        right_smiles_list = right_unique.get_column(config.right_smiles_col).to_list()
        assert len(left_smiles_list) > 0 and len(right_smiles_list) > 0, (
            "No smiles found on left or right side. Please verify SMILES columns."
        )

        left_mols = [Chem.MolFromSmiles(s) for s in left_smiles_list]
        right_mols = [Chem.MolFromSmiles(s) for s in right_smiles_list]
        invalid_left = [s for s, m in zip(left_smiles_list, left_mols) if m is None]
        invalid_right = [s for s, m in zip(right_smiles_list, right_mols) if m is None]
        assert not invalid_left and not invalid_right, (
            f"Invalid SMILES encountered. Left invalid examples: {invalid_left[:3]}, "
            f"Right invalid examples: {invalid_right[:3]}"
        )

        fpgen = MorganFingerprintGenerator(radius=config.fp_radius, fpSize=config.fp_size)
        fps_left = fpgen.GetFingerprints(left_mols, num_threads=config.fp_num_threads)
        fps_right = fpgen.GetFingerprints(right_mols, num_threads=config.fp_num_threads)
        sims_matrix = crossTanimotoSimilarityMemoryConstrained(fps_left, fps_right)

        unique_pairs = pairs_sim.select([config.left_smiles_col, config.right_smiles_col]).unique().collect(engine="streaming")
        if unique_pairs.height == 0:
            print("No unique pairs found for Tanimoto computation; returning empty DataFrame.")
            return pl.DataFrame({"avg_tanimoto": [], "median_tanimoto": [], "count": []})

        pairs_indices = unique_pairs.join(left_unique, on=config.left_smiles_col).join(right_unique, on=config.right_smiles_col)
        l_idxs = pairs_indices.get_column("l_idx").to_numpy()
        r_idxs = pairs_indices.get_column("r_idx").to_numpy()
        tanimoto_values = sims_matrix[l_idxs, r_idxs]

        mapping_df = pairs_indices.select([config.left_smiles_col, config.right_smiles_col]).with_columns(
            tanimoto_similarity=pl.Series(tanimoto_values)
        )
        pairs_with_sim = pairs_sim.join(
            mapping_df.lazy(),
            left_on=[config.left_smiles_col, config.right_smiles_col],
            right_on=[config.left_smiles_col, config.right_smiles_col],
        )

        if config.metrics_config is not None:
            stats = []
            for metric_key, thresh, label, color, marker in config.metrics_config:
                res = (
                    pairs_with_sim.filter(pl.col(metric_key).ge(thresh))
                    .with_columns(info_bin_val=(pl.col(config.x_measure) / config.x_bin_width).floor() * config.x_bin_width)
                    .filter(
                        pl.col("info_bin_val").is_not_null(),
                        pl.col("info_bin_val").ge(float(x_min)),
                        pl.col("info_bin_val").le(float(x_max)),
                    )
                    .group_by("info_bin_val")
                    .agg(
                        avg_tanimoto=pl.col("tanimoto_similarity").mean(),
                        median_tanimoto=pl.col("tanimoto_similarity").median(),
                        count=pl.len(),
                    )
                    .with_columns(
                        metric_name=pl.lit(metric_key),
                        threshold_used=pl.lit(thresh),
                        metric_label=pl.lit(label),
                        plot_color=pl.lit(color),
                        plot_marker=pl.lit(marker),
                    )
                    .sort("info_bin_val")
                )
                stats.append(res)

            all_stats = pl.concat(stats).collect(engine="streaming")

            if config.tanimoto_output_path is not None:
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
                            label=f"{label} (threshold {thresh})",
                            color=color,
                        )

                ax.set_xlabel(_measure_label(config.x_measure))
                ax.set_ylabel("Average Tanimoto Similarity of Matches")
                ax.legend(loc="lower right")
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(str(config.tanimoto_output_path), facecolor="white", transparent=False)
                plt.close(fig)

            return all_stats

        if group_by_cols:
            assert isinstance(group_by_cols, list) and len(group_by_cols) > 0, "group_by_cols must be a non-empty list"
            return pairs_with_sim.group_by(*group_by_cols).agg(
                avg_tanimoto=pl.col("tanimoto_similarity").mean(),
                median_tanimoto=pl.col("tanimoto_similarity").median(),
                count=pl.col("tanimoto_similarity").count(),
            ).collect(engine="streaming")

        return pairs_with_sim.select(
            pl.col("tanimoto_similarity").mean().alias("avg_tanimoto"),
            pl.col("tanimoto_similarity").median().alias("median_tanimoto"),
            pl.col("tanimoto_similarity").count().alias("count"),
        ).collect(engine="streaming")
    return (
        FprVsInfoConfig,
        compute_fpr_vs_info_stats,
        compute_spearman_correlations,
        plot_fpr_vs_info_metrics,
        plot_matched_avg_info_diff,
    )


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

    config = FprVsInfoConfig(
        pairs_parquet_path=OUTPUT_PAIRS_PATH,
        x_measure="spectral_information_score",
        x_bin_width=0.2,
        x_range=(0.0, 3.0),
        metrics_config=[
            ("dotprod_similarity", 0.80, "Dot Product", "C0", "o"),
            ("dotprod_similarity", 0.90, "Dot Product", "C1", "o"),
        ],
        fpr_output_path="fpr_vs_spectral_information_score.png",
        matched_info_output_path="avg_matched_info_diff.png",
        tanimoto_output_path="avg_tanimoto_vs_measure.png",
        left_smiles_col="smiles",
        right_smiles_col="smiles_right",
        fp_radius=2,
        fp_size=2048,
        fp_num_threads=0,
        plot_show_std=False,
        plot_only_most_informative=False,
        show_molecule_cdf=False,
    )
    return (config,)


@app.cell
def _(compute_fpr_vs_info_stats, config, plot_fpr_vs_info_metrics):
    # plto the fpr vs the spectral info and the normalized info (only fpr vs info, no avg or tanimoto)
    info_config = config.copy(
        x_measure="spectral_information_score",
        fpr_output_path="fpr_vs_spectral_information_score.png",
    )
    plot_fpr_vs_info_metrics(*compute_fpr_vs_info_stats(info_config), info_config)   
    return


@app.cell
def _(compute_fpr_vs_info_stats, config, plot_fpr_vs_info_metrics):
    normalized_info_config = config.copy(
        x_measure="normalized_spectral_information_score",
        x_bin_width=0.1,
        fpr_output_path="fpr_vs_normalized_spectral_information_score.png"
    )
    plot_fpr_vs_info_metrics(*compute_fpr_vs_info_stats(normalized_info_config), normalized_info_config)    
    return


@app.cell
def _(compute_fpr_vs_info_stats, config, plot_fpr_vs_info_metrics):
    entropy_config = config.copy(x_measure="spectral_entropy", x_bin_width=0.5, x_range=(0.0, 5.0), fpr_output_path="fpr_vs_spectral_entropy.png")

    plot_fpr_vs_info_metrics(*compute_fpr_vs_info_stats(entropy_config), entropy_config)

    peaks_config = config.copy(x_measure="num_clean_peaks", x_bin_width=3.0, x_range=(0.0, 200.0), fpr_output_path="fpr_vs_num_clean_peaks.png")
    plot_fpr_vs_info_metrics(*compute_fpr_vs_info_stats(peaks_config), peaks_config)
    return


@app.cell
def _(config):
    # Why: Use config.copy to change only what matters for this specific plot.
    tanimoto_config = config.copy(
        x_bin_width=0.5,
        metrics_config=[
            ("dotprod_similarity", 0.90, "Dot Product", "C1", "o"),
        ],
        tanimoto_output_path="avg_tanimoto_vs_measure.png",
    )
    return


@app.cell
def _(compute_spearman_correlations, config):
    spearman_config = config.copy(
        metrics_config=[("dotprod_similarity", 0.90, "Dot Product", "C1", "o")],
        x_bin_width=0.01,
    )
    print(compute_spearman_correlations(spearman_config, pairs_input=spearman_config.pairs_parquet_path, compute_on_binned=True))
    return


@app.cell
def _(
    compute_fpr_vs_info_stats,
    compute_spearman_correlations,
    config,
    plot_fpr_vs_info_metrics,
    plot_matched_avg_info_diff,
):
    spearman_df = compute_spearman_correlations(
        config,
        pairs_input=config.pairs_parquet_path,
        compute_on_binned=True,
    )

    # If you want alternate x-axes, make a config copy and rerun stats/plots with that config.

    all_stats, cdf_x, cdf_y = compute_fpr_vs_info_stats(config)
    plot_fpr_vs_info_metrics(all_stats, cdf_x, cdf_y, config)
    plot_matched_avg_info_diff(all_stats, config, relative=True)

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


if __name__ == "__main__":
    app.run()
