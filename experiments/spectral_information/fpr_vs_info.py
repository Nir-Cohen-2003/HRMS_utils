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
    from dataclasses import dataclass, field, replace
    from typing import List, Tuple, Union, Optional, Literal
    import numpy.typing as npt
    from rdkit import Chem
    from nvmolkit.fingerprints import MorganFingerprintGenerator
    from nvmolkit.similarity import crossTanimotoSimilarityMemoryConstrained
    import math
    import os
    from scipy.stats import spearmanr, wilcoxon

    os.environ["RUST_BACKTRACE"] = "full"
    return (
        Chem,
        List,
        Literal,
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
        replace,
        spearmanr,
        wilcoxon,
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
                    pl.col("spectral_information_score").mean().over(["base_inchikey", "ion_mode"])
                    )
            )
        ).with_columns(
            most_informative=pl.col("normalized_spectral_information_score").eq(1.0),
            normalized_spectral_entropy=pl.col("spectral_entropy").truediv(
                pl.col("spectral_entropy").mean().over(["base_inchikey", "ion_mode"])
            ),
            normalized_num_clean_peaks=pl.col("num_clean_peaks").truediv(
                pl.col("num_clean_peaks").mean().over(["base_inchikey", "ion_mode"])
            ),
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
    Literal,
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
    replace,
    spearmanr,
):


    MeasureName = Literal[
        "spectral_information_score",
        "weighted_spectral_information_score",
        "normalized_spectral_information_score",
        "spectral_entropy",
        "normalized_spectral_entropy",
        "num_clean_peaks",
        "normalized_num_clean_peaks",
    ]

    InfoMeasureName = Literal[
        "spectral_information_score",
        "weighted_spectral_information_score",
        "normalized_spectral_information_score",
    ]

    FprYAxisStat = Literal[
        "avg_false_matches",
        "fraction_any_false_match",
    ]

    @dataclass(frozen=True)
    class SimilarityMetricThreshold:
        metric_column: str
        threshold: float
        plot_label: str
        plot_color: str = "C0"
        plot_marker: str = "o"

    @dataclass
    class FprVsMeasureConfig:
        pairs_parquet_path: Union[str, Path] = "/home/analytit_admin/Data/spectral_libs/all_pairs_with_similarities.parquet"

        # Unified x-axis configuration used by the plots and raw Spearman computation.
        x_measure: MeasureName = "spectral_information_score"
        x_bin_width: float = 0.1
        x_range: Optional[Tuple[float, float]] = None

        # Why: Sometimes the mean count is dominated by heavy-tail spectra; this option plots a probability instead.
        y_axis_stat: FprYAxisStat = "avg_false_matches"

        # Matched-info difference configuration (info-only; defaults to x_measure if it is an info measure).
        measure_difference_measure: Optional[InfoMeasureName] = None

        metrics: List[SimilarityMetricThreshold] = field(
            default_factory=lambda: [
                SimilarityMetricThreshold("dotprod_similarity", 0.8, "Dot Product", "C0", "o"),
                SimilarityMetricThreshold("entropy_similarity", 0.75, "Entropy", "C2", "^"),
            ]
        )
        min_count_threshold: int = 5

        # Output paths
        fpr_output_path: Union[str, Path] = "fpr_vs_measure.png"
        matched_info_output_path: Union[str, Path] = "avg_matched_measure_diff.png"

        # Optional plot features
        plot_show_percentile_band: bool = True  # 10th-90th band
        plot_only_most_informative: bool = False
        show_molecule_cdf: bool = False
        matched_info_relative: bool = False

        def copy(self, **changes) -> "FprVsMeasureConfig":
            valid_fields = set(self.__dataclass_fields__.keys())
            unknown = set(changes) - valid_fields
            assert not unknown, f"Unknown fields for FprVsMeasureConfig.copy(): {sorted(list(unknown))}"
            return replace(self, **changes)

    @dataclass(frozen=True)
    class TanimotoMetricThreshold:
        metric_column: str
        threshold: float
        plot_label: str
        plot_color: str = "C0"
        plot_marker: str = "o"

    @dataclass
    class TanimotoVsMeasureConfig:
        pairs_parquet_path: Union[str, Path]

        x_measure: MeasureName = "spectral_information_score"
        x_bin_width: float = 0.2
        x_range: Optional[Tuple[float, float]] = None

        metrics: List[TanimotoMetricThreshold] = field(default_factory=list)
        min_count_threshold: int = 5

        # Enable either running on all spectra or only the most informative ones.
        only_most_informative: bool = False

        # SMILES columns
        left_smiles_col: str = "smiles"
        right_smiles_col: str = "smiles_right"

        # Fingerprint parameters
        fp_radius: int = 2
        fp_size: int = 2048
        fp_num_threads: int = 0

        # Output: one figure per metric entry; written into this directory.
        output_dir: Union[str, Path] = "."
        filename_template: str = "avg_tanimoto_vs_{metric_label}_thr_{threshold}.png"

        def copy(self, **changes) -> "TanimotoVsMeasureConfig":
            valid_fields = set(self.__dataclass_fields__.keys())
            unknown = set(changes) - valid_fields
            assert not unknown, f"Unknown fields for TanimotoVsMeasureConfig.copy(): {sorted(list(unknown))}"
            return replace(self, **changes)

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
            case "normalized_spectral_entropy":
                return "Normalized Spectral Entropy"
            case "num_clean_peaks":
                return "Number of Clean Peaks"
            case "normalized_num_clean_peaks":
                return "Normalized Number of Clean Peaks"
            case _:
                raise AssertionError(
                    f"Unknown measure '{measure}'. Supported measures are: "
                    "spectral_information_score, weighted_spectral_information_score, normalized_spectral_information_score, "
                    "spectral_entropy, num_clean_peaks."
                )

    def _default_x_range(measure: MeasureName) -> tuple[float, float]:
        # Why: Provide deterministic defaults while still letting callers override with x_range.
        match measure:
            case "spectral_information_score" | "weighted_spectral_information_score":
                return (0.0, 3.0)
            case "spectral_entropy":
                return (0.0, 5.0)
            case "num_clean_peaks":
                return (0.0, 200.0)
            case "normalized_spectral_information_score":
                return (0.0, 5.0)
            case "normalized_spectral_entropy":
                return (0.0, 5.0)
            case "normalized_num_clean_peaks":
                return (0.0, 5.0)
            case _:
                raise AssertionError(f"No default x_range for measure '{measure}'.")

    def _resolve_x_range(x_measure: MeasureName, x_range: Optional[Tuple[float, float]]) -> tuple[float, float]:
        if x_range is not None:
            x_min, x_max = x_range
            assert x_max >= x_min, f"x_range must satisfy max>=min, got {x_range}"
            return (float(x_min), float(x_max))
        return _default_x_range(x_measure)

    def _resolve_x_range_for_measure(config: FprVsMeasureConfig, measure: MeasureName) -> tuple[float, float]:
        # Why: config.x_range semantically applies to the current plotting x_measure; other measures use defaults.
        if measure == config.x_measure:
            return _resolve_x_range(config.x_measure, config.x_range)
        return _resolve_x_range(measure, None)

    def _resolve_measure_difference_measure(config: FprVsMeasureConfig) -> InfoMeasureName:
        # Why: Matched-info-diff is only meaningful for info-based measures; allow override but keep sane default.
        if config.measure_difference_measure is not None:
            return config.measure_difference_measure
        match config.x_measure:
            case "spectral_information_score" | "weighted_spectral_information_score" | "normalized_spectral_information_score":
                return config.x_measure  # type: ignore[return-value]
            case _:
                return "spectral_information_score"


    def _compute_binned_fpr_stats_for_metric(
        *,
        pairs_sim: pl.LazyFrame,
        config: FprVsMeasureConfig,
        metric: SimilarityMetricThreshold,
    ) -> pl.LazyFrame:
        assert config.x_bin_width > 0, f"x_bin_width must be > 0, got {config.x_bin_width}"
        x_min, x_max = _resolve_x_range(config.x_measure, config.x_range)

        required_cols = [
            "idx",
            "base_inchikey",
            "base_inchikey_right",
            metric.metric_column,
            config.x_measure,
            "most_informative",
            f"{config.x_measure}_right",
        ]

        pairs_sim.select(required_cols).limit(1).collect(engine="streaming")

        if config.plot_only_most_informative:
            pairs_sim = pairs_sim.filter(pl.col("most_informative"))

        base_per_spectrum = (
            pairs_sim.select(["idx", "base_inchikey", config.x_measure])
            .unique(subset="idx", keep="any")
        )

        matched_per_spectrum = (
            pairs_sim.filter(pl.col(metric.metric_column).ge(metric.threshold))
            .group_by("idx")
            .agg(
                false_compound_count=pl.col("base_inchikey_right").n_unique(),
                avg_matched_measure_diff=(
                    (pl.col(f"{config.x_measure}_right") - pl.col(config.x_measure)).mean()
                ),
            )
        )

        per_spectrum = (
            base_per_spectrum.join(matched_per_spectrum, on="idx", how="left")
            .with_columns(false_compound_count=pl.col("false_compound_count").fill_null(0))
            .with_columns(info_bin_val=(pl.col(config.x_measure) / config.x_bin_width).floor() * config.x_bin_width)
            .filter(
                pl.col("info_bin_val").is_not_null(),
                pl.col("info_bin_val").ge(float(x_min)),
                pl.col("info_bin_val").le(float(x_max)),
            )
        )

        return (
            per_spectrum.group_by("info_bin_val")
            .agg(
                total_count=pl.len(),
                avg_false_matches=pl.col("false_compound_count").mean(),
                fraction_any_false_match=pl.col("false_compound_count").gt(0).mean(),
                lower_percentile=pl.col("false_compound_count").quantile(0.1),
                upper_percentile=pl.col("false_compound_count").quantile(0.9),
                avg_matched_measure_diff=pl.col("avg_matched_measure_diff").mean(),
            )
            .with_columns(
                metric_label=pl.lit(metric.plot_label),
                metric_name=pl.lit(metric.metric_column),
                threshold_used=pl.lit(metric.threshold),
                plot_color=pl.lit(metric.plot_color),
                plot_marker=pl.lit(metric.plot_marker),
            )
            .sort("info_bin_val")
        )

    def _compute_molecule_cdf(
        *,
        pairs_sim: pl.LazyFrame,
        config: FprVsMeasureConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not config.show_molecule_cdf:
            return np.array([], dtype=float), np.array([], dtype=float)

        assert config.x_measure in (
            "spectral_information_score",
            "weighted_spectral_information_score",
            "normalized_spectral_information_score",
        ), (
            "show_molecule_cdf is only supported for info-based x_measure values; "
            f"got x_measure={config.x_measure}."
        )

        x_min, x_max = _resolve_x_range(config.x_measure, config.x_range)
        cdf_x = np.arange(float(x_min), float(x_max) + float(config.x_bin_width), float(config.x_bin_width))

        if config.plot_only_most_informative:
            pairs_sim = pairs_sim.filter(pl.col("most_informative"))

        molecule_max_x = (
            pairs_sim.select(["base_inchikey", config.x_measure])
            .group_by("base_inchikey")
            .agg(max_x=pl.col(config.x_measure).max())
            .collect(engine="streaming")
        )
        total_molecules = molecule_max_x.height
        assert total_molecules > 0, "No molecules available for CDF computation; check input data."
        cdf_y = np.array([molecule_max_x.filter(pl.col("max_x") >= float(x)).height / total_molecules for x in cdf_x])
        return cdf_x, cdf_y

    def plot_fpr_vs_measure(config: FprVsMeasureConfig) -> None:
        pairs_sim = pl.scan_parquet(config.pairs_parquet_path)

        assert config.y_axis_stat in ("avg_false_matches", "fraction_any_false_match"), (
            "config.y_axis_stat must be one of: avg_false_matches, fraction_any_false_match; "
            f"got {config.y_axis_stat}."
        )

        stats_lfs: list[pl.LazyFrame] = []
        for metric in config.metrics:
            stats_lfs.append(_compute_binned_fpr_stats_for_metric(pairs_sim=pairs_sim, config=config, metric=metric))

        all_stats = pl.concat(stats_lfs).collect(engine="streaming")
        if config.show_molecule_cdf:
            cdf_x, cdf_y = _compute_molecule_cdf(pairs_sim=pairs_sim, config=config)

        fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")

        if config.y_axis_stat == "avg_false_matches":
            y_column_name = "avg_false_matches"
            y_axis_label = "Average Number of False-Matched Compounds"
            y_is_fraction = False
        else:
            y_column_name = "fraction_any_false_match"
            y_axis_label = "Fraction of Spectra With ≥1 False Match"
            y_is_fraction = True

        y_max_seen: float = float("nan")

        for metric in config.metrics:
            subset = all_stats.filter(
                (pl.col("metric_name") == metric.metric_column) &
                (pl.col("threshold_used") == metric.threshold) &
                (pl.col("total_count") > config.min_count_threshold)
            )
            if subset.height == 0:
                print(f"Warning: No data for {metric.plot_label} at threshold {metric.threshold}")
                continue

            info_x = subset.get_column("info_bin_val").to_numpy()
            y_vals = subset.get_column(y_column_name).to_numpy()

            if y_vals.size:
                local_max = float(np.nanmax(y_vals))
                if np.isfinite(local_max):
                    # Why: auto-scale based on what we actually plotted (post min_count_threshold filtering).
                    y_max_seen = local_max if not np.isfinite(y_max_seen) else max(y_max_seen, local_max)

            ax.plot(
                info_x,
                y_vals,
                marker=metric.plot_marker,
                label=f"{metric.plot_label} (threshold = {metric.threshold})",
                color=metric.plot_color,
            )

            if (not y_is_fraction) and config.plot_show_percentile_band and "lower_percentile" in subset.columns and "upper_percentile" in subset.columns:
                lower_y = subset.get_column("lower_percentile").to_numpy()
                upper_y = subset.get_column("upper_percentile").to_numpy()
                ax.fill_between(info_x, lower_y, upper_y, color=metric.plot_color, alpha=0.15)
                ax.plot(info_x, lower_y, linestyle="--", color=metric.plot_color, alpha=0.7)
                ax.plot(info_x, upper_y, linestyle="--", color=metric.plot_color, alpha=0.7)

        if np.isfinite(y_max_seen):
            # Auto-scale the top of the axis to the observed data (+5% headroom).
            # For fraction mode, allow the top to be smaller than 1.05 (e.g., max=0.2 -> top≈0.21),
            # but still don't exceed 1.05.
            y_upper = max(0.05, float(y_max_seen) * 1.05)
            if y_is_fraction:
                y_upper = min(1.05, y_upper)
            ax.set_ylim(0.0, y_upper)
        else:
            # No valid plotted y-values; keep sensible default for fraction mode.
            if y_is_fraction:
                ax.set_ylim(0.0, 1.05)

        if config.show_molecule_cdf:
            if cdf_x.size and cdf_y.size:
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
                print("Warning: CDF data is empty; skipping CDF plot.")
                ax.legend(loc="upper right")
        else:
            ax.legend(loc="upper right")

        ax.set_xlabel(_measure_label(config.x_measure))
        ax.set_ylabel(y_axis_label)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(config.fpr_output_path), facecolor="white", transparent=False)
        plt.close(fig)

    def plot_avg_matched_measure_diff(config: FprVsMeasureConfig) -> None:
        pairs_sim = pl.scan_parquet(config.pairs_parquet_path)

        stats_lfs: list[pl.LazyFrame] = []
        for metric in config.metrics:
            stats_lfs.append(_compute_binned_fpr_stats_for_metric(pairs_sim=pairs_sim, config=config, metric=metric))
        all_stats = pl.concat(stats_lfs).collect(engine="streaming")

        measure_diff_col = _resolve_measure_difference_measure(config)
        fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")

        for metric in config.metrics:
            subset = all_stats.filter(
                (pl.col("metric_name") == metric.metric_column) &
                (pl.col("threshold_used") == metric.threshold) &
                (pl.col("total_count") > config.min_count_threshold)
            )
            matched_subset = subset.filter(pl.col("avg_matched_measure_diff").is_not_null())
            if matched_subset.height == 0:
                print(f"Warning: No matched-info data for {metric.plot_label} at threshold {metric.threshold}")
                continue

            x_vals = matched_subset.get_column("info_bin_val").to_numpy()
            y_vals = matched_subset.get_column("avg_matched_measure_diff").to_numpy()
            if config.matched_info_relative:
                # Why: Caller opted into relative normalization; avoid silent divide-by-zero by relying on x-range defaults > 0.
                y_plot = y_vals / x_vals
            else:
                y_plot = y_vals

            ax.plot(
                x_vals,
                y_plot,
                linestyle="--",
                marker="x",
                label=f"{metric.plot_label} (threshold = {metric.threshold})",
                color=metric.plot_color,
                alpha=0.9,
            )

        ax.set_xlabel(_measure_label(config.x_measure))
        ax.set_ylabel(f"Average Matched Δ ({_measure_label(measure_diff_col)})")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(config.matched_info_output_path), facecolor="white", transparent=False)
        plt.close(fig)

    def compute_raw_spearman_false_matches_vs_measure(config: FprVsMeasureConfig, *, measures: Optional[List[MeasureName]] = None) -> pl.DataFrame:
        """
        Raw (non-binned) Spearman between each requested measure and per-spectrum false-compound counts.

        Notes:
          - Similarity metrics/thresholds define which pairs count as matches (and thus false positives).
          - Measures (SIS/entropy/peaks) are the X variables for correlation.
        """
        pairs_sim = pl.scan_parquet(config.pairs_parquet_path)

        measures_to_use: List[MeasureName] = (
            measures
            if measures is not None
            else [
                "spectral_information_score",
                "weighted_spectral_information_score",
                "normalized_spectral_information_score",
                "spectral_entropy",
                "num_clean_peaks",
            ]
        )
        assert len(measures_to_use) > 0, "measures must be non-empty"

        required_cols = ["idx", "base_inchikey_right", "most_informative", *measures_to_use]
        pairs_sim.select(required_cols).limit(1).collect(engine="streaming")

        # One row per spectrum with all measures attached (avoid scanning per measure).
        per_spectrum_measures = (
            pairs_sim.select(["idx", "most_informative", *measures_to_use])
            .unique(subset="idx", keep="any")
            .collect(engine="streaming")
        )
        if config.plot_only_most_informative:
            per_spectrum_measures = per_spectrum_measures.filter(pl.col("most_informative"))

        results: list[dict[str, object]] = []
        for metric in config.metrics:
            # Per-spectrum false-positive counts for this similarity metric threshold.
            pairs_sim.select([metric.metric_column]).limit(1).collect(engine="streaming")
            matched_counts = (
                pairs_sim.filter(pl.col(metric.metric_column).ge(metric.threshold))
                .group_by("idx")
                .agg(false_compound_count=pl.col("base_inchikey_right").n_unique())
                .collect(engine="streaming")
            )

            per_spectrum = (
                per_spectrum_measures.join(matched_counts, on="idx", how="left")
                .with_columns(false_compound_count=pl.col("false_compound_count").fill_null(0))
            )

            for measure in measures_to_use:
                xy = (
                    per_spectrum.select([pl.col(measure).alias("x"), pl.col("false_compound_count").alias("y")])
                    .filter(pl.col("x").is_not_null())
                )

                x = xy.get_column("x").to_numpy()
                y = xy.get_column("y").to_numpy()
                if x.size < 3:
                    rho, p_val = float("nan"), float("nan")
                else:
                    rho, p_val = spearmanr(x, y)

                print(
                    "Raw Spearman (false matches vs measure) - "
                    f"{metric.plot_label} [col={metric.metric_column}, thr={metric.threshold}] vs "
                    f"{_measure_label(measure)}: rho={rho:.4g}, p={p_val:.4g}, n={int(x.size)}"
                )

                results.append({
                    "metric_column": metric.metric_column,
                    "threshold": float(metric.threshold),
                    "plot_label": metric.plot_label,
                    "measure": measure,
                    "measure_label": _measure_label(measure),
                    "spearman_rho": float(rho),
                    "p_value": float(p_val),
                    "n_points": int(x.size),
                })

        return pl.DataFrame(results)

    def plot_avg_tanimoto_vs_measure(config: TanimotoVsMeasureConfig) -> pl.DataFrame:
        assert len(config.metrics) > 0, "TanimotoVsMeasureConfig.metrics must be non-empty."
        pairs_sim = pl.scan_parquet(config.pairs_parquet_path)
        assert config.x_bin_width > 0, f"x_bin_width must be > 0, got {config.x_bin_width}"
        x_min, x_max = _resolve_x_range(config.x_measure, config.x_range)

        # Fail fast on required columns for tanimoto + measure binning.
        pairs_sim.select([config.left_smiles_col, config.right_smiles_col, config.x_measure, "most_informative"]).limit(1).collect(engine="streaming")

        if config.only_most_informative:
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

        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        all_stats: list[pl.DataFrame] = []
        for metric in config.metrics:
            # Why: Metric label is explicitly provided to avoid relying on column name semantics.
            pairs_with_sim.select([metric.metric_column]).limit(1).collect(engine="streaming")

            filtered = (
                pairs_with_sim.filter(pl.col(metric.metric_column).ge(metric.threshold))
                .with_columns(info_bin_val=(pl.col(config.x_measure) / pl.col(config.x_bin_width)).floor() * pl.col(config.x_bin_width))
                .filter(
                    pl.col("info_bin_val").is_not_null(),
                    pl.col("info_bin_val").ge(float(x_min)),
                    pl.col("info_bin_val").le(float(x_max)),
                )
            )

            # Raw Spearman on non-binned values (SciPy)
            raw_df = (
                filtered.select([pl.col(config.x_measure).alias("x"), pl.col("tanimoto_similarity").alias("y")])
                .filter(pl.col("x").is_not_null(), pl.col("y").is_not_null())
                .collect(engine="streaming")
            )
            x = raw_df.get_column("x").to_numpy()
            y = raw_df.get_column("y").to_numpy()
            if x.size < 3:
                rho, p_val = float("nan"), float("nan")
            else:
                rho, p_val = spearmanr(x, y)
            print(
                "Raw Spearman (tanimoto vs measure) - "
                f"{metric.plot_label} [col={metric.metric_column}, thr={metric.threshold}]: "
                f"rho={rho:.4g}, p={p_val:.4g}, n={int(x.size)}"
            )

            stats = (
                filtered.group_by("info_bin_val")
                .agg(
                    avg_tanimoto=pl.col("tanimoto_similarity").mean(),
                    median_tanimoto=pl.col("tanimoto_similarity").median(),
                    count=pl.len(),
                )
                .with_columns(
                    metric_label=pl.lit(metric.plot_label),
                    metric_name=pl.lit(metric.metric_column),
                    threshold_used=pl.lit(metric.threshold),
                    plot_color=pl.lit(metric.plot_color),
                    plot_marker=pl.lit(metric.plot_marker),
                )
                .sort("info_bin_val")
                .collect(engine="streaming")
            )
            all_stats.append(stats)

            plot_subset = stats.filter(pl.col("count") > config.min_count_threshold)
            fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
            if plot_subset.height > 0:
                ax.plot(
                    plot_subset.get_column("info_bin_val").to_numpy(),
                    plot_subset.get_column("avg_tanimoto").to_numpy(),
                    marker=metric.plot_marker,
                    label=f"{metric.plot_label} (threshold = {metric.threshold})",
                    color=metric.plot_color,
                )
            else:
                print(f"Warning: No binned tanimoto points above min_count_threshold for {metric.plot_label} (threshold = {metric.threshold})")

            ax.set_xlabel(_measure_label(config.x_measure))
            ax.set_ylabel("Average Tanimoto Similarity of Matches")
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            out_name = config.filename_template.format(
                metric_label=metric.plot_label.replace(" ", "_"),
                threshold=str(metric.threshold).replace(".", "p"),
            )
            fig.savefig(str(output_dir / out_name), facecolor="white", transparent=False)
            plt.close(fig)

        return pl.concat(all_stats) if all_stats else pl.DataFrame()
    return (
        FprVsMeasureConfig,
        SimilarityMetricThreshold,
        TanimotoMetricThreshold,
        TanimotoVsMeasureConfig,
        compute_raw_spearman_false_matches_vs_measure,
        plot_avg_matched_measure_diff,
        plot_avg_tanimoto_vs_measure,
        plot_fpr_vs_measure,
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
def _(FprVsMeasureConfig, OUTPUT_PAIRS_PATH, SimilarityMetricThreshold):
    fpr_config = FprVsMeasureConfig(
        pairs_parquet_path=OUTPUT_PAIRS_PATH,
        x_measure="spectral_information_score",
        x_bin_width=0.2,
        x_range=(0.0, 3.0),
        # y_axis_stat="fraction_any_false_match",  # optional: plot fraction with >=1 false match
        metrics=[
            SimilarityMetricThreshold("dotprod_similarity", 0.80, "Dot Product", "C0", "o"),
            SimilarityMetricThreshold("dotprod_similarity", 0.90, "Dot Product", "C1", "o"),
        ],
        fpr_output_path="fpr_vs_spectral_information_score.png",
        matched_info_output_path="avg_matched_measure_diff.png",
        plot_show_percentile_band=False,
        plot_only_most_informative=False,
        show_molecule_cdf=False,
        matched_info_relative=True,
    )
    return (fpr_config,)


@app.cell
def _(fpr_config, plot_fpr_vs_measure):
    info_config = fpr_config.copy(
        x_measure="spectral_information_score",
        fpr_output_path="fpr_vs_spectral_information_score.png",
    )
    plot_fpr_vs_measure(info_config)
    return


@app.cell
def _(SimilarityMetricThreshold, fpr_config, plot_fpr_vs_measure):
    normalized_info_config = fpr_config.copy(
        x_measure="normalized_spectral_information_score",
        x_bin_width=0.3,
        x_range=(0.0, 3.0),
        y_axis_stat="avg_false_matches",
        metrics=[
            SimilarityMetricThreshold("dotprod_similarity", 0.80, "Dot Product", "C0", "o"),
            SimilarityMetricThreshold("dotprod_similarity", 0.90, "Dot Product", "C1", "o"),
        ],
        fpr_output_path="avg_fpr_vs_normalized_spectral_information_score.png",
    )
    plot_fpr_vs_measure(normalized_info_config)
    return


@app.cell
def _(fpr_config, plot_fpr_vs_measure):
    entropy_config = fpr_config.copy(
        x_measure="spectral_entropy",
        x_bin_width=0.5,
        x_range=(0.0, 5.0),
        fpr_output_path="fpr_vs_spectral_entropy.png",
    )
    plot_fpr_vs_measure(entropy_config)
    normalized_entropy_config = fpr_config.copy(
        x_measure="normalized_spectral_entropy",
        x_bin_width=0.5,
        x_range=(0.0, 5.0),
        fpr_output_path="fpr_vs_normalized_spectral_entropy.png",
    )
    plot_fpr_vs_measure(normalized_entropy_config)
    peaks_config = fpr_config.copy(
        x_measure="num_clean_peaks",
        x_bin_width=3.0,
        x_range=(0.0, 200.0),
        fpr_output_path="fpr_vs_num_clean_peaks.png",
    )
    plot_fpr_vs_measure(peaks_config)
    normalized_num_peaks_config = fpr_config.copy(
        x_measure="normalized_num_clean_peaks",
        x_bin_width=0.3,
        x_range=(0.0, 3.0),
        fpr_output_path="fpr_vs_normalized_num_clean_peaks.png",
    )
    plot_fpr_vs_measure(normalized_num_peaks_config)
    return


@app.cell
def _(OUTPUT_PAIRS_PATH, TanimotoMetricThreshold, TanimotoVsMeasureConfig):
    tanimoto_config = TanimotoVsMeasureConfig(
        pairs_parquet_path=OUTPUT_PAIRS_PATH,
        x_measure="spectral_information_score",
        x_bin_width=0.5,
        x_range=(0.0, 3.0),
        metrics=[
            TanimotoMetricThreshold("dotprod_similarity", 0.90, "Dot Product", "C1", "o"),
            TanimotoMetricThreshold("entropy_similarity", 0.75, "Entropy", "C2", "^"),
        ],
        only_most_informative=False,
        left_smiles_col="smiles",
        right_smiles_col="smiles_right",
        fp_radius=2,
        fp_size=2048,
        fp_num_threads=0,
        output_dir=".",
    )
    return (tanimoto_config,)


@app.cell
def _(
    FprVsMeasureConfig,
    OUTPUT_PAIRS_PATH,
    SimilarityMetricThreshold,
    compute_raw_spearman_false_matches_vs_measure,
):

    spearman_config = FprVsMeasureConfig(
        pairs_parquet_path=OUTPUT_PAIRS_PATH,  # or Path("/path/to/all_pairs_with_similarities.parquet")
        # x_measure/x_bin_width/x_range are not required for the raw Spearman call,
        # but x_range will still be used for filtering when the measure == x_measure.
        x_measure="spectral_information_score",
        x_range=(0.0, 3.0),
        metrics=[
            # SimilarityMetricThreshold("dotprod_similarity", 0.80, "Dot Product", "C0", "o"),
            SimilarityMetricThreshold("dotprod_similarity", 0.90, "Dot Product", "C1", "o"),
        ],
        plot_only_most_informative=False,  # optional: matches the logic in the function
    )

    spearman_df = compute_raw_spearman_false_matches_vs_measure(
        spearman_config,
        measures=[
            "spectral_information_score",
            "normalized_spectral_information_score",
            "spectral_entropy",
            "normalized_spectral_entropy",
            "num_clean_peaks",
            "normalized_num_clean_peaks",
        ],
    )

    print(spearman_df)
    return


@app.cell
def _(
    compute_raw_spearman_false_matches_vs_measure,
    fpr_config,
    plot_avg_matched_measure_diff,
    plot_avg_tanimoto_vs_measure,
    plot_fpr_vs_measure,
    tanimoto_config,
):
    # Raw (non-binned) Spearman for false matches vs measure (SciPy)
    spearman_df = compute_raw_spearman_false_matches_vs_measure(fpr_config)
    print(spearman_df)

    # Plots compute their own stats now
    plot_fpr_vs_measure(fpr_config)
    plot_avg_matched_measure_diff(fpr_config)

    # Tanimoto: computes once, writes one figure per metric entry, and prints raw Spearman (SciPy)
    tanimoto_stats = plot_avg_tanimoto_vs_measure(tanimoto_config)
    print(tanimoto_stats.head())
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


@app.cell
def _(Literal, Optional, Path, Union, dataclass, np, pl, replace, wilcoxon):

    SelectionStrategy = Literal[
        "best_by_measure",          # pick the per-molecule spectrum that maximizes/minimizes `selection_measure`
        "random",                   # pick a uniformly-random spectrum per molecule (seeded)
        "closest_to_target_energy", # pick spectrum with minimal abs(energy - target_energy)
    ]

    SelectionDirection = Literal["max", "min"]
    WilcoxonAlternative = Literal["two-sided", "less", "greater"]

    # Why: Keep selection limited to the three validated/expected per-spectrum columns used in this analysis.
    SelectionMeasureName = Literal[
        "spectral_information_score",
        "spectral_entropy",
        "num_clean_peaks",
    ]


    @dataclass(frozen=True)
    class MoleculeSelectionEffectConfig:
        """
        Compare per-molecule selection strategies using paired tests.

        Grouping:
          - One comparison per (base_inchikey, ion_mode).

        Outcome:
          - false_compound_count per selected spectrum, based on similarity metric thresholding.

        Notes:
          - "Advantage" is defined as (baseline_fp - selected_fp), so positive means improvement.
        """

        pairs_parquet_path: Union[str, Path]
        exclude_molecules_without_false_positives: bool = True
        # Defines what counts as a "match" (and thus a potential false positive).
        similarity_metric_column: str = "dotprod_similarity"
        similarity_threshold: float = 0.9
        similarity_metric_label: str = "Dot Product"

        # How to select the "chosen" spectrum per molecule.
        selected_strategy: SelectionStrategy = "best_by_measure"
        selection_measure: SelectionMeasureName = "spectral_information_score"
        selection_direction: SelectionDirection = "max"

        # Baseline strategy to compare against.
        baseline_strategy: SelectionStrategy = "random"

        # Random selection control (reproducible).
        random_seed: int = 0

        # Energy-based baseline configuration.
        energy_column: str = "collision_energy"
        target_energy: Optional[float] = None  # required if any strategy uses closest_to_target_energy

        # Grouping columns (do not change unless you also change upstream parquet schema).
        molecule_id_column: str = "base_inchikey"
        ion_mode_column: str = "ion_mode"

        # Quality filters.
        min_spectra_per_group: int = 2  # must be >=2 to allow "exclude selected from baseline"
        drop_groups_with_null_measure: bool = True

        # Wilcoxon signed-rank parameters.
        wilcoxon_alternative: WilcoxonAlternative = "less"  # tests: selected_fp < baseline_fp
        wilcoxon_zero_method: Literal["wilcox", "pratt", "zsplit"] = "pratt"

        # Outputs
        output_dir: Union[str, Path] = "."
        file_prefix: str = "selection_effect"

        def copy(self, **changes) -> "MoleculeSelectionEffectConfig":
            valid_fields = set(self.__dataclass_fields__.keys())
            unknown = set(changes) - valid_fields
            assert not unknown, f"Unknown fields for MoleculeSelectionEffectConfig.copy(): {sorted(list(unknown))}"
            return replace(self, **changes)


    def _require_columns_or_fail(lf: pl.LazyFrame, required_columns: list[str]) -> None:
        # Why: make missing parquet columns fail fast with a clear message before heavy computation.
        lf.select(required_columns).limit(1).collect(engine="streaming")


    def _compute_per_spectrum_false_compound_counts(
        *,
        pairs_sim: pl.LazyFrame,
        similarity_metric_column: str,
        similarity_threshold: float,
        spectrum_columns: list[str],
        molecule_id_column: str,
        ion_mode_column: str,
    ) -> pl.DataFrame:
        required_cols = [
            "idx",
            molecule_id_column,
            ion_mode_column,
            "base_inchikey_right",  # required to count unique false compounds among matches
            similarity_metric_column,
            *spectrum_columns,
        ]
        _require_columns_or_fail(pairs_sim, required_cols)

        per_spectrum_base = (
            pairs_sim.select(["idx", molecule_id_column, ion_mode_column, *spectrum_columns])
            .unique(subset="idx", keep="any")
            .collect(engine="streaming")
        )

        matched_counts = (
            pairs_sim.filter(pl.col(similarity_metric_column).ge(float(similarity_threshold)))
            .group_by("idx")
            .agg(false_compound_count=pl.col("base_inchikey_right").n_unique())
            .collect(engine="streaming")
        )

        per_spectrum = (
            per_spectrum_base.join(matched_counts, on="idx", how="left")
            .with_columns(false_compound_count=pl.col("false_compound_count").fill_null(0).cast(pl.Int64))
        )
        return per_spectrum


    def _select_one_spectrum_per_group_best_by_measure(
        *,
        per_spectrum: pl.DataFrame,
        molecule_id_column: str,
        ion_mode_column: str,
        selection_measure: SelectionMeasureName,
        selection_direction: SelectionDirection = "max",
    ) -> pl.DataFrame:
        group_cols = [molecule_id_column, ion_mode_column]
        assert selection_direction in ("max", "min"), f"selection_direction must be max|min, got {selection_direction}"

        descending = selection_direction == "max"
        sorted_df = per_spectrum.sort(
            by=[*group_cols, selection_measure, "idx"],
            descending=[False, False, descending, False],
        )

        # Why: Polars includes group keys in the output; aggregating them again creates duplicate columns.
        first_exprs = [pl.col(c).first().alias(c) for c in sorted_df.columns if c not in group_cols]
        return sorted_df.group_by(group_cols, maintain_order=True).agg(first_exprs)


    def _select_one_spectrum_per_group_random(
        *,
        per_spectrum: pl.DataFrame,
        molecule_id_column: str,
        ion_mode_column: str,
        seed: int,
    ) -> pl.DataFrame:
        group_cols = [molecule_id_column, ion_mode_column]
        # with_key = per_spectrum.with_columns(_rand_key=pl.rand(seed=seed))
        # sorted_df = with_key.sort(by=[*group_cols, "_rand_key", "idx"], descending=[False, False, False, False])

        # # Why: exclude group keys to avoid DuplicateError; exclude the temp random key as well.
        # first_exprs = [pl.col(c).first().alias(c) for c in sorted_df.columns if c not in group_cols and c != "_rand_key"]
        # return sorted_df.group_by(group_cols, maintain_order=True).agg(first_exprs)
        return per_spectrum.filter(
            pl.int_range(pl.len())
            .shuffle()
            .over(group_cols)
            .eq(0)
        )


    def _select_one_spectrum_per_group_closest_to_target_energy(
        *,
        per_spectrum: pl.DataFrame,
        molecule_id_column: str,
        ion_mode_column: str,
        energy_column: str,
        target_energy: float,
    ) -> pl.DataFrame:
        group_cols = [molecule_id_column, ion_mode_column]
        assert target_energy is not None, "target_energy is required for closest_to_target_energy selection."

        with_dist = per_spectrum.with_columns(
            energy_distance=(pl.col(energy_column).cast(pl.Float64) - float(target_energy)).abs()
        )
        sorted_df = with_dist.sort(
            by=[*group_cols, "energy_distance", "idx"],
            descending=[False, False, False, False],
        )

        # Why: exclude group keys to avoid DuplicateError; exclude the temp distance column as well.
        first_exprs = [pl.col(c).first().alias(c) for c in sorted_df.columns if c not in group_cols and c != "energy_distance"]
        return sorted_df.group_by(group_cols, maintain_order=True).agg(first_exprs)


    def _select_one_spectrum_per_group(
        *,
        per_spectrum: pl.DataFrame,
        strategy: SelectionStrategy,
        molecule_id_column: str,
        ion_mode_column: str,
        selection_measure: SelectionMeasureName,
        selection_direction: SelectionDirection,
        seed: int,
        energy_column: str,
        target_energy: Optional[float],
    ) -> pl.DataFrame:
        if strategy == "best_by_measure":
            assert selection_measure in per_spectrum.columns, (
                f"selection_measure column '{selection_measure}' is missing from per_spectrum. "
                "Ensure it exists in the pairs parquet and is included in spectrum_columns."
            )
            return _select_one_spectrum_per_group_best_by_measure(
                per_spectrum=per_spectrum,
                molecule_id_column=molecule_id_column,
                ion_mode_column=ion_mode_column,
                selection_measure=selection_measure,
                selection_direction=selection_direction,
            )

        if strategy == "random":
            return _select_one_spectrum_per_group_random(
                per_spectrum=per_spectrum,
                molecule_id_column=molecule_id_column,
                ion_mode_column=ion_mode_column,
                seed=seed,
            )

        if strategy == "closest_to_target_energy":
            assert target_energy is not None, (
                "target_energy must be provided when using closest_to_target_energy. "
                "Example: target_energy=20.0"
            )
            assert energy_column in per_spectrum.columns, (
                f"energy_column '{energy_column}' is missing from per_spectrum. "
                "You likely need to include it in the parquet creation step."
            )
            return _select_one_spectrum_per_group_closest_to_target_energy(
                per_spectrum=per_spectrum,
                molecule_id_column=molecule_id_column,
                ion_mode_column=ion_mode_column,
                energy_column=energy_column,
                target_energy=float(target_energy),
            )

        raise AssertionError(f"Unknown selection strategy: {strategy}")


    def run_molecule_selection_effect_analysis(config: MoleculeSelectionEffectConfig) -> pl.DataFrame:
        """
        Per (molecule, ion_mode), compare chosen spectrum vs baseline spectrum.

        Prints:
          - n_groups
          - win/tie/loss rates
          - average/median advantage (baseline_fp - selected_fp)
          - Wilcoxon signed-rank test on paired fp counts

        Writes:
          - violin plot
          - CDF plot
          - delta histogram
        """
        pairs_sim = pl.scan_parquet(config.pairs_parquet_path)
        assert config.min_spectra_per_group >= 2, (
            f"min_spectra_per_group must be >= 2 to compare selected vs baseline, got {config.min_spectra_per_group}"
        )

        spectrum_columns: list[str] = [config.selection_measure]
        if config.selected_strategy == "closest_to_target_energy" or config.baseline_strategy == "closest_to_target_energy":
            assert config.target_energy is not None, (
                "target_energy must be set when any strategy is closest_to_target_energy."
            )
            spectrum_columns.append(config.energy_column)

        per_spectrum = _compute_per_spectrum_false_compound_counts(
            pairs_sim=pairs_sim,
            similarity_metric_column=config.similarity_metric_column,
            similarity_threshold=config.similarity_threshold,
            spectrum_columns=spectrum_columns,
            molecule_id_column=config.molecule_id_column,
            ion_mode_column=config.ion_mode_column,
        )

        group_cols = [config.molecule_id_column, config.ion_mode_column]

        if config.drop_groups_with_null_measure:
            per_spectrum = per_spectrum.filter(pl.col(config.selection_measure).is_not_null())

        # Filter to groups with enough spectra.
        per_spectrum = per_spectrum.with_columns(_group_size=pl.len().over(group_cols))
        per_spectrum = per_spectrum.filter(pl.col("_group_size").ge(config.min_spectra_per_group)).drop("_group_size")

        if config.exclude_molecules_without_false_positives:
            # Why: selection effects are undefined if a molecule never produces a false match under the chosen threshold.
            per_spectrum = per_spectrum.with_columns(
                _group_max_false_compound_count=pl.col("false_compound_count").max().over(group_cols)
            )
            per_spectrum = (
                per_spectrum.filter(pl.col("_group_max_false_compound_count").gt(0))
                .drop("_group_max_false_compound_count")
            )

        assert per_spectrum.height > 0, (
            "No spectra remain after filtering by min_spectra_per_group, null-measure handling, "
            "and (optionally) exclude_molecules_without_false_positives. "
            "Check your parquet contents and config."
        )

        selected = _select_one_spectrum_per_group(
            per_spectrum=per_spectrum,
            strategy=config.selected_strategy,
            molecule_id_column=config.molecule_id_column,
            ion_mode_column=config.ion_mode_column,
            selection_measure=config.selection_measure,
            selection_direction=config.selection_direction,
            seed=config.random_seed,
            energy_column=config.energy_column,
            target_energy=config.target_energy,
        ).rename(
            {"idx": "selected_idx", "false_compound_count": "selected_fp", config.selection_measure: "selected_measure"}
        )

        # Baseline is selected after excluding the selected spectrum to avoid trivial ties.
        per_spectrum_for_baseline = per_spectrum.join(
            selected.select([*group_cols, "selected_idx"]),
            on=group_cols,
            how="inner",
        ).filter(pl.col("idx") != pl.col("selected_idx")).drop("selected_idx")

        baseline = _select_one_spectrum_per_group(
            per_spectrum=per_spectrum_for_baseline,
            strategy=config.baseline_strategy,
            molecule_id_column=config.molecule_id_column,
            ion_mode_column=config.ion_mode_column,
            selection_measure=config.selection_measure,
            selection_direction=config.selection_direction,
            seed=config.random_seed + 1,  # Why: ensure baseline random differs from selected random when both are random.
            energy_column=config.energy_column,
            target_energy=config.target_energy,
        ).rename(
            {"idx": "baseline_idx", "false_compound_count": "baseline_fp", config.selection_measure: "baseline_measure"}
        )

        paired = (
            selected.join(baseline, on=group_cols, how="inner")
            .with_columns(delta_fp=(pl.col("baseline_fp") - pl.col("selected_fp")).cast(pl.Int64))
            .with_columns(
                selected_strategy=pl.lit(config.selected_strategy),
                baseline_strategy=pl.lit(config.baseline_strategy),
                selection_measure=pl.lit(config.selection_measure),
                similarity_metric_column=pl.lit(config.similarity_metric_column),
                similarity_threshold=pl.lit(float(config.similarity_threshold)),
            )
        )

        n_groups = paired.height
        assert n_groups > 0, (
            "No paired groups available after baseline exclusion/join. "
            "Likely too many groups have only one spectrum."
        )

        selected_fp = paired.get_column("selected_fp").to_numpy()
        baseline_fp = paired.get_column("baseline_fp").to_numpy()
        delta_fp = paired.get_column("delta_fp").to_numpy()

        win_rate = float((delta_fp > 0).mean())
        tie_rate = float((delta_fp == 0).mean())
        loss_rate = float((delta_fp < 0).mean())
        avg_advantage = float(np.mean(delta_fp))
        median_advantage = float(np.median(delta_fp))
        median_advantage_exclusing_ties = float(np.median(delta_fp[delta_fp != 0])) if np.any(delta_fp != 0) else 0.0

        # Wilcoxon signed-rank on paired samples: selected vs baseline.
        # We test whether selected_fp is stochastically smaller than baseline_fp (default alternative="less").
        w_stat, p_val = wilcoxon(
            x=selected_fp,
            y=baseline_fp,
            alternative=config.wilcoxon_alternative,
            zero_method=config.wilcoxon_zero_method,
        )

        print(
            "\nMolecule selection effect analysis\n"
            f"  groups (molecule, ion_mode): {n_groups}\n"
            f"  selected_strategy: {config.selected_strategy}\n"
            f"  baseline_strategy: {config.baseline_strategy}\n"
            f"  selection_measure: {config.selection_measure} ({config.selection_direction})\n"
            f"  match metric: {config.similarity_metric_label} [{config.similarity_metric_column} >= {config.similarity_threshold}]\n"
            f"  win/tie/loss: {win_rate:.4f} / {tie_rate:.4f} / {loss_rate:.4f}\n"
            f"  avg advantage (baseline_fp - selected_fp): {avg_advantage:.4g}\n"
            f"  avg advantage excluding ties : {avg_advantage/(1 - tie_rate):.4g}\n"
            f"  median advantage (baseline_fp - selected_fp): {median_advantage:.4g}\n"
            f"  median advantage excluding ties : {median_advantage_exclusing_ties:.4g}\n"
            f"  Wilcoxon signed-rank (alt='{config.wilcoxon_alternative}', zero_method='{config.wilcoxon_zero_method}'): "
            f"stat={float(w_stat):.4g}, p={float(p_val):.4g}\n"
        )

        out_dir = Path(config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        selected_label = f"Selected ({config.selected_strategy})"
        baseline_label = f"Baseline ({config.baseline_strategy})"
        title_base = (
            f"{config.similarity_metric_label} FP comparison per molecule\n"
            f"measure={config.selection_measure}, thr={config.similarity_threshold}"
        )


        return paired
    return (
        MoleculeSelectionEffectConfig,
        run_molecule_selection_effect_analysis,
    )


@app.cell
def _(
    MoleculeSelectionEffectConfig,
    OUTPUT_PAIRS_PATH,
    run_molecule_selection_effect_analysis,
):
    # ...existing code...
    effect_cfg = MoleculeSelectionEffectConfig(
        pairs_parquet_path=OUTPUT_PAIRS_PATH,
        similarity_metric_column="dotprod_similarity",
        similarity_threshold=0.80,
        similarity_metric_label="Dot Product",
        selected_strategy="best_by_measure",
        selection_measure="spectral_information_score",
        selection_direction="max",
        baseline_strategy="random",
        random_seed=42,
        output_dir=".",
        file_prefix="sis_best_vs_random_dp0p90",
    )

    paired_df = run_molecule_selection_effect_analysis(effect_cfg)

    effect_cfg_peaks = MoleculeSelectionEffectConfig(
        pairs_parquet_path=OUTPUT_PAIRS_PATH,
        similarity_metric_column="dotprod_similarity",
        similarity_threshold=0.80,
        similarity_metric_label="Dot Product",
        selected_strategy="best_by_measure",
        selection_measure="num_clean_peaks",
        selection_direction="max",
        baseline_strategy="random",
        random_seed=42,
        output_dir=".",
        file_prefix="num_peaks_best_vs_random_dp0p90",
    )
    paired_df = run_molecule_selection_effect_analysis(effect_cfg_peaks)

    effect_cfg_entropy = MoleculeSelectionEffectConfig(
        pairs_parquet_path=OUTPUT_PAIRS_PATH,
        similarity_metric_column="dotprod_similarity",
        similarity_threshold=0.80,
        similarity_metric_label="Dot Product",
        selected_strategy="best_by_measure",
        selection_measure="spectral_entropy",
        selection_direction="max",
        baseline_strategy="random",
        random_seed=42,
        output_dir=".",
        file_prefix="entropy_best_vs_random_dp0p90",
    )
    paired_df = run_molecule_selection_effect_analysis(effect_cfg_entropy)
    return


if __name__ == "__main__":
    app.run()
