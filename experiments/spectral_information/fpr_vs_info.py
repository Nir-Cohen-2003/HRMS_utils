import marimo

__generated_with = "0.18.0"
app = marimo.App()


@app.cell
def _():
    import polars as pl
    import numpy as np
    import matplotlib.pyplot as plt
    import spectral_similarity
    from pathlib import Path
    return Path, pl, plt


@app.cell
def _(Path):
    # Configuration
    # Why: Define constants for reproducibility and easy adjustment.
    # Update PARQUET_PATH to point to your actual data file.
    PARQUET_PATH = Path("/home/analytit_admin/Data/spectral_libs/msp_for_Yonathan/NIST.parquet") 
    TOLERANCE_PPM = 5.0
    THRESHOLDS = [0.75, 0.85, 0.95]
    MIN_ISOMERS = 5
    return MIN_ISOMERS, PARQUET_PATH


@app.cell
def _(MIN_ISOMERS, PARQUET_PATH, pl):

    lf = pl.scan_parquet(PARQUET_PATH).filter(
        pl.col("clean_precursor"),
        pl.len().over("precursor_formula_array").ge(MIN_ISOMERS)
        )

    # Why: Filter for valid precursor m/z and select only needed columns to save memory
    # Why: Add row index to uniquely identify spectra for later grouping
    lf = lf.select([
        "precursor_type",
        "precursor_mz", 
        "precursor_formula_array",
        "ion_mode", 
        "base_inchikey", 
        "spectral_information_score",
        "cleaned_normalized_mz", 
        "cleaned_normalized_intensity"
    ]).with_row_index("idx")
    return (lf,)


@app.cell
def _(lf, pl):
    # Prepare for Self-Join
    # Why: To efficiently find pairs within 5ppm, we bin precursor m/z by 1 Da.
    # We explode the left side to adjacent bins (b-1, b, b+1) to ensure we catch 
    # all matches that might cross bin boundaries.
    # pairs_filtered = lf.join_where(
    #     lf,
    #     pl.col("precursor_mz").is_close(pl.col("precursor_mz_right"), rel_tol=TOLERANCE_PPM * 1e-6),
    #     pl.col("ion_mode") == pl.col("ion_mode_right"),
    #     pl.col("base_inchikey") != pl.col("base_inchikey_right"),
    #     suffix="_right"
    # )
    pairs_filtered = lf.join(
        other=lf,on=["precursor_formula_array","ion_mode"],suffix="_right"
    ).filter(
        pl.col("base_inchikey") != pl.col("base_inchikey_right")
    )
    return (pairs_filtered,)


@app.cell
def _(pairs_filtered, pl):

    pairs_filtered.with_columns(
        spectra=pl.struct(
            mz1=pl.col("cleaned_normalized_mz").alias("mz1"),
            intensities1=pl.col("cleaned_normalized_intensity").alias("intensities1"),
            mz2=pl.col("cleaned_normalized_mz_right").alias("mz2"),
            intensities2=pl.col("cleaned_normalized_intensity_right").alias("intensities2"),
            precursor_mz1=pl.col("precursor_mz").alias("precursor_mz1"),
            precursor_mz2=pl.col("precursor_mz_right").alias("precursor_mz2")
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
            ms2_tolerance_in_ppm=10.0,
            clean_spectra_first=False,
            ignore_precursor=True,
        ),
        entropy_similarity=pl.col("spectra").spectral_similarity.entropy_similarity(
            ms2_tolerance_in_ppm=10.0,
            clean_spectra_first=False,
            ignore_precursor=True,
        ),
        mass_sqrt_cosine_similarity=pl.col("spectra").spectral_similarity.general_cosine_similarity(
            ms2_tolerance_in_ppm=10.0,
            clean_spectra_first=False,
            ignore_precursor=True,
            mass_power=0.5,
            intensity_power=0.5,
        ),
    ).sink_parquet("/home/analytit_admin/Data/spectral_libs/NIST_pairs_with_similarities.parquet",engine="streaming")
    return


@app.cell
def _(pl, plt):
    # Analysis and Plotting
    # Why: Bin spectral info score and calculate FPR for different similarity metrics.
    # We aggregate by the left spectrum ('idx') to determine if it had ANY false positive match above the specific threshold.

    pairs_sim = pl.scan_parquet("/home/analytit_admin/Data/spectral_libs/NIST_pairs_with_similarities.parquet")

    # Why: Calculate max similarity per query spectrum for EACH metric
    unique_spectra_stats = pairs_sim.group_by("idx").agg(
        max_dotprod=pl.col("dotprod_similarity").max(),
        max_entropy=pl.col("entropy_similarity").max(),
        max_mass_sqrt=pl.col("mass_sqrt_cosine_similarity").max(),
        spectral_information_score=pl.col("spectral_information_score").first()
    )

    bin_width = 0.1
    analysis_df = unique_spectra_stats.with_columns(
        info_bin_val=(pl.col("spectral_information_score") / bin_width).floor() * bin_width
    )

    # Filter to range 0-3 (standard range for spectral entropy/info)
    analysis_df = analysis_df.filter(
        (pl.col("info_bin_val") >= 0) & (pl.col("info_bin_val") <= 3)
    )

    grouped = analysis_df.group_by("info_bin_val")

    stats = []
    # Configuration: (Aggregated Column, Threshold, Display Label)
    # Why: Define specific thresholds for each similarity metric here.
    # Allows comparing different metrics at their optimal or standard thresholds.
    METRICS_CONFIG = [
        ("max_dotprod", 0.85, "Dot Product"),
        # ("max_dotprod", 0.9, "Dot Product"),
        ("max_dotprod", 0.95, "Dot Product"),
        ("max_entropy", 0.85, "Entropy"),
        ("max_entropy", 0.95, "Entropy"),
        # ("max_mass_sqrt", 0.9, "Mass Sqrt Cosine"),
    ]

    for agg_col, thresh, label in METRICS_CONFIG:
        # Why: FPR = (Count of spectra with max_metric >= thresh) / (Total count of spectra)
        res = grouped.agg(
            total_count=pl.len(),
            fp_count=(pl.col(agg_col) >= thresh).sum()
        ).with_columns(
            fpr=pl.col("fp_count") / pl.col("total_count"),
            metric_label=pl.lit(label),
            threshold_used=pl.lit(thresh)
        ).sort("info_bin_val")
        stats.append(res)

    # Why: Collect results into memory for plotting
    all_stats = pl.concat(stats).collect()
    print(all_stats)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    MIN_COUNT_THRESHOLD = 5

    # Why: Iterate through the config again to plot each series separately.
    # We filter by both label and threshold to distinguish cases where the same metric 
    # is used with different thresholds (e.g., Entropy 0.9 vs 0.85).
    for _, thresh, label in METRICS_CONFIG:
        subset = all_stats.filter(
            (pl.col("metric_label") == label) & 
            (pl.col("threshold_used") == thresh) & 
            (pl.col("total_count") > MIN_COUNT_THRESHOLD)
        )
    
        if subset.height > 0:
            ax.plot(
                subset["info_bin_val"], 
                subset["fpr"], 
                marker='o', 
                label=f"{label} (Thresh: {thresh})"
            )
        else:
            print(f"Warning: No data for {label} at threshold {thresh}")

    ax.set_xlabel("Spectral Information Score")
    ax.set_ylabel("False Positive Rate")
    ax.set_title("FPR vs Spectral Information Score by Similarity Metric")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Why: Save using the figure object in the same cell to prevent blank output
    fig.savefig("fpr_vs_info_metrics.png")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
