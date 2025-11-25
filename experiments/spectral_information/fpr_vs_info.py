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
    return Path, np, pl, plt


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
    return MIN_ISOMERS, PARQUET_PATHS


@app.cell
def _(MIN_ISOMERS, PARQUET_PATHS, pl):


    lf_list = []
    for PARQUET_PATH in PARQUET_PATHS:
        # Load spectral library data
        lf = pl.scan_parquet(PARQUET_PATH)
        lf_list.append(lf)
        # Preprocess spectral library
    lf = pl.union(lf_list).filter(
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
    ).sink_parquet("/home/analytit_admin/Data/spectral_libs/all_pairs_with_similarities.parquet",engine="streaming")
    return


@app.cell
def _(np, pl, plt):
    # Analysis and Plotting
    # Why: Bin spectral info score and calculate FPR for different similarity metrics.
    # We aggregate by the left spectrum ('idx') to determine if it had ANY false positive match above the specific threshold.
    MAX_INFO = 2.0
    pairs_sim = pl.scan_parquet("/home/analytit_admin/Data/spectral_libs/all_pairs_with_similarities.parquet")
    print(f"number of pairs:{pairs_sim.select(pl.len()).collect().item()}")
    print("number of unique spectra:", pairs_sim.select(pl.col("idx")).unique().select(pl.len()).collect().item())
    print("number of unique molecules:", pairs_sim.select(pl.col("base_inchikey")).unique().select(pl.len()).collect().item())
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

    # Filter to range 
    analysis_df = analysis_df.filter(
        (pl.col("info_bin_val") >= 0) & (pl.col("info_bin_val") <= MAX_INFO)
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
        # ("max_entropy", 0.75, "Entropy"),
        ("max_entropy", 0.81, "Entropy"),
        ("max_entropy", 0.93, "Entropy"),
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

    # ---------------------------------------------------------
    # NEW: Calculate Molecule Max Info CDF
    # ---------------------------------------------------------
    # Why: Determine the distribution of maximal information content per molecule.
    # We want to see what percentage of molecules have at least one spectrum with info score >= X.
    molecule_max_info = pairs_sim.group_by("base_inchikey").agg(
        max_info=pl.col("spectral_information_score").max()
    ).collect()

    # Define X axis for CDF (using the same range/bins as the main plot for alignment)
    # We generate a regular range to ensure the line is smooth and covers the whole plot area
    cdf_x = np.arange(0, MAX_INFO+0.1, bin_width)
    total_molecules = molecule_max_info.height
    cdf_y = []

    # Calculate reverse CDF (Survival Function)
    for x in cdf_x:
        count = molecule_max_info.filter(pl.col("max_info") >= x).height
        cdf_y.append(count / total_molecules)

    # ---------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------
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

    # NEW: Add secondary axis for Molecule Coverage CDF
    ax2 = ax.twinx()
    ax2.plot(
        cdf_x, 
        cdf_y, 
        color='black', 
        linestyle='--', 
        linewidth=2, 
        alpha=0.6,
        label='Molecule Coverage (Max Info ≥ X)'
    )
    ax2.set_ylabel("Fraction of Molecules")
    ax2.set_ylim(0, 1.05)  # Ensure range starts at 0 and allows seeing the top at 1

    ax.set_xlabel("Spectral Information Score")
    ax.set_ylabel("False Positive Rate")
    ax.set_title("FPR vs Spectral Information Score by Similarity Metric")

    # Combine legends from both axes
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    ax.grid(True, alpha=0.3)

    # Why: Save using the figure object in the same cell to prevent blank output
    fig.savefig("fpr_vs_info_metrics.png")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
