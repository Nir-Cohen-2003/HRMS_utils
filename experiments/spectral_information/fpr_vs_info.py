import marimo

__generated_with = "0.18.1"
app = marimo.App()


@app.cell
def _():
    import polars as pl
    import numpy as np
    import matplotlib.pyplot as plt
    import hrms_utils
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
    ]).with_row_index("idx").with_columns(
        nominal_mass=pl.col("precursor_mz").round(0)
    )
    return (lf,)


@app.cell
def _(lf, pl):

    pairs_filtered = lf.join(
        other=lf,on=["nominal_mass","ion_mode"],suffix="_right"
    ).filter(
        pl.col("precursor_mz").is_close(
            pl.col("precursor_mz_right"),rel_tol=5e-6
        ),
        pl.col("base_inchikey") != pl.col("base_inchikey_right")
    ).with_columns(
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
        dotprod_similarity_with_precursor=pl.col("spectra").spectral_similarity.dotprod_similarity(
            ms2_tolerance_in_ppm=10.0,
            clean_spectra_first=False,
            ignore_precursor=False,
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
    # Why: Bin spectral info score and calculate average number of false matches for different similarity metrics.
    # We aggregate by the left spectrum ('idx') to count how many false positive matches exist above the threshold.
    MAX_INFO = 3.0
    pairs_sim = pl.scan_parquet("/home/analytit_admin/Data/spectral_libs/all_pairs_with_similarities.parquet")
    # pairs_sim = pairs_filtered
    print(f"number of pairs:{pairs_sim.select(pl.len()).collect().item()}")
    print("number of unique spectra:", pairs_sim.select(pl.col("idx")).unique().select(pl.len()).collect().item())
    print("number of unique molecules:", pairs_sim.select(pl.col("base_inchikey")).unique().select(pl.len()).collect().item())

    bin_width = 0.1

    # Why: Define specific thresholds for each similarity metric here.
    METRICS_CONFIG = [
        # metric_key, threshold, human_label, color, marker
        ("dotprod_similarity", 0.8, "Dot Product (ignoring precursor)", "C0", "o"),
        ("dotprod_similarity_with_precursor", 0.8, "Dot Product (including precursor)", "C1", "s"),
        ("entropy_similarity", 0.75, "Entropy", "C2", "^"),
    ]

    stats = []
    for sim_col, thresh, label, color, marker in METRICS_CONFIG:
        unique_spectra_stats = pairs_sim.with_columns(
            is_false_match=(pl.col(sim_col) >= thresh).cast(pl.Int64)
        ).group_by("idx").agg(
            false_match_count=pl.col("is_false_match").sum(),
            spectral_information_score=pl.col("spectral_information_score").first()
        ).with_columns(
            info_bin_val=(pl.col("spectral_information_score") / bin_width).floor() * bin_width
        ).filter(
            (pl.col("info_bin_val") >= 0) & (pl.col("info_bin_val") <= MAX_INFO)
        )

        res = unique_spectra_stats.group_by("info_bin_val").agg(
            total_count=pl.len(),
            avg_false_matches=pl.col("false_match_count").mean()
        ).with_columns(
            metric_label=pl.lit(label),              # human readable
            metric_name=pl.lit(sim_col),             # unambiguous metric key
            threshold_used=pl.lit(thresh),
            plot_color=pl.lit(color),
            plot_marker=pl.lit(marker)
        ).sort("info_bin_val")
        stats.append(res)

    # Why: Collect results into memory for plotting
    all_stats = pl.concat(stats).collect(engine="streaming")
    pl.Comfig.set_tbl_rows(100)
    print(all_stats)

    # ---------------------------------------------------------
    # Molecule Max Info CDF
    # ---------------------------------------------------------
    # Why: Determine the distribution of maximal information content per molecule.
    molecule_max_info = pairs_sim.group_by("base_inchikey").agg(
        max_info=pl.col("spectral_information_score").max()
    ).collect()

    cdf_x = np.arange(0, MAX_INFO + 0.1, bin_width)
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

    for metric_key, thresh, label, color, marker in METRICS_CONFIG:
        subset = all_stats.filter(
            (pl.col("metric_name") == metric_key) &
            (pl.col("threshold_used") == thresh) &
            (pl.col("total_count") > MIN_COUNT_THRESHOLD)
        )

        if subset.height > 0:
            # Use explicit color/marker so the two dotprod lines are visually distinct
            ax.plot(
                subset["info_bin_val"],
                subset["avg_false_matches"],
                marker=marker,
                label=f"{label} (Thresh: {thresh})",
                color=color
            )
        else:
            print(f"Warning: No data for {label} at threshold {thresh}")

    # Secondary axis for Molecule Coverage CDF
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
    ax2.set_ylim(0, 1.05)

    ax.set_xlabel("Spectral Information Score")
    ax.set_ylabel("Average Number of False Matches")
    ax.set_title("Average Number of False Matches vs Spectral Information Score")

    # Combine legends from both axes
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    ax.grid(True, alpha=0.3)

    # plt.show()
    fig.savefig("fpr_vs_info_metrics.png")
    return


if __name__ == "__main__":
    app.run()
