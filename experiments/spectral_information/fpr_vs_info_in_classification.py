import marimo

__generated_with = "0.18.2"
app = marimo.App()


@app.cell
def _():
    import polars as pl
    import matplotlib.pyplot as plt
    plt.style.use('default')

    return pl, plt


@app.cell
def _(pl):
    nist = pl.scan_parquet("/home/analytit_admin/Data/spectral_libs/msp_for_Yonathan/NIST.parquet").select(
        ["nist_id","spectral_information_score","base_inchikey"]
    )
    model_results= pl.scan_parquet("/home/analytit_admin/dev/tidy_amphetamines/logs/gnn_run/version_126/test_predictions.parquet")
    return model_results, nist


@app.cell
def _(model_results, nist, pl):
    model_results_with_info= model_results.join(nist, left_on="spectrum_id", right_on="nist_id", how="inner").collect()
    model_results_with_info.filter(~pl.col("target").eq(1))
    return (model_results_with_info,)


@app.cell
def _(model_results_with_info, pl):
    ampthetamines= model_results_with_info.filter(pl.col("target").eq(0))
    non_amhetamines= model_results_with_info.filter(pl.col("target").eq(1))
    return ampthetamines, non_amhetamines


@app.cell
def _(pl):

    def compute_error_rate_by_bin(df: pl.DataFrame, bin_width: float, upper_bound: float) -> pl.DataFrame:
        """
        Computes the likelihood of false prediction binned by spectral information score.
    
        Args:
            df: Input dataframe containing 'target', 'prediction', and 'spectral_information_score'.
            bin_width: Width of the spectral information bins.
            upper_bound: Maximum spectral information score to include.
        
        Returns:
            pl.DataFrame: Aggregated stats with 'bin_start' and 'error_rate'.
        """
        # Why: Convert to lazy to optimize the query plan, though input is likely eager.
        # We calculate the error (target != prediction) and bin the spectral info score.
        return (
            df.lazy()
            .filter(pl.col("spectral_information_score") <= upper_bound)
            .with_columns([
                (pl.col("target") != pl.col("prediction")).cast(pl.Float64).alias("is_error"),
                ((pl.col("spectral_information_score") / bin_width).floor() * bin_width).alias("bin_start")
            ])
            .group_by("bin_start")
            .agg(pl.col("is_error").mean().alias("error_rate"))
            .sort("bin_start")
            .collect()
        )

    return (compute_error_rate_by_bin,)


@app.cell
def _(ampthetamines, compute_error_rate_by_bin, non_amhetamines, plt):
    # Constants for binning configuration
    BIN_WIDTH: float = 0.5
    UPPER_BOUND: float = 3.0
    PLOT_AMPHETAMINES: bool = False

    # Compute statistics for both groups
    # Note: Using variable names from previous cell context
    non_amp_stats = compute_error_rate_by_bin(non_amhetamines, BIN_WIDTH, UPPER_BOUND)

    if PLOT_AMPHETAMINES:
        amp_stats = compute_error_rate_by_bin(ampthetamines, BIN_WIDTH, UPPER_BOUND)

    # Generate Plot
    # Why: Set facecolor to white for consistent styling
    plt.figure(figsize=(10, 6), facecolor="white")

    # Plot Amphetamines line
    if PLOT_AMPHETAMINES:
        plt.plot(
            amp_stats["bin_start"], 
            amp_stats["error_rate"], 
            label="Amphetamines (Target=0)", 
            marker='o',
            linestyle='-',

        )

    # Plot Non-Amphetamines line
    plt.plot(
        non_amp_stats["bin_start"], 
        non_amp_stats["error_rate"], 
        label="Non-Amphetamines (Target=1)", 
        marker='x',
        linestyle='-'
    )

    plt.xlabel("Spectral Information Score")
    plt.ylabel("Likelihood of False Prediction")
    plt.title(f"Error Rate vs Spectral Information (Bin Width: {BIN_WIDTH})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # Why: Ensure white background on saved file
    plt.savefig("model_results_vs_info.png", dpi=600, facecolor="white",transparent=False)
    return BIN_WIDTH, PLOT_AMPHETAMINES, UPPER_BOUND


@app.cell
def _(
    BIN_WIDTH: float,
    PLOT_AMPHETAMINES: bool,
    UPPER_BOUND: float,
    compute_error_rate_by_bin,
    model_results_with_info,
    pl,
    plt,
):
    best_spectra_per_compound = (
        model_results_with_info.lazy()
        .sort("spectral_information_score", descending=True)
        .unique(subset=["base_inchikey"], keep="first")
        .collect()
    )

    amp_best = best_spectra_per_compound.filter(pl.col("target").eq(0))
    non_amp_best = best_spectra_per_compound.filter(pl.col("target").eq(1))

    # Compute stats for best spectra
    non_amp_best_stats = compute_error_rate_by_bin(non_amp_best, BIN_WIDTH, UPPER_BOUND)

    if PLOT_AMPHETAMINES:
        amp_best_stats = compute_error_rate_by_bin(amp_best, BIN_WIDTH, UPPER_BOUND)

    # Generate Plot for Best Spectra
    plt.figure(figsize=(10, 6), facecolor="white")

    if PLOT_AMPHETAMINES:
        plt.plot(
            amp_best_stats["bin_start"], 
            amp_best_stats["error_rate"], 
            label="Amphetamines (Target=0, Max Info)", 
            marker='o',
            linestyle='-'
        )

    plt.plot(
        non_amp_best_stats["bin_start"], 
        non_amp_best_stats["error_rate"], 
        label="Non-Amphetamines (Target=1, Max Info)", 
        marker='x',
        linestyle='-'
    )

    plt.xlabel("Spectral Information Score")
    plt.ylabel("Likelihood of False Prediction")
    plt.title(f"Error Rate vs Spectral Information (Max per Compound, Bin Width: {BIN_WIDTH})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("model_results_vs_info_max_per_compound.png", dpi=600, facecolor="white",transparent=False)
    plt.close()
    return


if __name__ == "__main__":
    app.run()
