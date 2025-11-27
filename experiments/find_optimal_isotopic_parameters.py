import marimo

__generated_with = "0.18.1"
app = marimo.App()


@app.cell
def _():
    import polars as pl
    from pathlib import Path
    import optuna
    from hrms_utils.formats import get_chromatogram, annotate_chromatogram_with_formulas
    from hrms_utils.formula_annotation.isotopic_pattern import deduce_isotopic_pattern
    from hrms_utils.formula_annotation.element_table import ELEMENT_INDEX
    from hrms_utils.hrms_core import NUM_ELEMENTS
    return (
        ELEMENT_INDEX,
        NUM_ELEMENTS,
        Path,
        deduce_isotopic_pattern,
        get_chromatogram,
        optuna,
        pl,
    )


@app.cell
def _(get_chromatogram, pl):
    chromatogram_path = "/home/analytit_admin/dev/HRMS_utils/tests/data/MSDIAL_output.txt"
    chromatogram_lf = get_chromatogram(str(chromatogram_path)).filter(
            pl.col("ms1_isotopes_m/z").is_not_null(),
            pl.col("msms_m/z").is_not_null(),
            pl.col("Isotope").eq(0),  # only monoisotopic peaks
            pl.col("Height") > 1e6,
        ).lazy().with_columns(
            nominal_mass = pl.col("Precursor_mz_MSDIAL").round(0).cast(pl.Int64)
            )

    return (chromatogram_lf,)


@app.cell
def _(Path, pl):
    PARQUET_PATHS =[
            Path("/home/analytit_admin/Data/spectral_libs/msp_for_Yonathan/NIST.parquet"),
            Path("/home/analytit_admin/Data/spectral_libs/msp_for_Yonathan/fraghub.parquet")
        ]
    spectral_lib = pl.union([pl.scan_parquet(p) for p in PARQUET_PATHS]).filter(
        pl.col("ion_mode").eq("P")
    ).with_columns(
        nominal_mass = pl.col("precursor_mz").round(0).cast(pl.Int64)
    )
    return (spectral_lib,)


@app.cell
def _(chromatogram_lf, pl, spectral_lib):
    suspects_lf = chromatogram_lf.join(
        other=spectral_lib,
        on="nominal_mass",
        how="inner",
        # suffix="_lib"
    ).filter(
        pl.col("Precursor_mz_MSDIAL").is_close(pl.col("precursor_mz"), rel_tol=5e-6)
    ).with_columns(
        pl.struct(
            pl.col("Precursor_mz_MSDIAL").alias("precursor_mz1"),
            pl.col("msms_m/z").alias("mz1"),
            pl.col("msms_intensity").alias("intensities1"),
            pl.col("precursor_mz").alias("precursor_mz2"),
            pl.col("cleaned_normalized_mz").alias("mz2"),
            pl.col("cleaned_normalized_intensity").alias("intensities2"),
        ).spectral_similarity.dotprod_similarity(
            ms2_tolerance_in_ppm=10.0,
            ignore_precursor=True
        ).alias("dotprod_similarity")
    ).filter(
        pl.col("dotprod_similarity") > 0.9,
        pl.col("dotprod_similarity").eq(pl.col("dotprod_similarity").max().over("Peak ID"))
    )
    suspects_df = suspects_lf.collect()
    return (suspects_df,)


@app.cell
def _(suspects_df):
    print(suspects_df.select(
        "Peak ID",
        "Precursor_mz_MSDIAL",
        "name",
        "nist_id",
        "dotprod_similarity",
        "formula"
    ))
    return


@app.cell
def _(ELEMENT_INDEX, pl, suspects_df):
    # Calculate spectral information score for high-quality hits
    # Why: We need to filter for spectra that are informative enough to trust their formula assignment
    INFO_THRESHOLD = 0.5

    # Get carbon index from element table
    CARBON_INDEX = ELEMENT_INDEX["C"]

    suspects_with_info = suspects_df.with_columns(
        pl.struct(
            pl.col("precursor_formula_array").alias("precursor_formula"),
            # pl.col("Precursor_mz_MSDIAL").alias("precursor_mz1"),
            pl.col("msms_m/z").alias("mz"),
            pl.col("msms_intensity").alias("intensities"),
        ).mass_decomposition.clean_and_normalize_spectrum(
            raw_fragment_tolerance_ppm = 5.0,
            normalized_fragment_tolerance_ppm = 5.0,
            min_dbe = -0.5,
            max_dbe = 30.0,
            dbe_mode="half_integer",
            water_absorption=True,
        ).alias("cleaned_spectrum").struct.unpack() 
    ).with_columns(
        pl.struct(
            pl.col("precursor_formula_array").alias("precursor_formula"),
            pl.col("formulas").alias("fragment_formulas"),
        ).spectral_info.spectral_info_score(
            distance_metric="l2",
            ignore_hydrogens=True
        ).alias("spectral_info_score")
    ).filter(
        pl.col("spectral_info_score") > INFO_THRESHOLD
    )

    print(f"Filtered to {suspects_with_info.height} spectra with info score > {INFO_THRESHOLD}")
    return CARBON_INDEX, suspects_with_info


@app.cell
def _(CARBON_INDEX, pl, suspects_with_info):
    # Prepare data for optimization
    # Why: Extract the data we need for isotopic pattern evaluation - precursor m/z, MS1 isotopes, and true carbon count

    optimization_data = suspects_with_info.select(
        "Peak ID",
        "Precursor_mz_MSDIAL",
        "ms1_isotopes_m/z",
        "ms1_isotopes_intensity",
        "molecular_formula_array",  # Ground truth formula from library match
    ).with_columns(
        # Extract true carbon count from the molecular formula
        pl.col("molecular_formula_array").arr.get(CARBON_INDEX).alias("true_carbon_count")
    ).filter(
        # Ensure we have valid data
        pl.col("true_carbon_count").is_not_null(),
        pl.col("true_carbon_count") > 0
    )

    print(f"Optimization dataset: {optimization_data.height} compounds")
    print(optimization_data.select("Peak ID", "Precursor_mz_MSDIAL", "true_carbon_count"))
    return (optimization_data,)


@app.cell
def _():
    return


@app.cell
def _(evaluate_isotopic_parameters, optimization_data, optuna):
    def objective(trial: optuna.Trial) -> float:
        """
        Optuna objective function to find minimal parameters that still give correct carbon bounds.
    
        Why minimize parameters: Lower tolerance values mean tighter constraints on the isotopic
        pattern, which leads to more specific formula predictions. We want the lowest values
        that still correctly bound the true carbon count.
        """
        # Suggest parameters - we want to find the minimum values that work
        absolute_tolerance = trial.suggest_float(
            "isotopic_intensity_absolute_tolerance", 
            1e3, 1e6, 
            log=True
        )
        relative_tolerance = trial.suggest_float(
            "isotopic_intensity_relative_tolerance", 
            0.01, 0.5
        )
    
        # Evaluate with these parameters
        success_rate = evaluate_isotopic_parameters(
            optimization_data,
            isotopic_intensity_absolute_tolerance=absolute_tolerance,
            isotopic_intensity_relative_tolerance=relative_tolerance,
        )
    
        # We need at least 95% success rate
        MIN_SUCCESS_RATE = 0.95
    
        if success_rate < MIN_SUCCESS_RATE:
            # Penalize configurations that don't achieve minimum success rate
            # Return a high value to discourage this configuration
            return float("inf")
    
        # Objective: minimize the sum of normalized parameters while maintaining success
        # Why: We want the tightest constraints that still work
        # Normalize to make them comparable
        normalized_absolute = absolute_tolerance / 1e6
        normalized_relative = relative_tolerance
    
        return normalized_absolute + normalized_relative
    return (objective,)


@app.cell
def _(objective, optuna):
    # Run Optuna optimization
    # Why: Optuna efficiently searches the parameter space to find optimal values

    study = optuna.create_study(
        direction="minimize",
        study_name="isotopic_pattern_optimization",
        sampler=optuna.samplers.TPESampler(seed=42),  # Why: deterministic for reproducibility
    )

    study.optimize(
        objective, 
        n_trials=100,
        show_progress_bar=True,
    )

    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    print(f"Best parameters found:")
    print(f"  isotopic_intensity_absolute_tolerance: {study.best_params['isotopic_intensity_absolute_tolerance']:.2e}")
    print(f"  isotopic_intensity_relative_tolerance: {study.best_params['isotopic_intensity_relative_tolerance']:.4f}")
    print(f"Best objective value: {study.best_value:.6f}")
    return (study,)


@app.cell
def _(
    CARBON_INDEX,
    NUM_ELEMENTS,
    deduce_isotopic_pattern,
    optimization_data,
    pl,
    study,
):
    # Validate the best parameters and show detailed results
    best_absolute = study.best_params["isotopic_intensity_absolute_tolerance"]
    best_relative = study.best_params["isotopic_intensity_relative_tolerance"]

    validation_result = optimization_data.with_columns(
        pl.struct(
            ["Precursor_mz_MSDIAL", "ms1_isotopes_m/z", "ms1_isotopes_intensity"]
        ).map_batches(
            lambda batch: deduce_isotopic_pattern(
                batch.struct.field("Precursor_mz_MSDIAL"),
                batch.struct.field("ms1_isotopes_m/z"),
                batch.struct.field("ms1_isotopes_intensity"),
                ms1_mass_tolerance_ppm=5.0,
                isotopic_mass_tolerance_ppm=2.0,
                minimum_intensity=5e4,
                intensity_absolute_tolerance=best_absolute,
                intensity_relative_tolerance=best_relative,
                max_bounds=None,
            ),
            return_dtype=pl.Array(inner=pl.Int32, shape=(2 * NUM_ELEMENTS,)),
            is_elementwise=True
        ).alias("bounds")
    ).filter(
        pl.col("bounds").arr.min().ge(0)
    ).with_columns(
        pl.col("bounds").arr.get(CARBON_INDEX).alias("min_carbon"),
        pl.col("bounds").arr.get(NUM_ELEMENTS + CARBON_INDEX).alias("max_carbon"),
    ).with_columns(
        (
            (pl.col("true_carbon_count") >= pl.col("min_carbon")) &
            (pl.col("true_carbon_count") <= pl.col("max_carbon"))
        ).alias("carbon_in_bounds"),
        (pl.col("max_carbon") - pl.col("min_carbon")).alias("carbon_range")
    )

    print("\nValidation with optimal parameters:")
    print(validation_result.select(
        "Peak ID",
        "Precursor_mz_MSDIAL",
        "true_carbon_count",
        "min_carbon",
        "max_carbon",
        "carbon_range",
        "carbon_in_bounds"
    ))

    success_rate = validation_result.select(pl.col("carbon_in_bounds").mean()).item()
    avg_carbon_range = validation_result.select(pl.col("carbon_range").mean()).item()

    print(f"\nFinal success rate: {success_rate:.2%}")
    print(f"Average carbon range (specificity): {avg_carbon_range:.2f}")
    print(f"\nOptimal parameters to use:")
    print(f"  isotopic_intensity_absolute_tolerance = {best_absolute:.2e}")
    print(f"  isotopic_intensity_relative_tolerance = {best_relative:.4f}")
    return


if __name__ == "__main__":
    app.run()
