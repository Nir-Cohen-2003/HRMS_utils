import marimo

__generated_with = "0.18.1"
app = marimo.App()


@app.cell
def _():
    import polars as pl
    from pathlib import Path
    import optuna
    from hrms_utils.formats import get_chromatogram, annotate_chromatogram_with_formulas
    from hrms_utils.formula_annotation.element_table import ELEMENT_INDEX
    from hrms_utils.hrms_core import NUM_ELEMENTS
    import numpy as np
    import matplotlib.pyplot as plt
    from dataclasses import dataclass
    from hrms_utils.formula_annotation.isotopic_pattern import (
        iso_mass_diffs, iso_zero_probs, iso_first_probs)
    CARBON_INDEX = ELEMENT_INDEX["C"]


    MASS_ACCURACY_PPM_TO_DA_THRESHOLD = 200.0

    return (
        CARBON_INDEX,
        MASS_ACCURACY_PPM_TO_DA_THRESHOLD,
        Path,
        dataclass,
        get_chromatogram,
        iso_first_probs,
        iso_mass_diffs,
        iso_zero_probs,
        np,
        pl,
        plt,
    )


@app.cell
def _(get_chromatogram, pl):
    chromatogram_path = "/home/analytit_admin/dev/HRMS_utils/tests/data/MSDIAL_output.txt"
    chromatogram_lf = get_chromatogram(str(chromatogram_path)).filter(
            pl.col("ms1_isotopes_m/z").is_not_null(),
            pl.col("msms_m/z").is_not_null(),
            pl.col("Isotope").eq(0),  # only monoisotopic peaks
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
        pl.col("dotprod_similarity") > 0.85,
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
def _(pl, suspects_df):
    # Calculate spectral information score for high-quality hits
    # Why: We need to filter for spectra that are informative enough to trust their formula assignment
    INFO_THRESHOLD = 1.0

    # Get carbon index from element table


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
        ).alias("cleaned_spectrum").struct.unnest() 
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
    return (suspects_with_info,)


@app.cell
def _(CARBON_INDEX, pl, suspects_with_info):
    # Prepare data for optimization
    # Why: Extract the data we need for isotopic pattern evaluation - precursor m/z, MS1 isotopes, and true carbon count

    optimization_data = suspects_with_info.select(
        "Peak ID",
        "Precursor_mz_MSDIAL",
        "Height",  # Why: Include height for validation output
        "ms1_isotopes_m/z",
        "ms1_isotopes_intensity",
        "precursor_formula_array",  # Ground truth formula from library match
    ).with_columns(
        # Extract true carbon count from the molecular formula
        pl.col("precursor_formula_array").arr.get(CARBON_INDEX).alias("true_carbon_count")
    ).filter(
        # Ensure we have valid data
        pl.col("true_carbon_count").is_not_null(),
        pl.col("true_carbon_count") > 0
    )

    print(f"Optimization dataset: {optimization_data.height} compounds")
    print(optimization_data.select("Peak ID", "Precursor_mz_MSDIAL", "true_carbon_count"))
    return


@app.cell
def _(
    CARBON_INDEX,
    MASS_ACCURACY_PPM_TO_DA_THRESHOLD,
    dataclass,
    iso_first_probs,
    iso_mass_diffs,
    iso_zero_probs,
    np,
    pl,
    plt,
):


    @dataclass
    class IsotopicToleranceModel:
        """
        Models isotopic intensity tolerance as: tolerance = relative * intensity + absolute
    
        Why this model: Measurement errors in mass spectrometry typically have both
        a fixed component (detector noise, baseline) and a proportional component
        (ion statistics, saturation effects).
        """
        absolute_tolerance: float
        relative_tolerance: float
        success_rate: float


    def fit_isotopic_tolerance_parameters(
        library_hits: pl.DataFrame,
        ms1_mass_tolerance_ppm: float = 5.0,
        isotopic_mass_tolerance_ppm: float = 3.0,
        minimum_c13_intensity: float = 5e4,
        target_success_rate: float = 0.99,
        precursor_mz_column: str = "Precursor_mz_MSDIAL",
        ms1_mz_column: str = "ms1_isotopes_m/z",
        ms1_intensity_column: str = "ms1_isotopes_intensity",
        formula_array_column: str = "precursor_formula_array",
    ) -> IsotopicToleranceModel:
        """
        Fit a linear tolerance model for isotopic pattern matching from library hits.
    
        Why: Given compounds with known formulas (from spectral library matching), we can
        compute the expected vs observed C13 intensity and fit a tolerance model that
        covers the target_success_rate of compounds.
    
        Required DataFrame Schema (using default column names):
            {
                "Precursor_mz_MSDIAL": pl.Float64,
                "ms1_isotopes_m/z": pl.List(pl.Float64),
                "ms1_isotopes_intensity": pl.List(pl.Float64),
                "precursor_formula_array": pl.Array(pl.<int type>, NUM_ELEMENTS),
            }
    
        Notes:
            - precursor_formula_array must have carbon count at index ELEMENT_INDEX["C"]
            - ms1_isotopes_m/z and ms1_isotopes_intensity arrays must be aligned (same length per row)
            - Column names are configurable via the *_column parameters
    
        Args:
            library_hits: DataFrame with MS1 isotope data and known molecular formulas
            ms1_mass_tolerance_ppm: Tolerance for finding precursor in MS1 spectrum
            isotopic_mass_tolerance_ppm: Tolerance for finding C13 peak
            minimum_c13_intensity: Skip compounds where expected C13 is below this threshold
            target_success_rate: Fraction of compounds the tolerance should cover (0.0 to 1.0)
            precursor_mz_column: Column name for precursor m/z (expects pl.Float64)
            ms1_mz_column: Column name for MS1 isotope m/z array (expects pl.List(pl.Float64))
            ms1_intensity_column: Column name for MS1 isotope intensity array (expects pl.List(pl.Float64))
            formula_array_column: Column name for formula array (expects pl.Array(<int type>, NUM_ELEMENTS))
    
        Returns:
            IsotopicToleranceModel with fitted parameters
    
        Raises:
            AssertionError: If no valid compounds with carbon count and MS1 isotope data are found,
                            or if no compounds could be processed after filtering.
        """
        # Why: Extract true carbon count and filter for valid data
        data = library_hits.with_columns(
            pl.col(formula_array_column).arr.get(CARBON_INDEX).alias("_true_carbon_count")
        ).filter(
            pl.col("_true_carbon_count").is_not_null(),
            pl.col("_true_carbon_count") > 0,
            pl.col(ms1_mz_column).is_not_null(),
            pl.col(ms1_intensity_column).is_not_null(),
        )
    
        assert data.height > 0, "No valid compounds with carbon count and MS1 isotope data found"
    
        # Why: Compute errors for each compound
        precursor_intensities: list[float] = []
        absolute_errors: list[float] = []
    
        for row in data.iter_rows(named=True):
            precursor_mz = row[precursor_mz_column]
            ms1_mzs = np.atleast_1d(np.array(row[ms1_mz_column]))
            ms1_intensities = np.atleast_1d(np.array(row[ms1_intensity_column]))
            true_carbon = row["_true_carbon_count"]
        
            ms1_absolute_tolerance = max(precursor_mz, MASS_ACCURACY_PPM_TO_DA_THRESHOLD) * ms1_mass_tolerance_ppm * 1e-6
            isotopic_absolute_tolerance = max(precursor_mz, MASS_ACCURACY_PPM_TO_DA_THRESHOLD) * isotopic_mass_tolerance_ppm * 1e-6
        
            # Why: Find precursor peak in MS1 spectrum
            precursor_idx = np.where(np.isclose(ms1_mzs, precursor_mz, atol=ms1_absolute_tolerance, rtol=0.0))[0]
            if len(precursor_idx) == 0:
                continue
        
            precursor_ms1_mz = ms1_mzs[precursor_idx[ms1_intensities[precursor_idx].argmax()]]
            precursor_ms1_intensity = ms1_intensities[precursor_idx].max()
        
            # Why: Compute expected C13 intensity from known carbon count
            expected_c13_intensity = (true_carbon * iso_first_probs[0] * precursor_ms1_intensity) / iso_zero_probs[0]
        
            if expected_c13_intensity < minimum_c13_intensity:
                continue
        
            # Why: Find observed C13 peak
            c13_peak_mz = precursor_ms1_mz + iso_mass_diffs[0]
            c13_peaks_idx = np.where(np.isclose(ms1_mzs, c13_peak_mz, atol=isotopic_absolute_tolerance, rtol=0.0))[0]
            observed_c13_intensity = ms1_intensities[c13_peaks_idx].max() if len(c13_peaks_idx) > 0 else 0.0
        
            precursor_intensities.append(precursor_ms1_intensity)
            absolute_errors.append(abs(observed_c13_intensity - expected_c13_intensity))
    
        assert len(precursor_intensities) > 0, "No compounds could be processed - check MS1 isotope data validity"
    
        precursor_arr = np.array(precursor_intensities)
        error_arr = np.array(absolute_errors)
    
        # Why: Grid search for minimal (relative, absolute) covering target_success_rate
        max_absolute_error = np.max(error_arr)
        max_relative_error = np.max(error_arr / precursor_arr)
        median_intensity = np.median(precursor_arr)
    
        best_relative, best_absolute, best_objective = 0.0, max_absolute_error * 1.1, float("inf")
    
        for log_abs in np.linspace(2, np.log10(max_absolute_error * 2), 80):
            absolute = 10 ** log_abs
            for relative in np.linspace(0.0, min(max_relative_error * 1.5, 0.5), 80):
                coverage = np.mean(error_arr <= relative * precursor_arr + absolute)
                if coverage >= target_success_rate:
                    objective = relative * median_intensity + absolute
                    if objective < best_objective:
                        best_objective, best_relative, best_absolute = objective, relative, absolute
    
        final_coverage = np.mean(error_arr <= best_relative * precursor_arr + best_absolute)
    
        # Why: Visualize the fit
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(precursor_arr, error_arr, alpha=0.6, label="Observed error per compound")
    
        x_range = np.linspace(precursor_arr.min(), precursor_arr.max(), 100)
        ax.plot(x_range, best_relative * x_range + best_absolute, 'r-', linewidth=2,
                label=f"Fitted: {best_relative:.6f}·I + {best_absolute:.2e}")
    
        ax.set_xlabel("Precursor (C12) Intensity")
        ax.set_ylabel("Absolute Error |observed - expected C13|")
        ax.set_title(f"Isotopic Intensity Error vs Precursor Intensity\n(Coverage: {final_coverage:.1%} of {len(error_arr)} compounds)")
        ax.legend()
        ax.set_xscale("log")
        ax.set_yscale("log")
        plt.tight_layout()
        plt.show()
    
        print(f"\nOptimal parameters for deduce_isotopic_pattern:")
        print(f"  intensity_absolute_tolerance: {best_absolute:.2e}")
        print(f"  intensity_relative_tolerance: {best_relative:.6f}")
        print(f"  Success rate achieved: {final_coverage:.2%}")
    
        return IsotopicToleranceModel(
            absolute_tolerance=best_absolute,
            relative_tolerance=best_relative,
            success_rate=final_coverage,
        )
    return (fit_isotopic_tolerance_parameters,)


@app.cell
def _(fit_isotopic_tolerance_parameters, suspects_with_info):
    # In a new cell after suspects_with_info is computed:

    model = fit_isotopic_tolerance_parameters(
        library_hits=suspects_with_info,
        ms1_mass_tolerance_ppm=5.0,
        isotopic_mass_tolerance_ppm=2.0,
        minimum_c13_intensity=1e4,
        target_success_rate=1.0,
    )
    return


if __name__ == "__main__":
    app.run()
