import polars as pl 
import numpy as np
import spectral_similarity
import optuna
from typing import List, Tuple


def calculate_manual_general_cosine_similarity(
    mz1: np.ndarray,
    intensities1: np.ndarray,
    mz2: np.ndarray,
    intensities2: np.ndarray,
    precursor_mz1: float,
    precursor_mz2: float,
    intensity_power: float,
    mass_power: float,
    ms2_tolerance_in_ppm: float,
    change_denominator: bool = True,
) -> float:
    """
    Calculate general cosine similarity manually using numpy.
    
    Why: This provides a reference implementation to verify the compiled version.
    """
    # Why: apply intensity power transformation to all peaks
    weighted_intensities1 = intensities1 ** intensity_power
    weighted_intensities2 = intensities2 ** intensity_power
    
    # Why: apply mass power transformation to all peaks
    if mass_power != 0.0:
        weighted_intensities1 = weighted_intensities1 * (mz1 ** mass_power)
        weighted_intensities2 = weighted_intensities2 * (mz2 ** mass_power)
    
    # Why: find matching peaks within ppm tolerance
    numerator = 0.0
    for i in range(len(mz1)):
        for j in range(len(mz2)):
            ppm_error = abs(mz1[i] - mz2[j]) / mz2[j] * 1e6
            if ppm_error <= ms2_tolerance_in_ppm:
                numerator += weighted_intensities1[i] * weighted_intensities2[j]
                break  # Why: only match each peak once to the first match found

    if change_denominator:
        # Why: compute denominator using norms of all peaks in both spectra
        denominator = np.linalg.norm(weighted_intensities1) * np.linalg.norm(weighted_intensities2)
    else:
        # Why: use non-weighted intensities of all peaks
        denominator = np.linalg.norm(intensities1 ** intensity_power) * np.linalg.norm(intensities2 ** intensity_power)

    assert denominator != 0.0, "Denominator cannot be zero in cosine similarity calculation"
    return numerator / denominator


def get_simulated_nist_similarity_pair(
        nist_id1: int, 
        nist_id2: int, 
        expected_similarity: float,
        mass_power: float,
        intensity_power: float,
        tolerance_ppm: float = 10.0,
) -> pl.DataFrame:
    """
    Get a pair of simulated spectra from NIST and compute their similarity.
    
    Why: Compare the compiled Rust implementation against manual numpy calculation and expected NIST value.
    """
    nist_lf = pl.scan_parquet("/home/analytit_admin/Data/NIST_hr_msms/NIST23.parquet")
    first_spectrum = nist_lf.filter(
        (pl.col("NIST_ID") == nist_id1)
    ).collect()
    second_spectrum = nist_lf.filter(
        (pl.col("NIST_ID") == nist_id2)
    ).collect()
    assert not first_spectrum.is_empty() and not second_spectrum.is_empty(), (
        f"No data found for NIST IDs {nist_id1} and/or {nist_id2}"
    )
    
    pair_df = pl.concat([
        first_spectrum.select([
            pl.col("PrecursorMZ").alias("precursor_mz1"),
            pl.col("Name").alias("name1"),
            pl.col("NIST_ID").alias("nist_id1"),
            pl.col("raw_spectrum_mz").alias("mz1"),
            pl.col("raw_spectrum_intensity").alias("intensities1")
        ]),
        second_spectrum.select([
            pl.col("PrecursorMZ").alias("precursor_mz2"),
            pl.col("Name").alias("name2"),
            pl.col("NIST_ID").alias("nist_id2"),
            pl.col("raw_spectrum_mz").alias("mz2"),
            pl.col("raw_spectrum_intensity").alias("intensities2")
        ])
    ], how="horizontal")
    
    # Why: compute similarity using the compiled Rust implementation
    result = pair_df.with_columns(
        rust_similarity=pl.struct([
            pl.col("precursor_mz1"),
            pl.col("mz1"),
            pl.col("intensities1"),
            pl.col("precursor_mz2"),
            pl.col("mz2"),
            pl.col("intensities2"),
        ]).spectral_similarity.general_cosine_similarity(
            intensity_power=intensity_power,
            mass_power=mass_power,
            ms2_tolerance_in_ppm=tolerance_ppm,
            clean_spectra_first=False,
            ignore_precursor=False
        )
    )
    
    # Why: compute similarity manually using numpy for verification
    row = result.row(0, named=True)
    manual_similarity = calculate_manual_general_cosine_similarity(
        mz1=np.array(row["mz1"]),
        intensities1=np.array(row["intensities1"]),
        mz2=np.array(row["mz2"]),
        intensities2=np.array(row["intensities2"]),
        precursor_mz1=row["precursor_mz1"],
        precursor_mz2=row["precursor_mz2"],
        intensity_power=intensity_power,
        mass_power=mass_power,
        ms2_tolerance_in_ppm=tolerance_ppm,
        change_denominator=True,
    )
    
    result = result.with_columns(
        python_similarity=pl.lit(manual_similarity),
        expected_nist_similarity=pl.lit(expected_similarity),
        rust_vs_python_diff=pl.col("rust_similarity") - pl.lit(manual_similarity),
        rust_vs_nist_diff=pl.col("rust_similarity") - pl.lit(expected_similarity),
        python_vs_nist_diff=pl.lit(manual_similarity) - pl.lit(expected_similarity)
    )
    
    return result.select([
        "nist_id1",
        "nist_id2",
        "rust_similarity",
        "python_similarity",
        "expected_nist_similarity",
        "rust_vs_python_diff",
        "rust_vs_nist_diff",
        "python_vs_nist_diff",
        "name1",
        "name2"
    ])


def optimize_similarity_parameters(
    nist_id1: int,
    test_cases: List[Tuple[int, float]],
    tolerance_ppm: float = 10.0,
    n_trials: int = 100,
) -> Tuple[float, float, pl.DataFrame]:
    """
    Optimize mass_power and intensity_power to minimize maximum difference between Rust results and expected NIST values.
    
    Why: Find the best parameters that make our implementation match NIST's similarity scores.
    
    Args:
        nist_id1: Reference NIST ID to compare against
        test_cases: List of (nist_id2, expected_similarity) tuples
        tolerance_ppm: MS2 tolerance in ppm
        n_trials: Number of Optuna trials to run
        
    Returns:
        Tuple of (best_mass_power, best_intensity_power, results_dataframe)
    """
    # Why: cache spectra data to avoid repeated database queries during optimization
    cached_spectra = {}
    nist_lf = pl.scan_parquet("/home/analytit_admin/Data/NIST_hr_msms/NIST23.parquet")
    
    for test_id, _ in [(nist_id1, 0.0)] + test_cases:
        if test_id not in cached_spectra:
            spectrum = nist_lf.filter(pl.col("NIST_ID") == test_id).collect()
            assert not spectrum.is_empty(), f"No data found for NIST_ID {test_id}"
            cached_spectra[test_id] = spectrum
    
    def objective(trial: optuna.Trial) -> float:
        """
        Objective function to minimize: maximum absolute difference between Rust and expected similarities.
        
        Why: We want to minimize the worst-case error across all test cases.
        """
        mass_power = trial.suggest_float("mass_power", 0.0, 3.0)
        intensity_power = trial.suggest_float("intensity_power", 0.0, 2.0)
        
        max_difference = 0.0
        
        for nist_id2, expected_similarity in test_cases:
            # Why: construct pair dataframe from cached spectra
            first_spectrum = cached_spectra[nist_id1]
            second_spectrum = cached_spectra[nist_id2]
            
            pair_df = pl.concat([
                first_spectrum.select([
                    pl.col("PrecursorMZ").alias("precursor_mz1"),
                    pl.col("raw_spectrum_mz").alias("mz1"),
                    pl.col("raw_spectrum_intensity").alias("intensities1")
                ]),
                second_spectrum.select([
                    pl.col("PrecursorMZ").alias("precursor_mz2"),
                    pl.col("raw_spectrum_mz").alias("mz2"),
                    pl.col("raw_spectrum_intensity").alias("intensities2")
                ])
            ], how="horizontal")
            
            # Why: compute similarity using current trial parameters
            result = pair_df.with_columns(
                rust_similarity=pl.struct([
                    pl.col("precursor_mz1"),
                    pl.col("mz1"),
                    pl.col("intensities1"),
                    pl.col("precursor_mz2"),
                    pl.col("mz2"),
                    pl.col("intensities2"),
                ]).spectral_similarity.general_cosine_similarity(
                    intensity_power=intensity_power,
                    mass_power=mass_power,
                    ms2_tolerance_in_ppm=tolerance_ppm,
                    clean_spectra_first=False,
                    ignore_precursor=False
                )
            )
            
            rust_similarity = result["rust_similarity"][0]
            difference = abs(rust_similarity - expected_similarity)
            max_difference = max(max_difference, difference)
        
        return max_difference
    
    # Why: use Optuna to find optimal parameters that minimize maximum error
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_mass_power = study.best_params["mass_power"]
    best_intensity_power = study.best_params["intensity_power"]
    
    # Why: compute final results with best parameters for reporting
    results = []
    for nist_id2, expected_similarity in test_cases:
        result_df = get_simulated_nist_similarity_pair(
            nist_id1, 
            nist_id2, 
            expected_similarity, 
            mass_power=best_mass_power,
            intensity_power=best_intensity_power,
            tolerance_ppm=tolerance_ppm
        )
        results.append(result_df)
    
    final_df = pl.concat(results)
    
    return best_mass_power, best_intensity_power, final_df


if __name__ == "__main__":
    nist_id1 = 1
    
    # Why: test several expected similarity values from NIST across different similarity ranges
    # fill in the values from the nistms program, take the DotProd column
    test_cases = [
        (2, 0.111),
    ]
    
    print("\n" + "="*100)
    print("Optimizing mass_power and intensity_power parameters...")
    print("="*100 + "\n")
    
    best_mass_power, best_intensity_power, optimized_results = optimize_similarity_parameters(
        nist_id1=nist_id1,
        test_cases=test_cases,
        tolerance_ppm=10.0,
        n_trials=2000
    )
    
    print(f"\nBest parameters found:")
    print(f"  mass_power = {best_mass_power:.6f}")
    print(f"  intensity_power = {best_intensity_power:.6f}")
    print(f"\nResults with optimized parameters:")
    print("="*100)
    print(optimized_results.select([
        "name1",
        "name2",
        "rust_similarity",
        "python_similarity",
        "expected_nist_similarity",
        "rust_vs_nist_diff"
    ]))
    
    max_error = optimized_results["rust_vs_nist_diff"].abs().max()
    print(f"\nMaximum absolute error: {max_error:.6f}")
    print("best_mass_power =", best_mass_power)
    print("best_intensity_power =", best_intensity_power)
    print("\n" + "="*100)
    # print(f"Comparing with parameters set to defaults (mass_power={}, intensity_power=0.502)...")
    print("="*100 + "\n")
    
    default_results = []
    for nist_id2, expected_similarity in test_cases:
        result_df = get_simulated_nist_similarity_pair(
            nist_id1, 
            nist_id2, 
            expected_similarity, 
            mass_power=0, 
            intensity_power=0.50
        )
        default_results.append(result_df)

    default_final_df = pl.concat(default_results)
    print(default_final_df.select([
        "name1",
        "name2",
        "rust_similarity",
        "python_similarity",
        "expected_nist_similarity",
        "rust_vs_nist_diff"
    ]))
    
    default_max_error = default_final_df["rust_vs_nist_diff"].abs().max()
    print(f"\nMaximum absolute error with defaults: {default_max_error:.6f}")
    print(f"Improvement: {(default_max_error - max_error):.6f}")