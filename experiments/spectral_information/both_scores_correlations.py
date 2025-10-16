import polars as pl
import numpy as np
from hrms_utils.spectral_information.space_spectral_info_score import spectral_info_polars
from hrms_utils.spectral_information.tree_spectral_info_score import tree_spectral_info_score_polars
from ms_entropy import calculate_entropy_similarity
from time import perf_counter
import matplotlib.pyplot as plt

def entropy_score_polars(
        spec1_mz: pl.Series, spec1_intensity: pl.Series,
        spec2_mz: pl.Series, spec2_intensity: pl.Series,
        ms2_mass_tolerance: float, noise_threshold: float) -> pl.Series:
    """
    Calculate entropy similarity for batches of spectra pairs.
    
    Why: Convert entire batch from Polars nested structure to ms_entropy's expected format,
    then iterate serially through the Cython function. No parallelism needed since ms_entropy
    releases the GIL internally and the conversion overhead dominates for small batches.
    
    Args:
        spec1_mz: Series of arrays containing m/z values for first spectra
        spec1_intensity: Series of arrays containing intensities for first spectra
        spec2_mz: Series of arrays containing m/z values for second spectra
        spec2_intensity: Series of arrays containing intensities for second spectra
        ms2_mass_tolerance: Mass tolerance in ppm (values < 0.5 treated as already scaled)
        noise_threshold: Minimum intensity threshold for peak cleaning
        
    Returns:
        pl.Series(dtype=pl.Float64) with one similarity score per spectrum pair
    """
    # Why: Convert tolerance to ppm if needed (match existing entropy_score logic)
    if ms2_mass_tolerance < 0.5:
        # Why: Value already contains the 1e-6 factor, convert to ppm
        ms2_tolerance_ppm = ms2_mass_tolerance * 1e6
    else:
        ms2_tolerance_ppm = ms2_mass_tolerance
    
    # Why: Convert Polars Series to numpy object arrays for iteration
    mz1_arrays = spec1_mz.to_numpy()
    intensity1_arrays = spec1_intensity.to_numpy()
    mz2_arrays = spec2_mz.to_numpy()
    intensity2_arrays = spec2_intensity.to_numpy()
    
    num_pairs = len(mz1_arrays)
    assert len(intensity1_arrays) == num_pairs, f"spec1_intensity length {len(intensity1_arrays)} != spec1_mz length {num_pairs}"
    assert len(mz2_arrays) == num_pairs, f"spec2_mz length {len(mz2_arrays)} != spec1_mz length {num_pairs}"
    assert len(intensity2_arrays) == num_pairs, f"spec2_intensity length {len(intensity2_arrays)} != spec1_mz length {num_pairs}"
    
    # Why: Pre-allocate result array for efficiency
    similarities = np.zeros(num_pairs, dtype=np.float64)
    
    # Why: Iterate serially - ms_entropy's Cython implementation releases GIL internally,
    # and batch conversion overhead is minimal for typical HRMS workloads
    for idx in range(num_pairs):
        # Why: Convert each spectrum pair to ms_entropy's expected format (N x 2 float32 arrays)
        peaks_query = np.column_stack((
            np.asarray(mz1_arrays[idx], dtype=np.float32),
            np.asarray(intensity1_arrays[idx], dtype=np.float32)
        ))
        peaks_reference = np.column_stack((
            np.asarray(mz2_arrays[idx], dtype=np.float32),
            np.asarray(intensity2_arrays[idx], dtype=np.float32)
        ))
        
        # Why: Call Cython function directly - it handles its own optimizations
        similarity = calculate_entropy_similarity(
            peaks_query,
            peaks_reference,
            ms2_tolerance_in_ppm=ms2_tolerance_ppm,
            clean_spectra=True,
            noise_threshold=noise_threshold,
        )
        
        # Why: Handle invalid results (NaN or None) by setting to 0.0
        if np.isnan(similarity) or similarity is None:
            similarities[idx] = 0.0
        else:
            similarities[idx] = similarity
    
    # Why: Return as Polars Series with explicit Float64 dtype for consistency
    return pl.Series(similarities, dtype=pl.Float64)

def main():
    # Load data
    nist_path = "/home/analytit_admin/Data/NIST_hr_msms/NIST23_info_scores.parquet"
    try:
        nist = pl.read_parquet(nist_path)
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        print("Please ensure the NIST23.parquet file is at the correct location.")
        return

    # 1. Filter to protonated only
    protonated_only = nist.filter(pl.col("Precursor_type") == "[M+H]+")

    # 2. Filter by precursor formula array
    formula_counts = protonated_only.group_by("Formula_array").agg(pl.n_unique("base_InChIKey").alias("formula_count"))
    formulas_to_keep = formula_counts.filter(pl.col("formula_count") >= 10)
    filtered_by_formula = protonated_only.join(formulas_to_keep, on="Formula_array")

    # 3. Calculate information scores
    print("Calculating information scores...")
    information_scores = filtered_by_formula

    # Check if information scores are already present
    if "spectral_tree_info_score" in information_scores.columns and "spectral_space_info_score" in information_scores.columns:
        information_scores = information_scores.rename({
            "spectral_tree_info_score": "tree_info_score",
            "spectral_space_info_score": "space_info_score"
        })
        print("Using existing information scores.")
    else:
        print("Calculating new information scores.")
        information_scores = information_scores.with_columns([
            pl.struct(["Formula_array", "fragment_formulas"])
              .map_batches(
                  lambda s: tree_spectral_info_score_polars(s.struct.field("Formula_array"), s.struct.field("fragment_formulas")),
                  return_dtype=pl.Float64).alias("tree_info_score"),
            pl.struct(["Formula_array", "fragment_formulas"])
              .map_batches(
                  lambda s: spectral_info_polars(s.struct.field("Formula_array"), s.struct.field("fragment_formulas")),
                  return_dtype=pl.Float64).alias("space_info_score")
        ])

    # 4. All-to-all spectral similarity search
    print("Running similarity search...")
    
    df1 = information_scores.select(["NIST_ID", "base_InChIKey", "masses_normalized", "cleaned_intensities", "tree_info_score", "space_info_score", "Formula_array"])
    df2 = df1.clone().rename({"NIST_ID": "NIST_ID_2", "base_InChIKey": "base_InChIKey_2", "masses_normalized": "masses_normalized_2", "cleaned_intensities": "cleaned_intensities_2"})

    all_pairs = df1.join(df2, on="Formula_array")

    # Filter out self-comparisons
    all_pairs = all_pairs.filter(pl.col("base_InChIKey") != pl.col("base_InChIKey_2"))

    if all_pairs.height > 0:
        print(f"Calculating similarities for {all_pairs.height} spectrum pairs...")
        start = perf_counter()
        # Calculate similarity using map_batches with the non-vectorized entropy_score
        all_results = all_pairs.with_columns(
            pl.struct([
                pl.col("masses_normalized"),
                pl.col("cleaned_intensities"),
                pl.col("masses_normalized_2"),
                pl.col("cleaned_intensities_2"),
            ]).map_batches(
                lambda s: entropy_score_polars(
                    s.struct.field("masses_normalized"),
                    s.struct.field("cleaned_intensities"),
                    s.struct.field("masses_normalized_2"),
                    s.struct.field("cleaned_intensities_2"),
                    ms2_mass_tolerance=0.5,
                    noise_threshold=1e-3
                ),
                return_dtype=pl.Float64
            ).alias("similarity")
        )
        end = perf_counter()
        print(f"Similarity calculation completed in {end - start:.2f} seconds for {all_results.height} pairs.")

        # Why: Save all pairwise similarity results for downstream analysis and debugging
        print("Saving all-pairs similarity results...")
        all_results.write_parquet("all_pairs_similarity_results.parquet")
        print(f"Saved {all_results.height} spectrum pair comparisons to all_pairs_similarity_results.parquet")

        # Why: Calculate number of high-similarity matches at different thresholds for each spectrum
        # to understand how information scores relate to spectrum uniqueness/redundancy
        print("Calculating match counts at similarity thresholds...")
        match_counts = all_results.group_by("NIST_ID").agg([
            pl.col("similarity").filter(pl.col("similarity") > 0.80).count().alias("match_count_above_0p80"),
            pl.col("similarity").filter(pl.col("similarity") > 0.85).count().alias("match_count_above_0p85"),
            pl.col("similarity").filter(pl.col("similarity") > 0.90).count().alias("match_count_above_0p90"),
            pl.col("similarity").filter(pl.col("similarity") > 0.95).count().alias("match_count_above_0p95"),
            pl.max("similarity").alias("max_similarity")
        ])
        
        # Why: Join match counts with information scores and molecule identifiers for correlation analysis
        data_for_correlation = information_scores.join(match_counts, on="NIST_ID")

    else:
        print("No similar spectra found between different molecules.")
        return

    # Why: Calculate Spearman correlations between information scores and both max similarity
    # and match counts at various thresholds to understand information-redundancy relationship
    print("Calculating correlations...")
    correlations = data_for_correlation.group_by("base_InChIKey").agg([
        pl.corr("tree_info_score", "max_similarity", method="spearman").alias("spearman_tree_vs_max_similarity"),
        pl.corr("space_info_score", "max_similarity", method="spearman").alias("spearman_space_vs_max_similarity"),
        pl.corr("tree_info_score", "match_count_above_0p80", method="spearman").alias("spearman_tree_vs_match_count_0p80"),
        pl.corr("space_info_score", "match_count_above_0p80", method="spearman").alias("spearman_space_vs_match_count_0p80"),
        pl.corr("tree_info_score", "match_count_above_0p85", method="spearman").alias("spearman_tree_vs_match_count_0p85"),
        pl.corr("space_info_score", "match_count_above_0p85", method="spearman").alias("spearman_space_vs_match_count_0p85"),
        pl.corr("tree_info_score", "match_count_above_0p90", method="spearman").alias("spearman_tree_vs_match_count_0p90"),
        pl.corr("space_info_score", "match_count_above_0p90", method="spearman").alias("spearman_space_vs_match_count_0p90"),
        pl.corr("tree_info_score", "match_count_above_0p95", method="spearman").alias("spearman_tree_vs_match_count_0p95"),
        pl.corr("space_info_score", "match_count_above_0p95", method="spearman").alias("spearman_space_vs_match_count_0p95"),
    ])

    # Why: Save comprehensive correlation results for analysis
    print("Saving results...")
    correlations.write_parquet("spearman_correlations.parquet")

    # Why: Visualize distribution of correlations for max similarity (primary metric)
    print("Generating histogram...")
    plt.figure(figsize=(10, 6))
    plt.hist(correlations["spearman_tree_vs_max_similarity"].drop_nulls(), bins=50, alpha=0.7, label="Tree Info Score")
    plt.hist(correlations["spearman_space_vs_max_similarity"].drop_nulls(), bins=50, alpha=0.7, label="Space Info Score")
    plt.xlabel("Spearman Correlation")
    plt.ylabel("Frequency")
    plt.title("Distribution of Spearman Correlations (vs Max Similarity)")
    plt.legend()
    plt.savefig("spearman_correlations_histogram.png")
    print("Histogram saved to spearman_correlations_histogram.png")
    # plt.show() is commented out to avoid blocking in a script
    
if __name__ == "__main__":
    main()
