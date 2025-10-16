import polars as pl
import numpy as np
from ms_entropy import calculate_entropy_similarity


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

