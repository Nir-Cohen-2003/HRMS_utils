use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::derive::polars_expr;
use rayon::prelude::*;

// Keep the Peak and Spectrum structs from the previous version
#[derive(Debug, Clone, PartialEq)]
pub struct Peak {
    pub mz: f32,
    pub intensity: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Spectrum {
    pub peaks: Vec<Peak>,
}

impl Spectrum {
    pub fn new(peaks: Vec<Peak>) -> Self {
        Spectrum { peaks }
    }

    // Helper to create Spectrum from Polars Series (List<Float32>)
    fn from_polars_lists(mz_series: &Series, intensity_series: &Series) -> PolarsResult<Self> {
        let mz_ca = mz_series.f32()?;
        let int_ca = intensity_series.f32()?;

        let peaks: Vec<Peak> = mz_ca
            .into_iter()
            .zip(int_ca.into_iter())
            .filter_map(|(mz_opt, int_opt)| {
                match (mz_opt, int_opt) {
                    (Some(mz), Some(intensity)) => Some(Peak { mz, intensity }),
                    _ => None, // Skip nulls if any
                }
            })
            .collect();
        Ok(Spectrum::new(peaks))
    }
}

// --- Keep the core spectral entropy functions (clean_spectrum, centroid_spectrum, etc.) ---
// --- They remain largely the same as in the previous example ---

/// Cleans a spectrum with the given parameters. (Identical to previous version)
pub fn clean_spectrum(
    peaks: &[Peak],
    min_mz: Option<f32>,
    max_mz: Option<f32>,
    noise_threshold: Option<f32>,
    min_ms2_difference_in_da: f32,
    max_peak_num: Option<usize>,
    normalize_intensity: bool,
) -> Spectrum {
    let mut cleaned_peaks = peaks.to_vec();

    // 1. Remove empty peaks and filter by mz
    cleaned_peaks.retain(|p| {
        p.intensity > 0.0
            && p.mz > min_mz.unwrap_or(0.0)
            && p.mz < max_mz.unwrap_or(f32::MAX)
    });

    // 2. Centroid the spectrum
    if min_ms2_difference_in_da > 0.0 {
        centroid_spectrum(&mut cleaned_peaks, min_ms2_difference_in_da);
    }

    // 3. Remove noise
    if let Some(threshold) = noise_threshold {
        if let Some(max_intensity) = cleaned_peaks.iter().map(|p| p.intensity).max_by(|a, b| a.partial_cmp(b).unwrap()) {
            let noise_level = threshold * max_intensity;
            cleaned_peaks.retain(|p| p.intensity >= noise_level);
        }
    }

    // 4. Keep top N peaks
    if let Some(n) = max_peak_num {
        if n < cleaned_peaks.len() {
            cleaned_peaks.sort_by(|a, b| b.intensity.partial_cmp(&a.intensity).unwrap());
            cleaned_peaks.truncate(n);
        }
    }

    // 5. Normalize intensity
    if normalize_intensity {
        let sum_intensity: f32 = cleaned_peaks.iter().map(|p| p.intensity).sum();
        if sum_intensity > 0.0 {
            for p in &mut cleaned_peaks {
                p.intensity /= sum_intensity;
            }
        } else {
             // Handle case where sum is zero after cleaning/filtering
            cleaned_peaks.clear();
        }
    }

    cleaned_peaks.sort_by(|a, b| a.mz.partial_cmp(&b.mz).unwrap());

    Spectrum::new(cleaned_peaks)
}

fn centroid_spectrum(peaks: &mut Vec<Peak>, tolerance: f32) {
     if peaks.len() <= 1 {
        return;
    }
    peaks.sort_by(|a, b| a.mz.partial_cmp(&b.mz).unwrap());

    let mut merged_peaks = Vec::with_capacity(peaks.len());
    let mut i = 0;
    while i < peaks.len() {
        let current_peak = &peaks[i];
        let mut group_end = i + 1;
        while group_end < peaks.len() && peaks[group_end].mz - current_peak.mz < tolerance {
            group_end += 1;
        }

        if group_end > i + 1 { // If there's more than one peak in the group
            let (sum_intensity, weighted_sum_mz) = peaks[i..group_end]
                .iter()
                .fold((0.0, 0.0), |(s_i, s_mz), p| {
                    (s_i + p.intensity, s_mz + p.intensity * p.mz)
                });

            if sum_intensity > 0.0 {
                 merged_peaks.push(Peak { mz: weighted_sum_mz / sum_intensity, intensity: sum_intensity });
            }
            i = group_end; // Skip merged peaks
        } else {
            merged_peaks.push(current_peak.clone());
            i += 1;
        }
    }
     *peaks = merged_peaks;
}


/// Calculates the spectral entropy of a spectrum. (Identical)
pub fn calculate_spectral_entropy(spectrum: &Spectrum) -> f32 {
    if spectrum.peaks.is_empty() { return 0.0; }
    let sum_intensity: f32 = spectrum.peaks.iter().map(|p| p.intensity).sum();
    if sum_intensity <= 0.0 { return 0.0; } // Avoid division by zero or log(0)
    spectrum.peaks.iter().map(|p| {
        if p.intensity > 0.0 {
            let intensity = p.intensity / sum_intensity;
            -intensity * intensity.log2()
        } else { 0.0 }
    }).sum()
}

/// Applies weight to intensity. (Identical)
pub fn apply_weight_to_intensity(spectrum: &mut Spectrum) {
    if spectrum.peaks.is_empty() { return; }
    let entropy = calculate_spectral_entropy(spectrum);
    if entropy < 3.0 && entropy >= 0.0 { // Added check for non-negative entropy
        let weight = 0.25 + 0.25 * entropy;
        for p in &mut spectrum.peaks {
            p.intensity = p.intensity.powf(weight);
        }
        let sum_intensity: f32 = spectrum.peaks.iter().map(|p| p.intensity).sum();
        if sum_intensity > 0.0 {
            for p in &mut spectrum.peaks {
                p.intensity /= sum_intensity;
            }
        } else {
             // Handle case where sum becomes zero after weighting (unlikely but possible with tiny values)
             spectrum.peaks.clear();
        }
    }
}


/// Calculates unweighted entropy similarity. (Identical)
pub fn calculate_unweighted_entropy_similarity(
    spec_a: &Spectrum,
    spec_b: &Spectrum,
    ms2_tolerance_in_da: f32,
) -> f32 {
    let mut a = 0;
    let mut b = 0;
    let mut similarity = 0.0;

    while a < spec_a.peaks.len() && b < spec_b.peaks.len() {
        let mass_diff = spec_a.peaks[a].mz - spec_b.peaks[b].mz;
        if mass_diff < -ms2_tolerance_in_da { a += 1; }
        else if mass_diff > ms2_tolerance_in_da { b += 1; }
        else {
            let intensity_a = spec_a.peaks[a].intensity;
            let intensity_b = spec_b.peaks[b].intensity;
            let intensity_ab = intensity_a + intensity_b;
            if intensity_a > 0.0 && intensity_b > 0.0 && intensity_ab > 0.0 { // Avoid log2(0)
                 similarity += intensity_ab * intensity_ab.log2() - intensity_a * intensity_a.log2() - intensity_b * intensity_b.log2();
            }
            a += 1; b += 1;
        }
    }
    (similarity / 2.0).max(0.0).min(1.0) // Ensure result is in [0, 1]
}

/// Calculates entropy similarity. (Identical)
pub fn calculate_entropy_similarity(
    spec_a: &Spectrum,
    spec_b: &Spectrum,
    ms2_tolerance_in_da: f32,
    clean_spectra_first: bool,
) -> f32 {
    let (mut a, mut b) = if clean_spectra_first {
        let cleaned_a = clean_spectrum(&spec_a.peaks, None, None, Some(0.01), 2.0 * ms2_tolerance_in_da, None, true);
        let cleaned_b = clean_spectrum(&spec_b.peaks, None, None, Some(0.01), 2.0 * ms2_tolerance_in_da, None, true);
        (cleaned_a, cleaned_b)
    } else {
        (spec_a.clone(), spec_b.clone()) // Clone if not cleaning to avoid modifying originals if passed by ref elsewhere
    };

    if a.peaks.is_empty() || b.peaks.is_empty() { return 0.0; }
    apply_weight_to_intensity(&mut a);
    apply_weight_to_intensity(&mut b);
    calculate_unweighted_entropy_similarity(&a, &b, ms2_tolerance_in_da)
}


// --- Polars Plugin Function ---

#[polars_expr(output_type=Float32)]
fn calculate_similarity_struct(inputs: &[Series]) -> PolarsResult<Series> {
    let struct_series = &inputs[0];
    let ca: &StructChunked = struct_series.struct_()?;

    // Extract the inner list series for mz1, int1, mz2, int2
    // Adjust field names based on your actual DataFrame structure
    let mz1_series = ca.field_by_name("mz1")?;
    let int1_series = ca.field_by_name("intensities1")?;
    let mz2_series = ca.field_by_name("mz2")?;
    let int2_series = ca.field_by_name("intensities2")?;

    // Convert ListChunked to Vec<Option<Series>> for easier parallel iteration
    let mz1_list = mz1_series.list()?;
    let int1_list = int1_series.list()?;
    let mz2_list = mz2_series.list()?;
    let int2_list = int2_series.list()?;

    let mz1_vec: Vec<Option<Series>> = mz1_list.into_iter().collect();
    let int1_vec: Vec<Option<Series>> = int1_list.into_iter().collect();
    let mz2_vec: Vec<Option<Series>> = mz2_list.into_iter().collect();
    let int2_vec: Vec<Option<Series>> = int2_list.into_iter().collect();

    // Parameters (you might want to pass these as arguments to the polars_function)
    let ms2_tolerance_in_da: f32 = 0.02;
    let clean_spectra_first: bool = true;

    // Parallel calculation using rayon
    let out: Float32Chunked = mz1_vec
        .into_par_iter()
        .zip(int1_vec.into_par_iter())
        .zip(mz2_vec.into_par_iter())
        .zip(int2_vec.into_par_iter())
        .map(|(((opt_mz1, opt_int1), opt_mz2), opt_int2)| {
            match (opt_mz1, opt_int1, opt_mz2, opt_int2) {
                (Some(mz1), Some(int1), Some(mz2), Some(int2)) => {
                    // Create Spectrum objects
                    let spec1 = Spectrum::from_polars_lists(&mz1, &int1).ok()?; // Use ok() to convert Result to Option
                    let spec2 = Spectrum::from_polars_lists(&mz2, &int2).ok()?;

                    // Calculate similarity
                    let similarity = calculate_entropy_similarity(
                        &spec1,
                        &spec2,
                        ms2_tolerance_in_da,
                        clean_spectra_first,
                    );
                    Some(similarity)
                }
                _ => None, // Handle rows with missing data
            }
        })
        .collect();

    Ok(out.into_series())
}