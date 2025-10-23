use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::derive::polars_expr;
use pyo3_polars::PolarsAllocator;
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

const MASS_THRESHOLD_FOR_PPM: f32 = 200.0;

/// Cleans a spectrum with the given parameters.
pub fn clean_spectrum(
    peaks: &[Peak],
    min_mz: Option<f32>,
    max_mz: Option<f32>,
    noise_threshold: Option<f32>,
    ms2_tolerance_in_ppm: f32,
    max_peak_num: Option<usize>,
    normalize_intensity: bool,
) -> Spectrum {
    let mut cleaned_peaks = peaks.to_vec();

    // 1. Remove empty peaks and filter by mz
    cleaned_peaks.retain(|p| {
        p.intensity > 0.0
            && p.mz >= min_mz.unwrap_or(0.0)
            && p.mz <= max_mz.unwrap_or(f32::MAX)
    });
    cleaned_peaks.sort_by(|a, b| a.mz.partial_cmp(&b.mz).unwrap());

    // 2. Centroid the spectrum (C-like iterative approach)
    if ms2_tolerance_in_ppm > 0.0 {
        while need_centroid(&cleaned_peaks, ms2_tolerance_in_ppm) {
            centroid_spectrum(&mut cleaned_peaks, ms2_tolerance_in_ppm);
        }
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
    
    // This step is needed if noise removal or top-k creates zero-intensity peaks, but Rust's retain/truncate avoids this.
    // However, to match the C code flow, we ensure it's sorted by mz before normalization.
    cleaned_peaks.retain(|p| p.intensity > 0.0);
    cleaned_peaks.sort_by(|a, b| a.mz.partial_cmp(&b.mz).unwrap());


    // 5. Normalize intensity
    if normalize_intensity {
        let sum_intensity: f32 = cleaned_peaks.iter().map(|p| p.intensity).sum();
        if sum_intensity > 0.0 {
            for p in &mut cleaned_peaks {
                p.intensity /= sum_intensity;
            }
        } else {
            cleaned_peaks.clear();
        }
    }

    // Final sort by m/z, which should already be the case but ensures correctness.
    cleaned_peaks.sort_by(|a, b| a.mz.partial_cmp(&b.mz).unwrap());

    Spectrum::new(cleaned_peaks)
}

/// Checks if a spectrum needs centroiding based on C's logic.
/// Assumes peaks are sorted by mz.
fn need_centroid(peaks: &[Peak], ms2_tolerance_in_ppm: f32) -> bool {
    if peaks.len() < 2 || ms2_tolerance_in_ppm <= 0.0 {
        return false;
    }
    for i in 0..(peaks.len() - 1) {
        // C code: min_ms2_difference_in_da = spectrum_2d[i + 1][0] * min_ms2_difference_in_ppm * 1e-6;
        let tolerance = peaks[i + 1].mz * ms2_tolerance_in_ppm * 1e-6;
        if peaks[i + 1].mz - peaks[i].mz <= tolerance {
            return true;
        }
    }
    false
}


/// Centroids a spectrum using the logic from the C code.
fn centroid_spectrum(peaks: &mut Vec<Peak>, ms2_tolerance_in_ppm: f32) {
    if peaks.is_empty() {
        return;
    }

    // The calling loop ensures the spectrum is sorted by m/z.

    // 1. Create an argsort of the spectrum by intensity, descending.
    let mut argsort: Vec<usize> = (0..peaks.len()).collect();
    argsort.sort_by(|&a, &b| peaks[b].intensity.partial_cmp(&peaks[a].intensity).unwrap_or(std::cmp::Ordering::Equal));

    for i in 0..argsort.len() {
        let idx = argsort[i];

        if peaks[idx].intensity <= 0.0 {
            continue; // Already merged
        }

        let current_peak_mz = peaks[idx].mz;

        // 2. Determine tolerance based on C code's logic
        let (mz_delta_allowed_left, mz_delta_allowed_right) = if ms2_tolerance_in_ppm > 0.0 {
            let left = current_peak_mz * ms2_tolerance_in_ppm * 1e-6;
            let right = current_peak_mz * ms2_tolerance_in_ppm / (1e6 - ms2_tolerance_in_ppm);
            (left, right)
        } else {
            (0.0, 0.0) // No DA tolerance parameter in Rust version
        };

        // 3. Find left and right bounds for merging in the m/z sorted array.
        let mut idx_left = idx;
        while idx_left > 0 && (current_peak_mz - peaks[idx_left - 1].mz) <= mz_delta_allowed_left {
            idx_left -= 1;
        }

        let mut idx_right = idx;
        while idx_right < peaks.len() - 1 && (peaks[idx_right + 1].mz - current_peak_mz) <= mz_delta_allowed_right {
            idx_right += 1;
        }

        // 4. Merge peaks in the window [idx_left, idx_right]
        // Only merge if there's more than one peak to merge with the current one.
        let mut merge_candidates = 0;
        for j in idx_left..=idx_right {
            if peaks[j].intensity > 0.0 {
                merge_candidates += 1;
            }
        }

        if merge_candidates > 1 {
            let mut intensity_sum = 0.0;
            let mut intensity_weighted_sum = 0.0;

            for j in idx_left..=idx_right {
                if peaks[j].intensity > 0.0 {
                    intensity_sum += peaks[j].intensity;
                    intensity_weighted_sum += peaks[j].intensity * peaks[j].mz;
                    peaks[j].intensity = 0.0; // Mark as merged
                }
            }

            // Write the new peak into the output spectrum at the original high-intensity peak's position
            if intensity_sum > 0.0 {
                peaks[idx].mz = intensity_weighted_sum / intensity_sum;
                peaks[idx].intensity = intensity_sum;
            }
        }
    }

    // Remove the zeroed-out peaks
    peaks.retain(|p| p.intensity > 0.0);
    // Re-sort by m/z as m/z values have changed
    peaks.sort_by(|a, b| a.mz.partial_cmp(&b.mz).unwrap());
}


/// Calculates the spectral entropy of a spectrum. (Identical)
pub fn calculate_spectral_entropy(spectrum: &Spectrum) -> f32 {
    if spectrum.peaks.is_empty() { return 0.0; }
    let sum_intensity: f32 = spectrum.peaks.iter().map(|p| p.intensity).sum();
    if sum_intensity <= 0.0 { return 0.0; } // Avoid division by zero or log(0)
    spectrum.peaks.iter().map(|p| {
        if p.intensity > 0.0 {
            let intensity = p.intensity / sum_intensity;
            -intensity * intensity.ln()
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
    ms2_tolerance_in_ppm: f32,
) -> f32 {
    let mut a = 0;
    let mut b = 0;
    let mut similarity = 0.0;

    while a < spec_a.peaks.len() && b < spec_b.peaks.len() {
        let mass = spec_a.peaks[a].mz;
        let tolerance = if mass < MASS_THRESHOLD_FOR_PPM {
            ms2_tolerance_in_ppm * 1e-6 * MASS_THRESHOLD_FOR_PPM
        } else {
            ms2_tolerance_in_ppm * 1e-6 * mass
        };
        let mass_diff = spec_a.peaks[a].mz - spec_b.peaks[b].mz;
        if mass_diff < -tolerance { a += 1; }
        else if mass_diff > tolerance { b += 1; }
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
    ms2_tolerance_in_ppm: f32,
    clean_spectra_first: bool,
    noise_threshold: Option<f32>,
) -> f32 {
    let (mut a, mut b) = if clean_spectra_first {
        let cleaned_a = clean_spectrum(&spec_a.peaks, None, None, noise_threshold, 2.0 * ms2_tolerance_in_ppm, None, true);
        let cleaned_b = clean_spectrum(&spec_b.peaks, None, None, noise_threshold, 2.0 * ms2_tolerance_in_ppm, None, true);
        (cleaned_a, cleaned_b)
    } else {
        (spec_a.clone(), spec_b.clone()) // Clone if not cleaning to avoid modifying originals if passed by ref elsewhere
    };

    if a.peaks.is_empty() || b.peaks.is_empty() { return 0.0; }
    apply_weight_to_intensity(&mut a);
    apply_weight_to_intensity(&mut b);
    calculate_unweighted_entropy_similarity(&a, &b, ms2_tolerance_in_ppm)
}


// --- Polars Plugin Function ---

#[derive(serde::Deserialize, Debug, Default)]
#[serde(default)]
struct SimilarityKwargs {
    ms2_tolerance_in_ppm: Option<f32>,
    clean_spectra_first: Option<bool>,
    noise_threshold: Option<f32>,
}

#[polars_expr(output_type=Float32)]
fn calculate_similarity_struct(inputs: &[Series], kwargs: SimilarityKwargs) -> PolarsResult<Series> {
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
    let ms2_tolerance_in_ppm = kwargs.ms2_tolerance_in_ppm.unwrap_or(5.0);
    let clean_spectra_first = kwargs.clean_spectra_first.unwrap_or(true);
    let noise_threshold = kwargs.noise_threshold;

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
                        ms2_tolerance_in_ppm,
                        clean_spectra_first,
                        noise_threshold,
                    );
                    Some(similarity)
                }
                _ => None, // Handle rows with missing data
            }
        })
        .collect();

    Ok(out.into_series())
}

#[pymodule]
fn _internal(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();