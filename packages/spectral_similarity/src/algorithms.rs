use crate::common::*;

/// Calculates the spectral entropy of a spectrum. (Identical)
pub fn calculate_spectral_entropy(spectrum: &Spectrum) -> f64 {
    if spectrum.peaks.is_empty() { return 0.0; }
    let sum_intensity: f64 = spectrum.peaks.iter().map(|p| p.intensity).sum();
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
        let sum_intensity: f64 = spectrum.peaks.iter().map(|p| p.intensity).sum();
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
    ms2_tolerance_in_ppm: f64,
) -> f64 {
    let mut a = 0;
    let mut b = 0;
    let mut similarity = 0.0;

    while a < spec_a.peaks.len() && b < spec_b.peaks.len() {
        let mass = spec_a.peaks[a].mz;
        let tolerance = (ms2_tolerance_in_ppm * 1e-6 * MASS_THRESHOLD_FOR_PPM).max(ms2_tolerance_in_ppm * 1e-6 * mass);
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
    ms2_tolerance_in_ppm: f64,
    clean_spectra_first: bool,
    noise_threshold: Option<f64>,
    precursor_mz: Option<f64>,
    ignore_precursor: Option<bool>,
) -> f64 {
    let (mut a, mut b) = if clean_spectra_first {
        let cleaned_a = clean_spectrum(&spec_a.peaks, None, precursor_mz, noise_threshold, 2.0 * ms2_tolerance_in_ppm, None, true, precursor_mz, ignore_precursor);
        let cleaned_b = clean_spectrum(&spec_b.peaks, None, precursor_mz, noise_threshold, 2.0 * ms2_tolerance_in_ppm, None, true, precursor_mz, ignore_precursor);
        (cleaned_a, cleaned_b)
    } else {
        (spec_a.clone(), spec_b.clone()) // Clone if not cleaning to avoid modifying originals if passed by ref elsewhere
    };

    if !clean_spectra_first {
        a.peaks.sort_by(|a, b| a.mz.partial_cmp(&b.mz).unwrap());
        b.peaks.sort_by(|a, b| a.mz.partial_cmp(&b.mz).unwrap());
    }

    if a.peaks.is_empty() || b.peaks.is_empty() { return 0.0; }
    apply_weight_to_intensity(&mut a);
    apply_weight_to_intensity(&mut b);
    calculate_unweighted_entropy_similarity(&a, &b, ms2_tolerance_in_ppm)
}

pub fn general_cosine_similarity(
    spec_a: &Spectrum,
    spec_b: &Spectrum,
    ms2_tolerance_in_ppm: f64,
    intensity_power: f64,
    mass_power: f64,
    clean_spectra_first: bool,
    noise_threshold: Option<f64>,
    precursor_mz: Option<f64>,
    ignore_precursor: Option<bool>,
) -> f64 {
    let (mut a_cleaned, mut b_cleaned) = if clean_spectra_first {
        let cleaned_a = clean_spectrum(&spec_a.peaks, None, precursor_mz, noise_threshold, 2.0 * ms2_tolerance_in_ppm, None, false, precursor_mz, ignore_precursor);
        let cleaned_b = clean_spectrum(&spec_b.peaks, None, precursor_mz, noise_threshold, 2.0 * ms2_tolerance_in_ppm, None, false, precursor_mz, ignore_precursor);
        (cleaned_a, cleaned_b)
    } else {
        (spec_a.clone(), spec_b.clone())
    };

    if !clean_spectra_first {
        a_cleaned.peaks.sort_by(|a, b| a.mz.partial_cmp(&b.mz).unwrap());
        b_cleaned.peaks.sort_by(|a, b| a.mz.partial_cmp(&b.mz).unwrap());
    }

    if a_cleaned.peaks.is_empty() || b_cleaned.peaks.is_empty() { return 0.0; }

    let weighted_peaks_a: Vec<(f64, f64)> = a_cleaned.peaks.iter().map(|p| {
        let weighted_intensity = p.intensity.powf(intensity_power) * p.mz.powf(mass_power);
        (p.mz, weighted_intensity)
    }).collect();

    let weighted_peaks_b: Vec<(f64, f64)> = b_cleaned.peaks.iter().map(|p| {
        let weighted_intensity = p.intensity.powf(intensity_power) * p.mz.powf(mass_power);
        (p.mz, weighted_intensity)
    }).collect();

    let mut dot_product = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for (_, intensity) in &weighted_peaks_a {
        norm_a += intensity * intensity;
    }
    norm_a = norm_a.sqrt();

    for (_, intensity) in &weighted_peaks_b {
        norm_b += intensity * intensity;
    }
    norm_b = norm_b.sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    let mut a_idx = 0;
    let mut b_idx = 0;

    while a_idx < weighted_peaks_a.len() && b_idx < weighted_peaks_b.len() {
        let mass_a = weighted_peaks_a[a_idx].0;
        let tolerance = (ms2_tolerance_in_ppm * 1e-6 * MASS_THRESHOLD_FOR_PPM).max(ms2_tolerance_in_ppm * 1e-6 * mass_a);
        let mass_diff = weighted_peaks_a[a_idx].0 - weighted_peaks_b[b_idx].0;

        if mass_diff < -tolerance {
            a_idx += 1;
        } else if mass_diff > tolerance {
            b_idx += 1;
        } else {
            dot_product += weighted_peaks_a[a_idx].1 * weighted_peaks_b[b_idx].1;
            a_idx += 1;
            b_idx += 1;
        }
    }

    (dot_product / (norm_a * norm_b)).max(0.0).min(1.0)
}
