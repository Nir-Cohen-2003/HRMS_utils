use super::common::*;
use crate::common::NUM_ELEMENTS;
use crate::spectral_information::algorithms::calculate_score_for_spectrum;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};

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
    precursor_mz_a: f64, // Required
    precursor_mz_b: f64, // Required
    ms2_tolerance_in_ppm: f64,
    clean_spectra_first: bool,
    noise_threshold: Option<f64>,
    ignore_precursor: bool,
) -> f64 {
    // Why: Each spectrum cleaned relative to its own precursor for symmetric filtering
    let mut a = clean_spectrum(&spec_a.peaks, precursor_mz_a, None, None, noise_threshold, 2.0 * ms2_tolerance_in_ppm, None, true, ignore_precursor, clean_spectra_first);
    let mut b = clean_spectrum(&spec_b.peaks, precursor_mz_b, None, None, noise_threshold, 2.0 * ms2_tolerance_in_ppm, None, true, ignore_precursor, clean_spectra_first);

    if a.peaks.is_empty() || b.peaks.is_empty() { return 0.0; }
    apply_weight_to_intensity(&mut a);
    apply_weight_to_intensity(&mut b);
    calculate_unweighted_entropy_similarity(&a, &b, ms2_tolerance_in_ppm)
}

// Why: Apply same pattern to other similarity functions
pub fn general_cosine_similarity(
    spec_a: &Spectrum,
    spec_b: &Spectrum,
    precursor_mz_a: f64,
    precursor_mz_b: f64,
    ms2_tolerance_in_ppm: f64,
    intensity_power: f64,
    mass_power: f64,
    clean_spectra_first: bool,
    noise_threshold: Option<f64>,
    ignore_precursor: bool,
) -> f64 {
    let (a_cleaned, b_cleaned) = (
        clean_spectrum(&spec_a.peaks, precursor_mz_a, None, None, noise_threshold, 2.0 * ms2_tolerance_in_ppm, None, false, ignore_precursor, clean_spectra_first),
        clean_spectrum(&spec_b.peaks, precursor_mz_b, None, None, noise_threshold, 2.0 * ms2_tolerance_in_ppm, None, false, ignore_precursor, clean_spectra_first)
    );

    if a_cleaned.peaks.is_empty() || b_cleaned.peaks.is_empty() { return 0.0; }

    let weighted_peaks_a: Vec<(f64, f64)> = a_cleaned.peaks.iter().map(|p| {
        let weighted_intensity = p.intensity.powf(intensity_power) * p.mz.powf(mass_power);
        (p.mz, weighted_intensity)
    }).collect();

    let weighted_peaks_b: Vec<(f64, f64)> = b_cleaned.peaks.iter().map(|p| {
        let weighted_intensity = p.intensity.powf(intensity_power) * p.mz.powf(mass_power);
        (p.mz, weighted_intensity)
    }).collect();

    let mut matching_pairs = Vec::new();
    for (i, &(mz_a, intensity_a)) in weighted_peaks_a.iter().enumerate() {
        let tolerance = (ms2_tolerance_in_ppm * 1e-6 * MASS_THRESHOLD_FOR_PPM).max(ms2_tolerance_in_ppm * 1e-6 * mz_a);
        for (j, &(mz_b, intensity_b)) in weighted_peaks_b.iter().enumerate() {
            if (mz_a - mz_b).abs() <= tolerance {
                matching_pairs.push((i, j, intensity_a * intensity_b));
            }
        }
    }

    matching_pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

    let mut dot_product = 0.0;
    let mut used_a = vec![false; weighted_peaks_a.len()];
    let mut used_b = vec![false; weighted_peaks_b.len()];

    for (i, j, score) in matching_pairs {
        if !used_a[i] && !used_b[j] {
            dot_product += score;
            used_a[i] = true;
            used_b[j] = true;
        }
    }

    let mut norm_a = 0.0;
    for &(_, intensity) in &weighted_peaks_a {
        norm_a += intensity * intensity;
    }
    norm_a = norm_a.sqrt();

    let mut norm_b = 0.0;
    for &(_, intensity) in &weighted_peaks_b {
        norm_b += intensity * intensity;
    }
    norm_b = norm_b.sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    (dot_product / (norm_a * norm_b)).max(0.0).min(1.0)
}

pub fn calculate_explained_intensity(
    spec_a: &Spectrum,
    spec_b: &Spectrum,
    precursor_mz_a: f64,
    precursor_mz_b: f64,
    ms2_tolerance_in_ppm: f64,
    intensity_power: f64,
    mass_power: f64,
    clean_spectra_first: bool,
    noise_threshold: Option<f64>,
    ignore_precursor: bool,
    permissive: bool,
) -> f64 {
    let (a_cleaned, b_cleaned) = (
        clean_spectrum(&spec_a.peaks, precursor_mz_a, None, None, noise_threshold, 2.0 * ms2_tolerance_in_ppm, None, false, ignore_precursor, clean_spectra_first),
        clean_spectrum(&spec_b.peaks, precursor_mz_b, None, None, noise_threshold, 2.0 * ms2_tolerance_in_ppm, None, false, ignore_precursor, clean_spectra_first)
    );

    if a_cleaned.peaks.is_empty() { return 0.0; }
    if b_cleaned.peaks.is_empty() { return 0.0; }

    let weighted_peaks_a: Vec<(f64, f64)> = a_cleaned.peaks.iter().map(|p| {
        let weighted_intensity = p.intensity.powf(intensity_power) * p.mz.powf(mass_power);
        (p.mz, weighted_intensity)
    }).collect();

    let weighted_peaks_b: Vec<(f64, f64)> = b_cleaned.peaks.iter().map(|p| {
        let weighted_intensity = p.intensity.powf(intensity_power) * p.mz.powf(mass_power);
        (p.mz, weighted_intensity)
    }).collect();

    let mut a_idx = 0;
    let mut b_idx = 0;
    let mut sum_a = 0.0;

    while a_idx < weighted_peaks_a.len() && b_idx < weighted_peaks_b.len() {
        let mass_a = weighted_peaks_a[a_idx].0;
        let tolerance = (ms2_tolerance_in_ppm * 1e-6 * MASS_THRESHOLD_FOR_PPM).max(ms2_tolerance_in_ppm * 1e-6 * mass_a);
        let mass_diff = weighted_peaks_a[a_idx].0 - weighted_peaks_b[b_idx].0;

        if mass_diff < -tolerance {
            if !permissive {
                return -1.0;
            }
            a_idx += 1;
        } else if mass_diff > tolerance {
            b_idx += 1;
        } else {
            sum_a += weighted_peaks_a[a_idx].1;
            a_idx += 1;
            b_idx += 1;
        }
    }

    if !permissive && a_idx < weighted_peaks_a.len() {
        return -1.0;
    }

    let sum_b: f64 = weighted_peaks_b.iter().map(|(_, intensity)| *intensity).sum();

    if sum_b == 0.0 {
        return 0.0;
    }

    (sum_a / sum_b).max(0.0).min(1.0)
}

#[derive(Clone, Copy)]
struct FormulaWrapper<'a>(&'a [f64]);

impl<'a> Hash for FormulaWrapper<'a> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for &x in self.0 {
            x.to_bits().hash(state);
        }
    }
}

impl<'a> PartialEq for FormulaWrapper<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.0
            .iter()
            .zip(other.0.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits())
    }
}

impl<'a> Eq for FormulaWrapper<'a> {}

pub fn calculate_info_similarity(
    precursor1: &[f64],
    fragments1: &[f64],
    precursor2: &[f64],
    fragments2: &[f64],
    distance_metric: &str,
    ignore_hydrogens: bool,
) -> InfoSimilarity {
    if precursor1 != precursor2 {
        return InfoSimilarity::default();
    }

    let frags1_set: HashSet<FormulaWrapper> = fragments1
        .chunks_exact(NUM_ELEMENTS)
        .map(FormulaWrapper)
        .collect();
    let frags2_set: HashSet<FormulaWrapper> = fragments2
        .chunks_exact(NUM_ELEMENTS)
        .map(FormulaWrapper)
        .collect();

    let spec1_info = calculate_score_for_spectrum(
        precursor1.to_vec(),
        fragments1.to_vec(),
        distance_metric,
        ignore_hydrogens,
    )
    .unwrap_or(0.0);
    let spec2_info = calculate_score_for_spectrum(
        precursor2.to_vec(),
        fragments2.to_vec(),
        distance_metric,
        ignore_hydrogens,
    )
    .unwrap_or(0.0);

    let union_frags: Vec<f64> = frags1_set
        .union(&frags2_set)
        .flat_map(|f| f.0)
        .copied()
        .collect();
    let union_info = calculate_score_for_spectrum(
        precursor1.to_vec(),
        union_frags,
        distance_metric,
        ignore_hydrogens,
    )
    .unwrap_or(0.0);

    let diff1_frags: Vec<f64> = frags1_set
        .difference(&frags2_set)
        .flat_map(|f| f.0)
        .copied()
        .collect();
    let diff1_info = calculate_score_for_spectrum(
        precursor1.to_vec(),
        diff1_frags,
        distance_metric,
        ignore_hydrogens,
    )
    .unwrap_or(0.0);

    let diff2_frags: Vec<f64> = frags2_set
        .difference(&frags1_set)
        .flat_map(|f| f.0)
        .copied()
        .collect();
    let diff2_info = calculate_score_for_spectrum(
        precursor1.to_vec(),
        diff2_frags,
        distance_metric,
        ignore_hydrogens,
    )
    .unwrap_or(0.0);

    InfoSimilarity {
        spec1_info,
        spec2_info,
        union_info,
        diff1_info,
        diff2_info,
    }
}
