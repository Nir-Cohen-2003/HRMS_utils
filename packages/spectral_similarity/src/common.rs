use polars::prelude::*;

#[derive(Debug, Clone, PartialEq)]
pub struct Peak {
    pub mz: f64,
    pub intensity: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Spectrum {
    pub peaks: Vec<Peak>,
}

impl Spectrum {
    pub fn new(peaks: Vec<Peak>) -> Self {
        Spectrum { peaks }
    }

    pub fn from_polars_lists(mz_series: &Series, intensity_series: &Series) -> PolarsResult<Self> {
        let mz_ca = mz_series.f64()?;
        let int_ca = intensity_series.f64()?;

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

pub const MASS_THRESHOLD_FOR_PPM: f64 = 200.0;

/// Cleans a spectrum with the given parameters.
pub fn clean_spectrum(
    peaks: &[Peak],
    precursor_mz: f64,
    min_mz: Option<f64>,
    max_mz: Option<f64>,
    noise_threshold: Option<f64>,
    ms2_tolerance_in_ppm: f64,
    max_peak_num: Option<usize>,
    normalize_intensity: bool,
    ignore_precursor: bool,
    apply_full_cleaning: bool,
) -> Spectrum {
    let mut cleaned_peaks: Vec<Peak> = peaks.to_vec();

    let mut effective_max_mz = max_mz.unwrap_or(f64::MAX);
    effective_max_mz = effective_max_mz.min(precursor_mz);

    // 1. Remove empty peaks and filter by mz
    cleaned_peaks.retain(|peak| {
        peak.intensity > 0.0
            && peak.mz >= min_mz.unwrap_or(0.0)
            && peak.mz <= effective_max_mz
    });

    if apply_full_cleaning {
        if ignore_precursor {
            // ignore fragments in the 1 Da range below the precursor, including it.
            cleaned_peaks.retain(|peak| peak.mz < precursor_mz - 1.0);
        } else {
            // ignore fragments in the 1 Da range below the precursor, not including it.
            let tolerance = ms2_tolerance_in_ppm * 1e-6 *(MASS_THRESHOLD_FOR_PPM).max(precursor_mz);
            cleaned_peaks.retain(|peak| peak.mz < precursor_mz - 1.0 || (peak.mz - precursor_mz).abs() < tolerance);
        }
    }

    cleaned_peaks.sort_by(|a, b| a.mz.partial_cmp(&b.mz).unwrap());

    // 2. Centroid the spectrum (C-like iterative approach)
    if ms2_tolerance_in_ppm > 0.0 {
        while need_centroid(&cleaned_peaks, ms2_tolerance_in_ppm) {
            centroid_spectrum(&mut cleaned_peaks, ms2_tolerance_in_ppm);
        }
    }

    if apply_full_cleaning {
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
    }
    
    cleaned_peaks.retain(|p| p.intensity > 0.0);
    cleaned_peaks.sort_by(|a, b| a.mz.partial_cmp(&b.mz).unwrap());


    // 5. Normalize intensity
    if normalize_intensity {
        let sum_intensity: f64 = cleaned_peaks.iter().map(|p| p.intensity).sum();
        if sum_intensity > 0.0 {
            for p in &mut cleaned_peaks {
                p.intensity /= sum_intensity;
            }
        } else {
            cleaned_peaks.clear();
        }
    }

    cleaned_peaks.sort_by(|a, b| a.mz.partial_cmp(&b.mz).unwrap());

    Spectrum::new(cleaned_peaks)
}

/// Checks if a spectrum needs centroiding based on C's logic.
/// Assumes peaks are sorted by mz.
pub fn need_centroid(peaks: &[Peak], ms2_tolerance_in_ppm: f64) -> bool {
    if peaks.len() < 2 || ms2_tolerance_in_ppm <= 0.0 {
        return false;
    }
    for i in 0..(peaks.len() - 1) {
        let tolerance = peaks[i + 1].mz * ms2_tolerance_in_ppm * 1e-6;
        if peaks[i + 1].mz - peaks[i].mz <= tolerance {
            return true;
        }
    }
    false
}


/// Centroids a spectrum using the logic from the C code.
pub fn centroid_spectrum(peaks: &mut Vec<Peak>, ms2_tolerance_in_ppm: f64) {
    if peaks.is_empty() {
        return;
    }

    let mut argsort: Vec<usize> = (0..peaks.len()).collect();
    argsort.sort_by(|&a, &b| peaks[b].intensity.partial_cmp(&peaks[a].intensity).unwrap_or(std::cmp::Ordering::Equal));

    for i in 0..argsort.len() {
        let idx = argsort[i];

        if peaks[idx].intensity <= 0.0 {
            continue; // Already merged
        }

        let current_peak_mz = peaks[idx].mz;

        let (mz_delta_allowed_left, mz_delta_allowed_right) = if ms2_tolerance_in_ppm > 0.0 {
            let left = current_peak_mz * ms2_tolerance_in_ppm * 1e-6;
            let right = current_peak_mz * ms2_tolerance_in_ppm / (1e6 - ms2_tolerance_in_ppm);
            (left, right)
        } else {
            (0.0, 0.0) 
        };

        let mut idx_left = idx;
        while idx_left > 0 && (current_peak_mz - peaks[idx_left - 1].mz) <= mz_delta_allowed_left {
            idx_left -= 1;
        }

        let mut idx_right = idx;
        while idx_right < peaks.len() - 1 && (peaks[idx_right + 1].mz - current_peak_mz) <= mz_delta_allowed_right {
            idx_right += 1;
        }

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

            if intensity_sum > 0.0 {
                peaks[idx].mz = intensity_weighted_sum / intensity_sum;
                peaks[idx].intensity = intensity_sum;
            }
        }
    }

    peaks.retain(|p| p.intensity > 0.0);
    peaks.sort_by(|a, b| a.mz.partial_cmp(&b.mz).unwrap());
}
