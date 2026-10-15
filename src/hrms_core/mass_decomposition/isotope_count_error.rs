//! Opt-in count margins around calibrated isotope estimates, never endpoint midpoints.
//!
//! The matching/inference path below mirrors mass-decomposition 0.1.0 bounds.rs,
//! using its public isotope constants and tolerance floor. Keep numerical parity
//! with that dependency (tested below) when upgrading it. The default plugin path
//! continues to call the dependency directly.
use super::common::IsotopicPatternParams;
use mass_decomposition::bounds::MIN_MASS_FOR_TOLERANCE;
use mass_decomposition::element::ELEMENT_ISOTOPES;
use std::collections::HashMap;

pub const TARGET_ELEMENTS: [(usize, &str); 4] = [(1, "C"), (7, "S"), (8, "Cl"), (10, "Br")];

pub fn validate_count_errors(
    minimum: Option<&HashMap<String, i32>>,
    maximum: Option<&HashMap<String, i32>>,
) -> Result<(), String> {
    for values in [minimum, maximum].into_iter().flatten() {
        for (element, value) in values {
            if !TARGET_ELEMENTS.iter().any(|(_, symbol)| symbol == element) || *value < 0 {
                return Err(
                    "Element count errors require C/S/Cl/Br and nonnegative Int32 values".into(),
                );
            }
        }
    }
    if let (Some(minimum), Some(maximum)) = (minimum, maximum) {
        for (element, value) in minimum {
            if maximum.get(element).is_some_and(|upper| value > upper) {
                return Err(format!(
                    "minimum element count error exceeds maximum for {element}"
                ));
            }
        }
    }
    Ok(())
}

/// Intersect adjusted margins with physical/global bounds; an empty interval stays
/// empty, rather than inventing a candidate at a cap. Float arithmetic avoids Int32
/// overflow before the final saturating conversion to the existing output schema.
pub fn clamp_carbon_tolerance(
    center: f64,
    tolerance: f64,
    global_minimum: i32,
    global_maximum: i32,
) -> (i32, i32) {
    let lower_bound = (center - tolerance)
        .ceil()
        .max(f64::from(global_minimum.max(0)));
    let upper_bound = (center + tolerance)
        .floor()
        .min(f64::from(global_maximum));
    if lower_bound > f64::from(i32::MAX) {
        return (i32::MAX, global_maximum.min(i32::MAX - 1));
    }
    (lower_bound as i32, upper_bound as i32)
}

pub fn clamp_count_margins(
    expected_count: f64,
    lower: f64,
    upper: f64,
    minimum_error: Option<i32>,
    maximum_error: Option<i32>,
    global_minimum: i32,
    global_maximum: i32,
) -> (i32, i32) {
    let center = (expected_count + 0.5).floor();
    let mut lower_error = (center - lower.ceil()).max(0.0);
    let mut upper_error = (upper.floor() - center).max(0.0);
    if let Some(minimum) = minimum_error {
        lower_error = lower_error.max(f64::from(minimum));
        upper_error = upper_error.max(f64::from(minimum));
    }
    if let Some(maximum) = maximum_error {
        lower_error = lower_error.min(f64::from(maximum));
        upper_error = upper_error.min(f64::from(maximum));
    }
    let lower_bound = (center - lower_error).max(f64::from(global_minimum.max(0)));
    let upper_bound = (center + upper_error).min(f64::from(global_maximum));
    if lower_bound > f64::from(i32::MAX) {
        // Saturating both endpoints to MAX would incorrectly turn an empty
        // representable interval into one candidate at MAX.
        return (i32::MAX, global_maximum.min(i32::MAX - 1));
    }
    (lower_bound as i32, upper_bound as i32)
}

pub fn deduce_isotopic_pattern_with_centers(
    precursor_mz: f64,
    ms1_mzs: &[f64],
    ms1_intensities: &[f64],
    params: &IsotopicPatternParams,
) -> ([f64; 8], [Option<f64>; 4]) {
    let ms1_absolute_tolerance =
        precursor_mz.max(MIN_MASS_FOR_TOLERANCE) * params.ms1_mass_tolerance_ppm * 1e-6;
    let isotopic_absolute_tolerance =
        precursor_mz.max(MIN_MASS_FOR_TOLERANCE) * params.isotopic_mass_tolerance_ppm * 1e-6;

    // Find precursor
    let precursor_indices: Vec<usize> = ms1_mzs
        .iter()
        .enumerate()
        .filter(|&(_, &mz)| (mz - precursor_mz).abs() <= ms1_absolute_tolerance)
        .map(|(i, _)| i)
        .collect();

    if precursor_indices.is_empty() {
        return ([-1.0; 8], [None; 4]);
    }

    // Find precursor intensity (max intensity among matches)
    let mut precursor_ms1_intensity = 0.0;
    let mut best_precursor_idx = 0;
    for &idx in &precursor_indices {
        if ms1_intensities[idx] > precursor_ms1_intensity {
            precursor_ms1_intensity = ms1_intensities[idx];
            best_precursor_idx = idx;
        }
    }
    let precursor_ms1_mz = ms1_mzs[best_precursor_idx];

    let mut result = [0.0; 8];
    let mut centers = [None; 4];

    // Element indices in ELEMENT_SYMBOLS
    // C: 1, S: 7, Cl: 8, Br: 10
    // Output indices: 0: C_low, 1: S_low, 2: Cl_low, 3: Br_low, 4: C_high, 5: S_high, 6: Cl_high, 7: Br_high

    let target_elements = [
        (1, 0),  // (Element Index, Result Index) -> C
        (7, 1),  // S
        (8, 2),  // Cl
        (10, 3), // Br
    ];

    for &(elem_idx, res_idx) in &target_elements {
        let iso_props = ELEMENT_ISOTOPES[elem_idx].expect("Element should have isotopic props");

        let peak_mz = precursor_ms1_mz + iso_props.mass_diff;
        let peak_indices: Vec<usize> = ms1_mzs
            .iter()
            .enumerate()
            .filter(|&(_, &mz)| (mz - peak_mz).abs() <= isotopic_absolute_tolerance)
            .map(|(i, _)| i)
            .collect();

        let peak_total_intensity = if peak_indices.is_empty() {
            0.0
        } else {
            peak_indices
                .iter()
                .map(|&i| ms1_intensities[i])
                .fold(0.0, f64::max)
        };

        let lower;
        let upper;

        if peak_total_intensity < params.minimum_intensity {
            lower = 0.0;
            upper = (params.minimum_intensity * iso_props.prob_0)
                / (iso_props.prob_1 * precursor_ms1_intensity);
        } else {
            let observed = peak_total_intensity;
            let expected_hat =
                params.model.alpha * (observed + params.model.offset).powf(params.model.beta);
            let count =
                (expected_hat * iso_props.prob_0) / (iso_props.prob_1 * precursor_ms1_intensity);
            if !peak_indices.is_empty() && count.is_finite() && count >= 0.0 {
                centers[res_idx] = Some(count);
            }
            let w_l = params.model.hetero_lower_a
                * (observed + params.model.offset).powf(params.model.hetero_lower_b);
            let w_u = params.model.hetero_upper_a
                * (observed + params.model.offset).powf(params.model.hetero_upper_b);
            let expected_lower = expected_hat * (-w_l).exp();
            let expected_upper = expected_hat * w_u.exp();
            lower =
                (expected_lower * iso_props.prob_0) / (iso_props.prob_1 * precursor_ms1_intensity);
            upper =
                (expected_upper * iso_props.prob_0) / (iso_props.prob_1 * precursor_ms1_intensity);
        }

        result[res_idx] = lower;
        result[res_idx + 4] = upper;
    }

    (result, centers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mass_decomposition::bounds::{CalibratedIsotopicModel, deduce_isotopic_pattern_inner};

    #[test]
    fn count_margin_rounding_asymmetry_and_caps() {
        assert_eq!(
            clamp_count_margins(10.1, 8.3, 18.4, Some(2), Some(2), 0, 50),
            (8, 12)
        );
        assert_eq!(
            clamp_count_margins(10.5, 10.5, 10.5, Some(0), Some(0), 0, 50),
            (11, 11)
        );
        assert_eq!(
            clamp_count_margins(10.1, 8.3, 18.4, Some(3), None, 0, 50),
            (7, 18)
        );
        assert_eq!(
            clamp_count_margins(10.1, 8.3, 18.4, None, Some(2), 0, 50),
            (9, 12)
        );
        assert_eq!(
            clamp_count_margins(1.1, 1.1, 1.1, Some(2), Some(2), 0, 50),
            (0, 3)
        );
        assert_eq!(
            clamp_count_margins(10.1, 8.3, 18.4, Some(2), Some(2), 9, 11),
            (9, 11)
        );
        assert_eq!(
            clamp_count_margins(10.1, 8.3, 18.4, Some(i32::MAX), None, 0, i32::MAX),
            (0, i32::MAX)
        );
        assert_eq!(
            clamp_count_margins(60.0, 60.0, 60.0, Some(2), Some(2), 0, 50),
            (58, 50)
        );
        assert_eq!(
            clamp_count_margins(1e300, 1e300, 1e300, None, Some(2), 0, i32::MAX),
            (i32::MAX, i32::MAX - 1)
        );
    }

    #[test]
    fn opted_inference_matches_dependency_and_detects_only_calibrated_centers() {
        let params = IsotopicPatternParams {
            ms1_mass_tolerance_ppm: 5.0,
            isotopic_mass_tolerance_ppm: 0.1,
            minimum_intensity: 100.0,
            model: CalibratedIsotopicModel {
                alpha: 1.5,
                beta: 0.98,
                offset: 0.0,
                hetero_lower_a: 1.7,
                hetero_lower_b: -0.1,
                hetero_upper_a: 18.0,
                hetero_upper_b: -0.29,
            },
        };
        for (element_index, _) in TARGET_ELEMENTS {
            let props = ELEMENT_ISOTOPES[element_index].unwrap();
            for precursor in [100.0, 200.0, 500.0] {
                for intensity in [0.0, 99.0, 100.0, 1e5] {
                    let masses = [precursor, precursor + props.mass_diff];
                    let intensities = [1e6, intensity];
                    let (bounds, centers) = deduce_isotopic_pattern_with_centers(
                        precursor,
                        &masses,
                        &intensities,
                        &params,
                    );
                    assert_eq!(
                        bounds,
                        deduce_isotopic_pattern_inner(precursor, &masses, &intensities, &params)
                    );
                    let result_index = TARGET_ELEMENTS
                        .iter()
                        .position(|(index, _)| *index == element_index)
                        .unwrap();
                    assert_eq!(centers[result_index].is_some(), intensity >= 100.0);
                    if let Some(center) = centers[result_index] {
                        let expected =
                            params.model.alpha * intensity.powf(params.model.beta) * props.prob_0
                                / (props.prob_1 * 1e6);
                        assert_eq!(center, expected);
                    }
                }
            }
        }
        let missing = deduce_isotopic_pattern_with_centers(100.0, &[99.0], &[1e6], &params);
        assert_eq!(missing, ([-1.0; 8], [None; 4]));
    }

    #[test]
    fn count_error_native_validation() {
        let carbon = HashMap::from([("C".to_string(), 2)]);
        assert!(validate_count_errors(Some(&carbon), Some(&carbon)).is_ok());
        for invalid in [
            HashMap::from([("N".to_string(), 2)]),
            HashMap::from([("C".to_string(), -1)]),
        ] {
            assert!(validate_count_errors(Some(&invalid), None).is_err());
        }
        let maximum = HashMap::from([("C".to_string(), 1)]);
        assert!(validate_count_errors(Some(&carbon), Some(&maximum)).is_err());
    }
}
