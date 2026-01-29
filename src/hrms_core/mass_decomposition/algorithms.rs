use super::common::{
    check_dbe, CleanedAndNormalizedSpectrumResult, CorrectedFragment, DecompositionParams, Formula,
    IsotopicPatternParams, SpectrumDecompositionParams, ATOMIC_MASSES, ELEMENT_ISOTOPES,
    MIN_MASS_FOR_TOLERANCE, NUM_ELEMENTS,
};
use super::precomputed::{find_best_precomputed, PrecomputedDecomposer};
use std::sync::Arc;

const MASS_ACCURACY_PPM_TO_DA_THRESHOLD: f64 = 200.0;

pub fn deduce_isotopic_pattern_inner(
    precursor_mz: f64,
    ms1_mzs: &[f64],
    ms1_intensities: &[f64],
    params: &IsotopicPatternParams,
) -> [f64; 8] {
    let ms1_absolute_tolerance =
        precursor_mz.max(MASS_ACCURACY_PPM_TO_DA_THRESHOLD) * params.ms1_mass_tolerance_ppm * 1e-6;
    let isotopic_absolute_tolerance = precursor_mz.max(MASS_ACCURACY_PPM_TO_DA_THRESHOLD)
        * params.isotopic_mass_tolerance_ppm
        * 1e-6;

    // Find precursor
    let precursor_indices: Vec<usize> = ms1_mzs
        .iter()
        .enumerate()
        .filter(|&(_, &mz)| (mz - precursor_mz).abs() <= ms1_absolute_tolerance)
        .map(|(i, _)| i)
        .collect();

    if precursor_indices.is_empty() {
        return [-1.0; 8];
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
            lower = ((peak_total_intensity * (1.0 - params.intensity_relative_tolerance)
                - params.intensity_absolute_tolerance)
                * iso_props.prob_0)
                / (iso_props.prob_1 * precursor_ms1_intensity);
            upper = ((peak_total_intensity * (1.0 + params.intensity_relative_tolerance)
                + params.intensity_absolute_tolerance)
                * iso_props.prob_0)
                / (iso_props.prob_1 * precursor_ms1_intensity);
        }

        result[res_idx] = lower;
        result[res_idx + 4] = upper;
    }

    // Special logic for Cl second peak (M+4)
    // Cl_lower index is 2, Cl_upper is 6
    let cl_lower = result[2];
    let cl_upper = result[6];

    if cl_lower > 0.0 && cl_upper >= 2.0 {
        let cl_props = ELEMENT_ISOTOPES[8].unwrap();
        let second_cl_peak_mz = precursor_ms1_mz + cl_props.mass_diff * 2.0;

        let peak_indices: Vec<usize> = ms1_mzs
            .iter()
            .enumerate()
            .filter(|&(_, &mz)| (mz - second_cl_peak_mz).abs() <= isotopic_absolute_tolerance)
            .map(|(i, _)| i)
            .collect();

        let second_cl_peak_total_intensity = if peak_indices.is_empty() {
            0.0
        } else {
            peak_indices
                .iter()
                .map(|&i| ms1_intensities[i])
                .fold(0.0, f64::max)
        };

        // We need the intensity of the first Cl peak again
        // Ideally we should have stored it, but we can recompute or just grab it.
        // Wait, we need it for the ratio.
        // Let's just recompute the first peak intensity to be safe and local.
        let first_cl_peak_mz = precursor_ms1_mz + cl_props.mass_diff;
        let first_peak_indices: Vec<usize> = ms1_mzs
            .iter()
            .enumerate()
            .filter(|&(_, &mz)| (mz - first_cl_peak_mz).abs() <= isotopic_absolute_tolerance)
            .map(|(i, _)| i)
            .collect();
        let first_cl_peak_total_intensity = if first_peak_indices.is_empty() {
            0.0
        } else {
            first_peak_indices
                .iter()
                .map(|&i| ms1_intensities[i])
                .fold(0.0, f64::max)
        };

        let expected_cl_ratio = cl_props.prob_1 / (2.0 * cl_props.prob_0);
        let actual_cl_ratio = second_cl_peak_total_intensity / first_cl_peak_total_intensity;

        if expected_cl_ratio * first_cl_peak_total_intensity
            > params.minimum_intensity * (1.0 + params.intensity_relative_tolerance)
        {
            let cl_deviation = ((actual_cl_ratio - expected_cl_ratio) / expected_cl_ratio).abs();
            if cl_deviation > params.intensity_relative_tolerance {
                result[2] = 0.0; // Cl_lower
                result[6] = 1.0; // Cl_upper
            }
        }
    }

    result
}

#[derive(Debug, Clone)]
struct Weight {
    original_index: usize,
    integer_mass: i64,
    min_count: i32,
    max_count: i32,
}

#[derive(Clone)]
pub struct MassDecomposer {
    weights: Vec<Weight>,
    ert: Arc<Vec<Vec<i64>>>, // Change to Arc
    precision: f64,
    min_error: f64,
    max_error: f64,
    is_initialized: bool,
    min_bounds: Formula,
    max_bounds: Formula,
    integer_weight_masses: Vec<i64>,
}

impl MassDecomposer {
    pub fn new(min_bounds: Formula, max_bounds: Formula) -> Self {
        let mut decomposer = MassDecomposer {
            weights: Vec::new(),
            ert: Arc::new(Vec::new()),
            precision: 1.0 / 80000.0,
            min_error: 0.0,
            max_error: 0.0,
            is_initialized: false,
            min_bounds,
            max_bounds,
            integer_weight_masses: Vec::new(),
        };

        // Always use precomputed data (guaranteed to find at least "ALL" preset)
        let precomp = find_best_precomputed(&max_bounds)
            .expect("Precomputed cache must contain at least the 'ALL' preset");
        decomposer.load_from_precomputed(precomp);

        decomposer
    }

    fn load_from_precomputed(&mut self, precomp: &PrecomputedDecomposer) {
        self.precision = precomp.precision;
        self.min_error = precomp.min_error;
        self.max_error = precomp.max_error;
        self.ert = Arc::clone(&precomp.ert); // Just clone the Arc, not the data

        // Build weights from precomputed data with actual bounds
        // IMPORTANT: We must include ALL elements from the precomputed decomposer,
        // even if max_bounds is 0, to maintain consistency with the ERT table
        self.weights.clear();
        for &(original_index, _mass, integer_mass) in &precomp.weights_data {
            // Always add the weight if it's in the precomputed decomposer
            // The bounds checking will filter out invalid formulas later
            self.weights.push(Weight {
                original_index,
                integer_mass,
                min_count: self.min_bounds[original_index],
                max_count: self.max_bounds[original_index],
            });
        }

        self.integer_weight_masses = self.weights.iter().map(|w| w.integer_mass).collect();
        self.is_initialized = true;
    }

    fn integer_bound(&self, mass_from: f64, mass_to: f64) -> (i64, i64) {
        let from_d = ((1.0 + self.min_error) * mass_from / self.precision).ceil();
        let to_d = ((1.0 + self.max_error) * mass_to / self.precision).floor();

        let start = from_d.max(0.0) as i64;
        let end = (start as f64).max(to_d) as i64;
        (start, end)
    }

    fn decomposable_fast(&self, i: usize, m: i64, remainder: i64) -> bool {
        if m < 0 {
            return false;
        }
        // Access through Arc - no performance penalty
        self.ert[remainder as usize][i] <= m
    }

    fn integer_decompose(
        &self,
        mass: i64,
        results: &mut Vec<(Formula, f64)>,
        target_mass: f64,
        tolerance_da: f64,
        params: &DecompositionParams,
    ) {
        let k = self.weights.len();
        if k == 0 {
            return;
        }

        let weight_masses = &self.integer_weight_masses;
        let a = weight_masses[0];
        if a <= 0 {
            return;
        }

        let mut temp_counts = vec![0; k];

        let mut i = (k - 1) as isize;
        let mut m = mass;

        // ...
        loop {
            if i >= k as isize {
                return;
            }
            let remainder = m % a;
            if !self.decomposable_fast(i as usize, m, remainder) {
                loop {
                    m += temp_counts[i as usize] as i64 * weight_masses[i as usize];
                    temp_counts[i as usize] = 0;
                    i += 1;

                    if i >= k as isize {
                        return; // Exit completely if we've exhausted all elements
                    }

                    if self.decomposable_fast(i as usize, m, m % a) {
                        break;
                    }
                }

                // Only execute this if we didn't exit via return above
                m -= weight_masses[i as usize];
                temp_counts[i as usize] += 1;
            } else {
                while i > 0 {
                    if !self.decomposable_fast((i - 1) as usize, m, remainder) {
                        break;
                    }
                    i -= 1;
                }

                if i == 0 {
                    if a > 0 {
                        temp_counts[0] = (m / a) as i32;
                    } else {
                        temp_counts[0] = 0;
                    }

                    // Single validation pass: check bounds and build formula
                    let mut res = [0; NUM_ELEMENTS];
                    let mut valid_formula = true;

                    for j in 0..k {
                        let count = temp_counts[j];
                        // Check weight bounds
                        if count < self.weights[j].min_count || count > self.weights[j].max_count {
                            valid_formula = false;
                            break;
                        }
                        res[self.weights[j].original_index] = count;
                    }

                    if valid_formula {
                        // Check DBE
                        if !check_dbe(
                            &res,
                            params.min_dbe,
                            params.max_dbe,
                            params.allow_half_integer,
                        ) {
                            valid_formula = false;
                        }
                    }

                    if valid_formula {
                        // Calculate mass and check tolerance
                        let formula_mass: f64 = res
                            .iter()
                            .enumerate()
                            .map(|(idx, &count)| ATOMIC_MASSES[idx] * count as f64)
                            .sum();
                        let error = formula_mass - target_mass;

                        if error.abs() <= tolerance_da {
                            let error_ppm = (error / formula_mass) * 1e6;
                            results.push((res, error_ppm));
                        }
                    }

                    i += 1;
                }

                while i < k as isize
                    && temp_counts[i as usize] >= self.weights[i as usize].max_count
                {
                    m += temp_counts[i as usize] as i64 * weight_masses[i as usize];
                    temp_counts[i as usize] = 0;
                    i += 1;
                }

                if i < k as isize {
                    m -= weight_masses[i as usize];
                    temp_counts[i as usize] += 1;
                } else {
                    return;
                }
            }
        }
    }

    pub fn decompose(
        &self,
        target_mass: f64,
        params: &DecompositionParams,
    ) -> (Vec<Formula>, Vec<f64>) {
        // Calculate tolerance in Daltons using the 200 Da rule
        let tolerance_da = params.tolerance_ppm * 1e-6 * target_mass.max(MIN_MASS_FOR_TOLERANCE);

        let mass_from = target_mass - tolerance_da;
        let mass_to = target_mass + tolerance_da;

        let (start, end) = self.integer_bound(mass_from, mass_to);

        let mut results = Vec::new();

        for m in start..=end {
            self.integer_decompose(m, &mut results, target_mass, tolerance_da, params);
        }

        let (formulas, errors_ppm) = results.into_iter().unzip();

        (formulas, errors_ppm)
    }
}

struct FitPoint {
    mass: f64,
    error: f64,
    weight: f64,
}

fn weighted_linear_regression(points: &[FitPoint]) -> (f64, f64) {
    let mut sw = 0.0;
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut sxx = 0.0;
    let mut sxy = 0.0;

    for p in points {
        sw += p.weight;
        sx += p.weight * p.mass;
        sy += p.weight * p.error;
        sxx += p.weight * p.mass * p.mass;
        sxy += p.weight * p.mass * p.error;
    }

    let denom = sw * sxx - sx * sx;
    if denom.abs() < 1e-12 {
        return (0.0, 0.0); // Not enough data or collinear
    }

    let b = (sw * sxy - sx * sy) / denom;
    let a = (sy - b * sx) / sw;
    (a, b)
}

pub struct SpectrumDecomposer {
    decomposer: Arc<MassDecomposer>,
}

impl SpectrumDecomposer {
    pub fn new(min_bounds: Formula, max_bounds: Formula) -> Self {
        SpectrumDecomposer {
            decomposer: Arc::new(MassDecomposer::new(min_bounds, max_bounds)),
        }
    }

    pub fn decompose_spectrum(
        &self,
        mz_values: &[f64],
        params: &SpectrumDecompositionParams,
    ) -> Vec<(Vec<Formula>, Vec<f64>)> {
        let params_arc = Arc::new(DecompositionParams {
            tolerance_ppm: params.tolerance_ppm,
            min_dbe: params.min_dbe,
            max_dbe: params.max_dbe,
            allow_half_integer: params.allow_half_integer,
        });

        // Parallel decomposition - all fragments share the same decomposer
        mz_values
            .iter()
            .map(|&mass| self.decomposer.decompose(mass, &params_arc))
            .collect()
    }

    pub fn clean_and_normalize_spectrum(
        &self,
        fragment_masses: &[f64],
        fragment_intensities: &[f64],
        params: &SpectrumDecompositionParams,
        max_allowed_normalized_mass_error_ppm: f64,
    ) -> CleanedAndNormalizedSpectrumResult {
        let fragment_solutions = self.decompose_spectrum(fragment_masses, params);

        // Try to fit a calibration line if we have enough fragments
        let use_normalization = fragment_solutions
            .iter()
            .filter(|(f, _)| !f.is_empty())
            .count()
            >= 3;

        let (intercept, slope) = if use_normalization {
            // Fit line through best formula for each fragment
            let mut fit_points = Vec::new();

            for (i, (formulas, _)) in fragment_solutions.iter().enumerate() {
                if formulas.is_empty() {
                    continue;
                }

                let measured_mass = fragment_masses[i];

                // Find formula with smallest absolute error
                let best_error = formulas
                    .iter()
                    .map(|formula| {
                        let calc_mass: f64 = formula
                            .iter()
                            .enumerate()
                            .map(|(j, &count)| ATOMIC_MASSES[j] * count as f64)
                            .sum();
                        calc_mass - measured_mass
                    })
                    .min_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap())
                    .unwrap();

                fit_points.push(FitPoint {
                    mass: measured_mass,
                    error: best_error,
                    weight: fragment_intensities[i],
                });
            }

            weighted_linear_regression(&fit_points)
        } else {
            (0.0, 0.0) // No correction
        };

        // Select best formula for each fragment
        let mut corrected_fragments = Vec::new();

        for (i, (formulas, _)) in fragment_solutions.iter().enumerate() {
            if formulas.is_empty() {
                continue;
            }

            let measured_mass = fragment_masses[i];
            let correction = intercept + slope * measured_mass;
            let normalized_mass = measured_mass + correction;

            // Calculate tolerance in Daltons
            let tolerance_da = max_allowed_normalized_mass_error_ppm
                * 1e-6
                * normalized_mass.max(MIN_MASS_FOR_TOLERANCE);

            // Find formula closest to normalized mass within tolerance
            let mut best_formula: Option<Formula> = None;
            let mut best_error_da = f64::INFINITY;
            let mut best_calc_mass = 0.0;

            for formula in formulas.iter() {
                let calc_mass: f64 = formula
                    .iter()
                    .enumerate()
                    .map(|(j, &count)| ATOMIC_MASSES[j] * count as f64)
                    .sum();

                let error_da = (calc_mass - normalized_mass).abs();

                if error_da <= tolerance_da && error_da < best_error_da {
                    best_error_da = error_da;
                    best_formula = Some(*formula);
                    best_calc_mass = calc_mass;
                }
            }

            if let Some(formula) = best_formula {
                let error_ppm = ((best_calc_mass - normalized_mass) / best_calc_mass) * 1e6;

                corrected_fragments.push(CorrectedFragment {
                    normalized_mass,
                    intensity: fragment_intensities[i],
                    formula,
                    error_ppm,
                });
            }
        }

        CleanedAndNormalizedSpectrumResult {
            fragments: corrected_fragments,
        }
    }
}
