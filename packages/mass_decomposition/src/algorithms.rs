
use crate::common::{Formula, DecompositionParams, NUM_ELEMENTS, ATOMIC_MASSES, check_dbe, MIN_MASS_FOR_TOLERANCE, SpectrumDecompositionParams, CleanedAndNormalizedSpectrumResult, CorrectedFragment};

#[derive(Debug, Clone)]
struct Weight {
    original_index: usize,
    mass: f64,
    integer_mass: i64,
    min_count: i32,
    max_count: i32,
}

pub struct MassDecomposer {
    weights: Vec<Weight>,
    ert: Vec<Vec<i64>>,
    precision: f64,
    min_error: f64,
    max_error: f64,
    is_initialized: bool,
    min_bounds: Formula,
    max_bounds: Formula,
    temp_counts: Vec<i32>,
}

impl MassDecomposer {
    pub fn new(min_bounds: Formula, max_bounds: Formula) -> Self {
        let mut decomposer = MassDecomposer {
            weights: Vec::new(),
            ert: Vec::new(),
            precision: 0.0,
            min_error: 0.0,
            max_error: 0.0,
            is_initialized: false,
            min_bounds,
            max_bounds,
            temp_counts: Vec::new(),
        };
        decomposer.init_money_changing();
        decomposer
    }

    fn init_money_changing(&mut self) {
        self.weights.clear();
        for i in 0..NUM_ELEMENTS {
            if self.max_bounds[i] > 0 {
                self.weights.push(Weight {
                    original_index: i,
                    mass: ATOMIC_MASSES[i],
                    integer_mass: 0,
                    min_count: self.min_bounds[i],
                    max_count: self.max_bounds[i],
                });
            }
        }
        self.weights.sort_by(|a, b| a.mass.partial_cmp(&b.mass).unwrap());
    }

    fn gcd(u: i64, v: i64) -> i64 {
        let mut u = u;
        let mut v = v;
        while v != 0 {
            let r = u % v;
            u = v;
            v = r;
        }
        u
    }

    fn discretize_masses(&mut self) {
        for weight in &mut self.weights {
            weight.integer_mass = (weight.mass / self.precision) as i64;
        }
    }

    fn divide_by_gcd(&mut self) {
        if self.weights.len() < 2 {
            return;
        }
        let mut d = self.weights[0].integer_mass;
        if self.weights.len() > 1 {
            d = Self::gcd(self.weights[0].integer_mass, self.weights[1].integer_mass);
            for i in 2..self.weights.len() {
                d = Self::gcd(d, self.weights[i].integer_mass);
                if d == 1 {
                    break;
                }
            }
        }
        if d > 1 {
            self.precision *= d as f64;
            for weight in &mut self.weights {
                weight.integer_mass /= d;
            }
        }
    }

    fn calc_ert(&mut self) {
        if self.weights.is_empty() {
            return;
        }
        let first_long_val = self.weights[0].integer_mass;
        if first_long_val <= 0 {
            // This should not happen in mass spectrometry data
            return;
        }

        self.ert = vec![vec![0; self.weights.len()]; first_long_val as usize];
        self.ert[0][0] = 0;
        for i in 1..first_long_val as usize {
            self.ert[i][0] = i64::MAX;
        }

        for j in 1..self.weights.len() {
            self.ert[0][j] = 0;
            let d = Self::gcd(first_long_val, self.weights[j].integer_mass);
            for p in 0..d {
                let mut n = i64::MAX;
                for i in (p..first_long_val).step_by(d as usize) {
                    if self.ert[i as usize][j - 1] < n {
                        n = self.ert[i as usize][j - 1];
                    }
                }

                if n == i64::MAX {
                    for i in (p..first_long_val).step_by(d as usize) {
                        self.ert[i as usize][j] = i64::MAX;
                    }
                } else {
                    for _ in 0..first_long_val / d {
                        n += self.weights[j].integer_mass;
                        let r = (n % first_long_val) as usize;
                        if self.ert[r][j - 1] < n {
                            n = self.ert[r][j - 1];
                        }
                        self.ert[r][j] = n;
                    }
                }
            }
        }
    }
    
    fn compute_errors(&mut self) {
        self.min_error = 0.0;
        self.max_error = 0.0;
        for weight in &self.weights {
            if weight.mass == 0.0 {
                continue;
            }
            let error = (self.precision * weight.integer_mass as f64 - weight.mass) / weight.mass;
            if error < self.min_error {
                self.min_error = error;
            }
            if error > self.max_error {
                self.max_error = error;
            }
        }
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
        self.ert[remainder as usize][i] <= m
    }

    fn integer_decompose(&mut self, mass: i64) -> Vec<Formula> {
        let mut results = Vec::with_capacity(100);
        let k = self.weights.len();
        if k == 0 {
            return results;
        }

        let weight_masses: Vec<i64> = self.weights.iter().map(|w| w.integer_mass).collect();
        let a = weight_masses[0];
        if a <= 0 {
            return results;
        }

        if self.temp_counts.len() < k {
            self.temp_counts.resize(k, 0);
        }
        self.temp_counts.fill(0);

        let mut i = (k - 1) as isize;
        let mut m = mass;

        loop {
            let remainder = m % a;
            if !self.decomposable_fast(i as usize, m, remainder) {
                loop {
                    if i >= (k - 1) as isize {
                        return results;
                    }
                    if self.decomposable_fast(i as usize, m, m % a) {
                        break;
                    }
                    m += self.temp_counts[i as usize] as i64 * weight_masses[i as usize];
                    self.temp_counts[i as usize] = 0;
                    i += 1;
                }

                if i < k as isize {
                    m -= weight_masses[i as usize];
                    self.temp_counts[i as usize] += 1;
                }
            } else {
                while i > 0 {
                    if !self.decomposable_fast((i - 1) as usize, m, remainder) {
                        break;
                    }
                    i -= 1;
                }

                if i == 0 {
                    if a > 0 {
                        self.temp_counts[0] = (m / a) as i32;
                    } else {
                        self.temp_counts[0] = 0;
                    }

                    let mut valid_formula = true;
                    for j in 0..k {
                        if self.temp_counts[j] < self.weights[j].min_count || self.temp_counts[j] > self.weights[j].max_count {
                            valid_formula = false;
                            break;
                        }
                    }

                    if valid_formula {
                        let mut res = [0; NUM_ELEMENTS];
                        for j in 0..k {
                            res[self.weights[j].original_index] = self.temp_counts[j];
                        }
                        results.push(res);
                    }
                    i += 1;
                }

                while i < k as isize && self.temp_counts[i as usize] >= self.weights[i as usize].max_count {
                    m += self.temp_counts[i as usize] as i64 * weight_masses[i as usize];
                    self.temp_counts[i as usize] = 0;
                    i += 1;
                }

                if i < k as isize {
                    m -= weight_masses[i as usize];
                    self.temp_counts[i as usize] += 1;
                } else {
                    return results;
                }
            }
        }
    }

    pub fn decompose(&mut self, target_mass: f64, params: &DecompositionParams) -> Vec<Formula> {
        if !self.is_initialized {
            // self.precision = (params.tolerance_ppm * 50.0f64 * 2.0f64) / 1_000_000.0f64;
            self.precision = 1.0 / 5963.337687;
            if self.precision == 0.0f64 {
                return Vec::new();
            }
            self.discretize_masses();
            self.divide_by_gcd();
            self.calc_ert();
            self.compute_errors();
            self.is_initialized = true;
        }

        let mass_from = target_mass - (params.tolerance_ppm * target_mass) / 1_000_000.0f64;
        let mass_to = target_mass + (params.tolerance_ppm * target_mass) / 1_000_000.0f64;
        
        let (start, end) = self.integer_bound(mass_from, mass_to);
        
        let mut all_results = Vec::new();
        for m in start..=end {
            let mut results = self.integer_decompose(m);
            all_results.append(&mut results);
        }
        
        let tolerance_da = params.tolerance_ppm * 1e-6f64 * target_mass.max(MIN_MASS_FOR_TOLERANCE);

        all_results.into_iter()
            .filter(|f| {
                let formula_mass: f64 = f.iter().enumerate().map(|(i, &count)| ATOMIC_MASSES[i] * count as f64).sum();
                (formula_mass - target_mass).abs() <= tolerance_da
            })
            .filter(|f| check_dbe(f, params.min_dbe, params.max_dbe, &params.dbe_mode))
                .collect()
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


pub struct SpectrumDecomposer {}

impl SpectrumDecomposer {
    pub fn new() -> Self {
        SpectrumDecomposer {}
    }

    pub fn decompose_spectrum_with_precursor(
        &mut self,
        mz_values: &[f64],
        precursor_formula: &Formula,
        params: &SpectrumDecompositionParams,
    ) -> Vec<Vec<Formula>> {
        let mut max_bounds = precursor_formula.clone();
        if params.water_absorption {
            max_bounds[0] += 2; // H
            max_bounds[4] += 1; // O
        }
        let max_bounds = max_bounds; // now immutable for the rest of the scope
        let min_bounds = [0; NUM_ELEMENTS];

        let mut all_results = Vec::with_capacity(mz_values.len());

        for &mass in mz_values {
            let mut decomposer = MassDecomposer::new(min_bounds, max_bounds);
            let formulas = decomposer.decompose(mass, &DecompositionParams {
                tolerance_ppm: params.tolerance_ppm,
                min_dbe: params.min_dbe,
                max_dbe: params.max_dbe,
                dbe_mode: params.dbe_mode.clone(),
                min_bounds,
                max_bounds,
            });
            all_results.push(formulas);
        }

        all_results
    }

    pub fn clean_and_normalize_spectrum_iterative(
        &mut self,
        fragment_masses: &[f64],
        fragment_intensities: &[f64],
        precursor_formula: &Formula,
        params: &SpectrumDecompositionParams,
        max_allowed_normalized_mass_error_ppm: f64,
        max_iterations: usize,
        convergence_tolerance: f64,
    ) -> CleanedAndNormalizedSpectrumResult {
        
        let fragment_solutions = self.decompose_spectrum_with_precursor(
            fragment_masses, 
            precursor_formula, 
            params
        );

        let mut linear_fit = (0.0, 0.0); // (a, b)

        for _ in 0..max_iterations {
            let mut fit_points = Vec::new();

            for (i, formulas) in fragment_solutions.iter().enumerate() {
                if formulas.is_empty() {
                    continue;
                }

                let measured_mass = fragment_masses[i];
                let predicted_error = linear_fit.0 + linear_fit.1 * measured_mass;

                let mut formula_errors = Vec::new();
                let mut formula_weights = Vec::new();

                for formula in formulas {
                    let calc_mass: f64 = formula.iter().enumerate().map(|(i, &c)| ATOMIC_MASSES[i] * c as f64).sum();
                    let error = calc_mass - measured_mass;
                    formula_errors.push(error);

                    let deviation = (error - predicted_error).abs();
                    formula_weights.push(1.0 / (deviation + 1e-9));
                }

                let total_weight: f64 = formula_weights.iter().sum();
                if total_weight == 0.0 {
                    continue;
                }

                let weighted_average_error: f64 = formula_errors.iter().zip(formula_weights.iter())
                    .map(|(&err, &w)| err * (w / total_weight))
                    .sum();

                fit_points.push(FitPoint {
                    mass: measured_mass,
                    error: weighted_average_error,
                    weight: fragment_intensities[i],
                });
            }

            let new_linear_fit = weighted_linear_regression(&fit_points);

            if (new_linear_fit.0 - linear_fit.0).abs() < convergence_tolerance &&
               (new_linear_fit.1 - linear_fit.1).abs() < convergence_tolerance {
                linear_fit = new_linear_fit;
                break;
            }
            linear_fit = new_linear_fit;
        }

        let mut corrected_fragments = Vec::new();
        for (i, formulas) in fragment_solutions.iter().enumerate() {
            if formulas.is_empty() {
                continue;
            }

            let measured_mass = fragment_masses[i];
            let predicted_error = linear_fit.0 + linear_fit.1 * measured_mass;

            let mut best_formula: Option<Formula> = None;
            let mut min_deviation = f64::INFINITY;

            for formula in formulas {
                let calc_mass: f64 = formula.iter().enumerate().map(|(i, &c)| ATOMIC_MASSES[i] * c as f64).sum();
                let error = calc_mass - measured_mass;
                let deviation = (error - predicted_error).abs();

                if deviation < min_deviation {
                    min_deviation = deviation;
                    best_formula = Some(formula.clone());
                }
            }

            if let Some(formula) = best_formula {
                let correction = linear_fit.0 + linear_fit.1 * measured_mass;
                let normalized_mass = measured_mass + correction;
                let calc_mass: f64 = formula.iter().enumerate().map(|(i, &c)| ATOMIC_MASSES[i] * c as f64).sum();
                let final_error = calc_mass - normalized_mass;
                let final_error_ppm = (final_error / normalized_mass) * 1e6;

                if final_error_ppm.abs() <= max_allowed_normalized_mass_error_ppm {
                    corrected_fragments.push(CorrectedFragment {
                        normalized_mass,
                        intensity: fragment_intensities[i],
                        formula,
                        error_ppm: final_error_ppm,
                    });
                }
            }
        }

        CleanedAndNormalizedSpectrumResult { fragments: corrected_fragments }
    }
}