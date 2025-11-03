use crate::common::{Formula, DecompositionParams, NUM_ELEMENTS, ATOMIC_MASSES, check_dbe};

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

    fn decomposable(&self, i: usize, m: i64, a1: i64) -> bool {
        if m < 0 {
            return false;
        }
        if a1 <= 0 {
            return false;
        }
        self.ert[(m % a1) as usize][i] <= m
    }

    fn integer_decompose_recursive(&self, mass: i64, k: usize, current_formula: &mut Formula, results: &mut Vec<Formula>) {
        if k == 0 {
            if mass % self.weights[0].integer_mass == 0 {
                let count = (mass / self.weights[0].integer_mass) as i32;
                if count >= self.weights[0].min_count && count <= self.weights[0].max_count {
                    current_formula[self.weights[0].original_index] = count;
                    results.push(current_formula.clone());
                }
            }
            return;
        }

        let w_k = self.weights[k].integer_mass;
        let min_c = self.weights[k].min_count;
        let max_c = self.weights[k].max_count;

        for c in min_c..=max_c {
            let remaining_mass = mass - c as i64 * w_k;
            if remaining_mass >= 0 && self.decomposable(k - 1, remaining_mass, self.weights[0].integer_mass) {
                current_formula[self.weights[k].original_index] = c;
                self.integer_decompose_recursive(remaining_mass, k - 1, current_formula, results);
            }
        }
    }
    
    pub fn decompose(&mut self, target_mass: f64, params: &DecompositionParams) -> Vec<Formula> {
        if !self.is_initialized {
            self.precision = (params.tolerance_ppm * target_mass * 2.0) / 1_000_000.0;
            if self.precision == 0.0 {
                return Vec::new();
            }
            self.discretize_masses();
            self.divide_by_gcd();
            self.calc_ert();
            self.compute_errors();
            self.is_initialized = true;
        }

        let mass_from = target_mass - (params.tolerance_ppm * target_mass) / 1_000_000.0;
        let mass_to = target_mass + (params.tolerance_ppm * target_mass) / 1_000_000.0;
        
        let (start, end) = self.integer_bound(mass_from, mass_to);
        
        let mut all_results = Vec::new();
        for m in start..=end {
            let mut results = Vec::new();
            let mut current_formula = [0; NUM_ELEMENTS];
            if self.weights.len() > 0 {
                self.integer_decompose_recursive(m, self.weights.len() - 1, &mut current_formula, &mut results);
            }
            all_results.extend(results);
        }
        
        all_results.into_iter().filter(|f| check_dbe(f, params.min_dbe, params.max_dbe, &params.dbe_mode)).collect()
    }
}