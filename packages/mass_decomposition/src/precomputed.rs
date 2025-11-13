use crate::common::{ATOMIC_MASSES, Formula, NUM_ELEMENTS};
use std::sync::{Arc, OnceLock};

#[derive(Debug, Clone)]
pub struct PrecomputedDecomposer {
    pub element_mask: [bool; NUM_ELEMENTS],
    pub weights_data: Vec<(usize, f64, i64)>, // (original_index, mass, integer_mass)
    pub ert: Arc<Vec<Vec<i64>>>,              // Wrap in Arc
    pub precision: f64,
    pub min_error: f64,
    pub max_error: f64,
}

// Define common element sets here - easily customizable
const COMMON_ELEMENT_SETS: &[([bool; NUM_ELEMENTS], &str)] = &[
    (
        [
            true, true, true, true, true, false, false, true, false, false, false, false,
        ],
        "CHNOFS",
    ),
    (
        [
            true, true, true, true, true, false, true, true, false, false, false, false,
        ],
        "CHNOFPS",
    ),
    (
        [
            true, true, true, true, true, false, true, true, true, false, false, false,
        ],
        "CHNOFPSCl",
    ),
    (
        [
            true, true, true, true, true, false, true, true, false, false, true, false,
        ],
        "CHNOFPSBr",
    ),
    (
        [
            true, true, true, true, true, false, true, true, true, false, true, false,
        ],
        "CHNOFPSClBr",
    ),
    // Full set - all elements
    ([true; NUM_ELEMENTS], "ALL"), //NEVER REMOVE THIS LINE
];

static PRECOMPUTED_CACHE: OnceLock<Vec<PrecomputedDecomposer>> = OnceLock::new();

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

/// Finds the optimal blowup factor (1/precision) to minimize relative error.
/// Based on the algorithm from Böcker & Lipták's mass decomposition paper.
/// Generates candidates until they become too sparse to improve the ratio.
fn find_optimal_blowup(atomic_masses: &[f64]) -> f64 {
    let mut candidates = Vec::new();

    // Generate candidates for each mass until the ratio stops improving
    // We'll generate candidates up to a point where they're very sparse
    let max_k = 100000; // Just to prevent infinite loops in edge cases

    for &mass in atomic_masses {
        if mass <= 0.0 {
            continue;
        }

        for k in 1..=max_k {
            let b_val = k as f64 / mass;
            candidates.push(b_val);
        }
    }

    // Remove duplicates and sort
    candidates.sort_by(|a, b| a.partial_cmp(b).unwrap());
    candidates.dedup_by(|a, b| (*a - *b).abs() < 1e-10);

    // Evaluate each candidate to find the one minimizing Delta(b)/b
    let mut best_b = 1.0 / 100000.0; // fallback default
    let mut min_ratio = f64::INFINITY;

    for b in candidates {
        if b <= 0.0 {
            continue;
        }

        // Calculate max relative error (Delta) for this b across all elements
        let mut current_max_error = 0.0;

        for &mass in atomic_masses {
            if mass <= 0.0 {
                continue;
            }

            let val = b * mass;
            // Add small epsilon for float stability at integer boundaries
            let floor_val = (val + 1e-9).floor();

            let error = b - (floor_val / mass);
            if error > current_max_error {
                current_max_error = error;
            }
        }

        // Calculate ratio Delta(b) / b
        let ratio = current_max_error / b;

        if ratio < min_ratio {
            min_ratio = ratio;
            best_b = b;
        }
    }
    // print ("Optimal blowup factor: ", best_b);
    eprintln!("Optimal blowup factor: {}", best_b);
    best_b
}

fn build_precomputed(element_mask: &[bool; NUM_ELEMENTS]) -> PrecomputedDecomposer {
    // Build weights for this element set
    let mut weights: Vec<(usize, f64, i64)> = Vec::new();
    let mut vocabulary_masses = Vec::new();

    for i in 0..NUM_ELEMENTS {
        if element_mask[i] {
            weights.push((i, ATOMIC_MASSES[i], 0));
            vocabulary_masses.push(ATOMIC_MASSES[i]);
        }
    }

    if weights.is_empty() {
        panic!("Invalid precomputed configuration: element_mask contains no elements");
    }

    weights.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    eprint!("Building precomputed decomposer, calling the find_optimal_blowup function: ");
    // Calculate optimal blowup factor (1/precision) based on vocabulary
    let optimal_blowup = find_optimal_blowup(&vocabulary_masses);
    let mut precision = 1.0 / optimal_blowup;

    // Discretize masses
    for w in &mut weights {
        w.2 = (w.1 / precision) as i64;
    }

    // Divide by GCD to further optimize
    if weights.len() >= 2 {
        let mut d = gcd(weights[0].2, weights[1].2);
        for i in 2..weights.len() {
            d = gcd(d, weights[i].2);
            if d == 1 {
                break;
            }
        }
        if d > 1 {
            precision *= d as f64;
            for w in &mut weights {
                w.2 /= d;
            }
        }
    }

    // Calculate ERT
    let first_long_val = weights[0].2;
    let mut ert = vec![vec![0i64; weights.len()]; first_long_val as usize];

    if first_long_val > 0 {
        ert[0][0] = 0;
        for i in 1..first_long_val as usize {
            ert[i][0] = i64::MAX;
        }

        for j in 1..weights.len() {
            ert[0][j] = 0;
            let d = gcd(first_long_val, weights[j].2);
            for p in 0..d {
                let mut n = i64::MAX;
                for i in (p..first_long_val).step_by(d as usize) {
                    if ert[i as usize][j - 1] < n {
                        n = ert[i as usize][j - 1];
                    }
                }

                if n == i64::MAX {
                    for i in (p..first_long_val).step_by(d as usize) {
                        ert[i as usize][j] = i64::MAX;
                    }
                } else {
                    for _ in 0..first_long_val / d {
                        n += weights[j].2;
                        let r = (n % first_long_val) as usize;
                        if ert[r][j - 1] < n {
                            n = ert[r][j - 1];
                        }
                        ert[r][j] = n;
                    }
                }
            }
        }
    }

    // Compute errors
    let mut min_error = 0.0;
    let mut max_error = 0.0;
    for w in &weights {
        if w.1 == 0.0 {
            continue;
        }
        let error = (precision * w.2 as f64 - w.1) / w.1;
        if error < min_error {
            min_error = error;
        }
        if error > max_error {
            max_error = error;
        }
    }

    PrecomputedDecomposer {
        element_mask: *element_mask,
        weights_data: weights,
        ert: Arc::new(ert),
        precision,
        min_error,
        max_error,
    }
}

pub fn init_precomputed_cache() {
    PRECOMPUTED_CACHE.get_or_init(|| {
        COMMON_ELEMENT_SETS
            .iter()
            .map(|(mask, _name)| build_precomputed(mask))
            .collect()
    });
}

pub fn find_best_precomputed(max_bounds: &Formula) -> Option<&'static PrecomputedDecomposer> {
    let cache = PRECOMPUTED_CACHE.get_or_init(|| {
        COMMON_ELEMENT_SETS
            .iter()
            .map(|(mask, _name)| build_precomputed(mask))
            .collect()
    });

    // Create element mask for current problem
    let mut required_mask = [false; NUM_ELEMENTS];
    for i in 0..NUM_ELEMENTS {
        if max_bounds[i] > 0 {
            required_mask[i] = true;
        }
    }

    // Find exact match first
    for precomp in cache.iter() {
        if precomp.element_mask == required_mask {
            return Some(precomp);
        }
    }

    // Find smallest superset
    let mut best_match: Option<&PrecomputedDecomposer> = None;
    let mut best_extra_count = usize::MAX;

    for precomp in cache.iter() {
        let mut is_superset = true;
        let mut extra_count = 0;

        for i in 0..NUM_ELEMENTS {
            if required_mask[i] && !precomp.element_mask[i] {
                is_superset = false;
                break;
            }
            if !required_mask[i] && precomp.element_mask[i] {
                extra_count += 1;
            }
        }

        if is_superset && extra_count < best_extra_count {
            best_extra_count = extra_count;
            best_match = Some(precomp);
        }
    }

    best_match
}
