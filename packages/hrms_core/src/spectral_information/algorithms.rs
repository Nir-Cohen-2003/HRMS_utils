use crate::common::NUM_ELEMENTS;

pub fn l1_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

pub fn l2_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

pub fn cosine_distance(a: &[f64], b: &[f64]) -> f64 {
    let dot_product = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f64>();
    let norm_a = a.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
    let norm_b = b.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        1.0
    } else {
        1.0 - (dot_product / (norm_a * norm_b))
    }
}

pub fn is_superformula(super_formula: &[f64], sub_formula: &[f64]) -> bool {
    const TOLERANCE: f64 = 1e-12;
    let is_super = super_formula
        .iter()
        .zip(sub_formula.iter())
        .all(|(sup, sub)| *sup >= *sub - TOLERANCE);
    let any_greater = super_formula
        .iter()
        .zip(sub_formula.iter())
        .any(|(sup, sub)| *sup > *sub + TOLERANCE);
    is_super && any_greater
}

pub fn calculate_score_for_spectrum(
    precursor_vec: Vec<f64>,
    fragments_flat: Vec<f64>,
    distance_metric: &str,
    ignore_hydrogens: bool,
) -> Option<f64> {
    if precursor_vec.is_empty() {
        return Some(0.0);
    }

    let active_mask: Vec<bool> = precursor_vec.iter().map(|&x| x > 0.0).collect();
    let n_active = active_mask.iter().filter(|&&b| b).count();

    if n_active == 0 {
        return Some(0.0);
    }

    let total_precursor_mass: f64 = precursor_vec
        .iter()
        .zip(active_mask.iter())
        .filter_map(|(mass, &is_active)| if is_active { Some(mass) } else { None })
        .sum();

    if total_precursor_mass <= 0.0 {
        return Some(0.0);
    }

    let mut all_formulas: Vec<Vec<f64>> = Vec::new();
    all_formulas.push(precursor_vec.clone());

    for fragment in fragments_flat.chunks(NUM_ELEMENTS) {
        if fragment.len() == NUM_ELEMENTS {
            all_formulas.push(fragment.to_vec());
        }
    }

    let mut norm_formulas: Vec<Vec<f64>> = Vec::with_capacity(all_formulas.len());
    for formula_vec in all_formulas.iter() {
        let f_active: Vec<f64> = formula_vec
            .iter()
            .zip(active_mask.iter())
            .filter_map(|(mass, &is_active)| if is_active { Some(*mass) } else { None })
            .collect();
        norm_formulas.push(f_active.iter().map(|&x| x / total_precursor_mass).collect());
    }

    let dist_fn = match distance_metric {
        "l1" => l1_distance,
        "l2" => l2_distance,
        "cosine" => cosine_distance,
        _ => panic!("Invalid distance metric"),
    };

    let mut unique_mask: Vec<bool> = vec![true; norm_formulas.len()];
    if ignore_hydrogens && n_active > 1 {
        for j in 0..norm_formulas.len() {
            if !unique_mask[j] {
                continue;
            }
            for l in (j + 1)..norm_formulas.len() {
                if unique_mask[l] {
                    let formula_j = &norm_formulas[j][1..];
                    let formula_l = &norm_formulas[l][1..];
                    
                    let is_same = formula_j.iter().zip(formula_l.iter()).all(|(a, b)| (a - b).abs() < 1e-12);

                    if is_same {
                        unique_mask[l] = false;
                    }
                }
            }
        }
    }

    let mut total_score = 0.0;

    for j in 0..norm_formulas.len() {
        if !unique_mask[j] {
            continue;
        }
        let node_a_norm = &norm_formulas[j];

        let node_a_compare = if ignore_hydrogens && n_active > 1 {
            &node_a_norm[1..]
        } else {
            &node_a_norm[..]
        };

        let mut min_dist = f64::INFINITY;

        for l in 0..norm_formulas.len() {
            if j == l || !unique_mask[l] {
                continue;
            }
            let node_b_norm = &norm_formulas[l];

            let node_b_compare = if ignore_hydrogens && n_active > 1 {
                &node_b_norm[1..]
            } else {
                &node_b_norm[..]
            };

            if is_superformula(node_b_compare, node_a_compare) {
                let dist = dist_fn(node_a_compare, node_b_compare);
                if dist < min_dist {
                    min_dist = dist;
                }
            }
        }

        if min_dist.is_finite() {
            let distance_cap = match distance_metric {
                "l1" => 2.0,
                "l2" => 2.0f64.sqrt(),
                "cosine" => 2.0,
                _ => 1.0,
            };

            let mut scaled_dist = min_dist / distance_cap;
            if scaled_dist <= 1e-12 {
                continue;
            }
            if scaled_dist >= 1.0 {
                scaled_dist = 1.0 - 1e-12;
            }

            let m: f64 = node_a_norm.iter().sum();
            if m > 0.0 {
                total_score += -scaled_dist * scaled_dist.ln() * m;
            }
        }
    }
    Some(total_score)
}
