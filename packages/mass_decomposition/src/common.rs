
use serde::Deserialize;

pub const NUM_ELEMENTS: usize = 15;

pub const ELEMENT_SYMBOLS: [&str; NUM_ELEMENTS] = [
    "H", "B", "C", "N", "O", "F", "Na", "Si", "P", "S", "Cl", "K", "As", "Br", "I"
];

pub const ATOMIC_MASSES: [f64; NUM_ELEMENTS] = [
    1.007825, 11.009305, 12.000000, 14.003074, 15.994915, 18.998403, 
    22.989770, 27.9769265, 30.973762, 31.972071, 34.96885271, 
    38.963707, 74.921596, 78.918338, 126.904468
];

pub const MIN_MASS_FOR_TOLERANCE: f64 = 200.0;

pub type Formula = [i32; NUM_ELEMENTS];

#[derive(Debug, Clone, Deserialize)]
pub struct DecompositionParams {
    pub tolerance_ppm: f64,
    pub min_dbe: f64,
    pub max_dbe: f64,
    pub dbe_mode: String,
    pub min_bounds: Formula,
    pub max_bounds: Formula,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DecompositionKwargs {
    pub tolerance_ppm: f64,
    pub min_dbe: f64,
    pub max_dbe: f64,
    pub dbe_mode: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SpectrumDecompositionParams {
    pub tolerance_ppm: f64,
    pub min_dbe: f64,
    pub max_dbe: f64,
    pub dbe_mode: String,
    pub water_absorption: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CleanAndNormalizeSpectrumKwargs {
    pub tolerance_ppm: f64,
    pub max_allowed_normalized_mass_error_ppm: f64,
    pub min_dbe: f64,
    pub max_dbe: f64,
    pub dbe_mode: String,
    pub water_absorption: bool,
}

pub fn formula_to_string(formula: &Formula) -> String {
    let mut s = String::new();
    for (i, &count) in formula.iter().enumerate() {
        if count > 0 {
            s.push_str(ELEMENT_SYMBOLS[i]);
            if count > 1 {
                s.push_str(&count.to_string());
            }
        }
    }
    s
}

pub fn check_dbe(formula: &Formula, min_dbe: f64, max_dbe: f64, dbe_mode: &str) -> bool {
    let h_count = formula[0];
    let c_count = formula[2];
    let n_count = formula[3];
    let f_count = formula[5];
    let cl_count = formula[10];
    let br_count = formula[13];
    let i_count = formula[14];
    let p_count = formula[8];

    let dbe = (c_count as f64) - (h_count as f64 / 2.0) + (n_count as f64 / 2.0) + (p_count as f64 / 2.0) - 
              ((f_count + cl_count + br_count + i_count) as f64 / 2.0) + 1.0;

    if dbe < min_dbe || dbe > max_dbe {
        return false;
    }

    match dbe_mode {
        "integer" => dbe.fract() == 0.0,
        "half_integer" => (dbe * 2.0).fract() == 0.0,
        "any" => true,
        _ => false,
    }
}

#[derive(Debug, Clone)]
pub struct CorrectedFragment {
    pub normalized_mass: f64,
    pub intensity: f64,
    pub formula: Formula,
    pub error_ppm: f64,
}

#[derive(Debug, Clone)]
pub struct CleanedAndNormalizedSpectrumResult {
    pub fragments: Vec<CorrectedFragment>,
}
