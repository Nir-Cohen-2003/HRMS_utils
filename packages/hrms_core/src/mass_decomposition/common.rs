use serde::Deserialize;
// Re-export NUM_ELEMENTS and ELEMENT_SYMBOLS so other modules can access them via this module.
pub use crate::common::{NUM_ELEMENTS, ELEMENT_SYMBOLS, ATOMIC_MASSES};




pub fn check_dbe(formula: &Formula, min_dbe: f64, max_dbe: f64, allow_half_integer: bool) -> bool {
    
    let c_count = formula[1];
    
    
    let n_count = formula[2];
    let p_count = formula[6];
    
    let h_count = formula[0];
    let na_count = formula[5];
    let k_count = formula[9];
    let f_count = formula[4];
    let cl_count = formula[8];
    let br_count = formula[10];
    let i_count = formula[11];
    
    let dbe = (
        (c_count) as f64) + 
        ((n_count + p_count) as f64 / 2.0) - 
        ((h_count+f_count + cl_count + br_count + i_count + na_count + k_count) as f64 / 2.0) 
        + 1.0;

    if dbe < min_dbe || dbe > max_dbe {
        return false;
    }

    if allow_half_integer {
        (dbe * 2.0).fract() == 0.0
    } else {
        dbe.fract() == 0.0
    }
}

pub const MIN_MASS_FOR_TOLERANCE: f64 = 200.0;

pub type Formula = [i32; NUM_ELEMENTS];

#[derive(Debug, Clone, Deserialize)]
pub struct DecompositionParams {
    pub tolerance_ppm: f64,
    pub min_dbe: f64,
    pub max_dbe: f64,
    pub allow_half_integer: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DecompositionKwargs {
    pub tolerance_ppm: f64,
    pub min_dbe: f64,
    pub max_dbe: f64,
    pub dbe_mode: String,
    pub min_bounds: Option<Formula>,
    pub max_bounds: Option<Formula>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SpectrumDecompositionParams {
    pub tolerance_ppm: f64,
    pub min_dbe: f64,
    pub max_dbe: f64,
    pub allow_half_integer: bool,
    pub water_absorption: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CleanAndNormalizeSpectrumKwargs {
    pub raw_fragment_tolerance_ppm: f64,
    pub normalized_fragment_tolerance_ppm: f64,
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

#[derive(Debug, Clone)]
pub struct CorrectedFragment {
    pub normalized_mass: f64,
    pub intensity: f64,
    pub formula: Formula,
    pub error_ppm: f64,
}

pub struct CleanedAndNormalizedSpectrumResult {
    pub fragments: Vec<CorrectedFragment>,
}

#[derive(Debug, Clone)]
pub struct DecomposedFragment {
    pub formula: Formula,
    pub mass: f64,
    pub error_ppm: f64,
}
