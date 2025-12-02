pub const NUM_ELEMENTS: usize = 12;

pub const ELEMENT_SYMBOLS: [&str; NUM_ELEMENTS] = [
    "H","C", "N", "O", "F", "Na", "P","S", "Cl", "K", "Br", "I"
];
pub const ATOMIC_MASSES: [f64; NUM_ELEMENTS] = [
    1.007825,12.000000, 14.003074, 15.994915, 18.998403, 
    22.989770, 30.973762, 31.972071, 34.96885271, 
    38.963707,78.918338, 126.904468
];

#[derive(Debug, Clone, Copy)]
pub struct IsotopeProps {
    pub mass_diff: f64,
    pub prob_0: f64,
    pub prob_1: f64,
}

pub const ELEMENT_ISOTOPES: [Option<IsotopeProps>; NUM_ELEMENTS] = [
    None, // H
    Some(IsotopeProps { mass_diff: 1.003355, prob_0: 0.9893, prob_1: 0.0107 }), // C
    Some(IsotopeProps { mass_diff: 0.997035, prob_0: 0.996, prob_1: 0.004 }), // N
    None, // O
    None, // F
    None, // Na
    None, // P
    Some(IsotopeProps { mass_diff: 1.995796, prob_0: 0.9493, prob_1: 0.0429 }), // S
    Some(IsotopeProps { mass_diff: 1.99705, prob_0: 0.7578, prob_1: 0.2422 }), // Cl
    None, // K
    Some(IsotopeProps { mass_diff: 1.99795, prob_0: 0.5069, prob_1: 0.4931 }), // Br
    None, // I
];