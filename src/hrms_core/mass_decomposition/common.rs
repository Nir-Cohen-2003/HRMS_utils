//! Re-export wrapper for the external `mass-decomposition` crate.
//!
//! This module re-exports the public surface area of the `mass-decomposition`
//! crate so that the rest of the codebase (notably the Polars/pyo3 interface)
//! can keep using `crate::mass_decomposition::common::*` import paths.

pub use mass_decomposition::element::{ATOMIC_MASSES, ELEMENT_SYMBOLS, NUM_ELEMENTS};
// `Formula` is re-exported at the crate root of `mass-decomposition` because the
// canonical definition in `formula.rs` is `pub(crate)`.
pub use mass_decomposition::formula::formula_to_string;
pub use mass_decomposition::Formula;
pub use mass_decomposition::bounds::{
    CalibratedIsotopicModel, CleanAndNormalizeSpectrumKwargs, CleanedAndNormalizedSpectrumResult,
    CorrectedFragment, DecomposedFragment, DecompositionKwargs, DecompositionParams,
    DeduceIsotopicPatternKwargs, IsotopicPatternParams, SpectrumDecompositionParams,
    check_dbe, default_max_bounds, default_min_bounds,
};

// Used by the deducer of isotopic patterns; `DeduceIsotopicPatternKwargs` carries
// `Option<HashMap<String, i32>>` fields.
pub use std::collections::HashMap;
