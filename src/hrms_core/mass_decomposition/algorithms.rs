//! Re-export wrapper for the external `mass-decomposition` crate.
//!
//! The core mass-decomposition algorithm implementations live in
//! `mass_decomposition::decomposer`; this module re-exports the public types
//! so existing `crate::mass_decomposition::algorithms::*` import paths continue
//! to work.

pub use mass_decomposition::bounds::deduce_isotopic_pattern_inner;
pub use mass_decomposition::decomposer::{MassDecomposer, SpectrumDecomposer};
