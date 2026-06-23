//! Re-export wrapper for the external `mass-decomposition` crate.
//!
//! The precomputed ERT cache and preset element-mask sets live in
//! `mass_decomposition::precomputed`; this module re-exports the public types
//! so existing `crate::mass_decomposition::precomputed::*` import paths continue
//! to work.

// This module is private (`mod precomputed;` in the parent) and serves only as
// a re-export surface for siblings within `mass_decomposition`. The `pub use`
// names are consumed by `polars_interface.rs` via `super::precomputed::*`, so
// suppress the "unused import" diagnostic on the file itself.
#[allow(unused_imports)]
pub use mass_decomposition::precomputed::{PrecomputedDecomposer, find_best_precomputed};
