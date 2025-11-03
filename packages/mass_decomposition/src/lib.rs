// packages/mass_decomposition/src/lib.rs

mod common;
mod algorithms;
mod polars_interface;

use pyo3::prelude::*;
use crate::polars_interface::decompose_mass;
use crate::common::NUM_ELEMENTS;

#[pymodule]
fn _internal(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(decompose_mass, m)?)?;
    m.add("NUM_ELEMENTS", NUM_ELEMENTS)?;
    Ok(())
}
