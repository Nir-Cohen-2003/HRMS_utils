use pyo3::prelude::*;
use pyo3_polars::PolarsAllocator;
use crate::common::{NUM_ELEMENTS, ATOMIC_MASSES, ELEMENT_SYMBOLS};

pub mod common;
pub mod mass_decomposition;
pub mod spectral_information;
pub mod spectral_similarity;

#[pymodule]
fn _internal(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("NUM_ELEMENTS", NUM_ELEMENTS)?;
    m.add("ATOMIC_MASSES", ATOMIC_MASSES)?;
    m.add("ELEMENT_SYMBOLS", ELEMENT_SYMBOLS)?;
    Ok(())
}

#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();