use pyo3::prelude::*;

pub mod algorithms;
pub mod polars_interface;

pub use polars_interface::*;
use mass_decomposition::common::NUM_ELEMENTS;

#[pymodule]
fn _internal_spectral_information(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("NUM_ELEMENTS", NUM_ELEMENTS)?;
    Ok(())
}

#[allow(unused_imports)]
use pyo3_polars::PolarsAllocator;
#[cfg(feature = "polars_global_allocator")]
#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();