use crate::common::{ATOMIC_MASSES, ELEMENT_SYMBOLS, NUM_ELEMENTS};
use pyo3::prelude::*;

pub mod common;
pub mod io_mzml;
pub mod io_thermo;
pub mod mass_decomposition;
pub mod spectral_information;
pub mod spectral_similarity;

#[pymodule]
fn _internal(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("NUM_ELEMENTS", NUM_ELEMENTS)?;
    m.add("ATOMIC_MASSES", ATOMIC_MASSES)?;
    m.add("ELEMENT_SYMBOLS", ELEMENT_SYMBOLS)?;
    m.add_function(wrap_pyfunction!(io_mzml::read_mzml_files, m)?)?;
    m.add_function(wrap_pyfunction!(io_thermo::read_thermo_files, m)?)?;
    Ok(())
}
// this allocator is not thread safe, and clashes with the thermo RAW file parser code which uses dotnet
// #[global_allocator]
// static ALLOC: PolarsAllocator = PolarsAllocator::new();
