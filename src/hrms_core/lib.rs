use crate::common::{ATOMIC_MASSES, ELEMENT_SYMBOLS, NUM_ELEMENTS};
use pyo3::prelude::*;

pub mod common;
pub mod io_mzml;
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
    Ok(())
}
// The global allocator remains disabled as a conservative default; mzML reading no longer
// pulls in the dotnet-backed Thermo RAW reader, but we keep the PolarsAllocator opt-in off.
// #[global_allocator]
// static ALLOC: PolarsAllocator = PolarsAllocator::new();
