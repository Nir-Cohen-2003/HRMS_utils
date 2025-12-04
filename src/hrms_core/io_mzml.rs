use mzdata::prelude::*;
use mzdata::spectrum::IsolationWindowState;
use mzdata::MzMLReader;
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use rayon::prelude::*;
use std::path::Path;

/// Read multiple mzML files in parallel and return a list of Polars DataFrames.
///
/// # Schema Extensions
/// To add more fields, see `src/hrms_core/RUST_SCHEMA_EXTENSIONS.md`.
#[pyfunction]
pub fn read_mzml_files(paths: Vec<String>) -> PyResult<Vec<PyDataFrame>> {
    let dfs: Vec<PyDataFrame> = paths
        .par_iter()
        .map(|path| {
            let path = Path::new(path);
            read_single_mzml(path).map(PyDataFrame)
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(dfs)
}

fn read_single_mzml(path: &Path) -> Result<DataFrame, Box<dyn std::error::Error + Send + Sync>> {
    let reader = MzMLReader::open_path(path)?;

    // Pre-allocate vectors for columns
    let mut ids = Vec::new();
    let mut scan_times = Vec::new();
    let mut ms_levels = Vec::new();
    let mut polarities = Vec::new();
    let mut mzs = Vec::new();
    let mut intensities = Vec::new();
    let mut precursor_mzs = Vec::new();
    let mut iso_window_lower = Vec::new();
    let mut iso_window_upper = Vec::new();
    let mut collision_energies = Vec::new();
    let mut collision_energy_units = Vec::new();

    for spectrum in reader {
        // ID
        ids.push(spectrum.id().to_string());

        // MS Level
        ms_levels.push(spectrum.ms_level());

        // Scan Time
        let st = spectrum.start_time();
        scan_times.push(if st > 0.0 { Some(st) } else { None });

        // Polarity
        let pol = match spectrum.polarity() {
            mzdata::spectrum::ScanPolarity::Positive => Some("positive".to_string()),
            mzdata::spectrum::ScanPolarity::Negative => Some("negative".to_string()),
            _ => None,
        };
        polarities.push(pol);

        // M/Z and Intensity Arrays
        let (mz_vec, int_vec) = if let Some(arrays) = &spectrum.arrays {
            let mzs = arrays.mzs().unwrap_or_default().into_owned();
            let ints = arrays
                .intensities()
                .unwrap_or_default()
                .iter()
                .map(|&x| x as f64)
                .collect();
            (mzs, ints)
        } else if let Some(peaks) = &spectrum.peaks {
            let mzs: Vec<f64> = peaks.iter().map(|p| p.mz).collect();
            let ints: Vec<f64> = peaks.iter().map(|p| p.intensity as f64).collect();
            (mzs, ints)
        } else {
            (Vec::new(), Vec::new())
        };
        
        mzs.push(Some(Series::new("mz".into(), mz_vec)));
        intensities.push(Some(Series::new("intensity".into(), int_vec)));

        // Precursor Info
        if let Some(precursor) = spectrum.precursor() {
            // Precursor M/Z
            if let Some(selected_ion) = precursor.ions.first() {
                precursor_mzs.push(Some(selected_ion.mz as f64));
            } else {
                // Fallback to isolation window target if selected ion is missing
                let iso = &precursor.isolation_window;
                if iso.target != 0.0 {
                     precursor_mzs.push(Some(iso.target as f64));
                } else {
                     precursor_mzs.push(None);
                }
            }

            // Isolation Window
            let iso = &precursor.isolation_window;
            if iso.target != 0.0 {
                let (lower, upper) = match iso.flags {
                    IsolationWindowState::Explicit => (iso.lower_bound, iso.upper_bound),
                    IsolationWindowState::Offset => (
                        iso.target - iso.lower_bound,
                        iso.target + iso.upper_bound,
                    ),
                    IsolationWindowState::Complete => (iso.lower_bound, iso.upper_bound),
                    _ => (0.0, 0.0),
                };
                iso_window_lower.push(Some(lower as f64));
                iso_window_upper.push(Some(upper as f64));
            } else {
                iso_window_lower.push(None);
                iso_window_upper.push(None);
            }

            // Collision Energy
            let activation = &precursor.activation;
            if activation.energy != 0.0 {
                collision_energies.push(Some(activation.energy as f64));
                
                // Try to find unit in params
                let mut unit = None;
                for param in activation.params() {
                    if param.is_ms() && param.accession == Some(1000045) {
                         unit = Some(format!("{:?}", param.unit));
                         break;
                    }
                }
                // Fallback search by name
                if unit.is_none() {
                    if let Some(p) = activation.params().iter().find(|p| p.name == "collision energy") {
                         unit = Some(p.unit.to_string());
                    }
                }
                collision_energy_units.push(unit);
            } else {
                collision_energies.push(None);
                collision_energy_units.push(None);
            }

        } else {
            precursor_mzs.push(None);
            iso_window_lower.push(None);
            iso_window_upper.push(None);
            collision_energies.push(None);
            collision_energy_units.push(None);
        }
    }

    // Create Series
    let s_id = Series::new("id".into(), ids);
    let s_ms_level = Series::new("ms_level".into(), ms_levels);
    let s_scan_time = Series::new("scan_time".into(), scan_times);
    let s_polarity = Series::new("polarity".into(), polarities);
    
    let s_mz = create_list_f64_series("mz", mzs)?;
    let s_intensity = create_list_f64_series("intensity", intensities)?;

    let s_precursor_mz = Series::new("precursor_mz".into(), precursor_mzs);
    let s_iso_lower = Series::new("isolation_window_lower_bound".into(), iso_window_lower);
    let s_iso_upper = Series::new("isolation_window_upper_bound".into(), iso_window_upper);
    let s_ce = Series::new("collision_energy".into(), collision_energies);
    let s_ce_unit = Series::new("collision_energy_unit".into(), collision_energy_units);

    let df = DataFrame::new(vec![
        s_id.into(),
        s_scan_time.into(),
        s_ms_level.into(),
        s_polarity.into(),
        s_mz.into(),
        s_intensity.into(),
        s_precursor_mz.into(),
        s_iso_lower.into(),
        s_iso_upper.into(),
        s_ce.into(),
        s_ce_unit.into(),
    ])?;

    Ok(df)
}

fn create_list_f64_series(name: &str, data: Vec<Option<Series>>) -> Result<Series, PolarsError> {
    let s: ListChunked = data.into_iter().collect();
    let mut s = s.into_series();
    s.rename(name.into());
    Ok(s)
}
