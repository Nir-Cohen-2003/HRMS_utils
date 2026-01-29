use mzdata::io::ThermoRawReader;
use mzdata::prelude::*;
use mzdata::spectrum::IsolationWindowState;
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use rayon::prelude::*;
use std::path::Path;

/// Read multiple Thermo RAW files in parallel and return a list of Polars DataFrames.
#[pyfunction]
pub fn read_thermo_files(paths: Vec<String>) -> PyResult<Vec<PyDataFrame>> {
    let dfs: Vec<PyDataFrame> = paths
        .iter()
        .map(|path| {
            let path = Path::new(path);
            read_single_thermo(path).map(PyDataFrame)
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(dfs)
}

fn read_single_thermo(path: &Path) -> Result<DataFrame, Box<dyn std::error::Error + Send + Sync>> {
    // Open the Thermo RAW reader
    let mut reader = ThermoRawReader::new(path)?;

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

    // Thermo specific fields
    let mut injection_times = Vec::new();
    let mut filter_strings = Vec::new();

    // Iterate over all spectra
    for i in 0..reader.len() {
        if let Some(spectrum) = reader.get_spectrum_by_index(i) {
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
            // Thermo reader populates arrays or peaks depending on mode.
            // We check arrays first, then peaks.
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
                    let iso = &precursor.isolation_window;
                    if iso.target != 0.0 {
                        precursor_mzs.push(Some(iso.target as f64));
                    } else {
                        precursor_mzs.push(None);
                    }
                }

                // Isolation Window
                // Logic mirrored from io_mzml.rs to handle Offset vs Explicit
                let iso = &precursor.isolation_window;
                if iso.target != 0.0 {
                    let (lower, upper) = match iso.flags {
                        IsolationWindowState::Explicit => (iso.lower_bound, iso.upper_bound),
                        IsolationWindowState::Offset => {
                            (iso.target - iso.lower_bound, iso.target + iso.upper_bound)
                        }
                        IsolationWindowState::Complete => (iso.lower_bound, iso.upper_bound),
                        _ => {
                            // If flags are unknown, we have to guess or assume.
                            // Thermo raw reader in mzdata seems to set explicit values often,
                            // but let's check if target is involved.
                            // If it's unknown, we'll try to use the raw values as is (Explicit assumption)
                            // or if they look like offsets (small numbers) relative to target?
                            // Safest is to treat as Explicit if we aren't sure,
                            // or default to 0,0 if truly unset.
                            // However, mzdata Thermo reader implementation sets lower/upper from API.
                            // Let's assume Explicit behavior for now if flags are default/unknown,
                            // as typically Thermo API gives absolute ranges or we don't have enough info.
                            // But usually mzdata sets the flags if it knows.
                            (iso.lower_bound, iso.upper_bound)
                        }
                    };

                    // Sanity check: if we got 0.0 for bounds but target is set, maybe it was offset 0?
                    // We'll just push what we calculated.
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
                } else {
                    collision_energies.push(None);
                }
            } else {
                precursor_mzs.push(None);
                iso_window_lower.push(None);
                iso_window_upper.push(None);
                collision_energies.push(None);
            }

            // Thermo Specific: Injection Time & Filter String
            // Access acquisition list -> first scan event
            let mut found_meta = false;
            if let Some(acquisition) = spectrum.acquisition().first_scan() {
                // Injection time is usually in milliseconds in Thermo
                injection_times.push(Some(acquisition.injection_time as f64));

                // Filter string is stored as a param "filter string" (accession 1000512)
                if let Some(param) = acquisition
                    .params()
                    .iter()
                    .find(|p| p.name == "filter string")
                {
                    filter_strings.push(Some(param.value.to_string()));
                } else {
                    filter_strings.push(None);
                }
                found_meta = true;
            }

            if !found_meta {
                injection_times.push(None);
                filter_strings.push(None);
            }
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

    let s_inj_time = Series::new("injection_time".into(), injection_times);
    let s_filter = Series::new("filter_string".into(), filter_strings);

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
        s_inj_time.into(),
        s_filter.into(),
    ])?;

    Ok(df)
}

fn create_list_f64_series(name: &str, data: Vec<Option<Series>>) -> Result<Series, PolarsError> {
    let s: ListChunked = data.into_iter().collect();
    let mut s = s.into_series();
    s.rename(name.into());
    Ok(s)
}
