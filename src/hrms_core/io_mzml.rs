use mzdata::prelude::*;
use mzdata::spectrum::{IsolationWindowState, SignalContinuity};
use mzdata::MzMLReader;
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use rayon::prelude::*;
use std::path::Path;

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
    // We don't know the exact size, but we can guess or just let Vec grow.
    // If we wanted to be faster, we could count spectra first, but that requires a pass.
    // Let's just collect into Vecs.

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
        // mzdata returns Cow<[f64]> or similar.
        // We need to convert to Series (List<f64>).
        // Polars ListBuilder expects values.
        
        // Note: mzdata might return f32 or f64 depending on the file, 
        // but the high-level `mzs()` usually returns `Cow<[f64]>`.
        // Let's check if we need to decode. `spectrum` from iterator is usually fully decoded if default.
        // But `mzdata` reader might be lazy.
        // The `spectrum` object has `mzs()` and `intensities()` methods from `SpectrumLike`?
        // Actually `SpectrumLike` doesn't enforce `mzs()`. `CentroidSpectrum` and `RawSpectrum` do.
        // `MzMLReader` yields `MultiLayerSpectrum`.
        // `MultiLayerSpectrum` has `mzs()` and `intensities()`.
        
        let mz_vec: Vec<f64> = spectrum.mzs().into_owned();
        let int_vec: Vec<f64> = spectrum.intensities().into_owned();
        
        mzs.push(Some(Series::new("mz".into(), mz_vec)));
        intensities.push(Some(Series::new("intensity".into(), int_vec)));

        // Precursor Info
        if let Some(precursor) = spectrum.precursor() {
            // Precursor M/Z
            // Usually in `selected_ion`.
            if let Some(selected_ion) = precursor.selected_ion().first() {
                precursor_mzs.push(Some(selected_ion.mz as f64));
            } else {
                // Fallback to isolation window target if selected ion is missing?
                // User asked for "precursor_mz". Usually selected ion mz.
                if let Some(iso) = precursor.isolation_window() {
                     precursor_mzs.push(Some(iso.target as f64));
                } else {
                     precursor_mzs.push(None);
                }
            }

            // Isolation Window
            if let Some(iso) = precursor.isolation_window() {
                let (lower, upper) = match iso.flags {
                    IsolationWindowState::Explicit => (iso.lower_bound, iso.upper_bound),
                    IsolationWindowState::Offset => (
                        iso.target - iso.lower_bound,
                        iso.target + iso.upper_bound,
                    ),
                    IsolationWindowState::Complete => (iso.lower_bound, iso.upper_bound), // Complete usually means explicit bounds are set
                    _ => (0.0, 0.0), // Unknown
                };
                
                // If flags are unknown or values are zero, we might want to emit None?
                // Assuming if target is non-zero, we have something.
                if iso.target != 0.0 {
                     iso_window_lower.push(Some(lower as f64));
                     iso_window_upper.push(Some(upper as f64));
                } else {
                     iso_window_lower.push(None);
                     iso_window_upper.push(None);
                }
            } else {
                iso_window_lower.push(None);
                iso_window_upper.push(None);
            }

            // Collision Energy
            if let Some(activation) = precursor.activation() {
                if activation.energy != 0.0 {
                    collision_energies.push(Some(activation.energy as f64));
                    // Unit? mzdata might not expose unit directly in the struct easily without looking at params.
                    // `Activation` struct has `energy`.
                    // We might need to look at `params` for unit.
                    // Let's check `Activation` struct in `mzdata`.
                    // It implements `ParamLike`.
                    // We can search params for collision energy unit.
                    // But `energy` field is f32.
                    // Let's try to find unit in params.
                    // Accession for collision energy is MS:1000045.
                    // We can look for that param and check its unit.
                    
                    let mut unit = None;
                    for param in activation.params() {
                        if param.is_ms() && param.accession.unwrap() == 1000045 {
                             unit = Some(format!("{:?}", param.unit));
                             break;
                        }
                    }
                    // If not found in specific param, maybe just "eV"?
                    // Let's leave unit as None or "unknown" if not found, or try to parse.
                    // `mzdata` `Unit` enum has `ElectronVolt`.
                    
                    // Actually, let's just look at the `unit` field of the param that corresponds to collision energy.
                    // Or we can just push None for now if it's hard.
                    // User asked for "collision energy and its unit".
                    
                    // Let's try to find the param.
                    let param = activation.params().iter().find(|p| p.name() == "collision energy" || (p.is_ms() && p.accession == Some(1000045)));
                    if let Some(p) = param {
                         collision_energy_units.push(Some(p.unit.to_string()));
                    } else {
                         collision_energy_units.push(None);
                    }

                } else {
                    collision_energies.push(None);
                    collision_energy_units.push(None);
                }
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
    
    // List Series for mz and intensity
    // We need to construct a ListChunked.
    // The easiest way in Polars is often to collect into a ChunkedArray of Series, but that's not direct.
    // We can use `ListPrimitiveChunkedBuilder` or just `Series::new` with `Vec<Series>`? No.
    // `Series::new` with `Vec<Vec<T>>` works? No.
    // We need to use `ListChunked`.
    
    // Helper to create List<f64> series
    let s_mz = create_list_f64_series("mz", &mzs)?;
    let s_intensity = create_list_f64_series("intensity", &intensities)?;

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

fn create_list_f64_series(name: &str, data: &[Option<Series>]) -> Result<Series, PolarsError> {
    // This is a bit inefficient, constructing Series for each row then collecting.
    // Better to use a builder if possible, but `polars` API for List construction can be tricky.
    // `ChunkedArray::from_iter` with `Option<Series>` might work?
    // Actually `ListChunked::from_iter` takes iterator of `Option<Box<dyn Array>>` or similar?
    // Let's try `Series::from_iter`.
    
    // Wait, `data` is `Vec<Option<Series>>`.
    // We can use `ListChunked::from_iter` where items are `Option<Series>`.
    // But `Series` inside must be compatible.
    
    // Let's try:
    let s: ListChunked = data.iter().map(|opt| opt.as_ref()).collect();
    let mut s = s.into_series();
    s.rename(name.into());
    Ok(s)
}
