use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rayon::prelude::*;
use super::common::*;
use super::algorithms::*;

#[derive(serde::Deserialize, Debug, Default)]
#[serde(default)]
pub struct SimilarityKwargs {
    ms2_tolerance_in_ppm: Option<f64>,
    clean_spectra_first: Option<bool>,
    noise_threshold: Option<f64>,
    ignore_precursor: Option<bool>,
    intensity_power: Option<f64>,
    mass_power: Option<f64>,
    permissive: Option<bool>,
}

#[polars_expr(output_type=Float64)]
pub fn calculate_similarity_struct(inputs: &[Series], kwargs: SimilarityKwargs) -> PolarsResult<Series> {
    let struct_series = &inputs[0];
    let ca: &StructChunked = struct_series.struct_()?;

    let mz1_series = ca.field_by_name("mz1")?;
    let int1_series = ca.field_by_name("intensities1")?;
    let mz2_series = ca.field_by_name("mz2")?;
    let int2_series = ca.field_by_name("intensities2")?;
    let precursor_mz1_series = ca.field_by_name("precursor_mz1")?;
    let precursor_mz2_series = ca.field_by_name("precursor_mz2")?;

    let mz1_list = mz1_series.list()?;
    let int1_list = int1_series.list()?;
    let mz2_list = mz2_series.list()?;
    let int2_list = int2_series.list()?;
    let precursor_mz1_ca = precursor_mz1_series.f64()?;
    let precursor_mz2_ca = precursor_mz2_series.f64()?;

    let mz1_vec: Vec<Option<Series>> = mz1_list.into_iter().collect();
    let int1_vec: Vec<Option<Series>> = int1_list.into_iter().collect();
    let mz2_vec: Vec<Option<Series>> = mz2_list.into_iter().collect();
    let int2_vec: Vec<Option<Series>> = int2_list.into_iter().collect();
    let precursor_mz1_vec: Vec<Option<f64>> = precursor_mz1_ca.into_iter().collect();
    let precursor_mz2_vec: Vec<Option<f64>> = precursor_mz2_ca.into_iter().collect();

    let ms2_tolerance_in_ppm = kwargs.ms2_tolerance_in_ppm.unwrap_or(5.0);
    let clean_spectra_first = kwargs.clean_spectra_first.unwrap_or(true);
    let noise_threshold = kwargs.noise_threshold;
    let ignore_precursor = kwargs.ignore_precursor.unwrap_or(false);

    let out: Float64Chunked = mz1_vec
        .into_par_iter()
        .zip(int1_vec.into_par_iter())
        .zip(mz2_vec.into_par_iter())
        .zip(int2_vec.into_par_iter())
        .zip(precursor_mz1_vec.into_par_iter())
        .zip(precursor_mz2_vec.into_par_iter())
        .map(|(((((opt_mz1, opt_int1), opt_mz2), opt_int2), precursor_mz1), precursor_mz2)| {
            match (opt_mz1, opt_int1, opt_mz2, opt_int2, precursor_mz1, precursor_mz2) {
                (Some(mz1), Some(int1), Some(mz2), Some(int2), Some(pmz1), Some(pmz2)) => {
                    let spec1: Spectrum = Spectrum::from_polars_lists(&mz1, &int1).ok()?;
                    let spec2: Spectrum = Spectrum::from_polars_lists(&mz2, &int2).ok()?;

                    let similarity: f64 = calculate_entropy_similarity(
                        &spec1,
                        &spec2,
                        pmz1,
                        pmz2,
                        ms2_tolerance_in_ppm,
                        clean_spectra_first,
                        noise_threshold,
                        ignore_precursor,
                    );
                    Some(similarity)
                }
                _ => None,
            }
        })
        .collect();

    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
pub fn cosine_similarity_struct(inputs: &[Series], kwargs: SimilarityKwargs) -> PolarsResult<Series> {
    let struct_series = &inputs[0];
    let ca: &StructChunked = struct_series.struct_()?;

    let mz1_series = ca.field_by_name("mz1")?;
    let int1_series = ca.field_by_name("intensities1")?;
    let mz2_series = ca.field_by_name("mz2")?;
    let int2_series = ca.field_by_name("intensities2")?;
    let precursor_mz1_series = ca.field_by_name("precursor_mz1")?;
    let precursor_mz2_series = ca.field_by_name("precursor_mz2")?;

    let mz1_list = mz1_series.list()?;
    let int1_list = int1_series.list()?;
    let mz2_list = mz2_series.list()?;
    let int2_list = int2_series.list()?;
    let precursor_mz1_ca = precursor_mz1_series.f64()?;
    let precursor_mz2_ca = precursor_mz2_series.f64()?;

    let mz1_vec: Vec<Option<Series>> = mz1_list.into_iter().collect();
    let int1_vec: Vec<Option<Series>> = int1_list.into_iter().collect();
    let mz2_vec: Vec<Option<Series>> = mz2_list.into_iter().collect();
    let int2_vec: Vec<Option<Series>> = int2_list.into_iter().collect();
    let precursor_mz1_vec: Vec<Option<f64>> = precursor_mz1_ca.into_iter().collect();
    let precursor_mz2_vec: Vec<Option<f64>> = precursor_mz2_ca.into_iter().collect();

    let ms2_tolerance_in_ppm = kwargs.ms2_tolerance_in_ppm.unwrap();
    let clean_spectra_first = kwargs.clean_spectra_first.unwrap();
    let noise_threshold = kwargs.noise_threshold;
    let ignore_precursor = kwargs.ignore_precursor.unwrap_or(false);
    let intensity_power = kwargs.intensity_power.unwrap();
    let mass_power = kwargs.mass_power.unwrap();

    let out: Float64Chunked = mz1_vec
        .into_par_iter()
        .zip(int1_vec.into_par_iter())
        .zip(mz2_vec.into_par_iter())
        .zip(int2_vec.into_par_iter())
        .zip(precursor_mz1_vec.into_par_iter())
        .zip(precursor_mz2_vec.into_par_iter())
        .map(|(((((opt_mz1, opt_int1), opt_mz2), opt_int2), precursor_mz1), precursor_mz2)| {
            match (opt_mz1, opt_int1, opt_mz2, opt_int2, precursor_mz1, precursor_mz2) {
                (Some(mz1), Some(int1), Some(mz2), Some(int2), Some(pmz1), Some(pmz2)) => {
                    let spec1: Spectrum = Spectrum::from_polars_lists(&mz1, &int1).ok()?;
                    let spec2: Spectrum = Spectrum::from_polars_lists(&mz2, &int2).ok()?;

                    let similarity: f64 = general_cosine_similarity(
                        &spec1,
                        &spec2,
                        pmz1,
                        pmz2,
                        ms2_tolerance_in_ppm,
                        intensity_power,
                        mass_power,
                        clean_spectra_first,
                        noise_threshold,
                        ignore_precursor,
                    );
                    Some(similarity)
                }
                _ => None,
            }
        })
        .collect();

    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
pub fn explained_intensity_struct(inputs: &[Series], kwargs: SimilarityKwargs) -> PolarsResult<Series> {
    let struct_series = &inputs[0];
    let ca: &StructChunked = struct_series.struct_()?;

    let mz1_series = ca.field_by_name("mz1")?;
    let int1_series = ca.field_by_name("intensities1")?;
    let mz2_series = ca.field_by_name("mz2")?;
    let int2_series = ca.field_by_name("intensities2")?;
    let precursor_mz1_series = ca.field_by_name("precursor_mz1")?;
    let precursor_mz2_series = ca.field_by_name("precursor_mz2")?;

    let mz1_list = mz1_series.list()?;
    let int1_list = int1_series.list()?;
    let mz2_list = mz2_series.list()?;
    let int2_list = int2_series.list()?;
    let precursor_mz1_ca = precursor_mz1_series.f64()?;
    let precursor_mz2_ca = precursor_mz2_series.f64()?;

    let mz1_vec: Vec<Option<Series>> = mz1_list.into_iter().collect();
    let int1_vec: Vec<Option<Series>> = int1_list.into_iter().collect();
    let mz2_vec: Vec<Option<Series>> = mz2_list.into_iter().collect();
    let int2_vec: Vec<Option<Series>> = int2_list.into_iter().collect();
    let precursor_mz1_vec: Vec<Option<f64>> = precursor_mz1_ca.into_iter().collect();
    let precursor_mz2_vec: Vec<Option<f64>> = precursor_mz2_ca.into_iter().collect();

    let ms2_tolerance_in_ppm = kwargs.ms2_tolerance_in_ppm.unwrap();
    let clean_spectra_first = kwargs.clean_spectra_first.unwrap();
    let noise_threshold = kwargs.noise_threshold;
    let ignore_precursor = kwargs.ignore_precursor.unwrap_or(false);
    let intensity_power = kwargs.intensity_power.unwrap();
    let mass_power = kwargs.mass_power.unwrap();
    let permissive = kwargs.permissive.unwrap_or(false);

    let out: Float64Chunked = mz1_vec
        .into_par_iter()
        .zip(int1_vec.into_par_iter())
        .zip(mz2_vec.into_par_iter())
        .zip(int2_vec.into_par_iter())
        .zip(precursor_mz1_vec.into_par_iter())
        .zip(precursor_mz2_vec.into_par_iter())
        .map(|(((((opt_mz1, opt_int1), opt_mz2), opt_int2), precursor_mz1), precursor_mz2)| {
            match (opt_mz1, opt_int1, opt_mz2, opt_int2, precursor_mz1, precursor_mz2) {
                (Some(mz1), Some(int1), Some(mz2), Some(int2), Some(pmz1), Some(pmz2)) => {
                    let spec1: Spectrum = Spectrum::from_polars_lists(&mz1, &int1).ok()?;
                    let spec2: Spectrum = Spectrum::from_polars_lists(&mz2, &int2).ok()?;

                    let similarity: f64 = calculate_explained_intensity(
                        &spec1,
                        &spec2,
                        pmz1,
                        pmz2,
                        ms2_tolerance_in_ppm,
                        intensity_power,
                        mass_power,
                        clean_spectra_first,
                        noise_threshold,
                        ignore_precursor,
                        permissive,
                    );
                    Some(similarity)
                }
                _ => None,
            }
        })
        .collect();

    Ok(out.into_series())
}


