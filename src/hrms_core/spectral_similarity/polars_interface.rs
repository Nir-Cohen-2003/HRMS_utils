use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rayon::prelude::*;
use super::common::*;
use super::algorithms::*;
use crate::common::NUM_ELEMENTS;


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

#[derive(serde::Deserialize, Debug)]
pub struct InfoSimilarityKwargs {
    pub distance_metric: String,
    pub ignore_hydrogens: bool,
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


pub fn info_similarity_output_type(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "info_similarity".into(),
        DataType::Struct(vec![
            Field::new("spec1_info".into(), DataType::Float64),
            Field::new("spec2_info".into(), DataType::Float64),
            Field::new("union_info".into(), DataType::Float64),
            Field::new("diff1_info".into(), DataType::Float64),
            Field::new("diff2_info".into(), DataType::Float64),
        ]),
    ))
}

#[polars_expr(output_type_func=info_similarity_output_type)]
pub fn info_similarity_struct(inputs: &[Series], kwargs: InfoSimilarityKwargs) -> PolarsResult<Series> {
    let struct_series = &inputs[0];
    let ca = struct_series.struct_()?;

    let precursor1_series = ca
        .field_by_name("precursor_formula1")?
        .cast(&DataType::Array(Box::new(DataType::Float64), NUM_ELEMENTS))?;
    let fragments1_series = ca
        .field_by_name("fragment_formulas1")?
        .cast(&DataType::List(Box::new(DataType::Array(
            Box::new(DataType::Float64),
            NUM_ELEMENTS,
        ))))?;
    let precursor2_series = ca
        .field_by_name("precursor_formula2")?
        .cast(&DataType::Array(Box::new(DataType::Float64), NUM_ELEMENTS))?;
    let fragments2_series = ca
        .field_by_name("fragment_formulas2")?
        .cast(&DataType::List(Box::new(DataType::Array(
            Box::new(DataType::Float64),
            NUM_ELEMENTS,
        ))))?;

    let precursors1_ca = precursor1_series.array()?;
    let fragments1_ca = fragments1_series.list()?;
    let precursors2_ca = precursor2_series.array()?;
    let fragments2_ca = fragments2_series.list()?;

    let distance_metric = kwargs.distance_metric;
    let ignore_hydrogens = kwargs.ignore_hydrogens;

    // Convert to Vec for parallel processing
    let precursors1_vec: Vec<Option<Series>> = precursors1_ca.into_iter().collect();
    let fragments1_vec: Vec<Option<Series>> = fragments1_ca.into_iter().collect();
    let precursors2_vec: Vec<Option<Series>> = precursors2_ca.into_iter().collect();
    let fragments2_vec: Vec<Option<Series>> = fragments2_ca.into_iter().collect();

    let results: Vec<Option<InfoSimilarity>> = precursors1_vec
        .into_par_iter()
        .zip(fragments1_vec.into_par_iter())
        .zip(precursors2_vec.into_par_iter())
        .zip(fragments2_vec.into_par_iter())
        .map(
            |(((precursor1_opt, fragments1_opt), precursor2_opt), fragments2_opt)| {
                match (
                    precursor1_opt,
                    fragments1_opt,
                    precursor2_opt,
                    fragments2_opt,
                ) {
                    (
                        Some(precursor1_s),
                        Some(fragments1_s),
                        Some(precursor2_s),
                        Some(fragments2_s),
                    ) => {
                        let precursor1_ca = precursor1_s.f64().unwrap();
                        let precursor1_vec: Vec<f64> =
                            precursor1_ca.into_no_null_iter().collect();

                        let fragments1_list = fragments1_s.array().unwrap();
                        let mut fragments1_flat: Vec<f64> = Vec::new();
                        for fragment_series_opt in fragments1_list.clone().into_iter() {
                            if let Some(fragment_series) = fragment_series_opt {
                                let fragment_ca = fragment_series.f64().unwrap();
                                fragments1_flat.extend(fragment_ca.into_no_null_iter());
                            }
                        }

                        let precursor2_ca = precursor2_s.f64().unwrap();
                        let precursor2_vec: Vec<f64> =
                            precursor2_ca.into_no_null_iter().collect();

                        let fragments2_list = fragments2_s.array().unwrap();
                        let mut fragments2_flat: Vec<f64> = Vec::new();
                        for fragment_series_opt in fragments2_list.clone().into_iter() {
                            if let Some(fragment_series) = fragment_series_opt {
                                let fragment_ca = fragment_series.f64().unwrap();
                                fragments2_flat.extend(fragment_ca.into_no_null_iter());
                            }
                        }

                        let result = calculate_info_similarity(
                            &precursor1_vec,
                            &fragments1_flat,
                            &precursor2_vec,
                            &fragments2_flat,
                            &distance_metric,
                            ignore_hydrogens,
                        );
                        Some(result)
                    }
                    _ => None,
                }
            },
        )
        .collect();

    let len = results.len();
    let mut spec1_info = PrimitiveChunkedBuilder::<Float64Type>::new("spec1_info".into(), len);
    let mut spec2_info = PrimitiveChunkedBuilder::<Float64Type>::new("spec2_info".into(), len);
    let mut union_info = PrimitiveChunkedBuilder::<Float64Type>::new("union_info".into(), len);
    let mut diff1_info = PrimitiveChunkedBuilder::<Float64Type>::new("diff1_info".into(), len);
    let mut diff2_info = PrimitiveChunkedBuilder::<Float64Type>::new("diff2_info".into(), len);

    for res in results {
        match res {
            Some(val) => {
                spec1_info.append_value(val.spec1_info);
                spec2_info.append_value(val.spec2_info);
                union_info.append_value(val.union_info);
                diff1_info.append_value(val.diff1_info);
                diff2_info.append_value(val.diff2_info);
            }
            None => {
                spec1_info.append_null();
                spec2_info.append_null();
                union_info.append_null();
                diff1_info.append_null();
                diff2_info.append_null();
            }
        }
    }

    let s1 = spec1_info.finish().into_series();
    let s2 = spec2_info.finish().into_series();
    let s3 = union_info.finish().into_series();
    let s4 = diff1_info.finish().into_series();
    let s5 = diff2_info.finish().into_series();

    let out = StructChunked::from_series(
        "info_similarity".into(),
        len,
        [&s1, &s2, &s3, &s4, &s5].iter().copied(),
    )?;
    Ok(out.into_series())
}
