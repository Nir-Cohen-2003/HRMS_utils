use polars::prelude::*;
use polars::datatypes::{DataType, Int32Type, Float64Type};
use pyo3_polars::derive::polars_expr;
use rayon::prelude::*;
use crate::algorithms::{MassDecomposer, SpectrumDecomposer};
use crate::common::{DecompositionParams, SpectrumDecompositionParams, formula_to_string, NUM_ELEMENTS, CleanAndNormalizeSpectrumKwargs, DecompositionKwargs};
use polars::series::Series;
use polars::frame::DataFrame;
use polars::chunked_array::builder::{
    ListPrimitiveChunkedBuilder, PrimitiveChunkedBuilder, StringChunkedBuilder, ListStringChunkedBuilder
};
use polars_arrow::array::Int32Array;

fn mass_decomposition_output(_fields: &[Field]) -> PolarsResult<Field> {
    let formula_field = Field::new("formulas".into(), DataType::List(Box::new(DataType::Array(Box::new(DataType::Int32), NUM_ELEMENTS))));
    let formula_str_field = Field::new("formulas_str".into(), DataType::List(Box::new(DataType::String)));
    let error_field = Field::new("errors".into(), DataType::List(Box::new(DataType::Float64)));
    let v = vec![formula_field, formula_str_field, error_field];
    Ok(Field::new("mass_decomposition".into(), DataType::Struct(v)))
}

#[polars_expr(output_type_func=mass_decomposition_output)]
fn mass_decomposition(inputs: &[Series], kwargs: DecompositionKwargs) -> PolarsResult<Series> {
    let masses = inputs[0].f64()?;
    let min_bounds = kwargs.min_bounds.unwrap_or([0; NUM_ELEMENTS]);
    let max_bounds = kwargs.max_bounds.unwrap_or([100; NUM_ELEMENTS]);

    let series_vec: Vec<Series> = masses
        .into_iter()
        .par_bridge()
        .map(|mass_opt| {
            mass_opt.map(|mass| {
                let params = DecompositionParams {
                    tolerance_ppm: kwargs.tolerance_ppm,
                    min_dbe: kwargs.min_dbe,
                    max_dbe: kwargs.max_dbe,
                    dbe_mode: kwargs.dbe_mode.clone(),
                    min_bounds,
                    max_bounds,
                };
                let mut decomposer = MassDecomposer::new(min_bounds, max_bounds);
                let results = decomposer.decompose(mass, &params);

                let mut formulas_builder = ListPrimitiveChunkedBuilder::<Int32Type>::new("formulas".into(), results.len(), NUM_ELEMENTS, DataType::Int32);
                let mut formulas_str_builder = StringChunkedBuilder::new("formulas_str".into(), results.len());
                let mut errors_builder = PrimitiveChunkedBuilder::<Float64Type>::new("errors".into(), results.len());

                for res in results {
                    formulas_builder.append_slice(&res.formula);
                    formulas_str_builder.append_value(formula_to_string(&res.formula));
                    errors_builder.append_value(res.error_ppm);
                }

                let s_formulas = formulas_builder.finish().into_series();
                let s_formulas_str = formulas_str_builder.finish().into_series();
                let s_errors = errors_builder.finish().into_series();

                let df = DataFrame::new(vec![s_formulas.into(), s_formulas_str.into(), s_errors.into()]).unwrap();
                df.into_struct("".into()).into_series()
            })
        })
        .flatten()
        .collect();

    let out = StructChunked::from_series("mass_decomposition".into(), series_vec.iter().collect::<Vec<_>>().as_slice())?;
    Ok(out.into_series())
}

#[polars_expr(output_type_func=mass_decomposition_output)]
fn mass_decomposition_with_bounds(inputs: &[Series], kwargs: DecompositionKwargs) -> PolarsResult<Series> {
    let s = &inputs[0];
    let ca = s.struct_()?;

    let mass_series = ca.field_by_name("mass")?;
    let masses = mass_series.f64()?;
    let min_bounds_series = ca.field_by_name("min_bounds")?;
    let min_bounds_ca = min_bounds_series.array()?;
    let max_bounds_series = ca.field_by_name("max_bounds")?;
    let max_bounds_ca = max_bounds_series.array()?;

    let series_vec: Vec<Series> = masses.into_iter()
        .zip(min_bounds_ca)
        .zip(max_bounds_ca)
        .par_bridge()
        .map(|((mass_opt, min_bounds_opt), max_bounds_opt)| {
            if let (Some(mass), Some(min_bounds_arr), Some(max_bounds_arr)) = (mass_opt, min_bounds_opt, max_bounds_opt) {
                let min_bounds_ca = min_bounds_arr.as_any().downcast_ref::<Int32Array>().unwrap();
                let min_bounds_sl: &[i32] = min_bounds_ca.values();
                let max_bounds_ca = max_bounds_arr.as_any().downcast_ref::<Int32Array>().unwrap();
                let max_bounds_sl: &[i32] = max_bounds_ca.values();

                let mut min_bounds = [0; NUM_ELEMENTS];
                let mut max_bounds = [0; NUM_ELEMENTS];
                min_bounds.copy_from_slice(min_bounds_sl);
                max_bounds.copy_from_slice(max_bounds_sl);

                let params = DecompositionParams {
                    tolerance_ppm: kwargs.tolerance_ppm,
                    min_dbe: kwargs.min_dbe,
                    max_dbe: kwargs.max_dbe,
                    dbe_mode: kwargs.dbe_mode.clone(),
                    min_bounds,
                    max_bounds,
                };
                let mut decomposer = MassDecomposer::new(min_bounds, max_bounds);
                let results = decomposer.decompose(mass, &params);

                let mut formulas_builder = ListPrimitiveChunkedBuilder::<Int32Type>::new("formulas".into(), results.len(), NUM_ELEMENTS, DataType::Int32);
                let mut formulas_str_builder = StringChunkedBuilder::new("formulas_str".into(), results.len());
                let mut errors_builder = PrimitiveChunkedBuilder::<Float64Type>::new("errors".into(), results.len());

                for res in results {
                    formulas_builder.append_slice(&res.formula);
                    formulas_str_builder.append_value(formula_to_string(&res.formula));
                    errors_builder.append_value(res.error_ppm);
                }

                let s_formulas = formulas_builder.finish().into_series();
                let s_formulas_str = formulas_str_builder.finish().into_series();
                let s_errors = errors_builder.finish().into_series();

                let df = DataFrame::new(vec![s_formulas.into(), s_formulas_str.into(), s_errors.into()]).unwrap();
                Some(df.into_struct("".into()).into_series())
            } else {
                None
            }
        })
        .flatten()
        .collect();
    
    let out = StructChunked::from_series("mass_decomposition_with_bounds".into(), series_vec.iter().collect::<Vec<_>>().as_slice())?;
    Ok(out.into_series())
}

fn spectrum_decomposition_output(_fields: &[Field]) -> PolarsResult<Field> {
    let formula_field = Field::new("formulas".into(), DataType::List(Box::new(DataType::List(Box::new(DataType::Array(Box::new(DataType::Int32), NUM_ELEMENTS))))));
    let formula_str_field = Field::new("formulas_str".into(), DataType::List(Box::new(DataType::List(Box::new(DataType::String)))));
    let error_field = Field::new("errors".into(), DataType::List(Box::new(DataType::List(Box::new(DataType::Float64)))));
    let v = vec![formula_field, formula_str_field, error_field];
    Ok(Field::new("spectrum_decomposition".into(), DataType::Struct(v)))
}

#[polars_expr(output_type_func=spectrum_decomposition_output)]
fn spectrum_decomposition(inputs: &[Series], kwargs: SpectrumDecompositionParams) -> PolarsResult<Series> {
    let s = &inputs[0];
    let ca = s.struct_()?;
    
    let mz_series = ca.field_by_name("mz_values")?;
    let mz_ca = mz_series.list()?;
    let precursor_series = ca.field_by_name("precursor_formula")?;
    let precursor_ca = precursor_series.array()?;

    let series_vec: Vec<Series> = mz_ca.into_iter()
        .zip(precursor_ca)
        .par_bridge()
        .map(|(mz_opt, precursor_opt)| {
            if let (Some(mz_list), Some(precursor_arr)) = (mz_opt, precursor_opt) {
                let mz_values: Vec<f64> = mz_list.f64().unwrap().into_no_null_iter().collect();
                let precursor_ca = precursor_arr.as_any().downcast_ref::<Int32Array>().unwrap();
                let precursor_sl: &[i32] = precursor_ca.values();
                let mut precursor_formula = [0; NUM_ELEMENTS];
                precursor_formula.copy_from_slice(precursor_sl);

                let mut decomposer = SpectrumDecomposer::new();
                let results = decomposer.decompose_spectrum_with_precursor(&mz_values, &precursor_formula, &kwargs);

                let mut list_of_formulas_builder = ListStringChunkedBuilder::new("formulas".into(), mz_ca.len(), results.len() * NUM_ELEMENTS);
                let mut list_of_formulas_str_builder = ListStringChunkedBuilder::new("formulas_str".into(), mz_ca.len(), results.iter().flatten().map(|r| formula_to_string(&r.formula).len()).sum());
                let mut list_of_errors_builder = ListPrimitiveChunkedBuilder::<Float64Type>::new("errors".into(), mz_ca.len(), results.iter().flatten().count(), DataType::Float64);

                for fragment_results in results {
                    if fragment_results.is_empty() {
                        list_of_formulas_builder.append_null();
                        list_of_formulas_str_builder.append_null();
                        list_of_errors_builder.append_null();
                    } else {
                        let mut formulas_builder = StringChunkedBuilder::new("".into(), fragment_results.len());
                        let mut errors_builder = PrimitiveChunkedBuilder::<Float64Type>::new("".into(), fragment_results.len());
                        for res in fragment_results {
                            formulas_builder.append_value(formula_to_string(&res.formula));
                            errors_builder.append_value(res.error_ppm);
                        }
                        list_of_formulas_builder.append_series(&formulas_builder.finish().into_series()).unwrap();
                        list_of_errors_builder.append_series(&errors_builder.finish().into_series()).unwrap();
                    }
                }

                let s_formulas = list_of_formulas_builder.finish().into_series();
                let s_formulas_str = list_of_formulas_str_builder.finish().into_series();
                let s_errors = list_of_errors_builder.finish().into_series();

                let df = DataFrame::new(vec![s_formulas.into(), s_formulas_str.into(), s_errors.into()]).unwrap();
                Some(df.into_struct("".into()).into_series())
            } else {
                None
            }
        })
        .flatten()
        .collect();

    let out = StructChunked::from_series("spectrum_decomposition".into(), series_vec.iter().collect::<Vec<_>>().as_slice())?;
    Ok(out.into_series())
}

fn spectrum_decomposition_normalized_output(_fields: &[Field]) -> PolarsResult<Field> {
    let formula_field = Field::new("formulas".into(), DataType::List(Box::new(DataType::Array(Box::new(DataType::Int32), NUM_ELEMENTS))));
    let formula_str_field = Field::new("formulas_str".into(), DataType::List(Box::new(DataType::String)));
    let normalized_masses_field = Field::new("normalized_masses".into(), DataType::List(Box::new(DataType::Float64)));
    let intensities_field = Field::new("intensities".into(), DataType::List(Box::new(DataType::Float64)));
    let error_field = Field::new("errors".into(), DataType::List(Box::new(DataType::Float64)));
    let v = vec![formula_field, formula_str_field, normalized_masses_field, intensities_field, error_field];
    Ok(Field::new("spectrum_decomposition_normalized".into(), DataType::Struct(v)))
}

#[polars_expr(output_type_func=spectrum_decomposition_normalized_output)]
fn spectrum_decomposition_normalized(inputs: &[Series], kwargs: CleanAndNormalizeSpectrumKwargs) -> PolarsResult<Series> {
    let s = &inputs[0];
    let ca = s.struct_()?;

    let masses_series = ca.field_by_name("fragment_masses")?;
    let masses_ca = masses_series.list()?;
    let intensities_series = ca.field_by_name("fragment_intensities")?;
    let intensities_ca = intensities_series.list()?;
    let precursor_series = ca.field_by_name("precursor_formula")?;
    let precursor_ca = precursor_series.array()?;

    let series_vec: Vec<Series> = masses_ca.into_iter()
        .zip(intensities_ca)
        .zip(precursor_ca)
        .par_bridge()
        .map(|((masses_opt, intensities_opt), precursor_opt)| {
            if let (Some(masses_list), Some(intensities_list), Some(precursor_arr)) = (masses_opt, intensities_opt, precursor_opt) {
                let masses: Vec<f64> = masses_list.f64().unwrap().into_no_null_iter().collect();
                let intensities: Vec<f64> = intensities_list.f64().unwrap().into_no_null_iter().collect();
                let precursor_ca = precursor_arr.as_any().downcast_ref::<Int32Array>().unwrap();
                let precursor_sl: &[i32] = precursor_ca.values();
                let mut precursor_formula = [0; NUM_ELEMENTS];
                precursor_formula.copy_from_slice(precursor_sl);

                let params = SpectrumDecompositionParams {
                    tolerance_ppm: kwargs.tolerance_ppm,
                    min_dbe: kwargs.min_dbe,
                    max_dbe: kwargs.max_dbe,
                    dbe_mode: kwargs.dbe_mode.clone(),
                    water_absorption: kwargs.water_absorption,
                };

                let mut decomposer = SpectrumDecomposer::new();
                let result = decomposer.clean_and_normalize_spectrum_iterative(
                    &masses,
                    &intensities,
                    &precursor_formula,
                    &params,
                    kwargs.max_allowed_normalized_mass_error_ppm,
                    10, // max_iterations
                    1e-9, // convergence_tolerance
                );

                let mut formulas_builder = ListPrimitiveChunkedBuilder::<Int32Type>::new("formulas".into(), result.fragments.len(), NUM_ELEMENTS, DataType::Int32);
                let mut formulas_str_builder = StringChunkedBuilder::new("formulas_str".into(), result.fragments.len());
                let mut normalized_masses_builder = PrimitiveChunkedBuilder::<Float64Type>::new("normalized_masses".into(), result.fragments.len());
                let mut intensities_builder = PrimitiveChunkedBuilder::<Float64Type>::new("intensities".into(), result.fragments.len());
                let mut errors_builder = PrimitiveChunkedBuilder::<Float64Type>::new("errors".into(), result.fragments.len());

                for frag in result.fragments {
                    formulas_builder.append_slice(&frag.formula);
                    formulas_str_builder.append_value(formula_to_string(&frag.formula));
                    normalized_masses_builder.append_value(frag.normalized_mass);
                    intensities_builder.append_value(frag.intensity);
                    errors_builder.append_value(frag.error_ppm);
                }

                let s_formulas = formulas_builder.finish().into_series();
                let s_formulas_str = formulas_str_builder.finish().into_series();
                let s_normalized_masses = normalized_masses_builder.finish().into_series();
                let s_intensities = intensities_builder.finish().into_series();
                let s_errors = errors_builder.finish().into_series();

                let df = DataFrame::new(vec![s_formulas.into(), s_formulas_str.into(), s_normalized_masses.into(), s_intensities.into(), s_errors.into()]).unwrap();
                Some(df.into_struct("".into()).into_series())
            } else {
                None
            }
        })
        .flatten()
        .collect();

    let out = StructChunked::from_series("spectrum_decomposition_normalized".into(), series_vec.iter().collect::<Vec<_>>().as_slice())?;
    Ok(out.into_series())
}