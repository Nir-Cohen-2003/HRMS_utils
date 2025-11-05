
use polars::prelude::*;

use rayon::prelude::*;

use pyo3_polars::derive::polars_expr;

use crate::algorithms::{MassDecomposer, SpectrumDecomposer};

use crate::common::{DecompositionParams, DecompositionKwargs, SpectrumDecompositionParams, NUM_ELEMENTS, Formula, CleanAndNormalizeSpectrumKwargs};

use polars::chunked_array::builder::{get_list_builder, ListPrimitiveChunkedBuilder, PrimitiveChunkedBuilder};

use itertools::izip;



pub fn decompose_mass_output(_: &[Field]) -> PolarsResult<Field> {

    Ok(Field::new(

        "decomposed_formulas".into(),

        DataType::List(Box::new(DataType::Array(Box::new(DataType::Int32), NUM_ELEMENTS))),

    ))

}



#[polars_expr(output_type_func=decompose_mass_output)]

pub fn decompose_mass(inputs: &[Series], kwargs: DecompositionParams) -> PolarsResult<Series> {

    let mass_series = &inputs[0];

    let ca: &Float64Chunked = mass_series.f64()?;

    let min_bounds = kwargs.min_bounds;

    let max_bounds = kwargs.max_bounds;

    let mut list_builder = get_list_builder(

        &DataType::Array(Box::new(DataType::Int32), NUM_ELEMENTS),

        ca.len(),

        ca.len() * 5,

        "decomposed_formulas".into(),

    );

    let results: PolarsResult<Vec<_>> = ca.into_iter().par_bridge().map(|opt_mass| {

        if let Some(mass) = opt_mass {

            let mut decomposer = MassDecomposer::new(min_bounds, max_bounds);

            let formulas = decomposer.decompose(mass, &kwargs);

            let mut list_primitive_builder = ListPrimitiveChunkedBuilder::<Int32Type>::new(

                "formulas_for_mass".into(),

                formulas.len(),

                formulas.len() * NUM_ELEMENTS,

                DataType::Int32,

            );

            for formula in &formulas {

                list_primitive_builder.append_slice(formula);

            }

            let list_primitive_chunked = list_primitive_builder.finish();

            let list_series = list_primitive_chunked.into_series();

            let array_series = list_series.cast(&DataType::Array(Box::new(DataType::Int32), NUM_ELEMENTS))?;

            Ok(Some(array_series))

        } else {

            Ok(None)

        }

    }).collect();

    for res in results? {

        if let Some(series) = res {

            list_builder.append_series(&series)?;

        } else {

            list_builder.append_null();

        }

    }

    let out = list_builder.finish();

    Ok(out.into_series())

}



#[polars_expr(output_type_func=decompose_mass_output)]

pub fn decompose_mass_with_bounds_struct(inputs: &[Series], kwargs: DecompositionKwargs) -> PolarsResult<Series> {

    let struct_series = &inputs[0];

    let struct_ca = struct_series.struct_()?;

    let mass_series = struct_ca.field_by_name("mass")?;

    let min_bounds_series = struct_ca.field_by_name("min_bounds")?;

    let max_bounds_series = struct_ca.field_by_name("max_bounds")?;

    let mass_ca: &Float64Chunked = mass_series.f64()?;

    let min_bounds_ca: &ArrayChunked = min_bounds_series.array()?;

    let max_bounds_ca: &ArrayChunked = max_bounds_series.array()?;

    let mut list_builder = get_list_builder(

        &DataType::Array(Box::new(DataType::Int32), NUM_ELEMENTS),

        mass_ca.len(),

        mass_ca.len() * 5,

        "decomposed_formulas".into(),

    );

    let results: PolarsResult<Vec<_>> = izip!(mass_ca.into_iter(), min_bounds_ca.into_iter(), max_bounds_ca.into_iter())

        .par_bridge()

        .map(|(opt_mass, opt_min_bounds, opt_max_bounds)| {

            if let (Some(mass), Some(min_bounds_arr), Some(max_bounds_arr)) = (opt_mass, opt_min_bounds, opt_max_bounds) {

                let mut min_bounds: Formula = [0; NUM_ELEMENTS];

                let s_min = Series::from(min_bounds_arr);

                let ca_min = s_min.i32()?;

                ca_min.into_no_null_iter().enumerate().for_each(|(i, v)| {

                    min_bounds[i] = v;

                });

                let mut max_bounds: Formula = [0; NUM_ELEMENTS];

                let s_max = Series::from(max_bounds_arr);

                let ca_max = s_max.i32()?;

                ca_max.into_no_null_iter().enumerate().for_each(|(i, v)| {

                    max_bounds[i] = v;

                });

                let mut decomposer = MassDecomposer::new(min_bounds, max_bounds);

                let params = DecompositionParams {

                    tolerance_ppm: kwargs.tolerance_ppm,

                    min_dbe: kwargs.min_dbe,

                    max_dbe: kwargs.max_dbe,

                    dbe_mode: kwargs.dbe_mode.clone(),

                    min_bounds,

                    max_bounds,

                };

                let formulas = decomposer.decompose(mass, &params);

                let mut list_primitive_builder = ListPrimitiveChunkedBuilder::<Int32Type>::new(

                    "formulas_for_mass".into(),

                    formulas.len(),

                    formulas.len() * NUM_ELEMENTS,

                    DataType::Int32,

                );

                for formula in &formulas {

                    list_primitive_builder.append_slice(formula);

                }

                let list_primitive_chunked = list_primitive_builder.finish();

                let list_series = list_primitive_chunked.into_series();

                let array_series = list_series.cast(&DataType::Array(Box::new(DataType::Int32), NUM_ELEMENTS))?;

                Ok(Some(array_series))

            } else {

                Ok(None)

            }

        }).collect();

    for res in results? {

        if let Some(series) = res {

            list_builder.append_series(&series)?;

        } else {

            list_builder.append_null();

        }

    }

    let out = list_builder.finish();

    Ok(out.into_series())

}



pub fn decompose_spectrum_output(_: &[Field]) -> PolarsResult<Field> {

    Ok(Field::new(

        "decomposed_spectrum".into(),

        DataType::List(Box::new(DataType::List(Box::new(DataType::Array(Box::new(DataType::Int32), NUM_ELEMENTS))))),

    ))

}



#[polars_expr(output_type_func=decompose_spectrum_output)]

pub fn decompose_spectrum_with_precursor_struct(inputs: &[Series], kwargs: SpectrumDecompositionParams) -> PolarsResult<Series> {

    let struct_series = &inputs[0];

    let struct_ca = struct_series.struct_()?;

    let mz_series = struct_ca.field_by_name("mz")?;

    let precursor_formula_series = struct_ca.field_by_name("precursor_formula")?;

    let mz_ca: &ListChunked = mz_series.list()?;

    let precursor_formula_ca: &ArrayChunked = precursor_formula_series.array()?;

    let mut outer_list_builder = get_list_builder(

        &DataType::List(Box::new(DataType::Array(Box::new(DataType::Int32), NUM_ELEMENTS))),

        mz_ca.len(),

        mz_ca.len(),

        "decomposed_spectrum".into(),

    );

    let results: PolarsResult<Vec<_>> = mz_ca.into_iter().zip(precursor_formula_ca.into_iter())

        .par_bridge()

        .map(|(mz_values_opt, precursor_formula_opt)| {

            if let (Some(mz_values_s), Some(precursor_formula_arr)) = (mz_values_opt, precursor_formula_opt) {

                let mz_values_ca = mz_values_s.f64()?;

                let mz_values: Vec<f64> = mz_values_ca.into_no_null_iter().collect();

                let mut precursor_formula: Formula = [0; NUM_ELEMENTS];

                let s = Series::from(precursor_formula_arr);

                let ca = s.i32()?;

                ca.into_no_null_iter().enumerate().for_each(|(i, v)| {

                    precursor_formula[i] = v;

                });

                let mut decomposer = SpectrumDecomposer::new();

                let formulas_per_mz = decomposer.decompose_spectrum_with_precursor(&mz_values, &precursor_formula, &kwargs);

                let mut inner_list_builder = get_list_builder(

                    &DataType::Array(Box::new(DataType::Int32), NUM_ELEMENTS),

                    formulas_per_mz.len(),

                    formulas_per_mz.iter().map(|f| f.len()).sum::<usize>(),

                    "formulas_for_mz".into(),

                );

                for formulas in formulas_per_mz {

                    let mut list_primitive_builder = ListPrimitiveChunkedBuilder::<Int32Type>::new(

                        "formulas".into(),

                        formulas.len(),

                        formulas.len() * NUM_ELEMENTS,

                        DataType::Int32,

                    );

                    for formula in &formulas {

                        list_primitive_builder.append_slice(formula);

                    }

                    let list_primitive_chunked = list_primitive_builder.finish();

                    let list_series = list_primitive_chunked.into_series();

                    let array_series = list_series.cast(&DataType::Array(Box::new(DataType::Int32), NUM_ELEMENTS))?;

                    inner_list_builder.append_series(&array_series)?;

                }

                let inner_list_series = inner_list_builder.finish().into_series();

                Ok(Some(inner_list_series))

            } else {

                Ok(None)

            }

        }).collect();

    for res in results? {

        if let Some(series) = res {

            outer_list_builder.append_series(&series)?;

        } else {

            outer_list_builder.append_null();

        }

    }

    let out = outer_list_builder.finish();

    Ok(out.into_series())

}



pub fn clean_and_normalize_spectrum_output(_: &[Field]) -> PolarsResult<Field> {

    let formula_field = Field::new("formula".into(), DataType::Array(Box::new(DataType::Int32), NUM_ELEMENTS));

    let normalized_mass_field = Field::new("normalized_mass".into(), DataType::Float64);

    let intensity_field = Field::new("intensity".into(), DataType::Float64);

    let error_ppm_field = Field::new("error_ppm".into(), DataType::Float64);

    let v: Vec<Field> = vec![normalized_mass_field, intensity_field, formula_field, error_ppm_field];

    let st = DataType::Struct(v);

    Ok(Field::new("corrected_spectrum".into(), DataType::List(Box::new(st))))

}



#[polars_expr(output_type_func=clean_and_normalize_spectrum_output)]

pub fn clean_and_normalize_spectrum(inputs: &[Series], kwargs: CleanAndNormalizeSpectrumKwargs) -> PolarsResult<Series> {

    let struct_series = &inputs[0];

    let struct_ca = struct_series.struct_()?;

    let mz_series = struct_ca.field_by_name("mz")?;

    let intensity_series = struct_ca.field_by_name("intensity")?;

    let precursor_formula_series = struct_ca.field_by_name("precursor_formula")?;

    let mz_ca: &ListChunked = mz_series.list()?;

    let intensity_ca: &ListChunked = intensity_series.list()?;

    let precursor_formula_ca: &ArrayChunked = precursor_formula_series.array()?;

    let mut decomposer = SpectrumDecomposer::new();

    let params = SpectrumDecompositionParams {

        tolerance_ppm: kwargs.tolerance_ppm,

        min_dbe: kwargs.min_dbe,

        max_dbe: kwargs.max_dbe,

        dbe_mode: kwargs.dbe_mode.clone(),

        water_absorption: kwargs.water_absorption,

    };

    let mut list_builder = get_list_builder(

        &DataType::Struct(vec![

            Field::new("normalized_mass".into(), DataType::Float64),

            Field::new("intensity".into(), DataType::Float64),

            Field::new("formula".into(), DataType::Array(Box::new(DataType::Int32), NUM_ELEMENTS)),

            Field::new("error_ppm".into(), DataType::Float64),

        ]),

        mz_ca.len(),

        mz_ca.len() * 10,

        "corrected_spectrum".into(),

    );

    for (mz_values_opt, intensity_values_opt, precursor_formula_opt) in izip!(mz_ca.into_iter(), intensity_ca.into_iter(), precursor_formula_ca.into_iter()) {

        if let (Some(mz_values_s), Some(intensity_values_s), Some(precursor_formula_arr)) = (mz_values_opt, intensity_values_opt, precursor_formula_opt) {

            let mz_values_ca = mz_values_s.f64()?;

            let fragment_masses: Vec<f64> = mz_values_ca.into_no_null_iter().collect();

            let intensity_values_ca = intensity_values_s.f64()?;

            let fragment_intensities: Vec<f64> = intensity_values_ca.into_no_null_iter().collect();

            let mut precursor_formula: Formula = [0; NUM_ELEMENTS];

            let s = Series::from(precursor_formula_arr);

            let ca = s.i32()?;

            ca.into_no_null_iter().enumerate().for_each(|(i, v)| {

                precursor_formula[i] = v;

            });

            let result = decomposer.clean_and_normalize_spectrum_iterative(

                &fragment_masses,

                &fragment_intensities,

                &precursor_formula,

                &params,

                kwargs.max_allowed_normalized_mass_error_ppm,

                20,

                1e-9,

            );

            let mut s_buffer: Vec<Series> = Vec::with_capacity(4);

            let mut normalized_mass_builder = PrimitiveChunkedBuilder::<Float64Type>::new("normalized_mass".into(), result.fragments.len());

            let mut intensity_builder = PrimitiveChunkedBuilder::<Float64Type>::new("intensity".into(), result.fragments.len());

            let mut formula_builder = ListPrimitiveChunkedBuilder::<Int32Type>::new("formula".into(), result.fragments.len(), result.fragments.len() * NUM_ELEMENTS, DataType::Int32);

            let mut error_ppm_builder = PrimitiveChunkedBuilder::<Float64Type>::new("error_ppm".into(), result.fragments.len());

            for fragment in &result.fragments {

                normalized_mass_builder.append_value(fragment.normalized_mass);

                intensity_builder.append_value(fragment.intensity);

                formula_builder.append_slice(&fragment.formula);

                error_ppm_builder.append_value(fragment.error_ppm);

            }

            s_buffer.push(normalized_mass_builder.finish().into_series());

            s_buffer.push(intensity_builder.finish().into_series());

            s_buffer.push(formula_builder.finish().into_series().cast(&DataType::Array(Box::new(DataType::Int32), NUM_ELEMENTS))?);

            s_buffer.push(error_ppm_builder.finish().into_series());

            let st = StructChunked::from_series("fragment".into(), result.fragments.len(), s_buffer.iter())?;

            list_builder.append_series(&st.into_series())?;

        } else {

            list_builder.append_null();

        }

    }

    let out = list_builder.finish();

    Ok(out.into_series())

}
