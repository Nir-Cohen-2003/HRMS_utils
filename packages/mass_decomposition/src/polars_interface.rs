use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use crate::algorithms::{MassDecomposer, SpectrumDecomposer};
use crate::common::{DecompositionParams, SpectrumDecompositionParams, NUM_ELEMENTS, Formula};
use polars::chunked_array::builder::{get_list_builder, ListPrimitiveChunkedBuilder};


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
        ca.len() * 5, // initial capacity for inner list, assuming avg 5 formulas per mass
        "decomposed_formulas".into(),
    );

    for opt_mass in ca.into_iter() {
        if let Some(mass) = opt_mass {
            let mut decomposer = MassDecomposer::new(min_bounds, max_bounds);
            let formulas = decomposer.decompose(mass, &kwargs);

            // 1. Build a List(Int32) series for the formulas of this mass
            let mut list_primitive_builder = ListPrimitiveChunkedBuilder::<Int32Type>::new(
                "formulas_for_mass".into(),
                formulas.len(), // capacity
                formulas.len() * NUM_ELEMENTS, // values capacity
                DataType::Int32,
            );
            for formula in &formulas {
                list_primitive_builder.append_slice(formula);
            }
            let list_primitive_chunked = list_primitive_builder.finish();
            let list_series = list_primitive_chunked.into_series();

            // 2. Cast to Array(Int32, NUM_ELEMENTS)
            let array_series = list_series.cast(&DataType::Array(Box::new(DataType::Int32), NUM_ELEMENTS))?;

            // 3. Append to the main list_builder
            list_builder.append_series(&array_series)?;
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

    let mz_iter = mz_ca.into_iter();
    let precursor_iter = precursor_formula_ca.into_iter();

    for (mz_values_opt, precursor_formula_opt) in mz_iter.zip(precursor_iter) {
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
            outer_list_builder.append_series(&inner_list_series)?;

        } else {
            outer_list_builder.append_null();
        }
    }

    let out = outer_list_builder.finish();
    Ok(out.into_series())
}