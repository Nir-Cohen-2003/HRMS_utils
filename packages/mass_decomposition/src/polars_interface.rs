use polars::datatypes::Field;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use crate::algorithms::MassDecomposer;
use crate::common::{DecompositionParams, NUM_ELEMENTS};

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