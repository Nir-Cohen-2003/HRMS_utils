use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use crate::algorithms::MassDecomposer;
use crate::common::{DecompositionParams, NUM_ELEMENTS};
use polars::builders::ListBuilder;
use polars::chunked_array::builder::get_list_builder;

#[polars_expr(output_type=List(Array(Int32, NUM_ELEMENTS)))]
pub fn decompose_mass(inputs: &[Series], kwargs: DecompositionParams) -> PolarsResult<Series> {
    let mass_series = &inputs[0];
    let ca: &Float64Chunked = mass_series.f64()?;

    let min_bounds = kwargs.min_bounds;
    let max_bounds = kwargs.max_bounds;

    let mut list_builder: ListBuilder<FixedSizeListBuilder<Int32Type>> = get_list_builder(
        &DataType::Array(Box::new(DataType::Int32), NUM_ELEMENTS),
        ca.len(),
        ca.len() * 5, // initial capacity for inner list, assuming avg 5 formulas per mass
        "decomposed_formulas",
    )?;

    for opt_mass in ca.into_iter() {
        if let Some(mass) = opt_mass {
            let mut decomposer = MassDecomposer::new(min_bounds, max_bounds);
            let formulas = decomposer.decompose(mass, &kwargs);
            let mut formula_builder = FixedSizeListBuilder::new("formula", Int32Type, NUM_ELEMENTS, formulas.len());
            for formula in formulas {
                formula_builder.append_slice(&formula);
            }
            list_builder.append_builder(&mut formula_builder)?;
        } else {
            list_builder.append_null();
        }
    }

    let out = list_builder.finish();
    Ok(out.into_series())
}