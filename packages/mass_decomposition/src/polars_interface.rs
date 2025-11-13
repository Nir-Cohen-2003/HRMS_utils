use polars::prelude::*;
use polars::datatypes::{DataType};
use pyo3_polars::derive::polars_expr;
use rayon::prelude::*;
use crate::algorithms::{MassDecomposer, SpectrumDecomposer};
use crate::common::{DecompositionParams, SpectrumDecompositionParams, formula_to_string, NUM_ELEMENTS, CleanAndNormalizeSpectrumKwargs,CleanedAndNormalizedSpectrumResult, DecompositionKwargs};
use polars::series::Series;
use polars_arrow::array::{Int32Array};
use std::collections::HashMap;
use std::sync::Arc;
use polars_arrow::array::PrimitiveArray;
use itertools::Itertools; // Add to Cargo.toml if not present



fn mass_decomposition_output(_fields: &[Field]) -> PolarsResult<Field> {
    let formula_field = Field::new("formulas".into(), DataType::List(Box::new(DataType::Array(Box::new(DataType::Int32), NUM_ELEMENTS))));
    let formula_str_field = Field::new("formulas_str".into(), DataType::List(Box::new(DataType::String)));
    let error_field = Field::new("errors_ppm".into(), DataType::List(Box::new(DataType::Float64)));
    let v = vec![formula_field, formula_str_field, error_field];
    Ok(Field::new("mass_decomposition".into(), DataType::Struct(v)))
}

#[polars_expr(output_type_func=mass_decomposition_output)]
fn mass_decomposition(inputs: &[Series], kwargs: DecompositionKwargs) -> PolarsResult<Series> {
    let masses = inputs[0].f64()?;
    let len = masses.len();
    let min_bounds: [i32; NUM_ELEMENTS] = kwargs.min_bounds.unwrap();
    let max_bounds: [i32; NUM_ELEMENTS] = kwargs.max_bounds.unwrap();

    let decomposer = Arc::new(MassDecomposer::new(min_bounds, max_bounds));
    
    let allow_half_integer = kwargs.dbe_mode == "half_integer";
    
    let params = Arc::new(DecompositionParams {
        tolerance_ppm: kwargs.tolerance_ppm,
        min_dbe: kwargs.min_dbe,
        max_dbe: kwargs.max_dbe,
        allow_half_integer,
    });

    // Get contiguous slice and process in parallel
    let masses_slice = masses.cont_slice().expect("masses should be contiguous");
    
    let mut indexed_results: Vec<(usize, Vec<[i32; NUM_ELEMENTS]>, Vec<f64>)> = masses_slice
        .par_iter()
        .enumerate()
        .map(|(idx, &mass)| {
            let (formulas, errors) = decomposer.decompose(mass, &params);
            (idx, formulas, errors)
        })
        .collect();

    // Sort by index to restore original order
    indexed_results.sort_unstable_by_key(|(idx, _, _)| *idx);

    // Build Series once after collecting all results
    let mut formulas_series_vec = Vec::with_capacity(len);
    let mut formulas_str_series_vec = Vec::with_capacity(len);
    let mut errors_series_vec = Vec::with_capacity(len);

    for (_idx, formulas, errors_ppm) in indexed_results {
        let formulas_vec: Vec<Option<Series>> = formulas
            .iter()
            .map(|formula| {
                let arr = Int32Array::from_slice(formula);
                let arr_boxed = Box::new(arr) as Box<dyn polars_arrow::array::Array>;
                Some(Series::try_from((PlSmallStr::EMPTY, arr_boxed)).unwrap())
            })
            .collect();
        
        let formulas_series = Series::new(
            PlSmallStr::from_static("formulas"), 
            formulas_vec
        ).cast(&DataType::Array(Box::new(DataType::Int32), NUM_ELEMENTS)).unwrap();

        let formulas_str: Vec<String> = formulas.iter()
            .map(|f| formula_to_string(f))
            .collect();

        formulas_series_vec.push(formulas_series);
        formulas_str_series_vec.push(Series::new("formulas_str".into(), formulas_str));
        errors_series_vec.push(Series::new("errors_ppm".into(), errors_ppm));
    }

    let out = StructChunked::from_series("mass_decomposition".into(), len, [
        &Series::new("formulas".into(), formulas_series_vec),
        &Series::new("formulas_str".into(), formulas_str_series_vec),
        &Series::new("errors_ppm".into(), errors_series_vec),
    ].iter().copied())?;
    
    Ok(out.into_series())
}

#[polars_expr(output_type_func=mass_decomposition_output)]
fn mass_decomposition_with_bounds(inputs: &[Series], kwargs: DecompositionKwargs) -> PolarsResult<Series> {
    let input_series = &inputs[0];
    let input_struct = input_series.struct_()?;
    let num_rows = input_struct.len();

    let mass_series = input_struct.field_by_name("mass")?;
    let min_bounds_series = input_struct.field_by_name("min_bounds")?;
    let max_bounds_series = input_struct.field_by_name("max_bounds")?;

    let masses_chunked = mass_series.rechunk();
    let min_bounds_chunked = min_bounds_series.rechunk();
    let max_bounds_chunked = max_bounds_series.rechunk();
    
    let masses_ca: &ChunkedArray<Float64Type> = masses_chunked.f64()?;
    let min_bounds_arrays: &ChunkedArray<FixedSizeListType> = min_bounds_chunked.array()?;
    let max_bounds_arrays: &ChunkedArray<FixedSizeListType> = max_bounds_chunked.array()?;

    let allow_half_integer = kwargs.dbe_mode == "half_integer";

    // Extract all data in parallel
    let bounds_data: Vec<_> = (0..num_rows)
        .into_par_iter()
        .map(|idx| {
            let mass = masses_ca.get(idx).unwrap();
            let min_bounds_arr = min_bounds_arrays.get(idx).unwrap();
            let max_bounds_arr = max_bounds_arrays.get(idx).unwrap();
            
            // Zero-copy slice access
            let min_bounds_values = min_bounds_arr
                .as_any()
                .downcast_ref::<PrimitiveArray<i32>>()
                .expect("min_bounds should be i32 array")
                .values();
            
            let max_bounds_values = max_bounds_arr
                .as_any()
                .downcast_ref::<PrimitiveArray<i32>>()
                .expect("max_bounds should be i32 array")
                .values();

            let mut min_bounds: [i32; NUM_ELEMENTS] = [0; NUM_ELEMENTS];
            let mut max_bounds: [i32; NUM_ELEMENTS] = [0; NUM_ELEMENTS];
            min_bounds.copy_from_slice(min_bounds_values);
            max_bounds.copy_from_slice(max_bounds_values);
            
            (idx, mass, (min_bounds, max_bounds))
        })
        .collect();

    // Group by bounds
    let mut bounds_to_data: HashMap<([i32; NUM_ELEMENTS], [i32; NUM_ELEMENTS]), Vec<(usize, f64)>> = HashMap::new();
    
    for (idx, mass, bounds) in bounds_data {
        bounds_to_data.entry(bounds)
            .or_insert_with(Vec::new)
            .push((idx, mass));
    }

    let params = Arc::new(DecompositionParams {
        tolerance_ppm: kwargs.tolerance_ppm,
        min_dbe: kwargs.min_dbe,
        max_dbe: kwargs.max_dbe,
        allow_half_integer,
    });
    
    // Process each unique bounds set in parallel
    let results_by_bounds: Vec<_> = bounds_to_data.into_par_iter()
        .flat_map(|((min_bounds, max_bounds), data)| {
            let decomposer = Arc::new(MassDecomposer::new(min_bounds, max_bounds));
            
            data.into_par_iter()
                .map(|(idx, mass)| {
                    let (formulas, errors) = decomposer.decompose(mass, &params);
                    (idx, formulas, errors)
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let mut indexed_results = results_by_bounds;
    indexed_results.sort_unstable_by_key(|(idx, _, _)| *idx);

    let mut formulas_series_vec = Vec::with_capacity(num_rows);
    let mut formulas_str_series_vec = Vec::with_capacity(num_rows);
    let mut errors_series_vec = Vec::with_capacity(num_rows);

    for (_idx, formulas, errors_ppm) in indexed_results {
        let formulas_vec: Vec<Option<Series>> = formulas
            .iter()
            .map(|formula| {
                let arr = Int32Array::from_slice(formula);
                let arr_boxed = Box::new(arr) as Box<dyn polars_arrow::array::Array>;
                Some(Series::try_from((PlSmallStr::EMPTY, arr_boxed)).unwrap())
            })
            .collect();
        
        let formulas_series = Series::new(
            PlSmallStr::from_static("formulas"), 
            formulas_vec
        ).cast(&DataType::Array(Box::new(DataType::Int32), NUM_ELEMENTS)).unwrap();

        let formulas_str: Vec<String> = formulas.iter()
            .map(|f| formula_to_string(f))
            .collect();

        formulas_series_vec.push(formulas_series);
        formulas_str_series_vec.push(Series::new("formulas_str".into(), formulas_str));
        errors_series_vec.push(Series::new("errors_ppm".into(), errors_ppm));
    }
    
    let out = StructChunked::from_series("mass_decomposition_with_bounds".into(), num_rows, [
        &Series::new("formulas".into(), formulas_series_vec),
        &Series::new("formulas_str".into(), formulas_str_series_vec),
        &Series::new("errors_ppm".into(), errors_series_vec),
    ].iter().copied())?;
    
    Ok(out.into_series())
}

fn spectrum_decomposition_normalized_output(_fields: &[Field]) -> PolarsResult<Field> {
    let formula_field = Field::new("formulas".into(), DataType::List(Box::new(DataType::Array(Box::new(DataType::Int32), NUM_ELEMENTS))));
    let formula_str_field = Field::new("formulas_str".into(), DataType::List(Box::new(DataType::String)));
    let normalized_masses_field = Field::new("normalized_masses".into(), DataType::List(Box::new(DataType::Float64)));
    let intensities_field = Field::new("intensities".into(), DataType::List(Box::new(DataType::Float64)));
    let error_field = Field::new("errors_ppm".into(), DataType::List(Box::new(DataType::Float64)));
    let v = vec![formula_field, formula_str_field, normalized_masses_field, intensities_field, error_field];
    Ok(Field::new("spectrum_decomposition_normalized".into(), DataType::Struct(v)))
}

#[polars_expr(output_type_func=spectrum_decomposition_normalized_output)]
fn spectrum_decomposition_normalized(inputs: &[Series], kwargs: CleanAndNormalizeSpectrumKwargs) -> PolarsResult<Series> {
    let s: &Series = &inputs[0];
    let ca: &ChunkedArray<StructType> = s.struct_()?;
    let len: usize = ca.len();

    // Don't rechunk - work with existing chunks directly
    let masses_series: Series = ca.field_by_name("mz")?;
    let intensities_series: Series = ca.field_by_name("intensities")?;
    let precursor_series: Series = ca.field_by_name("precursor_formula")?;

    let masses_ca: &ChunkedArray<ListType> = masses_series.list()?;
    let intensities_ca: &ChunkedArray<ListType> = intensities_series.list()?;
    let precursor_ca: &ChunkedArray<FixedSizeListType> = precursor_series.array()?;

    // Parallel extraction AND processing - work directly with Arrow arrays!
    let indexed_results: Vec<(usize, CleanedAndNormalizedSpectrumResult)> = (0..len)
        .into_par_iter()
        .filter_map(|idx| {
            
            // Get Arrow arrays directly
            let masses_list = masses_ca.get(idx)?;
            let intensities_list = intensities_ca.get(idx)?;
            let precursor_arr = precursor_ca.get(idx)?;
            
            // Extract f64 slices directly from Arrow PrimitiveArrays
            let masses_arr = masses_list
                .as_any()
                .downcast_ref::<PrimitiveArray<f64>>()?;
            let masses: &[f64] = masses_arr.values();
            
            let intensities_arr = intensities_list
                .as_any()
                .downcast_ref::<PrimitiveArray<f64>>()?;
            let intensities: &[f64] = intensities_arr.values();
            
            // Extract precursor formula directly
            let precursor_sl = precursor_arr
                .as_any()
                .downcast_ref::<PrimitiveArray<i32>>()?
                .values();
            
            let mut precursor_formula: [i32; NUM_ELEMENTS] = [0; NUM_ELEMENTS];
            precursor_formula.copy_from_slice(precursor_sl);
            
            let mut max_bounds = precursor_formula.clone();
            if kwargs.water_absorption {
                max_bounds[0] += 2;
                max_bounds[3] += 1;
            }
            let min_bounds = [0; NUM_ELEMENTS];
            
            let allow_half_integer = kwargs.dbe_mode == "half_integer";
            
            let params = SpectrumDecompositionParams {
                tolerance_ppm: kwargs.raw_fragment_tolerance_ppm,
                min_dbe: kwargs.min_dbe,
                max_dbe: kwargs.max_dbe,
                allow_half_integer,
                water_absorption: kwargs.water_absorption,
            };

            let decomposer = SpectrumDecomposer::new(min_bounds, max_bounds);
            
            let result = decomposer.clean_and_normalize_spectrum_iterative(
                masses,
                intensities,
                &params,
                kwargs.normalized_fragment_tolerance_ppm,
                10,
                1e-9,
            );
            
            Some((idx, result))
        })
        .collect();

    // Sort to restore order
    let mut sorted_results = indexed_results;
    sorted_results.sort_unstable_by_key(|(idx, _)| *idx);

    // Build Series in parallel
    let series_data: Vec<_> = sorted_results
        .into_par_iter()
        .map(|(_idx, result)| {
            let formulas_vec: Vec<Option<Series>> = result.fragments
                .iter()
                .map(|frag| {
                    let arr = Int32Array::from_slice(&frag.formula);
                    let arr_boxed = Box::new(arr) as Box<dyn polars_arrow::array::Array>;
                    Some(Series::try_from((PlSmallStr::EMPTY, arr_boxed)).unwrap())
                })
                .collect();
            
            let formulas_series = Series::new(
                PlSmallStr::from_static("formulas"), 
                formulas_vec
            ).cast(&DataType::Array(Box::new(DataType::Int32), NUM_ELEMENTS)).unwrap();

            let formulas_str: Vec<String> = result.fragments.iter()
                .map(|frag| formula_to_string(&frag.formula))
                .collect();
            
            let normalized_masses: Vec<f64> = result.fragments.iter()
                .map(|frag| frag.normalized_mass)
                .collect();
            
            let intensities: Vec<f64> = result.fragments.iter()
                .map(|frag| frag.intensity)
                .collect();
            
            let errors: Vec<f64> = result.fragments.iter()
                .map(|frag| frag.error_ppm)
                .collect();

            (
                formulas_series,
                Series::new("formulas_str".into(), formulas_str),
                Series::new("normalized_masses".into(), normalized_masses),
                Series::new("intensities".into(), intensities),
                Series::new("errors_ppm".into(), errors),
            )
        })
        .collect();

    let (formulas_series_vec, formulas_str_series_vec, normalized_masses_series_vec, intensities_series_vec, errors_series_vec): (Vec<_>, Vec<_>, Vec<_>, Vec<_>, Vec<_>) = 
        series_data.into_iter()
            .map(|(f, fs, nm, i, e)| (f, fs, nm, i, e))
            .multiunzip();

    let out = StructChunked::from_series("spectrum_decomposition_normalized".into(), len, [
        &Series::new("formulas".into(), formulas_series_vec),
        &Series::new("formulas_str".into(), formulas_str_series_vec),
        &Series::new("normalized_masses".into(), normalized_masses_series_vec),
        &Series::new("intensities".into(), intensities_series_vec),
        &Series::new("errors_ppm".into(), errors_series_vec),
    ].iter().copied())?;
    
    Ok(out.into_series())
}