use polars::prelude::*;
use polars::datatypes::{DataType};
use pyo3_polars::derive::polars_expr;
use rayon::prelude::*;
use crate::algorithms::{MassDecomposer, SpectrumDecomposer};
use crate::common::{DecompositionParams, SpectrumDecompositionParams, formula_to_string, NUM_ELEMENTS, CleanAndNormalizeSpectrumKwargs,CleanedAndNormalizedSpectrumResult, DecompositionKwargs};
use polars::series::Series;
use polars_arrow::array::{Int32Array};

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

    let decomposer = MassDecomposer::new(min_bounds, max_bounds);
    
    let params = DecompositionParams {
        tolerance_ppm: kwargs.tolerance_ppm,
        min_dbe: kwargs.min_dbe,
        max_dbe: kwargs.max_dbe,
        dbe_mode: kwargs.dbe_mode.clone(),
    };

    // Process masses in parallel with indices to preserve order
    let mut indexed_results: Vec<(usize, Vec<[i32; NUM_ELEMENTS]>, Vec<f64>)> = masses
        .into_no_null_iter()
        .enumerate()
        .par_bridge()
        .map(|(idx, mass)| {
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
        // Build formulas series
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

        // Build strings and errors
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
    // 1. Downcast to appropriate chunked arrays
    let masses_ca: &ChunkedArray<Float64Type> = masses_chunked.f64()?;
    let min_bounds_arrays: &ChunkedArray<FixedSizeListType> = min_bounds_chunked.array()?;
    let max_bounds_arrays: &ChunkedArray<FixedSizeListType> = max_bounds_chunked.array()?;

    // 2. Zip and enumerate to preserve order
    let zipped_iter = masses_ca.into_no_null_iter()
        .zip(min_bounds_arrays.into_no_null_iter())
        .zip(max_bounds_arrays.into_no_null_iter())
        .enumerate();

    // 3. Process in parallel with indices
    let mut indexed_results: Vec<(usize, Vec<[i32; NUM_ELEMENTS]>, Vec<f64>)> = zipped_iter
        .par_bridge()
        .map(|(idx, ((mass, min_bounds_arr), max_bounds_arr))| {
            let min_bounds_values: &[i32] = min_bounds_arr.i32()
                .expect("min_bounds should be i32 array")
                .cont_slice()
                .expect("min_bounds should be contiguous slice");
            let max_bounds_values: &[i32] = max_bounds_arr.i32()
                .expect("max_bounds should be i32 array")
                .cont_slice()
                .expect("max_bounds should be contiguous slice");

            let mut min_bounds: [i32; NUM_ELEMENTS] = [0; NUM_ELEMENTS];
            let mut max_bounds: [i32; NUM_ELEMENTS] = [0; NUM_ELEMENTS];
            min_bounds.copy_from_slice(min_bounds_values);
            max_bounds.copy_from_slice(max_bounds_values);

            let params = DecompositionParams {
                tolerance_ppm: kwargs.tolerance_ppm,
                min_dbe: kwargs.min_dbe,
                max_dbe: kwargs.max_dbe,
                dbe_mode: kwargs.dbe_mode.clone(),
            };
            
            let decomposer = MassDecomposer::new(min_bounds, max_bounds);
            let (formulas, errors) = decomposer.decompose(mass, &params);
            (idx, formulas, errors)
        })
        .collect();

    // 4. Sort by index to restore original order
    indexed_results.sort_unstable_by_key(|(idx, _, _)| *idx);

    // 5. Build Series in correct order
    let mut formulas_series_vec = Vec::with_capacity(num_rows);
    let mut formulas_str_series_vec = Vec::with_capacity(num_rows);
    let mut errors_series_vec = Vec::with_capacity(num_rows);

    for (_idx, formulas, errors_ppm) in indexed_results {
        // Build formulas series
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

        // Build strings and errors
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

    let masses_series: Series = ca.field_by_name("mz")?;
    let intensities_series: Series = ca.field_by_name("intensities")?;
    let precursor_series: Series = ca.field_by_name("precursor_formula")?;

    let masses_chunked: Series = masses_series.rechunk();
    let intensities_chunked: Series = intensities_series.rechunk();
    let precursor_chunked: Series = precursor_series.rechunk();

    let masses_ca: &ChunkedArray<ListType> = masses_chunked.list()?;
    let intensities_ca: &ChunkedArray<ListType> = intensities_chunked.list()?;
    let precursor_ca: &ChunkedArray<FixedSizeListType> = precursor_chunked.array()?;

    let zipped_iter = masses_ca.into_no_null_iter()
        .zip(intensities_ca.into_no_null_iter())
        .zip(precursor_ca.into_no_null_iter())
        .enumerate();  // Add enumeration

    let mut indexed_results: Vec<(usize, CleanedAndNormalizedSpectrumResult)> = zipped_iter
        .par_bridge()
        .map(|(idx, ((masses_list, intensities_list), precursor_arr))| {  // Add idx parameter
            let masses_ca = masses_list.f64().expect("masses should be f64 list");
            let intensities_ca = intensities_list.f64().expect("intensities should be f64 list");

            let masses: &[f64] = masses_ca.cont_slice().expect("masses should be contiguous slice");
            let intensities: &[f64] = intensities_ca.cont_slice().expect("intensities should be contiguous slice");
            
            let precursor_sl= precursor_arr.i32().expect("precursor_formula should be i32 array").downcast_as_array().values();
            
            let mut precursor_formula: [i32; NUM_ELEMENTS] = [0; NUM_ELEMENTS];
            precursor_formula.copy_from_slice(precursor_sl);
            let precursor_formula: [i32; NUM_ELEMENTS] = precursor_formula;

            let params = SpectrumDecompositionParams {
                tolerance_ppm: kwargs.raw_fragment_tolerance_ppm,
                min_dbe: kwargs.min_dbe,
                max_dbe: kwargs.max_dbe,
                dbe_mode: kwargs.dbe_mode.clone(),
                water_absorption: kwargs.water_absorption,
            };

            let decomposer = SpectrumDecomposer::new();
            let result = decomposer.clean_and_normalize_spectrum_iterative(
                &masses,
                &intensities,
                &precursor_formula,
                &params,
                kwargs.normalized_fragment_tolerance_ppm,
                10,
                1e-9,
            );
            
            (idx, result)  // Return tuple with index
        })
        .collect();

    // Sort by index to restore original order
    indexed_results.sort_unstable_by_key(|(idx, _)| *idx);

    // Build Series once after collecting all results
    let mut formulas_series_vec: Vec<Series> = Vec::with_capacity(len);
    let mut formulas_str_series_vec: Vec<Series> = Vec::with_capacity(len);
    let mut normalized_masses_series_vec: Vec<Series> = Vec::with_capacity(len);
    let mut intensities_series_vec: Vec<Series> = Vec::with_capacity(len);
    let mut errors_series_vec: Vec<Series> = Vec::with_capacity(len);

    for (_idx, result) in indexed_results {  // Iterate over sorted results
        // Build formulas series
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

        // Build other series
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

        formulas_series_vec.push(formulas_series);
        formulas_str_series_vec.push(Series::new("formulas_str".into(), formulas_str));
        normalized_masses_series_vec.push(Series::new("normalized_masses".into(), normalized_masses));
        intensities_series_vec.push(Series::new("intensities".into(), intensities));
        errors_series_vec.push(Series::new("errors_ppm".into(), errors));
    }

    let out = StructChunked::from_series("spectrum_decomposition_normalized".into(), len, [
        &Series::new("formulas".into(), formulas_series_vec),
        &Series::new("formulas_str".into(), formulas_str_series_vec),
        &Series::new("normalized_masses".into(), normalized_masses_series_vec),
        &Series::new("intensities".into(), intensities_series_vec),
        &Series::new("errors_ppm".into(), errors_series_vec),
    ].iter().copied())?;
    
    Ok(out.into_series())
}