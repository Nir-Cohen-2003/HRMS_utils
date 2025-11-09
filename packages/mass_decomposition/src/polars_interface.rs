use polars::prelude::*;
use polars::datatypes::{DataType, Int32Type, Float64Type};
use pyo3_polars::derive::polars_expr;
use rayon::prelude::*;
use crate::algorithms::{MassDecomposer, SpectrumDecomposer};
use crate::common::{DecompositionParams, SpectrumDecompositionParams, formula_to_string, NUM_ELEMENTS, CleanAndNormalizeSpectrumKwargs, DecompositionKwargs};
use polars::series::Series;
use polars::frame::DataFrame;
use polars::chunked_array::builder::{
    ListPrimitiveChunkedBuilder, PrimitiveChunkedBuilder, StringChunkedBuilder
};
use polars_arrow::array::{Int32Array, FixedSizeListArray};


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
    let len = masses.len();
    let min_bounds: [i32; 15] = kwargs.min_bounds.unwrap();
    let max_bounds: [i32; 15] = kwargs.max_bounds.unwrap();

    // Initialize decomposer once
    let mut decomposer = MassDecomposer::new(min_bounds, max_bounds);
    
    let params = DecompositionParams {
        tolerance_ppm: kwargs.tolerance_ppm,
        min_dbe: kwargs.min_dbe,
        max_dbe: kwargs.max_dbe,
        dbe_mode: kwargs.dbe_mode.clone(),
    };
    
    // Force initialization by doing a dummy decompose (or add explicit init method)
    let _ = decomposer.decompose(100.0, &params);

    // Process masses in parallel
    let results: Vec<(Series, Series, Series)> = masses
        .into_no_null_iter()
        .par_bridge()
        .map(|mass| {
            // Clone decomposer for each thread (shares read-only data, clones mutable state)
            let mut thread_decomposer = decomposer.clone();
            let (formulas, errors_ppm) = thread_decomposer.decompose(mass, &params);

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

            let mut formulas_str_builder = StringChunkedBuilder::new("formulas_str".into(), formulas.len());
            let mut errors_builder = PrimitiveChunkedBuilder::<Float64Type>::new("errors".into(), errors_ppm.len());

            for (formula, error) in formulas.iter().zip(errors_ppm.iter()) {
                formulas_str_builder.append_value(formula_to_string(formula));
                errors_builder.append_value(*error);
            }

            (
                formulas_series,
                formulas_str_builder.finish().into_series(),
                errors_builder.finish().into_series()
            )
        })
        .collect();

    // Unzip results
    let (formulas_series_vec, formulas_str_series_vec, errors_series_vec): (Vec<Series>, Vec<Series>, Vec<Series>) = 
        results.into_iter().fold(
            (Vec::with_capacity(len), Vec::with_capacity(len), Vec::with_capacity(len)),
            |(mut f, mut fs, mut e), (formula, formula_str, error)| {
                f.push(formula);
                fs.push(formula_str);
                e.push(error);
                (f, fs, e)
            }
        );

    let out = StructChunked::from_series("mass_decomposition".into(), len, [
        &Series::new("formulas".into(), formulas_series_vec),
        &Series::new("formulas_str".into(), formulas_str_series_vec),
        &Series::new("errors".into(), errors_series_vec),
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

    // 1. Downcast to appropriate chunked arrays
    let masses_ca = mass_series.f64()?;
    let min_bounds_arrays = min_bounds_series.array()?;
    let max_bounds_arrays = max_bounds_series.array()?;

    // 2. EFFICIENT STEP: Zip the iterators first
    let zipped_iter = masses_ca.into_iter()
        .zip(min_bounds_arrays.into_iter())
        .zip(max_bounds_arrays.into_iter());

    // 3. Process the zipped iterator in parallel
    // .par_bridge() converts a standard Iterator into a ParallelIterator
    let results: Vec<(Series, Series, Series)> = zipped_iter
        .par_bridge()
        .map(|((mass_opt, min_bounds_opt), max_bounds_opt)| {
            match (mass_opt, min_bounds_opt, max_bounds_opt) {
                (Some(mass), Some(min_bounds_arr), Some(max_bounds_arr)) => {
                    // Downcast to Int32Array
                    // print min_bounds_arr, and its type for debugging
                    // println!("min_bounds_arr: {:?}, type: {:?}", min_bounds_arr, min_bounds_arr.dtype());
                    
                    let min_bounds_values: &[i32] = min_bounds_arr.i32()
                        .expect("min_bounds should be i32 array")
                        .cont_slice()
                        .expect("min_bounds should be contiguous slice");
                    let max_bounds_values: &[i32] = max_bounds_arr.i32()
                        .expect("max_bounds should be i32 array")
                        .cont_slice()
                        .expect("max_bounds should be contiguous slice");
                    // now chek that we got result, pani

                    let mut min_bounds = [0; NUM_ELEMENTS];
                    let mut max_bounds = [0; NUM_ELEMENTS];
                    min_bounds.copy_from_slice(min_bounds_values);
                    max_bounds.copy_from_slice(max_bounds_values);

                    let params = DecompositionParams {
                        tolerance_ppm: kwargs.tolerance_ppm,
                        min_dbe: kwargs.min_dbe,
                        max_dbe: kwargs.max_dbe,
                        dbe_mode: kwargs.dbe_mode.clone(),
                    };
                    let mut decomposer = MassDecomposer::new(min_bounds, max_bounds);
                    let (formulas, errors_ppm) = decomposer.decompose(mass, &params);

                    // Build arrays directly for formulas
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

                    let mut formulas_str_builder = StringChunkedBuilder::new("formulas_str".into(), formulas.len());
                    let mut errors_builder = PrimitiveChunkedBuilder::<Float64Type>::new("errors".into(), errors_ppm.len());

                    for (formula, error) in formulas.iter().zip(errors_ppm.iter()) {
                        formulas_str_builder.append_value(formula_to_string(formula));
                        errors_builder.append_value(*error);
                    }

                    (
                        formulas_series,
                        formulas_str_builder.finish().into_series(),
                        errors_builder.finish().into_series()
                    )
                }
                _ => {
                    // Handle null input by creating empty lists with proper Array dtype
                    let empty_formulas = Series::new(PlSmallStr::from_static("formulas"), Vec::<Option<Series>>::new());
                    let empty_formulas_str = StringChunkedBuilder::new("formulas_str".into(), 0).finish().into_series();
                    let empty_errors = PrimitiveChunkedBuilder::<Float64Type>::new("errors".into(), 0).finish().into_series();
                    
                    (empty_formulas, empty_formulas_str, empty_errors)
                }
            }
        })
        .collect();

    // 4. Unzip results into separate vectors
    let (formulas_series_vec, formulas_str_series_vec, errors_series_vec): (Vec<Series>, Vec<Series>, Vec<Series>) = 
        results.into_iter().fold(
            (Vec::with_capacity(num_rows), Vec::with_capacity(num_rows), Vec::with_capacity(num_rows)),
            |(mut f, mut fs, mut e), (formula, formula_str, error)| {
                f.push(formula);
                fs.push(formula_str);
                e.push(error);
                (f, fs, e)
            }
        );
    
    let out = StructChunked::from_series("mass_decomposition_with_bounds".into(), num_rows, [
        &Series::new("formulas".into(), formulas_series_vec),
        &Series::new("formulas_str".into(), formulas_str_series_vec),
        &Series::new("errors".into(), errors_series_vec),
    ].iter().copied())?;
    
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
    let len = ca.len();

    let masses_series = ca.field_by_name("fragment_masses")?;
    let masses_ca = masses_series.list()?;
    let intensities_series = ca.field_by_name("fragment_intensities")?;
    let intensities_ca = intensities_series.list()?;
    let precursor_series = ca.field_by_name("precursor_formula")?;
    let precursor_ca = precursor_series.array()?;

    let mut formulas_series_vec: Vec<Series> = Vec::with_capacity(len);
    let mut formulas_str_series_vec: Vec<Series> = Vec::with_capacity(len);
    let mut normalized_masses_series_vec: Vec<Series> = Vec::with_capacity(len);
    let mut intensities_series_vec: Vec<Series> = Vec::with_capacity(len);
    let mut errors_series_vec: Vec<Series> = Vec::with_capacity(len);

    for ((masses_opt, intensities_opt), precursor_opt) in masses_ca.into_iter().zip(intensities_ca).zip(precursor_ca) {
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
                10, // Why: 10 iterations is sufficient for mass calibration convergence in typical HRMS data
                1e-9, // Why: Tight convergence tolerance ensures accurate mass calibration fit
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

            formulas_series_vec.push(formulas_builder.finish().into_series());
            formulas_str_series_vec.push(formulas_str_builder.finish().into_series());
            normalized_masses_series_vec.push(normalized_masses_builder.finish().into_series());
            intensities_series_vec.push(intensities_builder.finish().into_series());
            errors_series_vec.push(errors_builder.finish().into_series());
        } else {
            let empty_formulas = ListPrimitiveChunkedBuilder::<Int32Type>::new("formulas".into(), 0, NUM_ELEMENTS, DataType::Int32).finish().into_series();
            let empty_formulas_str = StringChunkedBuilder::new("formulas_str".into(), 0).finish().into_series();
            let empty_normalized_masses = PrimitiveChunkedBuilder::<Float64Type>::new("normalized_masses".into(), 0).finish().into_series();
            let empty_intensities = PrimitiveChunkedBuilder::<Float64Type>::new("intensities".into(), 0).finish().into_series();
            let empty_errors = PrimitiveChunkedBuilder::<Float64Type>::new("errors".into(), 0).finish().into_series();
            
            formulas_series_vec.push(empty_formulas);
            formulas_str_series_vec.push(empty_formulas_str);
            normalized_masses_series_vec.push(empty_normalized_masses);
            intensities_series_vec.push(empty_intensities);
            errors_series_vec.push(empty_errors);
        }
    }

    let out = StructChunked::from_series("spectrum_decomposition_normalized".into(), len, [
        &Series::new("formulas".into(), formulas_series_vec),
        &Series::new("formulas_str".into(), formulas_str_series_vec),
        &Series::new("normalized_masses".into(), normalized_masses_series_vec),
        &Series::new("intensities".into(), intensities_series_vec),
        &Series::new("errors".into(), errors_series_vec),
    ].iter().copied())?;
    
    Ok(out.into_series())
}