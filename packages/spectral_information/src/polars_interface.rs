use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rayon::prelude::*;

use crate::algorithms::calculate_score_for_spectrum;

use serde::Deserialize;
use mass_decomposition::common::NUM_ELEMENTS;

#[derive(Deserialize, Debug)]
pub struct SpectralInfoKwargs {
    pub distance_metric: String,
    pub ignore_hydrogens: bool,
}


#[polars_expr(output_type=Float64)]
pub fn tree_spectral_info_score(
    inputs: &[Series],
    kwargs: SpectralInfoKwargs,
) -> PolarsResult<Series> {

    let struct_series = &inputs[0];
    let ca: &StructChunked = struct_series.struct_()?;

    let precursor_series = ca
        .field_by_name("precursor_formula")?
        .cast(&DataType::Array(Box::new(DataType::Float64),NUM_ELEMENTS))?;
    let fragments_series = ca
        .field_by_name("fragment_formulas")?
        .cast(&DataType::List(Box::new(DataType::Float64)))?;

    let precursors_ca = precursor_series.array()?;
    let fragments_ca = fragments_series.list()?;

    let precursor_vec : Vec<_> = precursors_ca.into_no_null_iter().collect();
    let fragments_vec : Vec<_> = fragments_ca.into_no_null_iter().collect();
    // let precursors = inputs[0].list()?;
    // let fragments = inputs[1].list()?;

    let distance_metric = kwargs.distance_metric;
    let ignore_hydrogens = kwargs.ignore_hydrogens;

    let indexed_scores: Vec<(usize, f64)> = precursor_vec
        .into_par_iter()
        .zip(fragments_vec.into_par_iter())
        .enumerate()
        .map(|(index, (precursor_s, fragments_s))| {      
            let precursor_ca: &ChunkedArray<Float64Type> = precursor_s.f64().unwrap();
            let precursor_vec: Vec<f64> = precursor_ca.into_no_null_iter().collect();
            let fragments_list: &ChunkedArray<ListType> = fragments_s.list().unwrap();
            let score: Option<f64>    = calculate_score_for_spectrum(
                precursor_vec,
                fragments_list.clone(),
                &distance_metric,
                ignore_hydrogens,
            );
            (index, score.unwrap_or(0.0))
        }
        )
        .collect();
    
    let mut sorted_results = indexed_scores;
    sorted_results.sort_unstable_by_key(|(idx, _)| *idx);
    let scores: Vec<f64> = sorted_results.into_iter().map(|(_, score)| score).collect();
    let score_series: Series = Series::new("score".into(), scores.clone());
    Ok(score_series)
}
