from mass_decomposition_impl.mass_decomposer_cpp import (
    decompose_mass_parallel,
    decompose_mass_parallel_per_bounds,
    decompose_spectra_parallel,
    decompose_spectra_parallel_per_bounds,
    decompose_spectra_known_precursor_parallel, 
)
import polars as pl
import numpy as np
from typing import Iterable

def decompose_mass(
    mass_series:np.ndarray,
    min_bounds: np.ndarray | pl.Series,
    max_bounds: np.ndarray | pl.Series,
    tolerance_ppm: float = 5.0,
    min_dbe: float = 0.0,
    max_dbe: float = 40.0,
    max_hetero_ratio: float = 100.0,
    max_results: int = 100000,
):
    """
    Wrapper for decompose_mass_parallel. Handles both fixed and per-mass bounds.
    If min_bounds and max_bounds are 1D arrays, applies the same bounds to all masses.
    If they are Polars Series/DataFrame columns, applies per-mass bounds.
    Example usage:
    Unifrom bounds:
    min_formula = np.zeros(15, dtype=np.int32)
    max_formula = np.array([100,0,40,20,10,5,2,1,0,0,0,0,0,0,0], dtype=np.int32)            

    df = pl.DataFrame({
        "mass": [100.0, 200.0, 300.0, 400.0, 500.0]})       
    df = df.with_columns(pl.col("mass").map_batches(
        lambda x: decompose_mass(mass_series=x, min_bounds=min_formula, max_bounds=max_formula, tolerance_ppm=5.0, min_dbe=0.0, max_dbe=40.0),
        return_dtype=pl.List(pl.Array(pl.Int32, 15))
    ).alias("decomposed_formulas"))

    Per-mass bounds:
    min_formula = np.zeros(15, dtype=np.int32)
    max_formula = np.array([100,0,40,20,10,5,2,1,0,0,0,0,0,0,0], dtype=np.int32)
    df = pl.DataFrame({
        "mass": [100.0, 200.0, 300.0, 400.0, 500.0],
        "min_bounds": [min_formula] * 5,
        "max_bounds": [max_formula] * 5
    })  
    df = df.with_columns(
        pl.struct(["mass", "min_bounds", "max_bounds"]).map_batches(
            lambda row: decompose_mass(
                mass_series=row.struct.field("mass"),
                min_bounds=row.struct.field("min_bounds"),
                max_bounds=row.struct.field("max_bounds"),
            ),
            return_dtype=pl.List(pl.Array(pl.Int32, 15))
        ).alias("decomposed_formulas")
    )
    """
    mass_array = mass_series
    # Case 1: fixed bounds (1D numpy arrays)
    if isinstance(min_bounds, np.ndarray) and min_bounds.ndim == 1:
        results = decompose_mass_parallel(
            target_masses=mass_array,
            min_bounds=min_bounds,
            max_bounds=max_bounds,
            tolerance_ppm=tolerance_ppm,
            min_dbe=min_dbe,
            max_dbe=max_dbe,
            max_hetero_ratio=max_hetero_ratio,
            max_results=max_results,
        )
        return results
        # Output: list of lists of np.ndarray
        return pl.Series(results, dtype=pl.List(pl.Array(pl.Int32, len(min_bounds))))
    # elif isinstance(min_bounds, np.ndarray) and isinstance(max_bounds, np.ndarray):
    #     # so we have 2D arrays, we assume they are per-mass bounds
    #     if min_bounds.shape[0] != max_bounds.shape[0]:
    #         raise ValueError("min_bounds and max_bounds must have the same number of rows for per-mass bounds.")
    #     if min_bounds.shape[1] != max_bounds.shape[1]:
    #         raise ValueError("min_bounds and max_bounds must have the same number of columns for per-mass bounds.")
    #     per_mass_bounds = list(zip(min_bounds, max_bounds))
    #     mass_array = mass_series.to_numpy()
    #     results = decompose_mass_parallel_per_bounds(
    #         target_masses=mass_array,
    #         per_mass_bounds=per_mass_bounds,
    #         tolerance_ppm=tolerance_ppm,
    #         min_dbe=min_dbe,
    #         max_dbe=max_dbe,
    #         max_hetero_ratio=max_hetero_ratio,
    #         max_results=max_results,
    #     )
    # Case 2: per-mass bounds (Polars Series or DataFrame columns)
    elif isinstance(min_bounds, pl.Series| pl.Expr) and isinstance(max_bounds, pl.Series| pl.Expr):
        min_bounds_np = min_bounds.to_numpy()
        max_bounds_np = max_bounds.to_numpy()
        per_mass_bounds = list(zip(min_bounds_np, max_bounds_np))
        results = decompose_mass_parallel_per_bounds(
            target_masses=mass_array,
            per_mass_bounds=per_mass_bounds,
            tolerance_ppm=tolerance_ppm,
            min_dbe=min_dbe,
            max_dbe=max_dbe,
            max_hetero_ratio=max_hetero_ratio,
            max_results=max_results,
        )
        return pl.Series(results, dtype=pl.List(pl.Array(pl.Int32, min_bounds_np.shape[1])))
    else:
        raise ValueError("min_bounds and max_bounds must both be either 1D numpy arrays or Polars Series of arrays, but we got types: " \
        f"min_bounds: {type(min_bounds)}, max_bounds: {type(max_bounds)}")

def decompose_spectra(
    precursor_mass_series: pl.Series,
    fragment_masses_series: pl.Series,
    min_bounds,
    max_bounds,
    tolerance_ppm: float = 5.0,
    min_dbe: float = 0.0,
    max_dbe: float = 40.0,
    max_hetero_ratio: float = 100.0,
    max_results: int = 100000,
):
    """
    NOT IMPLEMENTED YET, DO NOT USE THIS FUNCTION.
    Wrapper for spectrum decomposition (unknown precursor).
    Handles both uniform and per-spectrum bounds.

    Example usage:
    Uniform bounds:
    min_formula = np.zeros(15, dtype=np.int32)
    max_formula = np.array([100,0,40,20,10,5,2,1,0,0,0,0,0,0,0], dtype=np.int32)

    df = pl.DataFrame({
        "precursor_mass": [500.0, 600.0],
        "fragment_masses": [[100.0, 200.0, 300.0], [150.0, 250.0, 350.0]]
    })
    df = df.with_columns(
        decompose_spectra(
            precursor_mass_series=pl.col("precursor_mass"),
            fragment_masses_series=pl.col("fragment_masses"),
            min_bounds=min_formula,
            max_bounds=max_formula,
            tolerance_ppm=5.0,
            min_dbe=0.0,
            max_dbe=40.0,
        ).alias("decomposed_spectra")
    )

    Per-spectrum bounds:
    min_formula = np.zeros(15, dtype=np.int32)
    max_formula = np.array([100,0,40,20,10,5,2,1,0,0,0,0,0,0,0], dtype=np.int32)
    df = pl.DataFrame({
        "precursor_mass": [500.0, 600.0],
        "fragment_masses": [[100.0, 200.0, 300.0], [150.0, 250.0, 350.0]],
        "min_bounds": [min_formula, min_formula],
        "max_bounds": [max_formula, max_formula]
    })
    df = df.with_columns(
        decompose_spectra(
            precursor_mass_series=pl.col("precursor_mass"),
            fragment_masses_series=pl.col("fragment_masses"),
            min_bounds=pl.col("min_bounds"),
            max_bounds=pl.col("max_bounds"),
            tolerance_ppm=5.0,
            min_dbe=0.0,
            max_dbe=40.0,
        ).alias("decomposed_spectra")
    )
    """
    raise NotImplementedError("This function is not implemented yet. docmpose the mass of the precursor, explode each option to different rows, and then use decompose_spectra_known_precursor instead.")
    precursor_masses = precursor_mass_series.to_numpy()
    fragment_masses_list = fragment_masses_series.to_list()
    # Uniform bounds
    if isinstance(min_bounds, np.ndarray) and min_bounds.ndim == 1:
        spectra_data = [
            {"precursor_mass": pm, "fragment_masses": fm}
            for pm, fm in zip(precursor_masses, fragment_masses_list)
        ]
        from mass_decomposition_impl.mass_decomposer_cpp import decompose_spectra_parallel
        results = decompose_spectra_parallel(
            spectra_data=spectra_data,
            min_bounds=min_bounds,
            max_bounds=max_bounds,
            tolerance_ppm=tolerance_ppm,
            min_dbe=min_dbe,
            max_dbe=max_dbe,
            max_hetero_ratio=max_hetero_ratio,
            max_results=max_results,
        )
    # Per-spectrum bounds
    elif isinstance(min_bounds, pl.Series) and isinstance(max_bounds, pl.Series):
        min_bounds_np = min_bounds.to_numpy()
        max_bounds_np = max_bounds.to_numpy()
        spectra_data = [
            {
                "precursor_mass": pm,
                "fragment_masses": fm,
                "min_bounds": min_b,
                "max_bounds": max_b,
            }
            for pm, fm, min_b, max_b in zip(
                precursor_masses, fragment_masses_list, min_bounds_np, max_bounds_np
            )
        ]
        
        results = decompose_spectra_parallel_per_bounds(
            spectra_data=spectra_data,
            tolerance_ppm=tolerance_ppm,
            min_dbe=min_dbe,
            max_dbe=max_dbe,
            max_hetero_ratio=max_hetero_ratio,
            max_results=max_results,
        )
    else:
        raise ValueError("min_bounds and max_bounds must both be either 1D numpy arrays or Polars Series of arrays.")
    # now, the type of the ruslts hsould be:
    # List[List[Dict]]
    # and the dict is of the format:
    #
    #     {
    #     'precursor': np.ndarray, # Shape: (NUM_ELEMENTS,), dtype: int32
    #     'precursor_mass': float, # Calculated mass of the precursor
    #     'precursor_error_ppm': float, # PPM error for precursor mass
    #     'fragments': List[List[np.ndarray]], # Nested structure of fragment formulas
    #     'fragment_masses': List[List[float]], # Corresponding calculated masses
    #     'fragment_errors_ppm': List[List[float]] # Corresponding PPM errors
    # }
    # now we print the format of the results
    # results should be  a list of lists of np.ndarray
    # but we only care its an iterable, not necessarily a list
    assert isinstance(results, Iterable), f"Results should be an iterable, but got {type(results)}"
    #now we care the same about the first element of results
    assert isinstance(results[0], Iterable), f"First entry (and all entries, but we check only the first) of results should be an iterable, but got {type(results[0])}"
    # and the first element of the first entry should be a dict
    assert isinstance(results[0][0], dict), f"First entry of the first result should be a dict, but got {type(results[0][0])}"
    # now validate each key, and the type nad shape of the values
    assert "precursor" in results[0][0].keys(), f"First entry of the first result should contain 'precursor' key, but got keys: {list(results[0][0].keys())}"
    assert isinstance(results[0][0]["precursor"], np.ndarray), f"Value of 'precursor' should be a numpy array, but got {type(results[0][0]['precursor'])}"
    assert results[0][0]["precursor"].dtype == np.int32, f"Value of 'precursor' should be a numpy array of int32, but got dtype {results[0][0]['precursor'].dtype}"
    assert results[0][0]["precursor"].shape[0] == len(min_bounds), f"Shape of 'precursor' should match the length of min_bounds, which is {len(min_bounds)}, but got {results[0][0]['precursor'].shape[0]}"
    assert "precursor_mass" in results[0][0].keys(), f"First entry of the first result should contain 'precursor_mass' key, but got keys: {list(results[0][0].keys())}"
    assert isinstance(results[0][0]["precursor_mass"], float), f"Value of 'precursor_mass' should be a float, but got {type(results[0][0]['precursor_mass'])}"
    assert "precursor_error_ppm" in results[0][0].keys(), f"First entry of the first result should contain 'precursor_error_ppm' key, but got keys: {list(results[0][0].keys())}"
    assert isinstance(results[0][0]["precursor_error_ppm"], float), f"Value of 'precursor_error_ppm' should be a float, but got {type(results[0][0]['precursor_error_ppm'])}"
    assert "fragments" in results[0][0].keys(), f"First entry of the first result should contain 'fragments' key, but got keys: {list(results[0][0].keys())}"
    assert isinstance(results[0][0]["fragments"], Iterable), f"Value of 'fragments' should be a list, but is {type(results[0][0]['fragments'])}"
    assert isinstance(results[0][0]["fragments"][0], Iterable), f"Each entry in 'fragments' should be a list, but got {type(results[0][0]['fragments'][0])} for the first fragment"
    ##### note that we might have no fragments retuned
    # check that each fragment is a numpy array
    if len(results[0][0]["fragments"][0]) > 0:
        for idx, f in enumerate(results[0][0]["fragments"][0]):
            assert isinstance(f, np.ndarray), f"Fragment at index {idx} should be a numpy array, but got {type(f)}"

    return pl.Series(
        results, 
        dtype=pl.List(
                pl.Struct(
                {
                    "precursor": pl.Array(pl.Int32, len(min_bounds)),  # Precursor formula as an array
                    "precursor_mass": pl.Float64,  # Calculated mass of the precursor
                    "precursor_error_ppm": pl.Float64,  # PPM error for precursor   
                    "fragments": pl.List(pl.List(pl.Array(pl.Int32, len(min_bounds)))),  # Nested structure of fragment formulas
                    "fragment_masses": pl.List(pl.List(pl.Float64)),  # Corresponding calculated masses
                    "fragment_errors_ppm": pl.List(pl.List(pl.Float64)),  # Correspond
                }
                )
            ),
            strict=False
        )


def decompose_spectra_known_precursor(
    precursor_formula_series: pl.Series,
    fragment_masses_series: pl.Series,
    min_bounds,
    max_bounds,
    tolerance_ppm: float = 5.0,
    max_results: int = 100000,
):
    """
    Wrapper for spectrum decomposition with known precursor formulas.
    Only supports uniform bounds.
    df = df.with_columns(
        pl.col("precursor_mass").map_batches(
            lambda x: decompose_mass(
                mass_series=x,
                min_bounds=min_formula,     
                max_bounds=max_formula,
                tolerance_ppm=5.0,
                min_dbe=0.0,
                max_dbe=40.0,
            ),
            return_dtype=pl.List(pl.Array(pl.Int32, len(min_formula)))
        ).alias("decomposed_formula")
    )
    df = df.explode("decomposed_formula")
    df = df.with_columns(
        pl.struct(["fragment_masses", "decomposed_formula"]).map_batches(
            lambda row: decompose_spectra_known_precursor(
                precursor_formula_series=row.struct.field("decomposed_formula"),
                fragment_masses_series=row.struct.field("fragment_masses"),
                min_bounds=min_formula,
                max_bounds=max_formula,
                tolerance_ppm=5.0,
            )).alias("decomposed_spectra")
        )
    
    """
    precursor_formulas = precursor_formula_series.to_numpy()
    fragment_masses_list = fragment_masses_series.to_numpy()
    spectra_data = [
        {"precursor_formula": pf, "fragment_masses": fm}
        for pf, fm in zip(precursor_formulas, fragment_masses_list)
    ]
    results = decompose_spectra_known_precursor_parallel(
        spectra_data=spectra_data,
        min_bounds=min_bounds,
        max_bounds=max_bounds,
        tolerance_ppm=tolerance_ppm,
        max_results=max_results,
    )
    return pl.Series(results, dtype= pl.List(pl.List(pl.Array(pl.Int32, len(min_bounds))))
    )


def test_mass_decomposition(size:int):  
    """
    Test function for mass decomposition.
    """
    if size == -1:
        nist = pl.read_parquet("/home/analytit_admin/Data/NIST_hr_msms/NIST_hr_msms.parquet")
    else:
        nist = pl.read_parquet("/home/analytit_admin/Data/NIST_hr_msms/NIST_hr_msms.parquet").sample(n=size)
    print(f"number of spectra: {nist.height}")
    print(f"number of spectra after filtering: {nist.height}")
    start = perf_counter()
    nist = nist.select(
        pl.col("NIST_ID"),
        pl.col("PrecursorMZ"))
    
    nist = nist.with_columns(
        pl.col("PrecursorMZ").map_batches(
            function=lambda x: decompose_mass(
                mass_series=x,
                min_bounds=np.zeros(15, dtype=np.int32),
                max_bounds=np.array([100, 0, 40, 20, 10, 5, 2, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32),
                tolerance_ppm=5.0,
                min_dbe=0.0,
                max_dbe=40.0,
            ),
            return_dtype=pl.List(pl.Array(pl.Int32, 15)),
            is_elementwise=True
        ).alias("decomposed_formula"))
    print(f"Decomposed formulas: {nist.height}")
    end = perf_counter()
    print(f"Time taken: {end - start:.2f} seconds")
    
    print(nist.head(5))
    print(nist.item(0, 2))

def test_spectra_decomposition(size:int):
    """
    Test function for spectra decomposition.
    """
    if size == -1:
        nist = pl.read_parquet("/home/analytit_admin/Data/NIST_hr_msms/NIST_hr_msms.parquet")
    else:
        nist = pl.read_parquet("/home/analytit_admin/Data/NIST_hr_msms/NIST_hr_msms.parquet").sample(n=size)
    print(f"number of spectra: {nist.height}")
    nist = nist.filter(
        pl.col("raw_spectrum_mz").list.len().gt(0),
        pl.col("Formula_array").arr.len().eq(15),
        )
    nist = nist.cast({
        "raw_spectrum_mz": pl.List(pl.Float64),
        "Formula_array": pl.Array(pl.Int32,15),
    })
    print(f"number of spectra after filtering: {nist.height}")
    start = perf_counter()
    nist_decomposed = nist.with_columns(
        pl.struct(["Formula_array", "raw_spectrum_mz"]).map_batches(
            lambda row: decompose_spectra_known_precursor(
                precursor_formula_series=row.struct.field("Formula_array"),
                fragment_masses_series=row.struct.field("raw_spectrum_mz"),
                min_bounds=np.zeros(15, dtype=np.int32),
                max_bounds=np.full(shape=15,fill_value=100, dtype=np.int32),
                tolerance_ppm=5.0,
            ),
            return_dtype=pl.List(pl.List(pl.Array(pl.Int32, 15)))
        ).alias("decomposed_spectra"))
    print(f"Decomposed spectra: {nist_decomposed.height}")
    end = perf_counter()
    print(f"Time taken: {end - start:.2f} seconds")

def compare_spectrum_decomposition_strategies(size=1000):
    """
    Compare different spectrum decomposition strategies.
    results:
    Strategy 1: Direct decomposition with decompose_spectra_known_precursor
    Strategy 2: Explode mass list, then decompose each mass using decompose_mass with per-mass bounds.
 
    for sampling of 100,000 spectra:
    Strategy 1: 29.41 seconds
    Strategy 2: 100.67 seconds
    
    Note that in the second strategy, we explode the mass list, then decompsoe, then group by NIST_ID and then join back to the original dataframe, which all introduces some overhead, but probavly not a lot, since most of the time is sepnt in the decomposition itself.
    """
    nist = pl.read_parquet("/home/analytit_admin/Data/NIST_hr_msms/NIST_hr_msms.parquet").sample(n=size)
    nist = nist.filter(
        pl.col("raw_spectrum_mz").list.len().gt(0),
        pl.col("Formula_array").arr.len().eq(15),
    )
    nist = nist.cast({
        "raw_spectrum_mz": pl.List(pl.Float64),
        "Formula_array": pl.Array(pl.Int32, 15),
    })
    #strategy 1- direct decomposition with "decompose_spectra_known_precursor"
    print("Strategy 1: Direct decomposition with decompose_spectra_known_precursor")
    start_1 = perf_counter()
    nist_decomposed = nist.with_columns(
        pl.struct(["Formula_array", "raw_spectrum_mz"]).map_batches(
            lambda row: decompose_spectra_known_precursor(
                precursor_formula_series=row.struct.field("Formula_array"),
                fragment_masses_series=row.struct.field("raw_spectrum_mz"),
                min_bounds=np.zeros(15, dtype=np.int32),
                max_bounds=np.full(shape=15, fill_value=100, dtype=np.int32),
                tolerance_ppm=5.0,
            ),
            return_dtype=pl.List(pl.List(pl.Array(pl.Int32, 15)))
        ).alias("decomposed_spectra")
    )
    end = perf_counter()
    print(f"Decomposed spectra: {nist_decomposed.height}")
    print(f"Time taken: {end - start_1:.2f} seconds")

    #strategy 2- explode mass list, then decompose spectra
    # do this by first selecting and then exploding and then joining the results, which does introduce some overhead
    print("Strategy 2: Explode mass list, then decompose spectra")
    start_2 = perf_counter()
    nist_exploded = nist.select(['Formula_array', 'raw_spectrum_mz','NIST_ID']).explode('raw_spectrum_mz')
    # create a column with min bounds
    # this the 
    nist_exploded = nist_exploded.with_columns(
        min_bounds=pl.lit(np.zeros((nist_exploded.height,15), dtype=np.int32)))
    nist_exploded = nist_exploded.with_columns(

        pl.struct(["Formula_array", "min_bounds","raw_spectrum_mz"]).map_batches(
            lambda row: decompose_mass(
                mass_series=row.struct.field("raw_spectrum_mz"),
                min_bounds=row.struct.field("min_bounds"),
                max_bounds=row.struct.field("Formula_array"),
                tolerance_ppm=5.0,
            ),
            return_dtype=pl.List(pl.Array(pl.Int32, 15))
        )
        .alias("decomposed_spectrum")
    )
    nist_decomposed_2= nist_exploded.group_by("NIST_ID").agg(
        pl.col("decomposed_spectrum"),
    )
    nist_decomposed_2 = nist.join(
        nist_decomposed_2,
        on="NIST_ID",
        how="left",
    )
    end = perf_counter()
    print(f"Decomposed spectra: {nist_decomposed_2.height}")
    print(f"Time taken: {end - start_2:.2f} seconds")
    



if __name__ == "__main__":
    from time import perf_counter

    test_mass_decomposition(size=100_000)