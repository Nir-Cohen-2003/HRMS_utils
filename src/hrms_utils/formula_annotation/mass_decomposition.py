import polars as pl
import numpy as np
from numpy.typing import NDArray
from typing import Iterable, Tuple

from .mass_decomposition_impl.mass_decomposer_cpp import (
    clean_and_normalize_spectra_known_precursor_parallel,
    clean_and_normalize_spectra_known_precursor_parallel_verbose,
    clean_spectra_known_precursor_parallel,
    clean_spectra_known_precursor_parallel_verbose,
    decompose_mass_parallel,
    decompose_mass_parallel_per_bounds,
    decompose_mass_parallel_per_bounds_verbose,
    decompose_mass_parallel_verbose,
    decompose_spectra_parallel,
    decompose_spectra_parallel_per_bounds,
    decompose_spectra_known_precursor_parallel,
    decompose_spectra_known_precursor_parallel_verbose,
    get_num_elements,
)

NUM_ELEMENTS = get_num_elements()




def decompose_mass(
    mass_series:pl.Series,
    min_bounds: NDArray[np.int32],
    max_bounds: NDArray[np.int32],
    tolerance_ppm: float = 5.0,
    min_dbe: float = 0.0,
    max_dbe: float = 40.0,

    
):
    """
    Wrapper for decompose_mass_parallel, fixed bounds only, with validation of input types.

    Example usage:

        df = pl.DataFrame({
            "mass": [100.0, 200.0, 300.0, 400.0, 500.0]
        })

        df = df.with_columns(
            pl.col("mass").map_batches(
                lambda x: decompose_mass(
                    mass_series=x,
                    min_bounds=min_formula,
                    max_bounds=max_formula,
                    tolerance_ppm=5.0,
                    min_dbe=0.0,
                    max_dbe=40.0
                ),
                return_dtype=pl.List(pl.Array(pl.Int32, 15))
            ).alias("decomposed_formulas")
        )

        min_formula = np.zeros(15, dtype=np.int32)
        max_formula = np.array([100, 0, 40, 20, 10, 5, 2, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)

        df = pl.DataFrame({
            "mass": [100.0, 200.0, 300.0, 400.0, 500.0],
            "min_bounds": [min_formula] * 5,
            "max_bounds": [max_formula] * 5
        })

        nist = nist.with_columns(
            pl.col("mass").map_batches(
                function=lambda x: decompose_mass(
                    mass_series=x,
                    min_bounds=np.zeros(15, dtype=np.int32),
                    max_bounds=np.array([100, 0, 40, 20, 10, 5, 2, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32),
                    tolerance_ppm=5.0,
                    min_dbe=0.0,
                    max_dbe=40.0
                ),
                return_dtype=pl.List(pl.Array(pl.Int32, 15)),
                is_elementwise=True
            ).alias("decomposed_formula")
        )
    """
    ## Validate input type and shapes
    assert isinstance(mass_series, pl.Series), f"mass_series should be a Polars Series, but got {type(mass_series)}"
    assert mass_series.dtype == pl.Float64, f"mass_series should be of type Float64, but got {mass_series.dtype}"
    assert isinstance(min_bounds, np.ndarray) and min_bounds.ndim == 1, f"min_bounds should be a 1D numpy array, but got {type(min_bounds)} with shape {min_bounds.shape}"
    assert isinstance(max_bounds, np.ndarray) and max_bounds.ndim == 1, f"max_bounds should be a 1D numpy array, but got {type(max_bounds)} with shape {max_bounds.shape}"
    if min_bounds.shape[0] != max_bounds.shape[0]:
        raise ValueError(f"min_bounds and max_bounds must have the same length for uniform bounds, but got lengths {min_bounds.shape[0]} and {max_bounds.shape[0]}.")
    assert min_bounds.dtype == np.int32, f"min_bounds should be of type int32, but got {min_bounds.dtype}"
    assert max_bounds.dtype == np.int32, f"max_bounds should be of type int32, but got {max_bounds.dtype}"
    assert isinstance(tolerance_ppm, (float, int)), f"tolerance_ppm should be a float or int, but got {type(tolerance_ppm)}"
    assert tolerance_ppm > 0, f"tolerance_ppm should be a positive value, but got {tolerance_ppm}"
    assert isinstance(min_dbe, (float, int)), f"min_dbe should be a float or int, but got {type(min_dbe)}"
    assert isinstance(max_dbe, (float, int)), f"max_dbe should be a float or int, but got {type(max_dbe)}"
    results = decompose_mass_parallel(
        target_masses=mass_series,
        min_bounds=min_bounds,
        max_bounds=max_bounds,
        tolerance_ppm=tolerance_ppm,
        min_dbe=min_dbe,
        max_dbe=max_dbe,
    )
    return results

def decompose_mass_verbose(
    mass_series: pl.Series,
    min_bounds: NDArray[np.int32],
    max_bounds: NDArray[np.int32],
    tolerance_ppm: float = 5.0,
    min_dbe: float = 0.0,
    max_dbe: float = 40.0,
    ) -> Tuple[pl.Series, pl.Series]:
    """Annotate masses and return both element counts and formatted formulas.

    Wrapper for decompose_mass_parallel_verbose with the same input validation
    as :func:`decompose_mass`. This verbose variant returns two Polars Series
    that share the same nested layout: for each input mass there may be zero
    or more candidate formulas.

    Parameters
    ----------
    mass_series : pl.Series
        Series of target masses (dtype Float64).
    min_bounds : np.ndarray(shape=(NUM_ELEMENTS,), dtype=int32)
        Minimum element counts (uniform for all masses).
    max_bounds : np.ndarray(shape=(NUM_ELEMENTS,), dtype=int32)
        Maximum element counts (uniform for all masses).
    tolerance_ppm : float
        Mass tolerance in ppm.
    min_dbe : float
        Minimum degree of unsaturation (DBE).
    max_dbe : float
        Maximum degree of unsaturation (DBE).
    max_results : int
        Upper bound on number of returned candidates per mass.

    Returns
    -------
    Tuple[pl.Series, pl.Series]
        - formula_series: pl.Series where each row is a List(Array(Int32, NUM_ELEMENTS))
          containing integer element counts for each candidate formula.
        - formula_string_series: pl.Series where each row is a List(Utf8)
          containing the human-readable string representation for each candidate
          formula in the corresponding position of `formula_series`.

    Notes
    -----
    - The two returned Series are aligned: element-count arrays at index i in
      `formula_series` correspond to the formula strings at index i in
      `formula_string_series`.
    """
    assert isinstance(mass_series, pl.Series), f"mass_series should be a Polars Series, but got {type(mass_series)}"
    assert mass_series.dtype == pl.Float64, f"mass_series should be of type Float64, but got {mass_series.dtype}"
    assert isinstance(min_bounds, np.ndarray) and min_bounds.ndim == 1, (
        f"min_bounds should be a 1D numpy array, but got {type(min_bounds)} with shape {min_bounds.shape}"
    )
    assert isinstance(max_bounds, np.ndarray) and max_bounds.ndim == 1, (
        f"max_bounds should be a 1D numpy array, but got {type(max_bounds)} with shape {max_bounds.shape}"
    )
    if min_bounds.shape[0] != max_bounds.shape[0]:
        raise ValueError(
            "min_bounds and max_bounds must have the same length for uniform bounds, "
            f"but got lengths {min_bounds.shape[0]} and {max_bounds.shape[0]}."
        )
    assert min_bounds.dtype == np.int32, f"min_bounds should be of type int32, but got {min_bounds.dtype}"
    assert max_bounds.dtype == np.int32, f"max_bounds should be of type int32, but got {max_bounds.dtype}"
    assert isinstance(tolerance_ppm, (float, int)), f"tolerance_ppm should be a float or int, but got {type(tolerance_ppm)}"
    assert tolerance_ppm > 0, f"tolerance_ppm should be a positive value, but got {tolerance_ppm}"
    assert isinstance(min_dbe, (float, int)), f"min_dbe should be a float or int, but got {type(min_dbe)}"
    assert isinstance(max_dbe, (float, int)), f"max_dbe should be a float or int, but got {type(max_dbe)}"
    return decompose_mass_parallel_verbose(
        target_masses=mass_series,
        min_bounds=min_bounds,
        max_bounds=max_bounds,
        tolerance_ppm=tolerance_ppm,
        min_dbe=min_dbe,
        max_dbe=max_dbe,
    )

def decompose_mass_per_bounds(
    mass_series: pl.Series,
    min_bounds: pl.Series,
    max_bounds: pl.Series,
    tolerance_ppm: float = 5.0,
    min_dbe: float = 0.0,
    max_dbe: float = 40.0,  
) -> pl.Series:
    """
    Return a Polars Series of possible formulas for the mass.

    The data type is:
        pl.Series(pl.List(pl.Array(inner=pl.int32, shape=(15,))))

    Example usage:

        min_formula = np.zeros(15, dtype=np.int32)
        max_formula = np.array([100, 0, 40, 20, 10, 5, 2, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)

        df = pl.DataFrame({
            "mass": [100.0, 200.0, 300.0, 400.0, 500.0],
            "min_bounds": [min_formula] * 5,
            "max_bounds": [max_formula] * 5
        })

        df = df.with_columns(
            pl.col("mass").map_batches(
                function=lambda x: decompose_mass_per_bounds(
                    mass_series=x,
                    min_bounds=pl.col("min_bounds"),
                    max_bounds=pl.col("max_bounds"),
                    tolerance_ppm=5.0,
                    min_dbe=0.0,
                    max_dbe=40.0
                ),
                return_dtype=pl.List(pl.Array(pl.Int32, 15)),
                is_elementwise=True
            ).alias("decomposed_formula")
        )
    """
    ## Validate input type and shapes
    assert isinstance(mass_series, pl.Series), f"mass_series should be a Polars Series, but got {type(mass_series)}"
    assert mass_series.dtype == pl.Float64, f"mass_series should be of type Float64, but got {mass_series.dtype}"
    assert isinstance(min_bounds, pl.Series) and min_bounds.dtype == pl.Array(pl.Int32,shape=(NUM_ELEMENTS,)), f"min_bounds should be a Polars Series of int32 arrays, but got {type(min_bounds)} with dtype {min_bounds.dtype}"
    assert isinstance(max_bounds, pl.Series) and max_bounds.dtype == pl.Array(pl.Int32,shape=(NUM_ELEMENTS,)), f"max_bounds should be a Polars Series of int32 arrays, but got {type(max_bounds)} with dtype {max_bounds.dtype}"
    assert isinstance(tolerance_ppm, (float, int)), f"tolerance_ppm should be a float or int, but got {type(tolerance_ppm)}"
    assert tolerance_ppm > 0, f"tolerance_ppm should be a positive value, but got {tolerance_ppm}"
    assert isinstance(min_dbe   , (float, int)), f"min_dbe should be a float or int, but got {type(min_dbe)}"
    assert isinstance(max_dbe   , (float, int)), f"max_dbe should be a float or int, but got {type(max_dbe)}"



    results = decompose_mass_parallel_per_bounds(
        target_masses=mass_series,
        min_bounds_per_mass=min_bounds,
        max_bounds_per_mass=max_bounds,
        tolerance_ppm=tolerance_ppm,
        min_dbe=min_dbe,
        max_dbe=max_dbe,
    )
    return results  

def decompose_mass_per_bounds_verbose(
    mass_series: pl.Series,
    min_bounds: pl.Series,
    max_bounds: pl.Series,
    tolerance_ppm: float = 5.0,
    min_dbe: float = 0.0,
    max_dbe: float = 40.0,
) -> Tuple[pl.Series, pl.Series]:
    """Per-row bounds variant that also reports formatted formula strings.

    This function is the per-row-bounds (per-mass) verbose variant of
    :func:`decompose_mass_per_bounds`. Inputs are validated to be Polars Series
    of fixed-size int32 arrays (shape NUM_ELEMENTS). The function delegates
    heavy computation to the compiled implementation and returns a tuple of
    two aligned Series.

    Parameters
    ----------
    mass_series : pl.Series
        Series of target masses (dtype Float64).
    min_bounds : pl.Series
        Per-mass minimum element counts (dtype: List/Array of Int32, shape NUM_ELEMENTS).
    max_bounds : pl.Series
        Per-mass maximum element counts (dtype: List/Array of Int32, shape NUM_ELEMENTS).
    tolerance_ppm : float
        Mass tolerance in ppm.
    min_dbe : float
        Minimum DBE.
    max_dbe : float
        Maximum DBE.

    Returns
    -------
    Tuple[pl.Series, pl.Series]
        - formula_series: pl.Series of List(Array(Int32, NUM_ELEMENTS)) containing
          candidate element counts for each mass.
        - formula_string_series: pl.Series of List(Utf8) containing the
          human-readable formula string for each candidate in the corresponding
          position of `formula_series`.
    """
    assert isinstance(mass_series, pl.Series), f"mass_series should be a Polars Series, but got {type(mass_series)}"
    assert mass_series.dtype == pl.Float64, f"mass_series should be of type Float64, but got {mass_series.dtype}"
    assert isinstance(min_bounds, pl.Series) and min_bounds.dtype == pl.Array(pl.Int32,shape=(NUM_ELEMENTS,)), f"min_bounds should be a Polars Series of int32 arrays, but got {type(min_bounds)} with dtype {min_bounds.dtype}"
    assert isinstance(max_bounds, pl.Series) and max_bounds.dtype == pl.Array(pl.Int32,shape=(NUM_ELEMENTS,)), f"max_bounds should be a Polars Series of int32 arrays, but got {type(max_bounds)} with dtype {max_bounds.dtype}"
    assert isinstance(tolerance_ppm, (float, int)), f"tolerance_ppm should be a float or int, but got {type(tolerance_ppm)}"
    assert tolerance_ppm > 0, f"tolerance_ppm should be a positive value, but got {tolerance_ppm}"
    assert isinstance(min_dbe   , (float, int)), f"min_dbe should be a float or int, but got {type(min_dbe)}"
    assert isinstance(max_dbe   , (float, int)), f"max_dbe should be a float or int, but got {type(max_dbe)}"



    return decompose_mass_parallel_per_bounds_verbose(
        target_masses=mass_series,
        min_bounds_per_mass=min_bounds,
        max_bounds_per_mass=max_bounds,
        tolerance_ppm=tolerance_ppm,
        min_dbe=min_dbe,
        max_dbe=max_dbe,
    )
                      
def decompose_spectra(
    precursor_mass_series: pl.Series,
    fragment_masses_series: pl.Series,
    min_bounds,
    max_bounds,
    tolerance_ppm: float = 5.0,
    min_dbe: float = 0.0,
    max_dbe: float = 40.0,
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
        results = decompose_spectra_parallel(
            spectra_data=spectra_data,
            min_bounds=min_bounds,
            max_bounds=max_bounds,
            tolerance_ppm=tolerance_ppm,
            min_dbe=min_dbe,
            max_dbe=max_dbe,
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
    tolerance_ppm: float = 5.0,
):
    """
    Decomposes the fragments, when the precursor was already decomposed.
    The bounds for fragment masses are determined by the precursor formula (not accounting for water absorption in orbitraps, WIP) and by the "zero fragment" from below.

    Example usage:

        # Decompose precursor masses to formulas
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

        # Explode formulas to one per row
        df = df.explode("decomposed_formula")

        # Annotate fragments for each precursor formula
        df = df.with_columns(
            pl.struct(["fragment_masses", "decomposed_formula"]).map_batches(
                lambda row: decompose_spectra_known_precursor(
                    precursor_formula_series=row.struct.field("decomposed_formula"),
                    fragment_masses_series=row.struct.field("fragment_masses"),
                    tolerance_ppm=5.0,
                )
            ).alias("decomposed_spectra")
        )
    """

    results = decompose_spectra_known_precursor_parallel(
        precursor_formula_series,
        fragment_masses_series,
        tolerance_ppm=tolerance_ppm,
    )
    return results

def decompose_spectra_known_precursor_verbose(
    precursor_formula_series: pl.Series,
    fragment_masses_series: pl.Series,
    tolerance_ppm: float = 5.0,
) -> Tuple[pl.Series, pl.Series]:
    """Verbose decomposition that includes human-readable formulas for fragments.

    Decomposes fragments for spectra where the precursor formula is already known.
    This verbose variant returns two aligned Series:
      - a structural Series describing the decomposition results, and
      - a Series of human-readable formula strings aligned with fragment formulas.

    Parameters
    ----------
    precursor_formula_series : pl.Series
        Series of precursor formulas (dtype: pl.Array(pl.Int32, NUM_ELEMENTS)).
    fragment_masses_series : pl.Series
        Series of per-spectrum fragment mass lists (dtype: pl.List(pl.Float64)).
    tolerance_ppm : float
        Mass tolerance in ppm for fragment annotation.


    Returns
    -------
    Tuple[pl.Series, pl.Series]
        - results_struct_series: pl.Series of Struct/List where each entry
          contains keys such as:
            {
                "precursor": Array(Int32, NUM_ELEMENTS),
                "precursor_mass": Float64,
                "precursor_error_ppm": Float64,
                "fragments": List(List(Array(Int32, NUM_ELEMENTS))),
                "fragment_masses": List(List(Float64)),
                "fragment_errors_ppm": List(List(Float64))
            }
        - results_formula_strings_series: pl.Series of List(List(Utf8)) where each
          inner string corresponds to the candidate array in the `fragments` field
          of the struct at the same position.

    Notes
    -----
    - The two returned Series are aligned so that string representations can be
      used directly for display or downstream reporting.
    """
    return decompose_spectra_known_precursor_parallel_verbose(
        precursor_formula_series=precursor_formula_series,
        fragment_masses_series=fragment_masses_series,
        tolerance_ppm=tolerance_ppm,
    )

def clean_spectra_known_precursor(
    precursor_formula_series: pl.Series,
    fragment_masses_series: pl.Series,
    fragment_intensities_series: pl.Series,
    *,
    tolerance_ppm: float = 5.0,
) -> pl.Series:
    """
    Parallel cleaner for spectra with known precursor formulas.

    Purpose:
        - Clean fragment lists for spectra where the precursor formula is already known.
        - Delegate heavy computation to the C++/OpenMP implementation while performing
          strict, fast validation of input schema here.

    Input (per-spectrum / row):
        - precursor_formula_series : pl.Series of fixed-size int32 arrays
            dtype: pl.Array(pl.Int32, NUM_ELEMENTS)
            shape: (NUM_ELEMENTS,)
            Description: integer element counts for the precursor formula.
        - fragment_masses_series : pl.Series of lists of floats
            dtype: pl.List(pl.Float64)
            Description: observed fragment m/z values for each spectrum.
        - fragment_intensities_series : pl.Series of lists of floats
            dtype: pl.List(pl.Float64)
            Description: observed intensities aligned with fragment_masses_series.

    Parameters:
        - tolerance_ppm (float): mass tolerance in ppm for fragment formula matching.

    Output:
        Returns a pl.Series with dtype pl.Struct containing, for each spectrum:
            {
                "masses": pl.List(pl.Float64),                    # kept fragment masses
                "intensities": pl.List(pl.Float64),               # corresponding intensities
                "fragment_formulas": pl.List(pl.List(pl.Array(pl.Int32, NUM_ELEMENTS))),  # candidate formulas per fragment
                "fragment_errors_ppm": pl.List(pl.List(pl.Float64)),                     # errors aligned with formulas
            }

    Notes:
        - Fails fast on mismatched lengths or invalid dtypes (explicit error messages).
        - This wrapper verifies schema and delegates compute-intensive work to the
          compiled implementation: clean_spectra_known_precursor_parallel.
        - Designed to be used in Polars map_batches / elementwise pipelines where each
          row represents a single spectrum.

    Example:
        cleaned = clean_spectra_known_precursor(
            precursor_formula_series=pl.col("decomposed_formula"),
            fragment_masses_series=pl.col("fragment_masses"),
            fragment_intensities_series=pl.col("fragment_intensities"),
            tolerance_ppm=5.0,
        )
    """
    # Validate inputs
    assert isinstance(precursor_formula_series, pl.Series), "precursor_formula_series must be a Polars Series"
    assert isinstance(fragment_masses_series, pl.Series), "fragment_masses_series must be a Polars Series"
    assert isinstance(fragment_intensities_series, pl.Series), "fragment_intensities_series must be a Polars Series"

    expected_arr_a = pl.Array(pl.Int32, NUM_ELEMENTS)
    expected_arr_b = pl.Array(pl.Int32, shape=(NUM_ELEMENTS,))
    assert precursor_formula_series.dtype in (expected_arr_a, expected_arr_b), (
        f"precursor_formula_series.dtype must be pl.Array(pl.Int32, {NUM_ELEMENTS}), got {precursor_formula_series.dtype}"
    )

    expected_list = pl.List(pl.Float64)
    assert fragment_masses_series.dtype == expected_list, (
        f"fragment_masses_series.dtype must be List(Float64), got {fragment_masses_series.dtype}"
    )
    assert fragment_intensities_series.dtype == expected_list, (
        f"fragment_intensities_series.dtype must be List(Float64), got {fragment_intensities_series.dtype}"
    )

    n = precursor_formula_series.len()
    if fragment_masses_series.len() != n or fragment_intensities_series.len() != n:
        raise ValueError("All input series must have the same length (one entry per spectrum).")

    # Delegate to Cython/C++ implementation
    return clean_spectra_known_precursor_parallel(
        precursor_formula_series=precursor_formula_series,
        fragment_masses_series=fragment_masses_series,
        fragment_intensities_series=fragment_intensities_series,
        tolerance_ppm=tolerance_ppm,
    )

def clean_spectra_known_precursor_verbose(
    precursor_formula_series: pl.Series,
    fragment_masses_series: pl.Series,
    fragment_intensities_series: pl.Series,
    *,
    tolerance_ppm: float = 5.0,
) -> pl.Series:
    """Clean spectra and include formula strings per fragment.

    Verbose variant of :func:`clean_spectra_known_precursor`. Performs the same
    validation and cleaning but also returns human-readable formula strings
    aligned with the fragment formulas.

    Inputs (per-spectrum / row):
        - precursor_formula_series : pl.Series of fixed-size int32 arrays
            dtype: pl.Array(pl.Int32, NUM_ELEMENTS)
        - fragment_masses_series : pl.Series of lists of floats (pl.List(pl.Float64))
        - fragment_intensities_series : pl.Series of lists of floats (pl.List(pl.Float64))

    Parameters
    ----------
    tolerance_ppm : float
        Mass tolerance in ppm.

    Returns
    -------
    pl.Series
        A Series of Struct for each spectrum containing:
            {
                "masses": pl.List(pl.Float64),
                "intensities": pl.List(pl.Float64),
                "fragment_formulas": pl.List(pl.List(pl.Array(pl.Int32, NUM_ELEMENTS))),
                "fragment_errors_ppm": pl.List(pl.List(pl.Float64)),
                "fragment_formulas_str": pl.List(pl.List(pl.Utf8)),  # human-readable strings aligned with fragment_formulas
            }

    Notes
    -----
    - Fails fast on mismatched lengths or invalid dtypes (explicit error messages).
    - Intended for use in Polars map_batches / elementwise pipelines where each
      row is a spectrum.
    """
    assert isinstance(precursor_formula_series, pl.Series), "precursor_formula_series must be a Polars Series"
    assert isinstance(fragment_masses_series, pl.Series), "fragment_masses_series must be a Polars Series"
    assert isinstance(fragment_intensities_series, pl.Series), "fragment_intensities_series must be a Polars Series"

    expected_arr = pl.Array(pl.Int32, NUM_ELEMENTS)
    assert precursor_formula_series.dtype in (expected_arr, pl.Array(pl.Int32, shape=(NUM_ELEMENTS,))), (
        f"precursor_formula_series.dtype must be pl.Array(pl.Int32, {NUM_ELEMENTS}), got {precursor_formula_series.dtype}"
    )

    expected_list = pl.List(pl.Float64)
    assert fragment_masses_series.dtype == expected_list, (
        f"fragment_masses_series.dtype must be List(Float64), got {fragment_masses_series.dtype}"
    )
    assert fragment_intensities_series.dtype == expected_list, (
        f"fragment_intensities_series.dtype must be List(Float64), got {fragment_intensities_series.dtype}"
    )

    n = precursor_formula_series.len()
    if fragment_masses_series.len() != n or fragment_intensities_series.len() != n:
        raise ValueError("All input series must share the same length.")

    return clean_spectra_known_precursor_parallel_verbose(
        precursor_formula_series=precursor_formula_series,
        fragment_masses_series=fragment_masses_series,
        fragment_intensities_series=fragment_intensities_series,
        tolerance_ppm=tolerance_ppm,
    )

def clean_and_normalize_spectra_known_precursor(
    precursor_formula_series: pl.Series,
    precursor_masses_series: pl.Series,
    fragment_masses_series: pl.Series,
    fragment_intensities_series: pl.Series,
    *,
    tolerance_ppm: float = 5.0,
    max_allowed_normalized_mass_error_ppm: float = 5.0,
) -> pl.Series:
    """
    Parallel cleaner for spectra with known precursor that:
    1) Estimates a spectrum-level mean mass error (systemic bias),
    2) Selects a single best formula per fragment (highest masses resolved first),
    3) Returns normalized fragment masses (target_mass + final_mean_error).

    Input schema per spectrum (row-wise):
    - precursor_formula_series: pl.Array(pl.Int32, NUM_ELEMENTS)
    - precursor_masses_series: pl.Float64 (observed precursor mass; for neutral-workflow pass non-ionized mass)
    - fragment_masses_series:   pl.List(pl.Float64)
    - fragment_intensities_series: pl.List(pl.Float64)

    Output:
    - pl.Series of Struct with fields:
        {
            "masses_normalized": pl.List(pl.Float64),
            "cleaned_intensities":       pl.List(pl.Float64),
            "fragment_formulas": pl.List(pl.Array(pl.Int32, NUM_ELEMENTS)),
            "fragment_errors_ppm": pl.List(pl.Float64),
        }
    """
    assert isinstance(precursor_formula_series, pl.Series), "precursor_formula_series must be a Polars Series"
    assert isinstance(precursor_masses_series, pl.Series), "precursor_masses_series must be a Polars Series"
    assert isinstance(fragment_masses_series, pl.Series), "fragment_masses_series must be a Polars Series"
    assert isinstance(fragment_intensities_series, pl.Series), "fragment_intensities_series must be a Polars Series"

    expected_arr_a = pl.Array(pl.Int32, NUM_ELEMENTS)
    expected_arr_b = pl.Array(pl.Int32, shape=(NUM_ELEMENTS,))
    if precursor_formula_series.dtype not in (expected_arr_a, expected_arr_b):
        raise TypeError(
            f"precursor_formula_series.dtype must be pl.Array(pl.Int32, {NUM_ELEMENTS}), "
            f"got {precursor_formula_series.dtype}"
        )

    if precursor_masses_series.dtype != pl.Float64:
        raise TypeError(f"precursor_masses_series.dtype must be Float64, got {precursor_masses_series.dtype}")

    expected_list = pl.List(pl.Float64)
    if fragment_masses_series.dtype != expected_list:
        raise TypeError(f"fragment_masses_series.dtype must be List(Float64), got {fragment_masses_series.dtype}")
    if fragment_intensities_series.dtype != expected_list:
        raise TypeError(f"fragment_intensities_series.dtype must be List(Float64), got {fragment_intensities_series.dtype}")

    n = precursor_formula_series.len()
    if (fragment_masses_series.len() != n or
        fragment_intensities_series.len() != n or
        precursor_masses_series.len() != n):
        raise ValueError("All input series must have the same length (one entry per spectrum).")

    return clean_and_normalize_spectra_known_precursor_parallel(
        precursor_formula_series=precursor_formula_series,
        precursor_masses_series=precursor_masses_series,
        fragment_masses_series=fragment_masses_series,
        fragment_intensities_series=fragment_intensities_series,
        tolerance_ppm=tolerance_ppm,
        max_allowed_normalized_mass_error_ppm=max_allowed_normalized_mass_error_ppm,
    )


def clean_and_normalize_spectra_known_precursor_verbose(
    precursor_formula_series: pl.Series,
    precursor_masses_series: pl.Series,
    fragment_masses_series: pl.Series,
    fragment_intensities_series: pl.Series,
    *,
    tolerance_ppm: float = 5.0,
    max_allowed_normalized_mass_error_ppm: float = 5.0,
) -> pl.Series:
    """Verbose cleaner + normalizer that adds ``fragment_formulas_str`` output.

    Performs the same steps as :func:`clean_and_normalize_spectra_known_precursor`
    but includes human-readable formula strings for the selected fragment formulas.

    Inputs (per-spectrum / row):
    - precursor_formula_series: pl.Array(pl.Int32, NUM_ELEMENTS)
    - precursor_masses_series: pl.Float64 (observed precursor mass)
    - fragment_masses_series:   pl.List(pl.Float64)
    - fragment_intensities_series: pl.List(pl.Float64)

    Parameters
    ----------
    tolerance_ppm : float
        Mass tolerance used for fragment matching.
    max_allowed_normalized_mass_error_ppm : float
        Maximum allowed mean mass error after normalization (safeguard).

    Returns
    -------
    pl.Series
        A Series of Struct for each spectrum containing:
            {
                "masses_normalized": pl.List(pl.Float64),
                "cleaned_intensities": pl.List(pl.Float64),
                "fragment_formulas": pl.List(pl.Array(pl.Int32, NUM_ELEMENTS)),
                "fragment_errors_ppm": pl.List(pl.Float64),
                "fragment_formulas_str": pl.List(pl.Utf8),  # strings aligned with fragment_formulas
            }

    Notes
    -----
    - The verbose output includes `fragment_formulas_str` which provides a
      human-readable representation for the chosen fragment formula per fragment.
    - The function fails fast on schema mismatches and delegates compute to the
      compiled verbose implementation.
    """
    assert isinstance(precursor_formula_series, pl.Series), "precursor_formula_series must be a Polars Series"
    assert isinstance(precursor_masses_series, pl.Series), "precursor_masses_series must be a Polars Series"
    assert isinstance(fragment_masses_series, pl.Series), "fragment_masses_series must be a Polars Series"
    assert isinstance(fragment_intensities_series, pl.Series), "fragment_intensities_series must be a Polars Series"

    expected_arr = pl.Array(pl.Int32, NUM_ELEMENTS)
    assert precursor_formula_series.dtype in (expected_arr, pl.Array(pl.Int32, shape=(NUM_ELEMENTS,))), (
        f"precursor_formula_series.dtype must be pl.Array(pl.Int32, {NUM_ELEMENTS}), got {precursor_formula_series.dtype}"
    )
    assert precursor_masses_series.dtype == pl.Float64, (
        f"precursor_masses_series.dtype must be Float64, got {precursor_masses_series.dtype}"
    )
    expected_list = pl.List(pl.Float64)
    assert fragment_masses_series.dtype == expected_list, (
        f"fragment_masses_series.dtype must be List(Float64), got {fragment_masses_series.dtype}"
    )
    assert fragment_intensities_series.dtype == expected_list, (
        f"fragment_intensities_series.dtype must be List(Float64), got {fragment_intensities_series.dtype}"
    )

    n = precursor_formula_series.len()
    if (
        fragment_masses_series.len() != n
        or fragment_intensities_series.len() != n
        or precursor_masses_series.len() != n
    ):
        raise ValueError("All input series must share the same length.")



    return clean_and_normalize_spectra_known_precursor_parallel_verbose(
        precursor_formula_series=precursor_formula_series,
        precursor_masses_series=precursor_masses_series,
        fragment_masses_series=fragment_masses_series,
        fragment_intensities_series=fragment_intensities_series,
        tolerance_ppm=tolerance_ppm,
        max_allowed_normalized_mass_error_ppm=max_allowed_normalized_mass_error_ppm,
    )


