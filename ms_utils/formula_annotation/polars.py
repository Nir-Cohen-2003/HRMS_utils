from mass_decomposition_impl.mass_decomposer_cpp import (
    decompose_mass_parallel,
    decompose_mass_parallel_per_bounds,
)
import polars as pl
import numpy as np


def decompose_mass(
    mass_series: pl.Series,
    min_bounds,
    max_bounds,
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
    mass_array = mass_series.to_numpy()
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
        # Output: list of lists of np.ndarray
        return pl.Series(results, dtype=pl.List(pl.Array(pl.Int32, len(min_bounds))))
    # Case 2: per-mass bounds (Polars Series or DataFrame columns)
    elif isinstance(min_bounds, pl.Series) and isinstance(max_bounds, pl.Series):
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
        raise ValueError("min_bounds and max_bounds must both be either 1D numpy arrays or Polars Series of arrays.")

