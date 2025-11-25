import polars as pl
import numpy as np
from time import perf_counter
import hrms_core  
import sys
import argparse


########################## H,C,  N,  O,  F, Na, P, S, Cl, K, Br, I
MIN_FORMULA: list[int] = [ 0, 0,  0,  0,  0, 0,  0, 0, 0,  0,0,  0]
MAX_FORMULA: list[int] = [100,60, 30, 30, 30, 0, 10, 5, 10, 0, 2,  3]
NUM_ELEMENTS = 12

def create_mock_nist(size:int = 10000) -> pl.DataFrame:
    """
    Create a mock NIST DataFrame with random precursor masses and formulas.
    The DataFrame will have size rows and the following columns:
    - NIST_ID: unique identifier
    - PrecursorMZ: precursor mass
    - precursor_formula: precursor formula (length 12)
    - mz: list of fragment m/z values (computed from the fragment formulas below)
    """
    np.random.seed(42)  # For reproducibility
    nist_ids = np.arange(1, size + 1)

    element_masses = np.array([
        1.007825,    # H
        12.0000,     # C
        14.003074,   # N
        15.994915,   # O
        18.998403,   # F
        22.989770,   # Na
        30.973762,   # P
        31.972071,   # S
        34.96885271, # Cl
        38.963707,   # K
        78.918338,   # Br
        126.904468,  # I
    ], dtype=np.float64)

    # Precursor formula (example: C6H13NO2 replicated 4× in this ordering: H,B,C,N,O,...)
    formula_array = np.array([
        13, 6, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0
    ], dtype=np.int32) * 4

    # Supplied fragment formulas (use as-is)
    fragments = [
        np.array([4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32) * 2,
        np.array([11, 6, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32),
        np.array([11, 6, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32) * 2,
    ]

    precursor_mass = float(np.sum(formula_array * element_masses))

    # Compute fragment masses from supplied fragment formulas
    fragment_masses = [float(np.sum(f * element_masses)) for f in fragments]

    formula_arrays = np.tile(formula_array, (size, 1))
    precursor_masses = np.full(size, precursor_mass, dtype=np.float64)

    # Repeat the same fragment list for each mock spectrum
    raw_spectrum_mz_col = [fragment_masses for _ in range(size)]

    df = pl.DataFrame({
        "NIST_ID": nist_ids,
        "PrecursorMZ": precursor_masses,
        "precursor_formula": [arr.tolist() for arr in formula_arrays],
        "mz": raw_spectrum_mz_col,
    })
    
    return df.cast({
        "mz": pl.List(pl.Float64),
        "precursor_formula": pl.Array(pl.Int32, 12),
    })

def test_roundtrip_decomposition(size: int = 10000):
    df = create_mock_nist(size=size)
    
    # Add intensities column (all ones)
    df = df.with_columns(
        intensities=pl.col("mz").list.eval(pl.lit(1.0))
    )
    
    # Create the spectrum struct
    df = df.with_columns(
        spectrum_struct=pl.struct(["mz", "intensities", "precursor_formula"])
    )
    
    start = perf_counter()
    
    result = df.with_columns(
        corrected=pl.col("spectrum_struct").mass_decomposition.clean_and_normalize_spectrum(
            raw_fragment_tolerance_ppm=5.0,
            normalized_fragment_tolerance_ppm=2.0,
            min_dbe=-0.5,
            max_dbe=30.0,
            dbe_mode="half_integer",
            water_absorption=False
        )
    )
    
    end = perf_counter()
    
    print(f"Spectrum decomposition for {size} rows took {end - start:.4f} seconds")
    assert result is not None
    assert len(result) == size
    assert 'corrected' in result.columns


def _min_bound(formula_arr:np.ndarray):
    min_formula = np.array(MIN_FORMULA, dtype=np.int32)
    result_arr = np.zeros((formula_arr.shape[0], NUM_ELEMENTS), dtype=np.int32)
    result_arr[:, 0] = min_formula[0]  # H
    result_arr[:, 1] = np.maximum(0,formula_arr[:, 1] - 1)  # C: C-1, clipped to [0, MAX]
    result_arr[:, 2] = np.where(formula_arr[:, 1] > 0, 1, 0)  # N: N if N>0, else 0
    result_arr[:, 3] = min_formula[3]  # O
    result_arr[:, 4] = min_formula[4]  # F
    result_arr[:, 5] = min_formula[5]  # Na
    result_arr[:, 6] = min_formula[6]  # P
    result_arr[:, 7] = np.where(formula_arr[:, 7] > 0, 1, 0)
    result_arr[:, 8] = formula_arr[:, 8]  # Cl: exact
    result_arr[:, 9] = min_formula[9]  # K
    result_arr[:, 10] = formula_arr[:, 10]  # Br: exact
    result_arr[:, 11] = min_formula[11]  # I
    return result_arr

def _max_bound(formula_arr:np.ndarray):
    max_formula = np.array(MAX_FORMULA, dtype=np.int32)
    result_arr = np.zeros((formula_arr.shape[0], NUM_ELEMENTS), dtype=np.int32)
    result_arr[:, 0] = max_formula[0]  # H
    result_arr[:, 1] = np.maximum(0,formula_arr[:, 1] + 1)  # C: C+1, clipped to [0, MAX]
    result_arr[:, 2] = np.where(formula_arr[:, 2] > 0, max_formula[3], 0)  # N: N if N>0, else 0
    result_arr[:, 3] = max_formula[3]  # O
    result_arr[:, 4] = max_formula[4]  # F
    result_arr[:, 5] = max_formula[5]  # Na
    result_arr[:, 6] = max_formula[6]  # P
    result_arr[:, 7] = np.where(formula_arr[:, 7] > 0, max_formula[9], 0)  # S: 0 if S==0 else MAX, evaluated elementwise
    result_arr[:, 8] = formula_arr[:, 8]  # Cl: exact
    result_arr[:, 9] = max_formula[9]  # K
    result_arr[:, 10] = formula_arr[:, 10]  # Br: exact
    result_arr[:, 11] = max_formula[11]  # I
    return result_arr

def create_isotopic_bounds(df:pl.DataFrame, formula_col: str = "precursor_formula") -> pl.DataFrame:
    """
    Create isotopic bounds for a given DataFrame with a formula column.
    Returns the DataFrame with two new columns: min_bounds and max_bounds.
    """
    df = df.with_columns(
        min_bounds=pl.col(formula_col).map_batches(
            lambda x: _min_bound(x.to_numpy()),
            return_dtype=pl.Array(pl.Int32, NUM_ELEMENTS),
        ),
        max_bounds=pl.col(formula_col).map_batches(
            lambda x: _max_bound(x.to_numpy()),
            return_dtype=pl.Array(pl.Int32, NUM_ELEMENTS),
        )
    )
    return df

def benchmark_mass_decomposition(size: int):
    nist = create_mock_nist(size=size).select(
        pl.col("NIST_ID"),
        pl.col("PrecursorMZ"),
        pl.col("precursor_formula"),
    )

    # Scenario 1: Uniform bounds
    print("\n--- Mass Decomposition Benchmark ---")
    print(f"Running for {size} rows.")
    print("\nScenario 1: Uniform bounds")
    start = perf_counter()
    nist_uniform_bounds = nist.with_columns(
        decomposed=pl.col("PrecursorMZ").mass_decomposition.decompose_mass(
            tolerance_ppm=5.0,
            min_bounds=MIN_FORMULA,
            max_bounds=MAX_FORMULA,
            min_dbe=-0.5,
            max_dbe=40.0,
        )
    )
    end = perf_counter()
    print(f"Uniform bounds decomposition time: {end - start:.4f} seconds")
    assert len(nist_uniform_bounds) == size

    # Scenario 2: Per-mass bounds (all same)
    print("\nScenario 2: Per-mass bounds (all same)")
    nist_with_bounds = nist.with_columns(
        min_bounds=pl.lit(np.tile(np.array(MIN_FORMULA, dtype=np.int32), (nist.height, 1))),
        max_bounds=pl.lit(np.tile(np.array(MAX_FORMULA, dtype=np.int32), (nist.height, 1)))
    )
    start = perf_counter()
    nist_non_uniform_bounds = nist_with_bounds.with_columns(
        decomposed=pl.struct([
            pl.col("PrecursorMZ").alias("mass"), 
            pl.col("min_bounds"), 
            pl.col("max_bounds")
        ]).mass_decomposition.decompose_mass_with_bounds(
            tolerance_ppm=5.0,
        )
    ).drop(["min_bounds", "max_bounds"])
    end = perf_counter()
    print(f"Per-mass bounds (all same) decomposition time: {end - start:.4f} seconds")
    assert len(nist_non_uniform_bounds) == size

    # Scenario 3: Per-mass bounds (isotopic)
    print("\nScenario 3: Per-mass bounds (isotopic)")
    nist_isotopic_bounds = create_isotopic_bounds(nist)
    start = perf_counter()
    nist_isotopic_bounds_decomposed = nist_isotopic_bounds.with_columns(
        decomposed=pl.struct([
            pl.col("PrecursorMZ").alias("mass"), 
            pl.col("min_bounds"), 
            pl.col("max_bounds")
        ]).mass_decomposition.decompose_mass_with_bounds(
            tolerance_ppm=5.0,
        )
    ).drop(["min_bounds", "max_bounds"])
    end = perf_counter()
    print(f"Isotopic bounds decomposition time: {end - start:.4f} seconds")
    assert len(nist_isotopic_bounds_decomposed) == size


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run spectrum roundtrip and mass decomposition benchmarks. Flags may be placed before or after the size argument."
    )
    parser.add_argument(
        "size",
        nargs="?",
        type=int,
        default=10000,
        help="Number of mock spectra (default: 10000)."
    )
    parser.add_argument(
        "--no-spectrum",
        action="store_true",
        dest="no_spectrum",
        help="Disable the spectrum roundtrip decomposition test."
    )
    parser.add_argument(
        "--no-mass",
        action="store_true",
        dest="no_mass",
        help="Disable the mass decomposition benchmark."
    )

    args = parser.parse_args()

    if not args.no_spectrum:
        test_roundtrip_decomposition(args.size)
    else:
        print("Skipping spectrum decomposition test")

    if not args.no_mass:
        benchmark_mass_decomposition(args.size)
    else:
        print("Skipping mass decomposition benchmark")