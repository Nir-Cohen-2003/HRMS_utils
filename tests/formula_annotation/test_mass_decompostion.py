import polars as pl
import numpy as np
from typing import Any
from pathlib import Path
from ms_utils.formula_annotation import (
    decompose_mass,
    decompose_mass_per_bounds,
    decompose_spectra_known_precursor
)

def mass_decomposition_test(size:int):
    """
    this function tests the mass decomposition functions, acros 3 scenarios:
    1. unifrom bounds, where the same bounds are used for all masses
    2. per-mass bounds, where each mass has its own bounds, but they happen to be all the same, and the same as in 1
    3. per-mass bounds, where each mass has its own bounds, and they are determined by the "isotopic pattern" of the mass, so we take the real formula of the mass, and then we assume that the number of carbons is +-1 or +-2, the number of Cl and Br is exacly the same, and add or remove S (so if S is not present, the upper bound is 0, if it is present, the lower bound is 1 and the upper is the one in the general constraints).
    
    all masses come from nist, and we filter them to be under 900 Da, since the complexity shoots up above that (and perhaps even before that, but we will see).

    """
    # check if the nist data is available, and if not, use mock data
    if Path("/home/analytit_admin/Data/NIST_hr_msms/NIST_hr_msms.parquet").exists():
        nist = pl.scan_parquet("/home/analytit_admin/Data/NIST_hr_msms/NIST_hr_msms.parquet").filter(
        pl.col("PrecursorMZ").le(900),
         pl.col("Precursor_type").eq("[M+H]+")
        )
    else:
        print(f"NIST not found, using mock data instead, with length {size}.")
        nist = create_mock_nist(size=size)
    if size != -1:
        nist = nist.sort(by="NIST_ID").head(n=size)
    nist = nist.select(
        pl.col("NIST_ID"),
        pl.col("PrecursorMZ"),
        pl.col("Formula_array"),
    ).collect()
    nist = nist.cast({
        "PrecursorMZ": pl.Float64,
        "Formula_array": pl.Array(pl.Int32, 15),
    })
    # scenario 1: uniform bounds
    start = perf_counter()
    nist_uniform_bounds = nist.with_columns(
        pl.col("PrecursorMZ").map_batches(
            function=lambda x: decompose_mass(
                mass_series=x,
                min_bounds=np.array(MIN_FORMULA, dtype=np.int32),
                max_bounds=np.array(MAX_FORMULA, dtype=np.int32),
                tolerance_ppm=5.0,
                min_dbe=0.0,
                max_dbe=40.0,
            ),
            return_dtype=pl.List(pl.Array(pl.Int32, 15)),
            is_elementwise=True
        ).alias("decomposed_formula"))
    end = perf_counter()
    unifrom_bounds_time = end - start
    print(f"Uniform bounds decomposition time: {unifrom_bounds_time:.2f} seconds")

    # scenario 2: per-mass bounds, where each mass has its own bounds, but they happen to be all the same as in scenario 1
    min_formula = np.array(MIN_FORMULA, dtype=np.int32)
    max_formula = np.array(MAX_FORMULA, dtype=np.int32)
    nist = nist.with_columns(
        max_bounds=pl.lit(
            np.tile(max_formula, (nist.height, 1)),
            dtype=pl.Array(pl.Int32, 15)
        ),
        min_bounds=pl.lit(
            np.tile(min_formula, (nist.height, 1)),
            dtype=pl.Array(pl.Int32, 15)
        )
    )
    start = perf_counter()
    nist_non_uniform_bounds = nist.with_columns(
        pl.struct(["PrecursorMZ", "min_bounds", "max_bounds"]).map_batches(
            lambda row: decompose_mass_per_bounds(
                mass_series=row.struct.field("PrecursorMZ"),
                min_bounds=row.struct.field("min_bounds"),
                max_bounds=row.struct.field("max_bounds"),
                tolerance_ppm=5.0,
            ),
            return_dtype=pl.List(pl.Array(pl.Int32, 15)),
            is_elementwise=True
        ).alias("decomposed_spectra")).drop(["min_bounds", "max_bounds"])
    end = perf_counter()
    per_mass_bounds_time = end - start
    print(f"Per-mass bounds decomposition time: {per_mass_bounds_time:.2f} seconds")

    # scenario 3: per-mass bounds, where each mass has its own bounds, and they are determined by the "isotopic pattern" of the mass
    # we will use the formula array to determine the bounds, and then we will use the formula to determine the bounds for each mass
    # we will assume that the number of carbons is +-1 or +-2, the number of Cl and Br is exactly the same, and add or remove S (so if S is not present, the upper bound is 0, if it is present, the lower bound is 1 and the upper is the one in the general constraints).
    nist_isotopic_bounds = create_isotopic_bounds(nist, formula_col="Formula_array")
    start = perf_counter()
    nist_isotopic_bounds = nist_isotopic_bounds.with_columns(
        pl.struct(["PrecursorMZ", "min_bounds", "max_bounds"]).map_batches(
            lambda row: decompose_mass_per_bounds(
                mass_series=row.struct.field("PrecursorMZ"),
                min_bounds=row.struct.field("min_bounds"),
                max_bounds=row.struct.field("max_bounds"),
                tolerance_ppm=5.0,
            ),
            return_dtype=pl.List(pl.Array(pl.Int32, 15)),
            is_elementwise=True
        ).alias("decomposed_spectra")).drop(["min_bounds", "max_bounds"])
    end = perf_counter()
    isotopic_bounds_time = end - start
    print(f"Isotopic bounds decomposition time: {isotopic_bounds_time:.2f} seconds")
    

def create_isotopic_bounds(df:pl.DataFrame, formula_col: str = "Formula_array"):
    """
    Create isotopic bounds for a given DataFrame with a formula column.
    Returns the DataFrame with two new columns: min_bounds and max_bounds.
    """
    df = df.with_columns(
        min_bounds=pl.col(formula_col).map_batches(
            lambda x: _min_bound(x.to_numpy()),
            return_dtype=pl.Array(pl.Int32, 15),
            is_elementwise=True
        ),
        max_bounds=pl.col(formula_col).map_batches(
            lambda x: _max_bound(x.to_numpy()),
            return_dtype=pl.Array(pl.Int32, 15),
            is_elementwise=True
        )
    )
    return df


def _min_bound(formula_arr:np.ndarray):
    #if MIN_FORMULA is not defined, we define it here
    # if 'MIN_FORMULA' not in globals():
    MIN_FORMULA = np.array([ 0,  0, 0,  0,  0,  0, 0, 0,  0, 0, 0,  0, 0, 0,  0], dtype=np.int32)
    # we take a 2d array, where each row is a formula, and we return a 2d array with the same shape, but with the min bounds for each formula
    result_arr = np.zeros((formula_arr.shape[0], 15), dtype=np.int32)
    result_arr[:, 0] = MIN_FORMULA[0]  # H
    result_arr[:, 1] = np.where(formula_arr[:, 1]>0, 1, 0)  # B: B if B>0, else 0
    result_arr[:, 2] = np.maximum(0,formula_arr[:, 2] - 1)  # C: C-2, clipped to [0, MAX]
    result_arr[:, 3] = np.where(formula_arr[:, 3] > 0, 1, 0)  # N: N if N>0, else 0
    result_arr[:, 4] = MIN_FORMULA[4]  # O
    result_arr[:, 5] = MIN_FORMULA[5]  # F
    result_arr[:, 6] = MIN_FORMULA[6]  # Na
    result_arr[:, 7] = MIN_FORMULA[7]  # Si
    result_arr[:, 8] = MIN_FORMULA[8]  # P
    # S: 0 if S==0 else MIN, evaluated elementwise
    result_arr[:, 9] = np.where(formula_arr[:, 9] > 0, 1, 0)
    result_arr[:, 10] = formula_arr[:, 10]  # Cl: exact
    result_arr[:, 11] = MIN_FORMULA[11]  # K
    result_arr[:, 12] = MIN_FORMULA[12]  # As
    result_arr[:, 13] = formula_arr[:, 13]  # Br: exact
    result_arr[:, 14] = MIN_FORMULA[14]  # I
    return result_arr


def _max_bound(formula_arr:np.ndarray):
    # if 'MAX_FORMULA' not in globals():
    ####################### H,   B, C,  N,  O,  F, Na,Si, P, S, Cl, K, As,Br, I
    MAX_FORMULA = np.array([100, 0, 60, 30, 30, 30, 0, 1, 2, 5, 10, 0, 0, 3,  3], dtype=np.int32)
    # we take a 2d array, where each row is a formula, and we return a 2d array with the same shape, but with the max bounds for each formula
    result_arr = np.zeros((formula_arr.shape[0], 15), dtype=np.int32)
    result_arr[:, 0] = MAX_FORMULA[0]  # H
    result_arr[:, 1] = np.maximum(0,formula_arr[:, 1])  # B
    result_arr[:, 2] = np.maximum(0,formula_arr[:, 2] + 1)  # C: C+2, clipped to [0, MAX]
    result_arr[:, 3] = np.where(formula_arr[:, 3] > 0, MAX_FORMULA[3], 0)  # N: N if N>0, else 0
    result_arr[:, 4] = MAX_FORMULA[4]  # O
    result_arr[:, 5] = MAX_FORMULA[5]  # F
    result_arr[:, 6] = MAX_FORMULA[6]  # Na
    result_arr[:, 7] = MAX_FORMULA[7]  # Si
    result_arr[:, 8] = MAX_FORMULA[8]  # P
    result_arr[:, 9] = np.where(formula_arr[:, 9] > 0, MAX_FORMULA[9], 0)  # S: 0 if S==0 else MAX, evaluated elementwise
    result_arr[:, 10] = formula_arr[:, 10]  # Cl: exact
    result_arr[:, 11] = MAX_FORMULA[11]  # K
    result_arr[:, 12] = MAX_FORMULA[12]  # As
    result_arr[:, 13] = formula_arr[:, 13]  # Br: exact
    result_arr[:, 14] = MAX_FORMULA[14]  # I
    return result_arr

def create_mock_nist(size:int = 1000) -> pl.DataFrame:
    """
    Create a mock NIST DataFrame with random precursor masses and formulas.
    The DataFrame will have size rows and the following columns:
    - NIST_ID: unique identifier
    - PrecursorMZ: precursor mass
    - Formula_array: precursor formula (length 15)
    - raw_spectrum_mz: list of fragment m/z values (computed from the fragment formulas below)
    """
    np.random.seed(42)  # For reproducibility
    nist_ids: np.ndarray[tuple[int], np.dtype[np.signedinteger[Any]]] = np.arange(1, size + 1)

    element_masses = np.array([
        1.007825,    # H
        11.009305,   # B
        12.0000,     # C
        14.003074,   # N
        15.994915,   # O
        18.998403,   # F
        22.989770,   # Na
        27.9769265,  # Si
        30.973762,   # P
        31.972071,   # S
        34.96885271, # Cl
        38.963707,   # K
        74.921596,   # As
        78.918338,   # Br
        126.904468,  # I
    ], dtype=np.float64)

    # Precursor formula (example: C6H13NO2 replicated 4× in this ordering: H,B,C,N,O,...)
    formula_array = np.array([
        13, 0, 6, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ], dtype=np.int32) * 4

    # Supplied fragment formulas (use as-is)
    fragments: list[np.ndarray] = [
        np.array([4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32) * 2,
        np.array([11, 0, 6, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32),
        np.array([11, 0, 6, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32) * 2,
    ]

    print(f"Using formula_array: {formula_array.tolist()}")
    precursor_mass = float(np.sum(formula_array * element_masses))
    print(f"resulting precursor mass: {precursor_mass:.4f} Da")

    # Compute fragment masses from supplied fragment formulas
    fragment_masses = [float(np.sum(f * element_masses)) for f in fragments]

    formula_arrays = np.tile(formula_array, (size, 1))
    precursor_masses = np.full(size, precursor_mass, dtype=np.float64)

    # Repeat the same fragment list for each mock spectrum
    raw_spectrum_mz_col = [fragment_masses for _ in range(size)]

    return pl.DataFrame({
        "NIST_ID": nist_ids,
        "PrecursorMZ": precursor_masses,
        "Formula_array": [arr.tolist() for arr in formula_arrays],
        "raw_spectrum_mz": raw_spectrum_mz_col,
    })

def test_spectra_decomposition(size:int) -> None:
    """
    Test function for spectra decomposition.
    """
    nist = create_mock_nist(size)
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
    ########################## H,  B, C,  N,  O,  F, Na,Si, P, S, Cl, K, As,Br, I
    MIN_FORMULA: list[int] = [ 0,  0, 0,  0,  0,  0, 0, 0,  0, 0, 0,  0, 0, 0,  0]
    MAX_FORMULA: list[int] = [100, 1, 60, 30, 30, 30, 0, 5, 10, 5, 10, 0, 1, 2,  3]
    mass_decomposition_test(size=10000)