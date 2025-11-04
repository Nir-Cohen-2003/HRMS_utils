
import polars as pl
import numpy as np
from time import perf_counter
import mass_decomposition
import sys

def create_mock_nist(size:int = 10000) -> pl.DataFrame:
    """
    Create a mock NIST DataFrame with random precursor masses and formulas.
    The DataFrame will have size rows and the following columns:
    - NIST_ID: unique identifier
    - PrecursorMZ: precursor mass
    - Formula_array: precursor formula (length 15)
    - raw_spectrum_mz: list of fragment m/z values (computed from the fragment formulas below)
    """
    np.random.seed(42)  # For reproducibility
    nist_ids = np.arange(1, size + 1)

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
    fragments = [
        np.array([4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32) * 2,
        np.array([11, 0, 6, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32),
        np.array([11, 0, 6, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32) * 2,
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
        "precursor_formula": pl.Array(pl.Int32, 15),
    })

def test_roundtrip_decomposition(size: int = 10000):
    df = create_mock_nist(size=size)
    
    start = perf_counter()
    
    result = df.with_columns(
        decomposed_formula=pl.struct([
            pl.col("precursor_formula"),
            pl.col("mz")
        ]).mass_decomposer.decompose_spectrum_with_precursor(
            tolerance_ppm=5.0
        )
    )
    
    end = perf_counter()
    
    print(f"Roundtrip for {size} rows took {end - start:.4f} seconds")
    assert result is not None
    assert len(result) == size
    assert 'decomposed_formula' in result.columns

if __name__ == "__main__":
    if len(sys.argv) > 1:
        size = int(sys.argv[1])
        test_roundtrip_decomposition(size)
    else:
        test_roundtrip_decomposition()
