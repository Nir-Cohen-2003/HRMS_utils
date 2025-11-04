import pytest
import polars as pl
from polars.testing import assert_frame_equal
import numpy as np
import mass_decomposition
import re

# The order of elements as defined in the Rust code
ELEMENT_SYMBOLS = [
    "H", "B", "C", "N", "O", "F", "Na", "Si", "P", "S", "Cl", "K", "As", "Br", "I"
]
NUM_ELEMENTS = len(ELEMENT_SYMBOLS)

def formula_string_to_array(formula_str: str) -> list[int]:
    """Converts a formula string to a formula array."""
    formula_arr = [0] * NUM_ELEMENTS
    for symbol, count_str in re.findall(r'([A-Z][a-z]?)(\d*)', formula_str):
        count = int(count_str) if count_str else 1
        if symbol in ELEMENT_SYMBOLS:
            formula_arr[ELEMENT_SYMBOLS.index(symbol)] = count
    return formula_arr


# Placeholder for test cases
# Each tuple contains: (mz_values, precursor_formula_str, tolerance_ppm, water_absorption, expected_formulas)
TEST_CASES = [
    (
        [78.046950, 104.062600, 128.062600],
        "C10H20O5",
        5.0,
        False,
        [ #expected formulas for each mz
            ["C6H6"],
            ["C8H8"],
            ["C10H8"]
        ]
    ),
]

@pytest.mark.parametrize("mz_values, precursor_formula_str, tolerance_ppm, water_absorption, expected_formulas_str", TEST_CASES)
def test_decompose_spectrum_with_precursor(mz_values, precursor_formula_str, tolerance_ppm, water_absorption, expected_formulas_str):
    """
    Tests the decompose_spectrum_with_precursor function with a set of predefined test cases.
    """
    precursor_formula = formula_string_to_array(precursor_formula_str)
    df = pl.DataFrame({
        "mz": [mz_values],
        "precursor_formula": [precursor_formula]
    },schema={
        "mz": pl.List(pl.Float64),
        "precursor_formula": pl.Array(pl.Int32, NUM_ELEMENTS)
    }).with_columns(
        spectrum_struct=pl.struct(["mz", "precursor_formula"])
    )

    result_df = df.with_columns(
        decomposed=pl.col("spectrum_struct").mass_decomposer.decompose_spectrum_with_precursor(
            tolerance_ppm=tolerance_ppm,
            water_absorption=water_absorption,
            min_dbe=0.0,
            max_dbe=50.0,
            dbe_mode="any",
        )
    )

    output_formulas = result_df.item(0, "decomposed").to_list()
    expected_formulas = [[formula_string_to_array(f) for f in formulas] for formulas in expected_formulas_str]
    
    assert output_formulas == expected_formulas
