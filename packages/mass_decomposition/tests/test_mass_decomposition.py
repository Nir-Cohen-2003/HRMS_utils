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

def formula_array_to_string(formula: list[int]) -> str:
    """Converts a formula array to a string representation, in ELEMENT_SYMBOLS order."""
    parts = []
    for i, count in enumerate(formula):
        if count > 0:
            parts.append(f"{ELEMENT_SYMBOLS[i]}{count}" if count > 1 else f"{ELEMENT_SYMBOLS[i]}")
    return "".join(parts)

def formula_string_to_array(formula_str: str) -> list[int]:
    """Converts a formula string to a formula array."""
    formula_arr = [0] * NUM_ELEMENTS
    # Use regex to find all element symbols followed by an optional number
    for symbol, count_str in re.findall(r'([A-Z][a-z]?)(\d*)', formula_str):
        count = int(count_str) if count_str else 1
        if symbol in ELEMENT_SYMBOLS:
            formula_arr[ELEMENT_SYMBOLS.index(symbol)] = count
    return formula_arr

# Test cases for mass decomposition
# Each tuple: (mass, tolerance_ppm, min_bounds, max_bounds, expected_formulas)
TEST_CASES = [
    (
        78.04695,  # Mass of C6H6
        5.0,
        {"C": 0, "H": 0},
        {"C": 10, "H": 10},
        ["C6H6"]
    ),
    (
        140.1062,
        10.0,
        {"C": 0, "H": 1, "O": 0, "N": 0},
        {"C": 20, "H": 40, "O": 10, "N": 5},
        ["C6H12N4", "C8H14NO"]
    ),
    (
        182.073165, 
        3.0,
        {"C": 0, "H": 0, "O": 0},
        {"C": 20, "H": 20, "O": 10, "P": 3, "N": 10, "S": 5, "Si": 5, "Cl": 5},
        ["C13H10O", "C9H13NOP", "C5H10N6Si", "C10H13ClN"]
    ),
    (
        182.073165, 
        3.0,
        {"C": 0, "H": 0, "O": 0},
        {"C": 20, "H": 20, "O": 10, "P": 3, "N": 10, "S": 5, "Si": 0, "Cl": 0},
        ["C13H10O", "C9H13NOP"]
    ),
    (
        182.0732, 
        3.0,
        {"C": 0, "H": 0, "O": 0},
        {"C": 20, "H": 20, "O": 10, "P": 0, "N": 10, "S": 5, "Si": 0, "Cl": 1},
        ["C13H10O", "C10H13ClN"]
    ),
    (
        112.007978, 
        5.0,
        {"C": 0, "H": 0, "O": 0,"Cl": 1},
        {"C": 20, "H": 20, "O": 10, "P": 0, "N": 10, "S": 5, "Si": 0, "Cl": 1},
        ["C6H5Cl"]
    ),
    (
        155.957461, 
        5.0,
        {"C": 0, "H": 0, "O": 0,"Cl": 1},
        {"C": 20, "H": 20, "O": 10, "P": 0, "N": 10, "S": 0, "Si": 0, "Cl": 0,"Br": 1},
        ["C6H5Br"]
    )
]

def bounds_to_array(bounds: dict[str, int]) -> list[int]:
    """Converts a dictionary of bounds to a fixed-size array."""
    bounds_arr = [0] * NUM_ELEMENTS
    for symbol, value in bounds.items():
        if symbol in ELEMENT_SYMBOLS:
            bounds_arr[ELEMENT_SYMBOLS.index(symbol)] = value
    return bounds_arr

@pytest.mark.parametrize("mass, tolerance_ppm, min_bounds_dict, max_bounds_dict, expected_formulas_str", TEST_CASES)
def test_decompose_mass(mass, tolerance_ppm, min_bounds_dict, max_bounds_dict, expected_formulas_str):
    """
    Tests the decompose_mass function with a set of predefined test cases.
    """
    df = pl.DataFrame({"mass": [mass]})

    min_bounds = bounds_to_array(min_bounds_dict)
    max_bounds = bounds_to_array(max_bounds_dict)

    result_df = df.with_columns(
        decomposed=pl.col("mass").mass_decomposer.decompose_mass(
            tolerance_ppm=tolerance_ppm,
            min_bounds=min_bounds,
            max_bounds=max_bounds,
            dbe_mode="half_integer",
        )
    )

    output_formulas_arr = result_df.item(0, "decomposed").to_list()
    expected_formulas_arr = [formula_string_to_array(s) for s in expected_formulas_str]

    sorted_output = sorted(output_formulas_arr)
    sorted_expected = sorted(expected_formulas_arr)

    output_formulas_str_sorted = sorted([formula_array_to_string(f) for f in output_formulas_arr])
    expected_formulas_str_sorted = sorted(expected_formulas_str)

    assert sorted_output == sorted_expected, \
        f"Expected {expected_formulas_str_sorted}, but got {output_formulas_str_sorted}"

def test_decompose_mass_no_result():
    """
    Tests that an empty list is returned when no formula can be found.
    """
    df = pl.DataFrame({"mass": [1.0]})  # A mass too small to form any formula

    min_bounds = bounds_to_array({"C": 1})
    max_bounds = bounds_to_array({"C": 10})

    result_df = df.with_columns(
        decomposed=pl.col("mass").mass_decomposer.decompose_mass(
            min_bounds=min_bounds,
            max_bounds=max_bounds,
        )
    )
    
    assert result_df.item(0, "decomposed").to_list() == []

def test_decompose_mass_output_type_and_shape():
    """
    Tests the output type and shape of the decompose_mass function.
    """
    df = pl.DataFrame({"mass": [100.0, 200.0]})
    
    result_df = df.with_columns(
        decomposed=pl.col("mass").mass_decomposer.decompose_mass()
    )

    decomposed_col = result_df["decomposed"]
    
    expected_dtype = pl.List(pl.Array(pl.Int32, NUM_ELEMENTS))
    assert decomposed_col.dtype == expected_dtype, f"Column dtype should be {expected_dtype}, but got {decomposed_col.dtype}"

    assert len(result_df) == len(df), "Number of rows should not change"

def test_decompose_mass_with_bounds():
    """
    Tests the decompose_mass_with_bounds function with per-mass bounds.
    """
    
    min_bounds_1 = bounds_to_array({"C": 0, "H": 0})
    max_bounds_1 = bounds_to_array({"C": 10, "H": 10})
    
    min_bounds_2 = bounds_to_array({"C": 0, "H": 1, "O": 0, "N": 0})
    max_bounds_2 = bounds_to_array({"C": 20, "H": 40, "O": 10, "N": 5})

    df = pl.DataFrame({
        "mass_data": [
            {
                "mass": 78.04695, # C6H6
                "min_bounds": min_bounds_1,
                "max_bounds": max_bounds_1,
            },
            {
                "mass": 140.1062,
                "min_bounds": min_bounds_2,
                "max_bounds": max_bounds_2,
            },
        ]
    }, 
    schema={
        "mass_data": pl.Struct([
            pl.Field("mass", pl.Float64),
            pl.Field("min_bounds", pl.Array(pl.Int32, NUM_ELEMENTS)),
            pl.Field("max_bounds", pl.Array(pl.Int32, NUM_ELEMENTS)),
        ])
    })

    result_df = df.with_columns(
        decomposed=pl.col("mass_data").mass_decomposer.decompose_mass_with_bounds(
            tolerance_ppm=10.0,
            dbe_mode="half_integer",
        )
    )

    # Check first result
    output_formulas_1 = result_df.item(0, "decomposed").to_list()
    expected_formulas_1_str = ["C6H6"]
    expected_formulas_1_arr = [formula_string_to_array(s) for s in expected_formulas_1_str]
    assert sorted(output_formulas_1) == sorted(expected_formulas_1_arr)

    # Check second result
    output_formulas_2 = result_df.item(1, "decomposed").to_list()
    expected_formulas_2_str = ["C6H12N4", "C8H14NO"]
    expected_formulas_2_arr = [formula_string_to_array(s) for s in expected_formulas_2_str]
    
    sorted_output = sorted(output_formulas_2)
    sorted_expected = sorted(expected_formulas_2_arr)

    output_formulas_str_sorted = sorted([formula_array_to_string(f) for f in output_formulas_2])
    expected_formulas_str_sorted = sorted(expected_formulas_2_str)

    assert sorted_output == sorted_expected, \
        f"Expected {expected_formulas_str_sorted}, but got {output_formulas_str_sorted}"

# Test cases for spectrum decomposition
# Each tuple contains: (mz_values, precursor_formula_str, tolerance_ppm, water_absorption, expected_formulas)
SPECTRUM_TEST_CASES = [
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
    (
        [78.046950,84.056172, 104.062600, 128.062600,152.1182],
        "C8H14N3",
        5.0,
        False,
        [ #expected formulas for each mz
            ["C6H6"],
            ["C3H6N3"],
            ["C8H8"],
            [],
            ["C8H14N3"]

        ]
    ),
]

@pytest.mark.parametrize("mz_values, precursor_formula_str, tolerance_ppm, water_absorption, expected_formulas_str", SPECTRUM_TEST_CASES)
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
            min_dbe=-0.5,
            max_dbe=50.0,
            dbe_mode="half_integer",
        )
    )

    output_formulas_arr = result_df.item(0, "decomposed").to_list()
    expected_formulas_arr = [[formula_string_to_array(f) for f in formulas] for formulas in expected_formulas_str]

    sorted_output = [sorted(mz_formulas) for mz_formulas in output_formulas_arr]
    sorted_expected = [sorted(mz_formulas) for mz_formulas in expected_formulas_arr]

    output_formulas_str_for_assert = [sorted([formula_array_to_string(f) for f in mz_formulas]) for mz_formulas in output_formulas_arr]
    expected_formulas_str_for_assert = [sorted(mz_formulas_str) for mz_formulas_str in expected_formulas_str]

    assert sorted_output == sorted_expected, \
        f"Expected {expected_formulas_str_for_assert}, but got {output_formulas_str_for_assert}"
