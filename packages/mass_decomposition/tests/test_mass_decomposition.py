import pytest
import polars as pl
from polars.testing import assert_frame_equal
import numpy as np
import  mass_decomposition
# The order of elements as defined in the Rust code
ELEMENT_SYMBOLS = [
    "H", "B", "C", "N", "O", "F", "Na", "Si", "P", "S", "Cl", "K", "As", "Br", "I"
]
NUM_ELEMENTS = len(ELEMENT_SYMBOLS)

def formula_array_to_string(formula: list[int]) -> str:
    """Converts a formula array to a string representation."""
    parts = []
    for i, count in enumerate(formula):
        if count > 0:
            parts.append(f"{ELEMENT_SYMBOLS[i]}{count}")
    return "".join(parts)

# Placeholder for test cases
# Each tuple contains: (mass, tolerance_ppm, min_bounds, max_bounds, expected_formulas)
TEST_CASES = [
    (
        78.04695,  # Mass of C6H6
        5.0,
        {"C": 0, "H": 0},
        {"C": 10, "H": 10},
        ["H6C6"]  # Expected formulas
    ),
    (
        140.1062,
        10.0,
        {"C": 0, "H": 1, "O": 0,"N":0},
        {"C": 20, "H": 40, "O": 10,"N":5},
        ["C6H12N4", "C8H14NO"] 
    ),
    (
        182.073165, 
        3.0,
        {"C": 0, "H": 0, "O": 0},
        {"C": 20, "H": 20, "O": 10,"P":3,"N":10,"S":5,"Si":5, "Cl":5},
        ["C13H10O","C9H13NOP","C5H10N6Si","C10H13ClN"]
    ),
    (
        182.073165, 
        3.0,
        {"C": 0, "H": 0, "O": 0},
        {"C": 20, "H": 20, "O": 10,"P":3,"N":10,"S":5,"Si":0, "Cl":0},
        ["C13H10O","C9H13NOP"]
    ),
    (
        182.073165, 
        3.0,
        {"C": 0, "H": 0, "O": 0},
        {"C": 20, "H": 20, "O": 10,"P":0,"N":10,"S":5,"Si":0, "Cl":0},
        ["C13H10O"]
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

    # Extract the list of formula arrays from the result
    output_formulas = result_df.item(0, "decomposed")
    
    # Convert formula arrays to string representations
    output_formulas_str = sorted([formula_array_to_string(f) for f in output_formulas])

import re

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
    for symbol, count_str in re.findall(r'([A-Z][a-z]?)(\d*)', formula_str):
        count = int(count_str) if count_str else 1
        if symbol in ELEMENT_SYMBOLS:
            formula_arr[ELEMENT_SYMBOLS.index(symbol)] = count
    return formula_arr

# Placeholder for test cases
# Each tuple contains: (mass, tolerance_ppm, min_bounds, max_bounds, expected_formulas)
TEST_CASES = [
    (
        78.04695,  # Mass of C6H6
        5.0,
        {"C": 0, "H": 0},
        {"C": 10, "H": 10},
        ["H6C6"]  # Expected formulas (H before C as per ELEMENT_SYMBOLS)
    ),
    (
        140.1062,
        10.0,
        {"C": 0, "H": 1, "O": 0,"N":0},
        {"C": 20, "H": 40, "O": 10,"N":5},
        ["H12C6N4", "H14C8NO"] # Expected formulas (H before C, N before O as per ELEMENT_SYMBOLS)
    ),
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

    # Extract the list of formula arrays from the result
    output_formulas_arr = result_df.item(0, "decomposed").to_list()
    
    # Convert expected formula strings to arrays
    expected_formulas_arr = [formula_string_to_array(s) for s in expected_formulas_str]

    # Sort both lists of arrays for comparison
    sorted_output = sorted(output_formulas_arr)
    sorted_expected = sorted(expected_formulas_arr)

    assert sorted_output == sorted_expected, \
        f"Expected {sorted(expected_formulas_str)}, but got {sorted([formula_array_to_string(f) for f in output_formulas_arr])}"

def test_decompose_mass_no_result():
    """
    Tests that an empty list is returned when no formula can be found.
    """
    df = pl.DataFrame({"mass": [1.0]}) # A mass too small to form any formula

    # Set bounds that are impossible to satisfy with the given mass
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
    
    # Check that the column has the correct data type
    expected_dtype = pl.List(pl.Array(pl.Int32, NUM_ELEMENTS))
    assert decomposed_col.dtype == expected_dtype, f"Column dtype should be {expected_dtype}, but got {decomposed_col.dtype}"

    # Check that the number of rows is correct
    assert len(result_df) == len(df), "Number of rows should not change"