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
        ["C9H13NOP", "C13H10O"]
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
]*2

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
        pl.col("mass").mass_decomposition.decompose_mass(
            tolerance_ppm=tolerance_ppm,
            min_bounds=min_bounds,
            max_bounds=max_bounds,
            dbe_mode="half_integer",
        ).alias("decomposed").struct.unnest()
    )
    # print("Result DataFrame:", result_df.schema)
    assert result_df.schema == pl.Schema({'mass': pl.Float64, 'formulas': pl.List(pl.Array(pl.Int32, shape=(15,))), 'formulas_str': pl.List(pl.String), 'errors': pl.List(pl.Float64)})
    
    
    output_formulas_arr = result_df.item(0, "formulas").to_list()
    assert isinstance(output_formulas_arr, list), f"Expected list of formulas, but got {type(output_formulas_arr)}, which is {output_formulas_arr}"
    # print("Output formulas (arrays):", output_formulas_arr, "type:", type(output_formulas_arr))
    expected_formulas_arr = [formula_string_to_array(s) for s in expected_formulas_str]
    # return

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
        pl.col("mass").mass_decomposition.decompose_mass(
            tolerance_ppm=5.0,
            min_bounds=min_bounds,
            max_bounds=max_bounds,
        ).alias("decomposed").struct.unnest()
    )
    
    assert result_df.item(0, "formulas").to_list() == []
    assert result_df.item(0, "formulas_str").to_list() == []
    assert result_df.item(0, "errors").to_list() == []

def test_decompose_mass_output_type_and_shape():
    """
    Tests the output type and shape of the decompose_mass function.
    """
    df = pl.DataFrame({"mass": [100.0, 200.0]})
    
    result_df = df.with_columns(
        pl.col("mass").mass_decomposition.decompose_mass().alias("decomposed")
    )

    decomposed_col = result_df["decomposed"]
    
    # Check that it returns a struct with the correct fields
    assert isinstance(decomposed_col.dtype, pl.Struct), f"Expected a struct, but got {decomposed_col.dtype}"
    
    expected_fields = {
        "formulas": pl.List(pl.Array(pl.Int32, NUM_ELEMENTS)),
        "formulas_str": pl.List(pl.String),
        "errors": pl.List(pl.Float64),
    }
    
    for field in decomposed_col.dtype.fields:
        assert field.name in expected_fields, f"Unexpected field {field.name}"
        assert field.dtype == expected_fields[field.name], f"Field {field.name} has wrong dtype: expected {expected_fields[field.name]}, got {field.dtype}"

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
        ]*2
    }, 
    schema={
        "mass_data": pl.Struct([
            pl.Field("mass", pl.Float64),
            pl.Field("min_bounds", pl.Array(pl.Int32, NUM_ELEMENTS)),
            pl.Field("max_bounds", pl.Array(pl.Int32, NUM_ELEMENTS)),
        ])
    })
    print("Input DataFrame schema:", df.schema)
    result_df = df.with_columns(
        pl.col("mass_data").mass_decomposition.decompose_mass_with_bounds(
            tolerance_ppm=10.0,
            dbe_mode="half_integer",
        ).alias("decomposed").struct.unnest()
    )
    print("Result DataFrame schema:", result_df.schema)
    # print("Result DataFrame:", result_df)

    # Check first result
    output_formulas_1 = result_df.item(0, "formulas").to_list()
    expected_formulas_1_str = ["C6H6"]
    expected_formulas_1_arr = [formula_string_to_array(s) for s in expected_formulas_1_str]
    assert sorted(output_formulas_1) == sorted(expected_formulas_1_arr)

    # Check second result
    output_formulas_2 = result_df.item(1, "formulas").to_list()
    expected_formulas_2_str = ["C6H12N4", "C8H14NO"]
    expected_formulas_2_arr = [formula_string_to_array(s) for s in expected_formulas_2_str]
    
    sorted_output = sorted(output_formulas_2)
    sorted_expected = sorted(expected_formulas_2_arr)

    output_formulas_str_sorted = sorted([formula_array_to_string(f) for f in output_formulas_2])
    expected_formulas_str_sorted = sorted(expected_formulas_2_str)

    assert sorted_output == sorted_expected, \
        f"Expected {expected_formulas_str_sorted}, but got {output_formulas_str_sorted}"

# Test cases for spectrum cleaning and normalization
# Each tuple contains: (mz_values, precursor_formula_str, tolerance_ppm, water_absorption, expected_formulas)
CLEANING_TEST_CASES = [
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
    (
        [78.046950,84.056172, 104.062600, 168.113687,152.1182],
        "C8H14N3",
        5.0,
        True,
        [ #expected formulas for each mz
            ["C6H6"],
            ["C3H6N3"],
            ["C8H8"],
            ["C8H14N3O"],
            ["C8H14N3"]

        ]
    ),
]*2


@pytest.mark.parametrize("mz_values, precursor_formula_str, tolerance_ppm, water_absorption, expected_formulas_str", CLEANING_TEST_CASES)
def test_clean_and_normalize_spectrum(mz_values, precursor_formula_str, tolerance_ppm, water_absorption, expected_formulas_str):
    precursor_formula = formula_string_to_array(precursor_formula_str)
    df = pl.DataFrame({
        "mz": [mz_values],
        "intensities": [[100.0] * len(mz_values)],
        "precursor_formula": [precursor_formula]
    }, schema={
        "mz": pl.List(pl.Float64),
        "intensities": pl.List(pl.Float64),
        "precursor_formula": pl.Array(pl.Int32, NUM_ELEMENTS)
    }).with_columns(
        spectrum_struct=pl.struct(["mz", "intensities", "precursor_formula"])
    )

    result_df = df.with_columns(
        corrected=pl.col("spectrum_struct").mass_decomposition.clean_and_normalize_spectrum(
            raw_fragment_tolerance_ppm=tolerance_ppm,
            normalized_fragment_tolerance_ppm=2.0,
            min_dbe=-10.0,
            max_dbe=100.0,
            dbe_mode="half_integer",
            water_absorption=water_absorption
        )
    )

    corrected_col = result_df["corrected"]
    assert isinstance(corrected_col.dtype, pl.Struct), f"Expected a struct, but got {corrected_col.dtype}"

    # Check that the struct has the correct fields and types
    expected_fields = {
        "normalized_masses": pl.List(pl.Float64),
        "intensities": pl.List(pl.Float64),
        "formulas": pl.List(pl.Array(pl.Int32, NUM_ELEMENTS)),
        "errors_ppm": pl.List(pl.Float64),
        "formulas_str": pl.List(pl.Utf8),
    }
    for field in corrected_col.dtype.fields:
        assert field.name in expected_fields, f"Unexpected field {field.name} in corrected spectrum struct"
        assert field.dtype == expected_fields[field.name], f"Field {field.name} has wrong dtype: expected {expected_fields[field.name]}, got {field.dtype}"

    # Check that only fragments with a formula are returned
    expected_formulas_flat = [f for formulas in expected_formulas_str for f in formulas]
    expected_formulas_arr = [formula_string_to_array(s) for s in expected_formulas_flat]
    
    output_struct = corrected_col.to_list()[0]
    output_formulas_arr = output_struct["formulas"]
    output_formulas_str = output_struct["formulas_str"]

    # Check that string formulas match array formulas
    for i, formula_arr in enumerate(output_formulas_arr):
        assert formula_array_to_string(formula_arr) == output_formulas_str[i], f"Formula string does not match array for formula {i}: expected {formula_array_to_string(formula_arr)}, got {output_formulas_str[i]}"

    assert len(output_formulas_arr) <= len(expected_formulas_arr), f"Expected at most {len(expected_formulas_arr)} formulas, but got {len(output_formulas_arr)}"

    # Check that the correct formula is chosen (compare arrays, but print strings on error)
    for formula_arr in output_formulas_arr:
        formula_str = formula_array_to_string(formula_arr)
        assert formula_arr in expected_formulas_arr, \
            f"Unexpected formula {formula_str} in output, expected one of {expected_formulas_flat}"

if __name__ == "__main__":
    test_decompose_mass(TEST_CASES[0][0], TEST_CASES[0][1], TEST_CASES[0][2], TEST_CASES[0][3], TEST_CASES[0][4])