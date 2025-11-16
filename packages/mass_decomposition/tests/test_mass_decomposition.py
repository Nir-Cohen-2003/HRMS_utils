import pytest
import polars as pl
from polars.testing import assert_frame_equal
import numpy as np
import mass_decomposition
import re

# The order of elements as defined in the Rust code
ELEMENT_SYMBOLS = [
    "H", "C", "N", "O", "F", "Na", "P", "S", "Cl", "K", "Br", "I"
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

def bounds_to_array(bounds: dict[str, int]) -> list[int]:
    """Converts a dictionary of bounds to a fixed-size array."""
    bounds_arr = [0] * NUM_ELEMENTS
    for symbol, value in bounds.items():
        if symbol in ELEMENT_SYMBOLS:
            bounds_arr[ELEMENT_SYMBOLS.index(symbol)] = value
    return bounds_arr

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
    assert result_df.item(0, "errors_ppm").to_list() == []

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
        "errors_ppm": pl.List(pl.Float64),
    }
    
    for field in decomposed_col.dtype.fields:
        assert field.name in expected_fields, f"Unexpected field {field.name}"
        assert field.dtype == expected_fields[field.name], f"Field {field.name} has wrong dtype: expected {expected_fields[field.name]}, got {field.dtype}"

    assert len(result_df) == len(df), "Number of rows should not change"

def test_decompose_mass_batch():
    """
    Tests the decompose_mass_with_bounds function with a batch of test cases in a single dataframe.
    This ensures that results are returned in the correct order when processing multiple masses
    with different bounds per mass.
    """
    test_cases = [
        (78.04695,{"C": 0, "H": 0}, {"C": 10, "H": 10}, ["C6H6"]),
        (140.1062, {"C": 0, "H": 1, "O": 0, "N": 0}, {"C": 20, "H": 40, "O": 10, "N": 5}, ["C6H12N4"]),
        (182.073165, {"C": 0, "H": 0, "O": 0}, {"C": 20, "H": 20, "O": 10, "P": 3, "N": 10, "S": 5, "Cl": 5}, ["C13H10O", "C9H13NOP", "C10H13ClN","C5H16N2OP2","C5H14N2O3S","C3H14N5P2","C6H16ClN2P"]),
        (182.073165,  {"C": 0, "H": 0, "O": 0}, {"C": 20, "H": 20, "O": 10, "P": 3, "N": 10, "S": 5, "Cl": 0}, ["C13H10O", "C9H13NOP", "C5H16N2OP2","C5H14N2O3S","C3H14N5P2"]),
        (182.0732, {"C": 0, "H": 0, "O": 0}, {"C": 20, "H": 20, "O": 10, "P": 0, "N": 10, "S": 5,  "Cl": 1}, ["C13H10O",  "C10H13ClN","C5H14N2O3S"]),
        (112.007978, {"C": 0, "H": 0, "O": 0, "Cl": 1}, {"C": 20, "H": 20, "O": 10, "P": 0, "N": 10, "S": 5,  "Cl": 1}, ["C6H5Cl"]),
        (155.957461,  {"C": 0, "H": 0, "O": 0, "Br": 1}, {"C": 20, "H": 20, "O": 10, "P": 0, "N": 10, "S": 0, "Cl": 0, "Br": 1}, ["C6H5Br"]),
        (432.9951, {"C": 0, "H": 0, "O": 0, "F": 0, "N": 0, "S": 0, "Cl": 0, "Br": 0, "I": 0}, {"C": 10, "H": 11, "O": 6, "F": 6, "N": 2, "S": 2, "Cl": 0, "Br": 0, "I": 0}, ["C10H11F6N2O6S2"]),
    ]
    
    masses = [tc[0] for tc in test_cases]
    min_bounds_list = [bounds_to_array(tc[1]) for tc in test_cases]
    max_bounds_list = [bounds_to_array(tc[2]) for tc in test_cases]
    expected_formulas_list = [tc[3] for tc in test_cases]
    
    # Create DataFrame with per-mass bounds
    df = pl.DataFrame({
        "mass_data": [
            {
                "mass": mass,
                "min_bounds": min_bounds,
                "max_bounds": max_bounds,
            }
            for mass, min_bounds, max_bounds in zip(masses, min_bounds_list, max_bounds_list)
        ],
        "row_id": list(range(len(masses)))  # Track original order
    }, schema={
        "mass_data": pl.Struct([
            pl.Field("mass", pl.Float64),
            pl.Field("min_bounds", pl.Array(pl.Int32, NUM_ELEMENTS)),
            pl.Field("max_bounds", pl.Array(pl.Int32, NUM_ELEMENTS)),
        ]),
        "row_id": pl.Int32
    })
    
    result_df = df.with_columns(
        pl.col("mass_data").mass_decomposition.decompose_mass_with_bounds(
            min_dbe=-0.5,
            max_dbe=40.0,
            tolerance_ppm=5.0,
            dbe_mode="half_integer",
        ).alias("decomposed").struct.unnest()
    )
    
    # Verify schema
    assert result_df.schema["formulas"] == pl.List(pl.Array(pl.Int32, shape=(12,)))
    assert result_df.schema["formulas_str"] == pl.List(pl.String)
    assert result_df.schema["errors_ppm"] == pl.List(pl.Float64)
    
    # Check that we got the same number of rows back in the same order
    assert len(result_df) == len(test_cases), f"Expected {len(test_cases)} rows, got {len(result_df)}"
    assert result_df["row_id"].to_list() == list(range(len(test_cases))), "Row order was not preserved"
    
    # For each row, verify that the expected formulas are present (order within formulas list may vary)
    for i, expected_formulas_str in enumerate(expected_formulas_list):
        output_formulas_arr = result_df.item(i, "formulas").to_list()
        expected_formulas_arr = [formula_string_to_array(s) for s in expected_formulas_str]
        
        sorted_output = sorted(output_formulas_arr)
        sorted_expected = sorted(expected_formulas_arr)

        output_formulas_str_sorted = sorted([formula_array_to_string(f) for f in output_formulas_arr])
        expected_formulas_str_sorted = sorted(expected_formulas_str)

        assert sorted_expected == sorted_output, \
            f"Row {i} (mass={masses[i]}): Expected {expected_formulas_str_sorted}, but got {output_formulas_str_sorted}"

def test_clean_and_normalize_spectrum_batch_no_water():
    """
    Tests the clean_and_normalize_spectrum function with a batch of spectra WITHOUT water absorption.
    This ensures that results are returned in the correct order when processing multiple spectra.
    """
    cleaning_test_cases = [
        ([78.046950, 104.062600, 128.062600], "C10H20O5", [["C6H6"], ["C8H8"], ["C10H8"]]),
        ([78.046950, 84.056172, 104.062600, 128.062600, 152.1182], "C8H14N3", [["C6H6"], ["C3H6N3"], ["C8H8"], [], ["C8H14N3"]]),
        ([ 53.039125 , 55.0542], "C10H25N2O2", [["C4H5"], ["C4H7"]]),
        ([72.0804, 76.0389, 118.0859], "C5H12NO2", [["C4H10N"], ["C2H6NO2"], ["C5H12NO2"]]),
        ([57.0571, 58.0649, 59.0489, 69.0696, 71.0727, 73.0644, 86.0961], "C6H14NO2", [["C3H7N"], ["C3H8N"], ["C3H7O"], ["C5H9"], ["C4H9N"], ["C4H9O"], ["C5H12N"]]),

        (
            [
                52.018724,53.0131, 54.0336, 55.0288, 55.0414, 56.0366, 56.0480,
                56.0492, 57.0444, 66.0335, 67.0287, 69.0080, 69.0443, 79.0296,
                80.0239, 81.0317, 83.0236, 93.0443, 94.0646, 105.0443, 139.0571
            ],
            "C5H8N4O2",
            [
                ["C3H2N"], ["C2HN2"], ["C3H4N"], ["C2H3N2"], ["C3H5N"], ["C2H4N2"], [],
                ["C3H6N"], ["C2H5N2"], ["C4H4N"], ["C3H3N2"], ["C2HN2O"], ["C3H5N2"], ["C4H3N2"],
                ["C3H2N3"], ["C3H3N3"], ["C3H3N2O"], ["C5H5N2"], [], [], []
            ]
        ),
        (

            [386.9900, 414.9846, 432.9582, 432.9698, 432.9951],
            "C10H11F6N2O6S2",
            [
                ["C9H9F6N2O4S2"],
                ["C10H9F6N2O5S2"],
                [],
                [],
                ["C10H11F6N2O6S2"]
            ]
        )
    ]
    
    mz_list = [tc[0] for tc in cleaning_test_cases]
    precursor_formulas_str = [tc[1] for tc in cleaning_test_cases]
    expected_formulas_list = [tc[2] for tc in cleaning_test_cases]
    
    precursor_formulas = [formula_string_to_array(s) for s in precursor_formulas_str]
    intensities_list = [[100.0] * len(mz) for mz in mz_list]
    
    df = pl.DataFrame({
        "row_id": list(range(len(mz_list))),
        "mz": mz_list,
        "intensities": intensities_list,
        "precursor_formula": precursor_formulas
    }, schema={
        "row_id": pl.Int32,
        "mz": pl.List(pl.Float64),
        "intensities": pl.List(pl.Float64),
        "precursor_formula": pl.Array(pl.Int32, 12)
    }).with_columns(
        spectrum_struct=pl.struct(["mz", "intensities", "precursor_formula"])
    )
    
    result_df = df.with_columns(
        corrected=pl.col("spectrum_struct").mass_decomposition.clean_and_normalize_spectrum(
            raw_fragment_tolerance_ppm=5.0,
            normalized_fragment_tolerance_ppm=5.0,
            min_dbe=-0.5,
            max_dbe=30.0,
            dbe_mode="half_integer",
            water_absorption=False
        )
    )
    
    # Check that we got the same number of rows back in the same order
    assert len(result_df) == len(cleaning_test_cases), f"Expected {len(cleaning_test_cases)} rows, got {len(result_df)}"
    assert result_df["row_id"].to_list() == list(range(len(cleaning_test_cases))), "Row order was not preserved"
    
    # Verify each spectrum
    for i, expected_formulas_str_list in enumerate(expected_formulas_list):
        output_struct = result_df["corrected"].to_list()[i]
        output_formulas_arr = output_struct["formulas"]
        output_formulas_str = output_struct["formulas_str"]
        
        # Check that string formulas match array formulas
        for j, formula_arr in enumerate(output_formulas_arr):
            assert formula_array_to_string(formula_arr) == output_formulas_str[j], \
                f"Row {i}: Formula string does not match array for formula {j}"
        
        # Flatten expected formulas (remove empty lists which represent invalid fragments)
        expected_formulas_flat_str = [f for formulas in expected_formulas_str_list for f in formulas]
        expected_formulas_flat_arr = [formula_string_to_array(s) for s in expected_formulas_flat_str]
        
        # Sort both lists of arrays for a canonical comparison
        sorted_output = sorted(output_formulas_arr)
        sorted_expected = sorted(expected_formulas_flat_arr)

        # For better error messages, convert back to strings for the assert message
        output_formulas_str_sorted = sorted([formula_array_to_string(f) for f in output_formulas_arr])
        expected_formulas_str_sorted = sorted(expected_formulas_flat_str)

        assert sorted_expected == sorted_output, \
            f"Row {i}: Cleaned spectrum formulas do not match expected.\n" \
            f"Expected: {expected_formulas_str_sorted}\n" \
            f"Got:      {output_formulas_str_sorted}"

def test_clean_and_normalize_spectrum_batch_with_water():
    """
    Tests the clean_and_normalize_spectrum function with a batch of spectra WITH water absorption.
    This ensures that results are returned in the correct order when processing multiple spectra.
    """
    cleaning_test_cases = [
        ([78.046950, 104.062600, 128.062600], "C10H20O5", [["C6H6"], ["C8H8"], ["C10H8"]]),
        ([78.046950, 84.056172, 104.062600, 168.113687, 152.1182], "C8H14N3", [["C6H6"], ["C3H6N3"], ["C8H8"], ["C8H14N3O"], ["C8H14N3"]]),
        ([53.0385, 55.0542], "C10H25N2O2", [["C4H5"], ["C4H7"]]),
        ([72.0804, 76.0389, 118.0859], "C5H12NO2", [["C4H10N"], ["C2H6NO2"], ["C5H12NO2"]]),
        ([57.0571, 58.0649, 59.0489, 69.0696, 71.0727, 73.0644, 86.0961], "C6H14NO2", [["C3H7N"], ["C3H8N"], ["C3H7O"], ["C5H9"], ["C4H9N"], ["C4H9O"], ["C5H12N"]])
    ]
    
    mz_list = [tc[0] for tc in cleaning_test_cases]
    precursor_formulas_str = [tc[1] for tc in cleaning_test_cases]
    expected_formulas_list = [tc[2] for tc in cleaning_test_cases]
    
    precursor_formulas = [formula_string_to_array(s) for s in precursor_formulas_str]
    intensities_list = [[100.0] * len(mz) for mz in mz_list]
    
    df = pl.DataFrame({
        "row_id": list(range(len(mz_list))),
        "mz": mz_list,
        "intensities": intensities_list,
        "precursor_formula": precursor_formulas
    }, schema={
        "row_id": pl.Int32,
        "mz": pl.List(pl.Float64),
        "intensities": pl.List(pl.Float64),
        "precursor_formula": pl.Array(pl.Int32, 12)
    }).with_columns(
        spectrum_struct=pl.struct(["mz", "intensities", "precursor_formula"])
    )
    
    result_df = df.with_columns(
        corrected=pl.col("spectrum_struct").mass_decomposition.clean_and_normalize_spectrum(
            raw_fragment_tolerance_ppm=5.0,
            normalized_fragment_tolerance_ppm=5.0,
            min_dbe=-0.5,
            max_dbe=40.0,
            dbe_mode="half_integer",
            water_absorption=True
        )
    )
    
    # Check that we got the same number of rows back in the same order
    assert len(result_df) == len(cleaning_test_cases), f"Expected {len(cleaning_test_cases)} rows, got {len(result_df)}"
    assert result_df["row_id"].to_list() == list(range(len(cleaning_test_cases))), "Row order was not preserved"
    
    # Verify each spectrum
    for i, expected_formulas_str_list in enumerate(expected_formulas_list):
        output_struct = result_df["corrected"].to_list()[i]
        output_formulas_arr = output_struct["formulas"]
        output_formulas_str = output_struct["formulas_str"]
        
        # Check that string formulas match array formulas
        for j, formula_arr in enumerate(output_formulas_arr):
            assert formula_array_to_string(formula_arr) == output_formulas_str[j], \
                f"Row {i}: Formula string does not match array for formula {j}"
        
        # Flatten expected formulas (remove empty lists which represent invalid fragments)
        expected_formulas_flat_str = [f for formulas in expected_formulas_str_list for f in formulas]
        expected_formulas_flat_arr = [formula_string_to_array(s) for s in expected_formulas_flat_str]
        
        # Sort both lists of arrays for a canonical comparison
        sorted_output = sorted(output_formulas_arr)
        sorted_expected = sorted(expected_formulas_flat_arr)

        # For better error messages, convert back to strings for the assert message
        output_formulas_str_sorted = sorted([formula_array_to_string(f) for f in output_formulas_arr])
        expected_formulas_str_sorted = sorted(expected_formulas_flat_str)

        assert sorted_expected == sorted_output, \
            f"Row {i}: Cleaned spectrum formulas do not match expected.\n" \
            f"Expected: {expected_formulas_str_sorted}\n" \
            f"Got:      {output_formulas_str_sorted}"
