import re
import numpy as np
import polars as pl
import math
from functools import lru_cache
from typing import TypeVar, overload
from .element_table import (
    NUM_ELEMENTS,
    ELEMENTS,
    ELEMENT_MASSES,
    ELEMENT_SYMBOLS,
    ELEMENT_REGEXES,
    validate_formula_array,
)

formula_array_element_dtype = np.int64

clean_formula_pattern = r'(([A-Z][a-z]?\d*))+'

element_masses = np.array(ELEMENT_MASSES, dtype=np.float64)  # For mass calculation
num_elements= NUM_ELEMENTS
#gets a formula string and a mass, returns True/False, and the formula
@lru_cache(maxsize = None)
def formula_fits_mass(formula: str, mass: float, mass_tolerance:float=3e-6) -> bool:
    if formula is None or formula == '':
        return False
    else:
        element_array = format_formula_string_to_array(formula)
        if np.any(element_array < 0):
            return False

    try:
        calculated_mass = np.inner(element_masses, element_array)
        calculated_mass = float(calculated_mass)
        mass = float(mass)
        return math.isclose(calculated_mass, mass, rel_tol=mass_tolerance, abs_tol=0.0)
    except Exception as e:
        print(f"Error in formula_fits_mass: {e}")
        print(f"Formula: {formula}, Mass: {mass}")
        return False

@lru_cache(maxsize = None)
def get_precursor_ion_formula_array(entry):
    formula_string = re.search(r'Formula: (.+)', entry)
    if formula_string is None:
        return np.zeros(NUM_ELEMENTS, dtype=formula_array_element_dtype)
    formula = formula_string.group(1)

    precursor_type = re.search(r'Precursor_type: (.+)', entry)
    if precursor_type is None:
        return format_formula_string_to_array(formula)
    precursor_type = (precursor_type.group(1)).removeprefix('[').removesuffix(']')
    precursor_ion_formula = precursor_type.replace('M',formula)

    return format_formula_string_to_array(precursor_ion_formula)

#gets a formula string, returns an array of the elements in the formula, can handle +- in formula.
@lru_cache(maxsize = None)
def format_formula_string_to_array(raw_formula : str) -> np.ndarray:
    global clean_formula_pattern
    main = re.search(clean_formula_pattern, raw_formula)
    if main is None:
        return np.zeros(NUM_ELEMENTS, dtype=formula_array_element_dtype)

    main = main.group()
    formula_array = clean_formula_string_to_array(main)
    add = re.search(r'[+]\d?'+clean_formula_pattern, raw_formula)
    if add is not None:
        add = add.group()
        multiplier = re.search(r'\d+', add)
        if multiplier is not None:
            multiplier = int(multiplier.group())
            formula_array = formula_array + multiplier*clean_formula_string_to_array(add)
        else:
            formula_array = formula_array + clean_formula_string_to_array(add)
    sub = re.search(r'[-]\d?'+clean_formula_pattern, raw_formula)
    if sub is not None:
        sub = sub.group()
        multiplier = re.search(r'\d+', sub)
        if multiplier is not None:
            multiplier = int(multiplier.group())
            formula_array = formula_array - multiplier*clean_formula_string_to_array(sub)
        else:
            formula_array = formula_array - clean_formula_string_to_array(sub)
    return formula_array

@lru_cache(maxsize = None)
def clean_formula_string_to_array(formula: str) -> np.ndarray:
    element_array = np.zeros(NUM_ELEMENTS, dtype=formula_array_element_dtype)
    for i, element in enumerate(ELEMENT_SYMBOLS):
        regex = ELEMENT_REGEXES[i]
        element_and_num = re.search(regex, formula)
        if element_and_num is not None:
            element_number = re.search(r'\d+', element_and_num.group())
            if element_number is not None:
                element_array[i] = int(element_number.group())
            else:
                element_array[i] = 1
        else:
            element_array[i] = 0
    return element_array

T = TypeVar("T", pl.DataFrame, pl.LazyFrame)

@overload
def formula_to_array(df: pl.DataFrame, input_col_name: str, output_col_name: str) -> pl.DataFrame: ...
@overload
def formula_to_array(df: pl.LazyFrame, input_col_name: str, output_col_name: str) -> pl.LazyFrame: ...

def formula_to_array(df: T, input_col_name: str, output_col_name: str) -> T:
    regex_expressions = []
    for i, element in enumerate(ELEMENT_SYMBOLS):
        regex = ELEMENT_REGEXES[i]
        extract_pattern = regex.replace(r'(\d+|[A-Z]|$){1}', r'(\d*)')
        regex_expressions.append(
            pl.when(pl.col(input_col_name).str.contains(regex))
            .then(
                pl.col(input_col_name).str.extract(extract_pattern, 1)
                .str.replace_all('^$', '1')
                .str.to_integer(strict=False)
            )
            .otherwise(0)
            .alias(element)
        )
    df = df.with_columns(*regex_expressions)
    df = df.with_columns(
        pl.concat_list([pl.col(e) for e in ELEMENT_SYMBOLS]).list.to_array(NUM_ELEMENTS).alias(output_col_name)
    )
    df = df.drop(list(ELEMENT_SYMBOLS))
    return df

if __name__ == '__main__':
    main_df = pl.DataFrame({
        'MOLECULAR_FORMULA': ['C11H14BrNO2', 'C9H12BrN', 'C9H12BN', 'C9H12Br','C9H12B','Br', 'Br2']
    }).lazy()
    main_df = formula_to_array(main_df, 'MOLECULAR_FORMULA', 'formula_array')
    print(main_df.collect())