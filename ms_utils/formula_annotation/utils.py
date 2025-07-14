import re
import numpy as np
import polars as pl
import math
from functools import lru_cache
from typing import TypeVar, overload

num_elements = 15
formula_array_element_dtype = np.int64

clean_formula_pattern = r'(([A-Z][a-z]?\d*))+'
element_data = {
    'H': (1.007825, r'H(\d+|[A-Z]|$){1}'), #0
    'B':(11.009305, r'B(\d+|[A-Z]|$){1}'), #1
    'C': (12.0000, r'C(\d+|[A-Z]|$){1}'), #2
    'N': (14.003074, r'N(\d+|[A-Z]|$){1}'), #3
    'O': (15.994915, r'O(\d+|[A-Z]|$){1}'), #4
    'F': (18.998403, r'F(\d+|[A-Z]|$){1}'), #5
    'Na': (22.989770, r'Na(\d+|[A-Z]|$){1}'), #6
    'Si':(27.9769265, r'Si(\d+|[A-Z]|$){1}'), #7
    'P': (30.973762, r'P(\d+|[A-Z]|$){1}'), #8
    'S': (31.972071, r'S(\d+|[A-Z]|$){1}'), #9
    'Cl': (34.96885271, r'Cl(\d+|[A-Z]|$){1}'), #10
    'K': (38.963707, r'K(\d+|[A-Z]|$){1}'), #11
    'As':(74.921596,r'As(\d+|[A-Z]|$){1}'), #12
    'Br': (78.918338, r'Br(\d+|[A-Z]|$){1}'), #13
    'I': (126.904468, r'I(\d+|[A-Z]|$){1}'), #14
    }

element_masses = np.array([value[0] for value in element_data.values()], dtype=np.float64) #just for accelerating the checking of the formula-mass fit for fragments.

#gets a formula string and a mass, returns True/False, and the formula
@lru_cache(maxsize = None)
def formula_fits_mass(formula: str, mass: float, mass_tolerance:float=3e-6) -> bool:
    if formula is None or formula == '':
        return False
    else :
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
        # If there's an error in the calculation, return False
        return False
    
@lru_cache(maxsize = None)
def get_precursor_ion_formula_array(entry):
    formula_string = re.search(r'Formula: (.+)', entry)
    if formula_string is None:
        return np.zeros(len(element_data), dtype=formula_array_element_dtype)
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
        return np.zeros(len(element_data), dtype=formula_array_element_dtype)
    
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
    element_array = np.zeros(num_elements, dtype=formula_array_element_dtype)
    i = 0
    for element in element_data:
        element_and_num = re.search(element_data[element][1], formula)
        if element_and_num is not None:
            element_number = re.search(r'\d+', element_and_num.group())
            if element_number is not None:
                element_array[i] = int(element_number.group())
            else:
                element_array[i] = 1
        else:
            element_array[i] = 0
        i = i+1
    return element_array


T = TypeVar("T", pl.DataFrame, pl.LazyFrame)

@overload
def formula_to_array(df: pl.DataFrame, input_col_name: str, output_col_name: str) -> pl.DataFrame: ...
@overload
def formula_to_array(df: pl.LazyFrame, input_col_name: str, output_col_name: str) -> pl.LazyFrame: ...

def formula_to_array(df: T, input_col_name: str, output_col_name: str) -> T:
    # Build columns dynamically using element_data
    regex_expressions = []
    for element, (mass, regex) in element_data.items():
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
        pl.concat_list([pl.col(e) for e in element_data.keys()]).list.to_array(num_elements).alias(output_col_name)
    )
    df = df.drop(list(element_data.keys()))
    return df



if __name__ == '__main__':

    # formulas = ['C11H14BrNO2', 'C9H12BrN', 'C9H12BN', 'C9H12Br','C9H12B','Br']

    # # clear_python_cache()
    # boron=element_data['B'][1]
    # for formula in formulas:
    #     print(formula)
    #     print(format_formula_string_to_array(formula))
    #     print(clean_formula_string_to_array(formula))
    #     if re.search(boron, formula) is not None:
    #         print('Boron found')
    #     print('\n')
    # print(boron)


    # testing for formula_to_array_EPA
    main_df = pl.DataFrame({
        'MOLECULAR_FORMULA': ['C11H14BrNO2', 'C9H12BrN', 'C9H12BN', 'C9H12Br','C9H12B','Br', 'Br2']
    }).lazy()
    main_df = formula_to_array(main_df)
    print(main_df.collect())