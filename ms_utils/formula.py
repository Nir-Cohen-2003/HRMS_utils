import re
import numpy as np
import polars as pl
import math
from functools import lru_cache

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
    except:
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


def formula_to_array_EPA(main_df):
    main_df = main_df.with_columns(
        pl.col('MOLECULAR_FORMULA').str.extract(element_data['H'][1],2).str.to_integer(strict=False).cast(pl.UInt16).fill_null(strategy='zero').alias('H'),
        pl.col('MOLECULAR_FORMULA').str.extract(element_data['B'][1],2).str.to_integer(strict=False).cast(pl.UInt16).fill_null(strategy='zero').alias('B'),
        pl.col('MOLECULAR_FORMULA').str.extract(element_data['C'][1],2).str.to_integer(strict=False).cast(pl.UInt16).fill_null(strategy='zero').alias('C'),
        pl.col('MOLECULAR_FORMULA').str.extract(element_data['N'][1],2).str.to_integer(strict=False).cast(pl.UInt16).fill_null(strategy='zero').alias('N'),
        pl.col('MOLECULAR_FORMULA').str.extract(element_data['O'][1],2).str.to_integer(strict=False).cast(pl.UInt16).fill_null(strategy='zero').alias('O'),
        pl.col('MOLECULAR_FORMULA').str.extract(element_data['F'][1],2).str.to_integer(strict=False).cast(pl.UInt16).fill_null(strategy='zero').alias('F'),
        pl.col('MOLECULAR_FORMULA').str.extract(element_data['Na'][1],2).str.to_integer(strict=False).cast(pl.UInt16).fill_null(strategy='zero').alias('Na'),
        pl.col('MOLECULAR_FORMULA').str.extract(element_data['Si'][1],2).str.to_integer(strict=False).cast(pl.UInt16).fill_null(strategy='zero').alias('Si'),
        pl.col('MOLECULAR_FORMULA').str.extract(element_data['P'][1],2).str.to_integer(strict=False).cast(pl.UInt16).fill_null(strategy='zero').alias('P'),
        pl.col('MOLECULAR_FORMULA').str.extract(element_data['S'][1],2).str.to_integer(strict=False).cast(pl.UInt16).fill_null(strategy='zero').alias('S'),
        pl.col('MOLECULAR_FORMULA').str.extract(element_data['Cl'][1],2).str.to_integer(strict=False).cast(pl.UInt16).fill_null(strategy='zero').alias('Cl'),
        pl.col('MOLECULAR_FORMULA').str.extract(element_data['K'][1],2).str.to_integer(strict=False).cast(pl.UInt16).fill_null(strategy='zero').alias('K'),
        pl.col('MOLECULAR_FORMULA').str.extract(element_data['As'][1],2).str.to_integer(strict=False).cast(pl.UInt16).fill_null(strategy='zero').alias('As'),
        pl.col('MOLECULAR_FORMULA').str.extract(element_data['Br'][1],2).str.to_integer(strict=False).cast(pl.UInt16).fill_null(strategy='zero').alias('Br'),
        pl.col('MOLECULAR_FORMULA').str.extract(element_data['I'][1],2).str.to_integer(strict=False).cast(pl.UInt16).fill_null(strategy='zero').alias('I'),
        )

    main_df = main_df.with_columns(
        pl.concat_list([
            pl.col('H'), pl.col('C'), pl.col('N'), pl.col('O'), pl.col('F'), pl.col('Na'),
            pl.col('P'), pl.col('S'), pl.col('Cl'), pl.col('K'), pl.col('Br'), pl.col('I'),
            pl.col('B'), pl.col('Si'), pl.col('As')
        ]).list.to_array(num_elements).alias('MOLECULAR_FORMULA_array')
    )
    main_df = main_df.drop(['H', 'B', 'C', 'N', 'O', 'F', 'Na','Si', 'P', 'S', 'Cl', 'K',  'As', 'Br', 'I'])
    
    return main_df



if __name__ == '__main__':

    formulas = ['C11H14BrNO2', 'C9H12BrN', 'C9H12BN', 'C9H12Br','C9H12B','Br']

    # clear_python_cache()
    boron=element_data['B'][1]
    for formula in formulas:
        print(formula)
        print(format_formula_string_to_array(formula))
        print(clean_formula_string_to_array(formula))
        if re.search(boron, formula) is not None:
            print('Boron found')
        print('\n')
    print(boron)
