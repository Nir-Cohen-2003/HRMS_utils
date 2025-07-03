# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Cython wrapper for the C++ mass decomposition implementation with OpenMP
parallelization.
"""
cimport cython
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.utility cimport pair
# The libcpp.array import is no longer needed
from typing import List, Dict, Tuple, Iterable
import numpy as np

cimport numpy as np

# C++ declarations from the header file
cdef extern from "mass_decomposer_common.hpp" namespace "FormulaAnnotation":
    cdef int NUM_ELEMENTS
    # Correctly declare as a pointer, not a fixed-size array
    cdef const char** ELEMENT_SYMBOLS
    cdef const double* ATOMIC_MASSES
    # The problematic typedef is removed from here.

cdef extern from "mass_decomposer_common.hpp":
    # Define Formula_cpp as a cppclass and declare the methods we use on it.
    # "Formula" is the actual C++ type name from the global using directive.
    cdef cppclass Formula_cpp "Formula":
        Formula_cpp() nogil
        void fill(int) nogil
        int& operator[](size_t) nogil

    cdef struct Spectrum:
        double precursor_mass
        vector[double] fragment_masses
    
    cdef struct SpectrumWithBounds:
        double precursor_mass
        vector[double] fragment_masses
        Formula_cpp precursor_min_bounds
        Formula_cpp precursor_max_bounds

    cdef struct SpectrumWithKnownPrecursor:
        Formula_cpp precursor_formula
        vector[double] fragment_masses
    
    cdef struct SpectrumDecomposition:
        Formula_cpp precursor
        vector[vector[Formula_cpp]] fragments
        double precursor_mass
        double precursor_error_ppm
        vector[vector[double]] fragment_masses
        vector[vector[double]] fragment_errors_ppm

    cdef struct ProperSpectrumResults:
        vector[SpectrumDecomposition] decompositions
    
    cdef struct DecompositionParams:
        double tolerance_ppm
        double min_dbe
        double max_dbe
        double max_hetero_ratio
        int max_results
        Formula_cpp min_bounds
        Formula_cpp max_bounds
    
    cdef cppclass MassDecomposer:
        MassDecomposer(const Formula_cpp&, const Formula_cpp&)
        vector[Formula_cpp] decompose(double, const DecompositionParams&)
        @staticmethod
        vector[vector[Formula_cpp]] decompose_parallel(const vector[double]&, const DecompositionParams&)
        @staticmethod
        vector[vector[Formula_cpp]] decompose_masses_parallel_per_bounds(const vector[double]&, const vector[pair[Formula_cpp, Formula_cpp]]&, const DecompositionParams&)
        ProperSpectrumResults decompose_spectrum(double, const vector[double]&, const DecompositionParams&)
        @staticmethod
        vector[ProperSpectrumResults] decompose_spectra_parallel(const vector[Spectrum]&, const DecompositionParams&)
        @staticmethod
        vector[ProperSpectrumResults] decompose_spectra_parallel_per_bounds(const vector[SpectrumWithBounds]&, const DecompositionParams&)
        vector[vector[Formula_cpp]] decompose_spectrum_known_precursor(const Formula_cpp&, const vector[double]&, const DecompositionParams&)
        @staticmethod
        vector[vector[vector[Formula_cpp]]] decompose_spectra_known_precursor_parallel(const vector[SpectrumWithKnownPrecursor]&, const DecompositionParams&)

# Typedef for numpy arrays
ctypedef np.int32_t F_DTYPE_t

# Helper functions for converting Python objects to C++ and vice-versa

cdef Formula_cpp _convert_numpy_to_formula(np.ndarray arr):
    """Convert a NumPy array to a C++ Formula."""
    cdef Formula_cpp formula
    cdef int* arr_ptr = <int*>np.PyArray_DATA(arr)
    for i in range(NUM_ELEMENTS):
        formula[i] = arr_ptr[i]
    return formula

cdef np.ndarray _convert_formula_to_array(const Formula_cpp& cpp_formula):
    """Convert C++ Formula to a NumPy array."""
    cdef np.ndarray arr = np.empty(NUM_ELEMENTS, dtype=np.int32)
    cdef int* arr_ptr = <int*>np.PyArray_DATA(arr)
    for i in range(NUM_ELEMENTS):
        arr_ptr[i] = cpp_formula[i]
    return arr

# cdef Formula_cpp _convert_dict_to_formula(dict py_formula):
#     """Convert Python dict to C++ Formula."""
#     cdef Formula_cpp cpp_formula
#     cpp_formula.fill(0)
#     for i in range(NUM_ELEMENTS):
#         # Remove the "FormulaAnnotation." prefix
#         symbol_str = ELEMENT_SYMBOLS[i].decode('utf-8')
#         if symbol_str in py_formula:
#             cpp_formula[i] = py_formula[symbol_str]
#     return cpp_formula

cdef void _validate_bounds_array(np.ndarray arr, str name):
    if arr.ndim != 1:
        raise TypeError(f"{name} must be a 1D array")
    if arr.shape[0] != NUM_ELEMENTS:
        raise ValueError(f"{name} must have length {NUM_ELEMENTS}")
    if arr.dtype != np.int32:
        raise TypeError(f"{name} must be of type numpy.int32")

cdef DecompositionParams _convert_params(
    double tolerance_ppm, double min_dbe, double max_dbe,
    double max_hetero_ratio, int max_results,
    np.ndarray min_bounds, np.ndarray max_bounds):
    """Convert Python parameters to C++ DecompositionParams."""
    _validate_bounds_array(min_bounds, "min_bounds")
    _validate_bounds_array(max_bounds, "max_bounds")
    
    cdef DecompositionParams params
    params.tolerance_ppm = tolerance_ppm
    params.min_dbe = min_dbe
    params.max_dbe = max_dbe
    params.max_hetero_ratio = max_hetero_ratio
    params.max_results = max_results
    params.min_bounds = _convert_numpy_to_formula(min_bounds)
    params.max_bounds = _convert_numpy_to_formula(max_bounds)
    return params

# Public Python functions

def get_element_info() -> dict:
    """Returns a dictionary with element information."""
    return {
        # Iterate C arrays by index
        'order': [ELEMENT_SYMBOLS[i].decode('utf-8') for i in range(NUM_ELEMENTS)],
        'masses': [ATOMIC_MASSES[i] for i in range(NUM_ELEMENTS)],
        'count': NUM_ELEMENTS
    }

def decompose_mass(
    target_mass: float,
    min_bounds: np.ndarray,
    max_bounds: np.ndarray,
    tolerance_ppm: float = 5.0,
    min_dbe: float = 0.0,
    max_dbe: float = 40.0,
    max_hetero_ratio: float = 100.0,
    max_results: int = 100000
) -> list[np.ndarray]:
    cdef DecompositionParams params = _convert_params(tolerance_ppm, min_dbe, max_dbe,
                                                     max_hetero_ratio, max_results,
                                                     min_bounds, max_bounds)
    cdef MassDecomposer* decomposer = new MassDecomposer(params.min_bounds, params.max_bounds)
    cdef vector[Formula_cpp] results
    
    try:
        results = decomposer.decompose(target_mass, params)
        python_results = [_convert_formula_to_array(res) for res in results]
        return python_results
    finally:
        del decomposer

def decompose_mass_parallel(
    target_masses: Iterable[float],
    min_bounds: np.ndarray,
    max_bounds: np.ndarray,
    tolerance_ppm: float = 5.0,
    min_dbe: float = 0.0,
    max_dbe: float = 40.0,
    max_hetero_ratio: float = 100.0,
    max_results: int = 100000
) -> list:
    # Convert iterable to list to allow checking for emptiness and getting length
    target_masses_list = list(target_masses)
    if not target_masses_list:
        return []
    
    cdef DecompositionParams params = _convert_params(tolerance_ppm, min_dbe, max_dbe,
                                                     max_hetero_ratio, max_results,
                                                     min_bounds, max_bounds)
    cdef vector[double] masses_vec = target_masses_list
    cdef vector[vector[Formula_cpp]] all_results
    
    all_results = MassDecomposer.decompose_parallel(masses_vec, params)
    python_results = [[_convert_formula_to_array(res) for res in mass_results] for mass_results in all_results]
    return python_results

def decompose_mass_parallel_per_bounds(
    target_masses: Iterable[float],
    per_mass_bounds: list, # List of (min_bounds_np, max_bounds_np) tuples
    tolerance_ppm: float = 5.0,
    min_dbe: float = 0.0,
    max_dbe: float = 40.0,
    max_hetero_ratio: float = 100.0,
    max_results: int = 100000
) -> list:
    # Convert iterable to list to allow checking for emptiness and getting length
    target_masses_list = list(target_masses)
    if not target_masses_list:
        return []
    if len(target_masses_list) != len(per_mass_bounds):
        raise ValueError("Length of target_masses and per_mass_bounds must be equal.")

    # Create a dummy params object just to pass some values
    cdef np.ndarray dummy_bounds = np.zeros(NUM_ELEMENTS, dtype=np.int32)
    cdef DecompositionParams params = _convert_params(tolerance_ppm, min_dbe, max_dbe,
                                                     max_hetero_ratio, max_results,
                                                     dummy_bounds, dummy_bounds)
    cdef vector[double] masses_vec = target_masses_list
    cdef vector[pair[Formula_cpp, Formula_cpp]] bounds_vec
    bounds_vec.reserve(len(per_mass_bounds))
    for min_b, max_b in per_mass_bounds:
        _validate_bounds_array(min_b, "min_bounds in list")
        _validate_bounds_array(max_b, "max_bounds in list")
        bounds_vec.push_back(pair[Formula_cpp, Formula_cpp](_convert_numpy_to_formula(min_b), _convert_numpy_to_formula(max_b)))

    cdef vector[vector[Formula_cpp]] all_results
    all_results = MassDecomposer.decompose_masses_parallel_per_bounds(masses_vec, bounds_vec, params)
    python_results = [[_convert_formula_to_array(res) for res in mass_results] for mass_results in all_results]
    return python_results

def decompose_spectrum(
    precursor_mass: float,
    fragment_masses: list,
    min_bounds: np.ndarray,
    max_bounds: np.ndarray,
    tolerance_ppm: float = 5.0,
    min_dbe: float = 0.0,
    max_dbe: float = 40.0,
    max_hetero_ratio: float = 100.0,
    max_results: int = 100000
) -> list:
    if not fragment_masses:
        return []
    cdef DecompositionParams params = _convert_params(tolerance_ppm, min_dbe, max_dbe,
                                                     max_hetero_ratio, max_results,
                                                     min_bounds, max_bounds)
    cdef vector[double] frag_masses_vec = fragment_masses
    cdef MassDecomposer* decomposer = new MassDecomposer(params.min_bounds, params.max_bounds)
    cdef ProperSpectrumResults cpp_results
    
    try:
        cpp_results = decomposer.decompose_spectrum(precursor_mass, frag_masses_vec, params)
        python_results = []
        for decomp in cpp_results.decompositions:
            py_decomp = {
                'precursor': _convert_formula_to_array(decomp.precursor),
                'precursor_mass': decomp.precursor_mass,
                'precursor_error_ppm': decomp.precursor_error_ppm,
                'fragments': [[_convert_formula_to_array(f) for f in frag_list] for frag_list in decomp.fragments],
                'fragment_masses': [list(fm) for fm in decomp.fragment_masses],
                'fragment_errors_ppm': [list(fe) for fe in decomp.fragment_errors_ppm]
            }
            python_results.append(py_decomp)
        return python_results
    finally:
        del decomposer

def decompose_spectra_parallel(
    spectra_data: Iterable[dict], # list of dicts with 'precursor_mass' and 'fragment_masses'
    min_bounds: np.ndarray,
    max_bounds: np.ndarray,
    tolerance_ppm: float = 5.0,
    min_dbe: float = 0.0,
    max_dbe: float = 40.0,
    max_hetero_ratio: float = 100.0,
    max_results: int = 100000
) -> list:
    # Convert iterable to list to allow checking for emptiness and getting length
    spectra_data_list = list(spectra_data)
    if not spectra_data_list:
        return []
    cdef DecompositionParams params = _convert_params(tolerance_ppm, min_dbe, max_dbe,
                                                     max_hetero_ratio, max_results,
                                                     min_bounds, max_bounds)
    cdef vector[Spectrum] spectra_vec
    spectra_vec.reserve(len(spectra_data_list))
    cdef Spectrum s
    for spec_data in spectra_data_list:
        s.precursor_mass = spec_data['precursor_mass']
        s.fragment_masses = spec_data['fragment_masses']
        spectra_vec.push_back(s)

    cdef vector[ProperSpectrumResults] all_cpp_results
    all_cpp_results = MassDecomposer.decompose_spectra_parallel(spectra_vec, params)
    all_python_results = []
    for cpp_results in all_cpp_results:
        python_results = []
        for decomp in cpp_results.decompositions:
            py_decomp = {
                'precursor': _convert_formula_to_array(decomp.precursor),
                'precursor_mass': decomp.precursor_mass,
                'precursor_error_ppm': decomp.precursor_error_ppm,
                'fragments': [[_convert_formula_to_array(f) for f in frag_list] for frag_list in decomp.fragments],
                'fragment_masses': [list(fm) for fm in decomp.fragment_masses],
                'fragment_errors_ppm': [list(fe) for fe in decomp.fragment_errors_ppm]
            }
            python_results.append(py_decomp)
        all_python_results.append(python_results)
    return all_python_results

def decompose_spectra_parallel_per_bounds(
    spectra_data: Iterable[dict], # list of dicts with 'precursor_mass', 'fragment_masses', 'min_bounds', 'max_bounds'
    tolerance_ppm: float = 5.0,
    min_dbe: float = 0.0,
    max_dbe: float = 40.0,
    max_hetero_ratio: float = 100.0,
    max_results: int = 100000
) -> list:
    # Convert iterable to list to allow checking for emptiness and getting length
    spectra_data_list = list(spectra_data)
    if not spectra_data_list:
        return []
    
    cdef np.ndarray dummy_bounds = np.zeros(NUM_ELEMENTS, dtype=np.int32)
    cdef DecompositionParams params = _convert_params(tolerance_ppm, min_dbe, max_dbe,
                                                     max_hetero_ratio, max_results,
                                                     dummy_bounds, dummy_bounds)
    cdef vector[SpectrumWithBounds] spectra_vec
    spectra_vec.reserve(len(spectra_data_list))
    cdef SpectrumWithBounds s
    for spec_data in spectra_data_list:
        s.precursor_mass = spec_data['precursor_mass']
        s.fragment_masses = spec_data['fragment_masses']
        _validate_bounds_array(spec_data['min_bounds'], "min_bounds in list")
        _validate_bounds_array(spec_data['max_bounds'], "max_bounds in list")
        s.precursor_min_bounds = _convert_numpy_to_formula(spec_data['min_bounds'])
        s.precursor_max_bounds = _convert_numpy_to_formula(spec_data['max_bounds'])
        spectra_vec.push_back(s)

    cdef vector[ProperSpectrumResults] all_cpp_results
    all_cpp_results = MassDecomposer.decompose_spectra_parallel_per_bounds(spectra_vec, params)
    all_python_results = []
    for cpp_results in all_cpp_results:
        python_results = []
        for decomp in cpp_results.decompositions:
            py_decomp = {
                'precursor': _convert_formula_to_array(decomp.precursor),
                'precursor_mass': decomp.precursor_mass,
                'precursor_error_ppm': decomp.precursor_error_ppm,
                'fragments': [[_convert_formula_to_array(f) for f in frag_list] for frag_list in decomp.fragments],
                'fragment_masses': [list(fm) for fm in decomp.fragment_masses],
                'fragment_errors_ppm': [list(fe) for fe in decomp.fragment_errors_ppm]
            }
            python_results.append(py_decomp)
        all_python_results.append(python_results)
    return all_python_results

def decompose_spectrum_known_precursor(
    precursor_formula: np.ndarray,
    fragment_masses: list,
    min_bounds: np.ndarray,
    max_bounds: np.ndarray,
    tolerance_ppm: float = 5.0,
    max_results: int = 100000
) -> list:
    _validate_bounds_array(precursor_formula, "precursor_formula")
    cdef DecompositionParams params = _convert_params(tolerance_ppm, -100.0, 100.0, 100.0, max_results, min_bounds, max_bounds)
    
    cdef Formula_cpp cpp_precursor = _convert_numpy_to_formula(precursor_formula)
    cdef Formula_cpp min_b = _convert_numpy_to_formula(min_bounds)

    cdef vector[double] frag_masses_vec = fragment_masses
    cdef MassDecomposer* decomposer = new MassDecomposer(min_b, cpp_precursor)
    cdef vector[vector[Formula_cpp]] results

    try:
        results = decomposer.decompose_spectrum_known_precursor(cpp_precursor, frag_masses_vec, params)
        return [[_convert_formula_to_array(f) for f in res] for res in results]
    finally:
        del decomposer

def decompose_spectra_known_precursor_parallel(
    spectra_data: Iterable[dict], # list of dicts with 'precursor_formula' (np.ndarray) and 'fragment_masses'
    min_bounds: np.ndarray,
    max_bounds: np.ndarray,
    tolerance_ppm: float = 5.0,
    max_results: int = 100000
) -> list:
    # Convert iterable to list to allow checking for emptiness and getting length
    spectra_data_list = list(spectra_data)
    if not spectra_data_list:
        return []
    cdef DecompositionParams params = _convert_params(tolerance_ppm, -100.0, 100.0, 100.0, max_results, min_bounds, max_bounds)
    
    cdef vector[SpectrumWithKnownPrecursor] spectra_vec
    spectra_vec.reserve(len(spectra_data_list))
    cdef SpectrumWithKnownPrecursor s
    for i, spec_data in enumerate(spectra_data_list):
        precursor_formula_arr = spec_data['precursor_formula']
        _validate_bounds_array(precursor_formula_arr, f"precursor_formula in spectra_data at index {i}")
        s.precursor_formula = _convert_numpy_to_formula(precursor_formula_arr)
        s.fragment_masses = spec_data['fragment_masses']
        spectra_vec.push_back(s)

    cdef vector[vector[vector[Formula_cpp]]] all_results
    all_results = MassDecomposer.decompose_spectra_known_precursor_parallel(spectra_vec, params)
    return [[[_convert_formula_to_array(f) for f in res] for res in spec_res] for spec_res in all_results]