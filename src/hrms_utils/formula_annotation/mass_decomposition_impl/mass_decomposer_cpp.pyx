# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Cython wrapper for the C++ mass decomposition implementation with OpenMP
parallelization.
"""
cimport cython
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.utility cimport pair
# import memcpy
from libc.string cimport memcpy
import pyarrow as pa
import polars as pl
# cimport pyarrow as pa
# The libcpp.array import is no longer needed
from typing import List, Dict, Tuple, Iterable
import numpy as np
cimport numpy as np

# C++ declarations from the header file
cdef extern from "mass_decomposer_common.hpp" namespace "FormulaAnnotation":
    cdef int NUM_ELEMENTS
    cdef const char** ELEMENT_SYMBOLS
    cdef const double* ATOMIC_MASSES
    size_t FORMULA_NBYTES() nogil
    const int* formula_data_const(const Formula_cpp&) nogil
    int* formula_data(Formula_cpp&) nogil

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
        # double max_hetero_ratio
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

def get_num_elements():
    return NUM_ELEMENTS

# Helper functions for converting Python objects to C++ and vice-versa

cdef Formula_cpp _convert_numpy_to_formula(np.ndarray arr):
    """Convert a contiguous NumPy int32 array to a C++ Formula via memcpy."""
    _validate_bounds_array(arr, "formula/bounds")
    # Ensure C-contiguous view; if not, make one-time contiguous copy (small).
    cdef np.ndarray contig = np.ascontiguousarray(arr, dtype=np.int32)
    cdef Formula_cpp formula
    # Typed memoryview guarantees C-contiguous layout for memcpy
    cdef F_DTYPE_t[::1] mv = contig
    cdef size_t nbytes = NUM_ELEMENTS * sizeof(F_DTYPE_t)
    memcpy(<void*>&formula[0], <const void*>&mv[0], nbytes)
    return formula

# Module-level cached byte size for memcpy of a single Formula
cdef size_t FORMULA_NBYTES_C = 0
FORMULA_NBYTES_C = FORMULA_NBYTES()

cdef np.ndarray _convert_formula_to_array(const Formula_cpp& cpp_formula):
    """Convert C++ Formula to a NumPy array using a single memcpy."""
    cdef np.ndarray arr = np.empty(NUM_ELEMENTS, dtype=np.int32)
    cdef void* dst = <void*> np.PyArray_DATA(arr)
    cdef const void* src = <const void*> formula_data_const(cpp_formula)
    memcpy(dst, src, FORMULA_NBYTES_C)
    return arr


cdef void _validate_bounds_array(np.ndarray arr, str name):
    if arr.ndim != 1:
        raise TypeError(f"{name} must be a 1D array")
    if arr.shape[0] != NUM_ELEMENTS:
        raise ValueError(f"{name} must have length {NUM_ELEMENTS}")
    if arr.dtype != np.int32:
        raise TypeError(f"{name} must be of type numpy.int32")

cdef DecompositionParams _convert_params(
    double tolerance_ppm, double min_dbe, double max_dbe,
    # double max_hetero_ratio,
    int max_results,
    np.ndarray min_bounds, np.ndarray max_bounds):
    """Convert Python parameters to C++ DecompositionParams."""
    _validate_bounds_array(min_bounds, "min_bounds")
    _validate_bounds_array(max_bounds, "max_bounds")
    
    cdef DecompositionParams params
    params.tolerance_ppm = tolerance_ppm
    params.min_dbe = min_dbe
    params.max_dbe = max_dbe
    # params.max_hetero_ratio = max_hetero_ratio
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
                                                    #  max_hetero_ratio,
                                                     max_results,
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
    target_masses: pl.Series, # 1D array of target masses
    min_bounds: np.ndarray, # 1D array of min bounds (shape must match NUM_ELEMENTS)
    max_bounds: np.ndarray, # 1D array of max bounds (shape must match NUM_ELEMENTS)
    tolerance_ppm: float = 5.0,
    min_dbe: float = 0.0,
    max_dbe: float = 40.0,
    max_hetero_ratio: float = 100.0,
    max_results: int = 100000
) -> pl.Series:
    target_masses = target_masses.to_numpy()

    cdef np.ndarray[double, ndim=1, mode="c"] contig_masses = np.ascontiguousarray(target_masses, dtype=np.float64)
    cdef double* masses_ptr = &contig_masses[0]
    cdef size_t n_masses = contig_masses.shape[0]
    
    cdef vector[double] masses_vec
    masses_vec.assign(masses_ptr, masses_ptr + n_masses)

    cdef DecompositionParams params = _convert_params(tolerance_ppm, min_dbe, max_dbe, max_results,min_bounds, max_bounds)
    cdef vector[vector[Formula_cpp]] all_results
    
    all_results = MassDecomposer.decompose_parallel(masses_vec, params)
    
    cdef size_t num_masses = all_results.size()
    cdef size_t total_formulas = 0
    cdef size_t i, j, k
    
    # First pass: calculate total number of formulas to pre-allocate memory
    for i in range(num_masses):
        total_formulas += all_results[i].size()
        
    # Allocate flat numpy arrays for offsets and the flattened formula data
    
    cdef np.ndarray offsets_array = np.empty(num_masses + 1, dtype=np.int32)
    cdef np.int32_t[::1] offsets_view = offsets_array  # define view for writing offsets
    cdef np.ndarray flat_formulas_array = np.empty(total_formulas * NUM_ELEMENTS, dtype=np.int32)

    # Use raw pointer + memcpy per formula (faster than k-loop)
    cdef F_DTYPE_t* dst_base = <F_DTYPE_t*> np.PyArray_DATA(flat_formulas_array)
    cdef size_t current_offset = 0
    cdef size_t formula_idx = 0
    cdef size_t num_formulas_for_mass

    for i in range(num_masses):
        offsets_view[i] = current_offset
        num_formulas_for_mass = all_results[i].size()
        for j in range(num_formulas_for_mass):
            memcpy(
                <void*> (dst_base + formula_idx * NUM_ELEMENTS),
                <const void*> formula_data_const(all_results[i][j]),
                FORMULA_NBYTES_C
            )
            formula_idx += 1
        current_offset += num_formulas_for_mass

    offsets_view[num_masses] = total_formulas
    
    # Create Arrow arrays from the numpy arrays (zero-copy).
    # These are Python objects, so we don't use cdef.
    value_array = pa.array(flat_formulas_array, type=pa.int32())
    offset_array = pa.array(offsets_array, type=pa.int32())
    
    # Build the nested array structure
    # 1. Innermost array: FixedSizeList for each formula
    formula_list_array = pa.FixedSizeListArray.from_arrays(value_array, NUM_ELEMENTS)
    # 2. Outermost array: ListArray for the list of formulas per mass
    final_array = pa.ListArray.from_arrays(offset_array, formula_list_array)
    
    return pl.from_arrow(
        data=final_array,
        schema={"decomposed_formula":pl.List(pl.Array(pl.Int32, NUM_ELEMENTS))})

def decompose_mass_parallel_per_bounds(
    target_masses: pl.Series, # 1D array of target masses
    min_bounds_per_mass: pl.Series, # series of 1D arrays of min bounds, each with shape (NUM_ELEMENTS,)
    max_bounds_per_mass: pl.Series, #   series of 1D arrays of max bounds, each with shape (NUM_ELEMENTS,)
    tolerance_ppm: float = 5.0,
    min_dbe: float = 0.0,
    max_dbe: float = 40.0,
    max_hetero_ratio: float = 100.0,
    max_results: int = 100000
) -> pl.Series:

    # target_masses = target_masses.to_numpy()
    # min_bounds_per_mass = min_bounds_per_mass.to_numpy()
    # max_bounds_per_mass = max_bounds_per_mass.to_numpy()
    
    cdef np.ndarray[double, ndim=1, mode="c"] contig_masses = np.ascontiguousarray(target_masses.to_numpy(), dtype=np.float64)
    cdef np.ndarray[np.int32_t, ndim=2, mode="c"] contig_min_bounds = np.ascontiguousarray(min_bounds_per_mass.to_numpy(), dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=2, mode="c"] contig_max_bounds = np.ascontiguousarray(max_bounds_per_mass.to_numpy(), dtype=np.int32)

    cdef int n_masses = contig_masses.shape[0]
    if n_masses == 0:
        # Returning an empty series is more consistent than raising an error
        # for an empty input, matching the behavior of other functions.
        return pl.Series("decomposed_formula", [], dtype=pl.List(pl.Array(pl.Int32, NUM_ELEMENTS)))
    if contig_min_bounds.shape[0] != n_masses or contig_max_bounds.shape[0] != n_masses:
        raise ValueError("Number of rows in min_bounds_per_mass and max_bounds_per_mass must match the number of target masses.")
    if contig_min_bounds.shape[1] != NUM_ELEMENTS or contig_max_bounds.shape[1] != NUM_ELEMENTS:
        raise ValueError(f"Number of columns in bounds arrays must be {NUM_ELEMENTS}.")

        # Create a dummy params object; min/max_bounds are ignored by the C++ function
    cdef np.ndarray dummy_bounds = np.zeros(NUM_ELEMENTS, dtype=np.int32)
    cdef DecompositionParams params = _convert_params(tolerance_ppm, min_dbe, max_dbe,
                                                     max_results,
                                                     dummy_bounds, dummy_bounds)
    
    # Efficiently populate C++ vectors from numpy arrays
    cdef vector[double] masses_vec
    cdef double* masses_ptr = &contig_masses[0]
    masses_vec.assign(masses_ptr, masses_ptr + n_masses)

    cdef vector[pair[Formula_cpp, Formula_cpp]] bounds_vec
    bounds_vec.reserve(n_masses)

    cdef np.int32_t* min_bounds_ptr = &contig_min_bounds[0, 0]
    cdef np.int32_t* max_bounds_ptr = &contig_max_bounds[0, 0]
    cdef size_t i
    cdef Formula_cpp min_f, max_f
    cdef size_t formula_size_bytes = NUM_ELEMENTS * sizeof(F_DTYPE_t)

    for i in range(n_masses):
        memcpy(<void*>&min_f[0], min_bounds_ptr + i * NUM_ELEMENTS, formula_size_bytes)
        memcpy(<void*>&max_f[0], max_bounds_ptr + i * NUM_ELEMENTS, formula_size_bytes)
        bounds_vec.push_back(pair[Formula_cpp, Formula_cpp](min_f, max_f))

    cdef vector[vector[Formula_cpp]] all_results
    all_results = MassDecomposer.decompose_masses_parallel_per_bounds(masses_vec, bounds_vec, params)

    cdef size_t num_masses = all_results.size()
    cdef size_t total_formulas = 0
    cdef size_t k

    for i in range(num_masses):
        total_formulas += all_results[i].size()

    cdef np.ndarray offsets_array = np.empty(num_masses + 1, dtype=np.int32)
    cdef np.int32_t[::1] offsets_view = offsets_array  # define view for writing offsets
    cdef np.ndarray flat_formulas_array = np.empty(total_formulas * NUM_ELEMENTS, dtype=np.int32)

    cdef F_DTYPE_t* dst_base = <F_DTYPE_t*> np.PyArray_DATA(flat_formulas_array)
    cdef size_t current_offset = 0
    cdef size_t formula_idx = 0
    cdef size_t num_formulas_for_mass

    for i in range(num_masses):
        offsets_view[i] = current_offset
        num_formulas_for_mass = all_results[i].size()
        for j in range(num_formulas_for_mass):
            memcpy(
                <void*> (dst_base + formula_idx * NUM_ELEMENTS),
                <const void*> formula_data_const(all_results[i][j]),
                FORMULA_NBYTES_C
            )
            formula_idx += 1
        current_offset += num_formulas_for_mass

    offsets_view[num_masses] = total_formulas

    value_array = pa.array(flat_formulas_array, type=pa.int32())
    offset_array = pa.array(offsets_array, type=pa.int32())

    formula_list_array = pa.FixedSizeListArray.from_arrays(value_array, NUM_ELEMENTS)
    final_array = pa.ListArray.from_arrays(offset_array, formula_list_array)

    return pl.from_arrow(
        data=final_array,
        schema={"decomposed_formula": pl.List(pl.Array(pl.Int32, NUM_ELEMENTS))})

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
                                                     max_results,
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
                                                     max_results,
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
    
    # Validate only the first bounds arrays
    _validate_bounds_array(spectra_data_list[0]['min_bounds'], "min_bounds in spectra_data[0]")
    _validate_bounds_array(spectra_data_list[0]['max_bounds'], "max_bounds in spectra_data[0]")

    cdef np.ndarray dummy_bounds = np.zeros(NUM_ELEMENTS, dtype=np.int32)
    cdef DecompositionParams params = _convert_params(tolerance_ppm, min_dbe, max_dbe,
                                                     max_results,
                                                     dummy_bounds, dummy_bounds)
    cdef vector[SpectrumWithBounds] spectra_vec
    spectra_vec.reserve(len(spectra_data_list))
    cdef SpectrumWithBounds s
    for spec_data in spectra_data_list:
        s.precursor_mass = spec_data['precursor_mass']
        s.fragment_masses = spec_data['fragment_masses']
        # No need to validate here, just convert
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
    cdef DecompositionParams params = _convert_params(tolerance_ppm, 0, 100.0, max_results, min_bounds, max_bounds)
    
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
    spectra_data: Iterable[dict],
    tolerance_ppm: float = 5.0,
    max_results: int = 100000
) -> list:
    # Convert iterable to list to allow checking for emptiness and getting length
    spectra_data_list = list(spectra_data)
    if not spectra_data_list:
        return []
    # Validate only the first precursor_formula
    precursor_formula_arr = spectra_data_list[0]['precursor_formula']
    _validate_bounds_array(precursor_formula_arr, "precursor_formula in spectra_data[0]")

    cdef np.ndarray min_bounds = np.zeros(NUM_ELEMENTS, dtype=np.int32)
    cdef DecompositionParams params = _convert_params(tolerance_ppm, 0, 30.0, max_results, min_bounds, np.zeros(NUM_ELEMENTS, dtype=np.int32))
    
    cdef vector[SpectrumWithKnownPrecursor] spectra_vec
    spectra_vec.reserve(len(spectra_data_list))
    cdef SpectrumWithKnownPrecursor s
    for spec_data in spectra_data_list:
        s.precursor_formula = _convert_numpy_to_formula(spec_data['precursor_formula'])
        s.fragment_masses = spec_data['fragment_masses']
        spectra_vec.push_back(s)

    cdef vector[vector[vector[Formula_cpp]]] all_results
    all_results = MassDecomposer.decompose_spectra_known_precursor_parallel(spectra_vec, params)
    return [[[_convert_formula_to_array(f) for f in res] for res in spec_res] for spec_res in all_results]