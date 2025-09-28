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

    cdef struct FormulaWithString:
        Formula_cpp formula
        string formula_string

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

    cdef struct SpectrumDecompositionVerbose:
        Formula_cpp precursor
        string precursor_string
        vector[vector[Formula_cpp]] fragments
        vector[vector[string] ] fragment_strings
        double precursor_mass
        double precursor_error_ppm
        vector[vector[double]] fragment_masses
        vector[vector[double]] fragment_errors_ppm

    cdef struct ProperSpectrumResultsVerbose:
        vector[SpectrumDecompositionVerbose] decompositions
    
    cdef struct DecompositionParams:
        double tolerance_ppm
        double min_dbe
        double max_dbe
        # double max_hetero_ratio
        Formula_cpp min_bounds
        Formula_cpp max_bounds

    # Declarations for cleaning API (nested types in C++ are aliased here)
    cdef cppclass CleanSpectrumWithKnownPrecursor_cpp "MassDecomposer::CleanSpectrumWithKnownPrecursor":
        Formula_cpp precursor_formula
        vector[double] fragment_masses
        vector[double] fragment_intensities
        double precursor_mass
        double max_allowed_normalized_mass_error_ppm

    cdef cppclass CleanedSpectrumResult_cpp "MassDecomposer::CleanedSpectrumResult":
        vector[double] masses
        vector[double] intensities
        vector[vector[Formula_cpp]] fragment_formulas
        vector[vector[double]] fragment_errors_ppm

    cdef cppclass CleanedSpectrumResultVerbose_cpp "MassDecomposer::CleanedSpectrumResultVerbose":
        vector[double] masses
        vector[double] intensities
        vector[vector[Formula_cpp]] fragment_formulas
        vector[vector[string] ] fragment_formulas_strings
        vector[vector[double]] fragment_errors_ppm

    # New: result for single-formula-per-fragment, normalized masses
    cdef cppclass CleanedAndNormalizedSpectrumResult_cpp "MassDecomposer::CleanedAndNormalizedSpectrumResult":
        vector[double] masses_normalized
        vector[double] intensities
        vector[Formula_cpp] fragment_formulas
        vector[double] fragment_errors_ppm

    cdef cppclass CleanedAndNormalizedSpectrumResultVerbose_cpp "MassDecomposer::CleanedAndNormalizedSpectrumResultVerbose":
        vector[double] masses_normalized
        vector[double] intensities
        vector[Formula_cpp] fragment_formulas
        vector[string] fragment_formulas_strings
        vector[double] fragment_errors_ppm

    cdef cppclass MassDecomposer:
        MassDecomposer(const Formula_cpp&, const Formula_cpp&)
        vector[Formula_cpp] decompose(double, const DecompositionParams&)
        vector[FormulaWithString] decompose_verbose(double, const DecompositionParams&)
        @staticmethod
        vector[vector[Formula_cpp]] decompose_parallel(const vector[double]&, const DecompositionParams&)
        @staticmethod
        vector[vector[Formula_cpp]] decompose_masses_parallel_per_bounds(const vector[double]&, const vector[pair[Formula_cpp, Formula_cpp]]&, const DecompositionParams&)
        @staticmethod
        vector[vector[FormulaWithString]] decompose_parallel_verbose(const vector[double]&, const DecompositionParams&)
        @staticmethod
        vector[vector[FormulaWithString]] decompose_masses_parallel_per_bounds_verbose(const vector[double]&, const vector[pair[Formula_cpp, Formula_cpp]]&, const DecompositionParams&)
        ProperSpectrumResults decompose_spectrum(double, const vector[double]&, const DecompositionParams&)
        ProperSpectrumResultsVerbose decompose_spectrum_verbose(double, const vector[double]&, const DecompositionParams&)
        @staticmethod
        vector[ProperSpectrumResults] decompose_spectra_parallel(const vector[Spectrum]&, const DecompositionParams&)
        @staticmethod
        vector[ProperSpectrumResults] decompose_spectra_parallel_per_bounds(const vector[SpectrumWithBounds]&, const DecompositionParams&)
        @staticmethod
        vector[ProperSpectrumResultsVerbose] decompose_spectra_parallel_verbose(const vector[Spectrum]&, const DecompositionParams&)
        @staticmethod
        vector[ProperSpectrumResultsVerbose] decompose_spectra_parallel_per_bounds_verbose(const vector[SpectrumWithBounds]&, const DecompositionParams&)
        vector[vector[Formula_cpp]] decompose_spectrum_known_precursor(const Formula_cpp&, const vector[double]&, const DecompositionParams&)
        vector[vector[FormulaWithString]] decompose_spectrum_known_precursor_verbose(const Formula_cpp&, const vector[double]&, const DecompositionParams&)
        @staticmethod
        vector[vector[vector[Formula_cpp]]] decompose_spectra_known_precursor_parallel(const vector[SpectrumWithKnownPrecursor]&, const DecompositionParams&)
        @staticmethod
        vector[vector[vector[FormulaWithString]]] decompose_spectra_known_precursor_parallel_verbose(const vector[SpectrumWithKnownPrecursor]&, const DecompositionParams&)
        @staticmethod
        vector[CleanedSpectrumResult_cpp] clean_spectra_known_precursor_parallel(const vector[CleanSpectrumWithKnownPrecursor_cpp]&, const DecompositionParams&)
        CleanedSpectrumResultVerbose_cpp clean_spectrum_known_precursor_verbose(const Formula_cpp&, const vector[double]&, const vector[double]&, const DecompositionParams&)
        @staticmethod
        vector[CleanedSpectrumResultVerbose_cpp] clean_spectra_known_precursor_parallel_verbose(const vector[CleanSpectrumWithKnownPrecursor_cpp]&, const DecompositionParams&)
        @staticmethod
        vector[CleanedAndNormalizedSpectrumResult_cpp] clean_and_normalize_spectra_known_precursor_parallel(const vector[CleanSpectrumWithKnownPrecursor_cpp]&, const DecompositionParams&)
        CleanedAndNormalizedSpectrumResultVerbose_cpp clean_and_normalize_spectrum_known_precursor_verbose(const Formula_cpp&, const vector[double]&, const vector[double]&, double, double, const DecompositionParams&)
        @staticmethod
        vector[CleanedAndNormalizedSpectrumResultVerbose_cpp] clean_and_normalize_spectra_known_precursor_parallel_verbose(const vector[CleanSpectrumWithKnownPrecursor_cpp]&, const DecompositionParams&)
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
    np.ndarray min_bounds, np.ndarray max_bounds):
    """Convert Python parameters to C++ DecompositionParams."""
    _validate_bounds_array(min_bounds, "min_bounds")
    _validate_bounds_array(max_bounds, "max_bounds")
    
    cdef DecompositionParams params
    params.tolerance_ppm = tolerance_ppm
    params.min_dbe = min_dbe
    params.max_dbe = max_dbe
    # params.max_hetero_ratio = max_hetero_ratio
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



def decompose_mass_parallel(
    target_masses: pl.Series, # 1D array of target masses
    min_bounds: np.ndarray, # 1D array of min bounds (shape must match NUM_ELEMENTS)
    max_bounds: np.ndarray, # 1D array of max bounds (shape must match NUM_ELEMENTS)
    tolerance_ppm: float = 5.0,
    min_dbe: float = 0.0,
    max_dbe: float = 40.0,
    max_hetero_ratio: float = 100.0,
) -> pl.Series:
    target_masses = target_masses.to_numpy()

    cdef np.ndarray[double, ndim=1, mode="c"] contig_masses = np.ascontiguousarray(target_masses, dtype=np.float64)
    cdef double* masses_ptr = &contig_masses[0]
    cdef size_t n_masses = contig_masses.shape[0]
    
    cdef vector[double] masses_vec
    masses_vec.assign(masses_ptr, masses_ptr + n_masses)

    cdef DecompositionParams params = _convert_params(tolerance_ppm, min_dbe, max_dbe, min_bounds, max_bounds)
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

def decompose_mass_parallel_verbose(
    target_masses: pl.Series,
    min_bounds: np.ndarray,
    max_bounds: np.ndarray,
    tolerance_ppm: float = 5.0,
    min_dbe: float = 0.0,
    max_dbe: float = 40.0,
    max_hetero_ratio: float = 100.0,
) -> pl.Series:
    target_masses = target_masses.to_numpy()

    cdef np.ndarray[double, ndim=1, mode="c"] contig_masses = np.ascontiguousarray(target_masses, dtype=np.float64)
    cdef double* masses_ptr = &contig_masses[0]
    cdef size_t n_masses = contig_masses.shape[0]

    if n_masses == 0:
        empty_formulas = pl.Series(
            "decomposed_formula",
            [],
            dtype=pl.List(pl.Array(pl.Int32, NUM_ELEMENTS)),
        )
        empty_strings = pl.Series(
            "decomposed_formula_str",
            [],
            dtype=pl.List(pl.Utf8),
        )
        return empty_formulas, empty_strings

    cdef vector[double] masses_vec
    masses_vec.assign(masses_ptr, masses_ptr + n_masses)

    cdef DecompositionParams params = _convert_params(tolerance_ppm, min_dbe, max_dbe, min_bounds, max_bounds)
    cdef vector[vector[FormulaWithString]] all_results = MassDecomposer.decompose_parallel_verbose(masses_vec, params)

    cdef size_t num_masses = all_results.size()
    cdef size_t total_formulas = 0
    cdef size_t total_chars = 0
    cdef size_t i, j

    for i in range(num_masses):
        total_formulas += all_results[i].size()
        for j in range(all_results[i].size()):
            total_chars += all_results[i][j].formula_string.size()

    if total_formulas > 2147483647:
        raise OverflowError("Number of formulas exceeds int32 limits for Arrow offsets")
    if total_chars > 2147483647:
        raise OverflowError("Total string length exceeds int32 limits for Arrow offsets")

    cdef np.ndarray offsets_array = np.empty(num_masses + 1, dtype=np.int32)
    cdef np.int32_t[::1] offsets_view = offsets_array
    cdef np.ndarray flat_formulas_array = np.empty(total_formulas * NUM_ELEMENTS, dtype=np.int32)

    cdef np.ndarray string_offsets_array = np.empty(total_formulas + 1, dtype=np.int32)
    cdef np.int32_t[::1] string_offsets_view = string_offsets_array
    cdef np.ndarray string_data_array = np.empty(total_chars, dtype=np.uint8)

    cdef F_DTYPE_t* dst_base = <F_DTYPE_t*> np.PyArray_DATA(flat_formulas_array)
    cdef unsigned char* string_data_ptr = <unsigned char*> np.PyArray_DATA(string_data_array)

    cdef size_t formula_idx = 0
    cdef size_t current_offset = 0
    cdef size_t char_cursor = 0
    cdef string formula_str
    cdef size_t length

    string_offsets_view[0] = 0

    for i in range(num_masses):
        offsets_view[i] = current_offset
        for j in range(all_results[i].size()):
            memcpy(
                <void*> (dst_base + formula_idx * NUM_ELEMENTS),
                <const void*> formula_data_const(all_results[i][j].formula),
                FORMULA_NBYTES_C
            )
            formula_str = all_results[i][j].formula_string
            length = formula_str.size()
            if length > 0:
                memcpy(
                    <void*>(string_data_ptr + char_cursor),
                    <const void*> formula_str.c_str(),
                    length
                )
            char_cursor += length
            formula_idx += 1
            string_offsets_view[formula_idx] = <np.int32_t>char_cursor
        current_offset += all_results[i].size()

    offsets_view[num_masses] = total_formulas
    string_offsets_view[total_formulas] = <np.int32_t>char_cursor

    value_array = pa.array(flat_formulas_array, type=pa.int32())
    offset_array = pa.array(offsets_array, type=pa.int32())
    formula_list_array = pa.FixedSizeListArray.from_arrays(value_array, NUM_ELEMENTS)
    final_array = pa.ListArray.from_arrays(offset_array, formula_list_array)

    string_offsets_buffer = pa.py_buffer(string_offsets_array)
    string_data_buffer = pa.py_buffer(string_data_array)
    string_values_array = pa.Array.from_buffers(pa.utf8(), total_formulas, [None, string_offsets_buffer, string_data_buffer])
    string_offset_array = pa.array(offsets_array, type=pa.int32())
    string_list_array = pa.ListArray.from_arrays(string_offset_array, string_values_array)

    formula_series = pl.Series("decomposed_formulas", final_array)
    string_series = pl.Series("decomposed_formulas_str", string_list_array)
    return pl.struct(formula_series, string_series,eager=True)

def decompose_mass_parallel_per_bounds(
    target_masses: pl.Series, # 1D array of target masses
    min_bounds_per_mass: pl.Series, # series of 1D arrays of min bounds, each with shape (NUM_ELEMENTS,)
    max_bounds_per_mass: pl.Series, #   series of 1D arrays of max bounds, each with shape (NUM_ELEMENTS,)
    tolerance_ppm: float = 5.0,
    min_dbe: float = 0.0,
    max_dbe: float = 40.0,
    max_hetero_ratio: float = 100.0,
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
    cdef DecompositionParams params = _convert_params(
        tolerance_ppm,
        min_dbe,
        max_dbe,
        dummy_bounds,
        dummy_bounds,
    )
    
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

def decompose_mass_parallel_per_bounds_verbose(
    target_masses: pl.Series,
    min_bounds_per_mass: pl.Series,
    max_bounds_per_mass: pl.Series,
    tolerance_ppm: float = 5.0,
    min_dbe: float = 0.0,
    max_dbe: float = 40.0,
    max_hetero_ratio: float = 100.0,
) -> pl.Series:

    cdef np.ndarray[double, ndim=1, mode="c"] contig_masses = np.ascontiguousarray(target_masses.to_numpy(), dtype=np.float64)
    cdef np.ndarray[np.int32_t, ndim=2, mode="c"] contig_min_bounds = np.ascontiguousarray(min_bounds_per_mass.to_numpy(), dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=2, mode="c"] contig_max_bounds = np.ascontiguousarray(max_bounds_per_mass.to_numpy(), dtype=np.int32)

    cdef int n_masses = contig_masses.shape[0]
    if n_masses == 0:
        empty_formulas = pl.Series(
            "decomposed_formula",
            [],
            dtype=pl.List(pl.Array(pl.Int32, NUM_ELEMENTS)),
        )
        empty_strings = pl.Series(
            "decomposed_formula_str",
            [],
            dtype=pl.List(pl.Utf8),
        )
        return empty_formulas, empty_strings
    if contig_min_bounds.shape[0] != n_masses or contig_max_bounds.shape[0] != n_masses:
        raise ValueError("Number of rows in min_bounds_per_mass and max_bounds_per_mass must match the number of target masses.")
    if contig_min_bounds.shape[1] != NUM_ELEMENTS or contig_max_bounds.shape[1] != NUM_ELEMENTS:
        raise ValueError(f"Number of columns in bounds arrays must be {NUM_ELEMENTS}.")

    cdef np.ndarray dummy_bounds = np.zeros(NUM_ELEMENTS, dtype=np.int32)
    cdef DecompositionParams params = _convert_params(
        tolerance_ppm,
        min_dbe,
        max_dbe,
        dummy_bounds,
        dummy_bounds,
    )

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

    cdef vector[vector[FormulaWithString]] all_results
    all_results = MassDecomposer.decompose_masses_parallel_per_bounds_verbose(masses_vec, bounds_vec, params)

    cdef size_t num_masses = all_results.size()
    cdef size_t total_formulas = 0
    cdef size_t total_chars = 0
    cdef size_t j

    for i in range(num_masses):
        total_formulas += all_results[i].size()
        for j in range(all_results[i].size()):
            total_chars += all_results[i][j].formula_string.size()

    if total_formulas > 2147483647:
        raise OverflowError("Number of formulas exceeds int32 limits for Arrow offsets")
    if total_chars > 2147483647:
        raise OverflowError("Total string length exceeds int32 limits for Arrow offsets")

    cdef np.ndarray offsets_array = np.empty(num_masses + 1, dtype=np.int32)
    cdef np.int32_t[::1] offsets_view = offsets_array
    cdef np.ndarray flat_formulas_array = np.empty(total_formulas * NUM_ELEMENTS, dtype=np.int32)

    cdef np.ndarray string_offsets_array = np.empty(total_formulas + 1, dtype=np.int32)
    cdef np.int32_t[::1] string_offsets_view = string_offsets_array
    cdef np.ndarray string_data_array = np.empty(total_chars, dtype=np.uint8)

    cdef F_DTYPE_t* dst_base = <F_DTYPE_t*> np.PyArray_DATA(flat_formulas_array)
    cdef unsigned char* string_data_ptr = <unsigned char*> np.PyArray_DATA(string_data_array)

    cdef size_t formula_idx = 0
    cdef size_t current_offset = 0
    cdef size_t char_cursor = 0
    cdef string formula_str
    cdef size_t length

    string_offsets_view[0] = 0

    for i in range(num_masses):
        offsets_view[i] = current_offset
        for j in range(all_results[i].size()):
            memcpy(
                <void*> (dst_base + formula_idx * NUM_ELEMENTS),
                <const void*> formula_data_const(all_results[i][j].formula),
                FORMULA_NBYTES_C
            )
            formula_str = all_results[i][j].formula_string
            length = formula_str.size()
            if length > 0:
                memcpy(
                    <void*>(string_data_ptr + char_cursor),
                    <const void*> formula_str.c_str(),
                    length
                )
            char_cursor += length
            formula_idx += 1
            string_offsets_view[formula_idx] = <np.int32_t>char_cursor
        current_offset += all_results[i].size()

    offsets_view[num_masses] = total_formulas
    string_offsets_view[total_formulas] = <np.int32_t>char_cursor

    value_array = pa.array(flat_formulas_array, type=pa.int32())
    offset_array = pa.array(offsets_array, type=pa.int32())
    formula_list_array = pa.FixedSizeListArray.from_arrays(value_array, NUM_ELEMENTS)
    final_array = pa.ListArray.from_arrays(offset_array, formula_list_array)

    string_offsets_buffer = pa.py_buffer(string_offsets_array)
    string_data_buffer = pa.py_buffer(string_data_array)
    string_values_array = pa.Array.from_buffers(pa.utf8(), total_formulas, [None, string_offsets_buffer, string_data_buffer])
    string_offset_array = pa.array(offsets_array, type=pa.int32())
    string_list_array = pa.ListArray.from_arrays(string_offset_array, string_values_array)

    formula_series = pl.Series("decomposed_formulas", final_array)
    string_series = pl.Series("decomposed_formulas_str", string_list_array)
    return pl.struct(formula_series, string_series, eager=True)

#TODO: make this run. currently its too nested, but this is the actual output we want- 
# for each spectrum, we want a list of possbile explanations, each consisting of a precursor formula and a list of fragment formulas, where each fragment can have several explanations! also we want the masses and errors.
# now this is very complicated, so it might be better to force the user to first decompose the precursor, then pass each precursor formula with the fragments to a function that decomposes the fragments with known precursor.
# we can't do it here, since this is ti be used as a polras expression, and either we get extremely nested data structures which is the current state, or we return a diferenct number of rows, which is not allowed.
def decompose_spectra_parallel(
    spectra_data: Iterable[dict], # list of dicts with 'precursor_mass' and 'fragment_masses'
    min_bounds: np.ndarray,
    max_bounds: np.ndarray,
    tolerance_ppm: float = 5.0,
    min_dbe: float = 0.0,
    max_dbe: float = 40.0,
) -> list:
    # Convert iterable to list to allow checking for emptiness and getting length
    spectra_data_list = list(spectra_data)
    if not spectra_data_list:
        return []
    cdef DecompositionParams params = _convert_params(
        tolerance_ppm,
        min_dbe,
        max_dbe,
        min_bounds,
        max_bounds,
    )
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

    return pl.Series(
        all_python_results, 
        dtype=pl.List(
                pl.Struct(
                {
                    "precursor": pl.Array(pl.Int32, len(min_bounds)),  # Precursor formula as an array
                    "precursor_mass": pl.Float64,  # Calculated mass of the precursor
                    "precursor_error_ppm": pl.Float64,  # PPM error for precursor   
                    "fragments": pl.List(pl.List(pl.Array(pl.Int32, len(min_bounds)))),  # Nested structure of fragment formulas
                    "fragment_masses": pl.List(pl.List(pl.Float64)),  # Corresponding calculated masses
                    "fragment_errors_ppm": pl.List(pl.List(pl.Float64)),  # Correspond
                }
                )
            ),
            strict=False
        )

# TODO: same as above for the uniform bounds version.
def decompose_spectra_parallel_per_bounds(
    spectra_data: Iterable[dict], # list of dicts with 'precursor_mass', 'fragment_masses', 'min_bounds', 'max_bounds'
    tolerance_ppm: float = 5.0,
    min_dbe: float = 0.0,
    max_dbe: float = 40.0,
) -> list:
    # Convert iterable to list to allow checking for emptiness and getting length
    spectra_data_list = list(spectra_data)
    if not spectra_data_list:
        return []
    
    # Validate only the first bounds arrays
    _validate_bounds_array(spectra_data_list[0]['min_bounds'], "min_bounds in spectra_data[0]")
    _validate_bounds_array(spectra_data_list[0]['max_bounds'], "max_bounds in spectra_data[0]")

    cdef np.ndarray dummy_bounds = np.zeros(NUM_ELEMENTS, dtype=np.int32)
    cdef DecompositionParams params = _convert_params(
        tolerance_ppm,
        min_dbe,
        max_dbe,
        dummy_bounds,
        dummy_bounds,
    )
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


def decompose_spectra_known_precursor_parallel(
    precursor_formula_series: pl.Series,  # series of 1D arrays shape (NUM_ELEMENTS,), dtype=int32 (pl.Array)
    fragment_masses_series: pl.Series,    # series of lists[float], variable length per spectrum
    tolerance_ppm: float = 5.0,
) -> pl.Series:
    """
    Convert Polars Series to contiguous buffers and pass to C++ parallel routine.
    - precursor_formula_series: pl.Series of fixed-size arrays (NUM_ELEMENTS) of int32.
    - fragment_masses_series: pl.Series of Python lists (variable length).
    Returns a Polars Series of List(List(Array(int32, NUM_ELEMENTS))) matching
    [spectrum][fragment][formula[NUM_ELEMENTS]] structure.
    """
    # Convert precursor formulas to a contiguous 2D int32 array
    cdef np.ndarray[np.int32_t, ndim=2, mode="c"] contig_precursors = np.ascontiguousarray(
        precursor_formula_series.to_numpy(), dtype=np.int32
    )

    cdef int n = <int>contig_precursors.shape[0]
    if n == 0:
        # Empty input -> empty series of nested list type
        return pl.Series(
            "fragment_formulas",
            [],
            dtype=pl.List(pl.List(pl.Array(pl.Int32, NUM_ELEMENTS))),
        )
    if contig_precursors.shape[1] != NUM_ELEMENTS:
        raise ValueError(f"Each precursor formula must have length {NUM_ELEMENTS} (got {contig_precursors.shape[1]}).")

    if fragment_masses_series.len() != n:
        raise ValueError("fragment_masses_series length must match precursor_formula_series length.")

    # Params: bounds are ignored by the C++ routine here; pass zeros.
    cdef np.ndarray min_bounds = np.zeros(NUM_ELEMENTS, dtype=np.int32)
    cdef np.ndarray max_bounds = np.zeros(NUM_ELEMENTS, dtype=np.int32)
    cdef DecompositionParams params = _convert_params(
        tolerance_ppm,
        0.0,
        30.0,  # dbe range for fragments with known precursor
        min_bounds,
        max_bounds,
    )

    # Prepare C++ input vector<SpectrumWithKnownPrecursor>
    cdef vector[SpectrumWithKnownPrecursor] spectra_vec
    spectra_vec.reserve(n)

    cdef np.int32_t* prec_ptr = &contig_precursors[0, 0]
    cdef size_t i
    cdef Formula_cpp prec
    cdef size_t formula_size_bytes = NUM_ELEMENTS * sizeof(F_DTYPE_t)

    # Extract fragment masses lists from Polars once (object Python lists)
    frag_lists = fragment_masses_series.to_list()

    cdef SpectrumWithKnownPrecursor s
    cdef np.ndarray[double, ndim=1, mode="c"] frag_contig
    cdef double* fptr
    cdef Py_ssize_t flen

    for i in range(n):
        # Copy precursor formula row i -> C++ Formula (memcpy for speed)
        memcpy(<void*>&prec[0], <const void*>(prec_ptr + i * NUM_ELEMENTS), formula_size_bytes)

        s.precursor_formula = prec

        # Convert fragment list i -> contiguous double buffer and assign into vector<double>
        seq = frag_lists[i] if frag_lists[i] is not None else []
        frag_contig = np.ascontiguousarray(seq, dtype=np.float64)
        flen = frag_contig.shape[0]
        if flen > 0:
            fptr = &frag_contig[0]
            s.fragment_masses.assign(fptr, fptr + flen)
        else:
            s.fragment_masses.clear()

        spectra_vec.push_back(s)

    # Call C++ parallel routine
    cdef vector[vector[vector[Formula_cpp]]] all_results
    all_results = MassDecomposer.decompose_spectra_known_precursor_parallel(spectra_vec, params)

    # Convert nested results -> Python nested lists of numpy int32 arrays
    # Shape: [n_spectra][n_fragments_for_spec][formula_array(NUM_ELEMENTS)]
    py_results = []
    cdef size_t si, fj, fk
    cdef size_t n_specs = all_results.size()
    for si in range(n_specs):
        spec_out = []
        for fj in range(all_results[si].size()):
            frag_out = []
            for fk in range(all_results[si][fj].size()):
                frag_out.append(_convert_formula_to_array(all_results[si][fj][fk]))
            spec_out.append(frag_out)
        py_results.append(spec_out)

    # Return as Polars Series with explicit nested dtype
    return pl.Series(py_results, dtype=pl.List(pl.List(pl.Array(pl.Int32, NUM_ELEMENTS))))

    
def clean_spectra_known_precursor_parallel(
    precursor_formula_series: pl.Series,   # Series of pl.Array(int32, NUM_ELEMENTS)
    fragment_masses_series: pl.Series,     # Series of list[float]
    fragment_intensities_series: pl.Series,# Series of list[float]
    tolerance_ppm: float = 5.0,
) -> pl.Series:
    """
    Parallel cleaner with known precursor.
    Returns a Series[Struct] with fields:
      - masses: List[Float64]
      - intensities: List[Float64]
      - fragment_formulas: List[List[Array(Int32, NUM_ELEMENTS)]]
      - fragment_errors_ppm: List[List[Float64]]
    Arrow-backed construction avoids per-row Python lists.
    """
    # Convert precursor formulas to 2D contiguous int32
    cdef np.ndarray[np.int32_t, ndim=2, mode="c"] contig_precursors = np.ascontiguousarray(
        precursor_formula_series.to_numpy(), dtype=np.int32
    )
    cdef int n = <int>contig_precursors.shape[0]
    if contig_precursors.shape[1] != NUM_ELEMENTS:
        raise ValueError(f"Each precursor formula must have length {NUM_ELEMENTS} (got {contig_precursors.shape[1]}).")
    if fragment_masses_series.len() != n or fragment_intensities_series.len() != n:
        raise ValueError("fragment_masses_series and fragment_intensities_series lengths must match precursor_formula_series length.")
    if n == 0:
        # Return empty struct series with correct schema
        s_masses = pl.Series("normalized_masses", [], dtype=pl.List(pl.Float64))
        s_intens = pl.Series("cleaned_intensities", [], dtype=pl.List(pl.Float64))
        s_frm = pl.Series("fragment_formulas", [], dtype=pl.List(pl.List(pl.Array(pl.Int32, NUM_ELEMENTS))))
        s_err = pl.Series("fragment_errors_ppm", [], dtype=pl.List(pl.List(pl.Float64)))
        return pl.struct(s_masses, s_intens, s_frm, s_err, eager=True)

    # Params: set DBE bounds for fragments; bounds ignored; pass zeros.
    cdef np.ndarray min_bounds = np.zeros(NUM_ELEMENTS, dtype=np.int32)
    cdef np.ndarray max_bounds = np.zeros(NUM_ELEMENTS, dtype=np.int32)
    cdef DecompositionParams params = _convert_params(
        tolerance_ppm, 0.0, 30.0,
        min_bounds, max_bounds
    )

    # Prepare input vector<CleanSpectrumWithKnownPrecursor>
    cdef vector[CleanSpectrumWithKnownPrecursor_cpp] spectra_vec
    spectra_vec.reserve(n)

    frag_mass_lists = fragment_masses_series.to_list()
    frag_int_lists = fragment_intensities_series.to_list()

    cdef np.int32_t* prec_ptr = &contig_precursors[0, 0]
    cdef size_t formula_size_bytes = NUM_ELEMENTS * sizeof(F_DTYPE_t)
    cdef size_t i
    cdef Formula_cpp prec
    cdef CleanSpectrumWithKnownPrecursor_cpp s

    cdef np.ndarray[double, ndim=1, mode="c"] mass_contig
    cdef np.ndarray[double, ndim=1, mode="c"] inten_contig
    cdef double* mptr
    cdef double* iptr
    cdef Py_ssize_t mlen, ilen

    for i in range(n):
        # Copy precursor row i
        memcpy(<void*>&prec[0], <const void*>(prec_ptr + i * NUM_ELEMENTS), formula_size_bytes)
        s.precursor_formula = prec

        # Masses list -> contiguous buffer
        seq_m = frag_mass_lists[i] if frag_mass_lists[i] is not None else []
        mass_contig = np.ascontiguousarray(seq_m, dtype=np.float64)
        mlen = mass_contig.shape[0]
        if mlen > 0:
            mptr = &mass_contig[0]
            s.fragment_masses.assign(mptr, mptr + mlen)
        else:
            s.fragment_masses.clear()

        # Intensities list -> contiguous buffer
        seq_i = frag_int_lists[i] if frag_int_lists[i] is not None else []
        inten_contig = np.ascontiguousarray(seq_i, dtype=np.float64)
        ilen = inten_contig.shape[0]
        if ilen > 0:
            iptr = &inten_contig[0]
            s.fragment_intensities.assign(iptr, iptr + ilen)
        else:
            s.fragment_intensities.clear()

        spectra_vec.push_back(s)

    # Call C++ parallel cleaner
    cdef vector[CleanedSpectrumResult_cpp] all_results
    all_results = MassDecomposer.clean_spectra_known_precursor_parallel(spectra_vec, params)

    # First pass: sizes for outer (per-spectrum) and inner (per-fragment) lists
    cdef size_t si, fj, fk
    cdef size_t n_specs = all_results.size()
    cdef size_t total_frags = 0
    cdef size_t total_formulas = 0
    cdef size_t total_masses = 0      # also equals total_frags
    cdef size_t total_intens = 0      # also equals total_frags
    cdef size_t nf = 0
    for si in range(n_specs):
        nf = all_results[si].fragment_formulas.size()
        total_frags += nf
        total_masses += all_results[si].masses.size()
        total_intens += all_results[si].intensities.size()
        for fj in range(nf):
            total_formulas += all_results[si].fragment_formulas[fj].size()

    # Offsets arrays
    cdef np.ndarray offs_frags = np.empty(n_specs + 1, dtype=np.int32)
    cdef np.int32_t[::1] offs_frags_v = offs_frags
    offs_frags_v[0] = 0

    cdef np.ndarray offs_formulas = np.empty(total_frags + 1, dtype=np.int32)
    cdef np.int32_t[::1] offs_formulas_v = offs_formulas
    offs_formulas_v[0] = 0

    # Flat buffers
    cdef np.ndarray flat_masses = np.empty(total_masses, dtype=np.float64)
    cdef double* masses_dst = <double*> np.PyArray_DATA(flat_masses)

    cdef np.ndarray flat_intens = np.empty(total_intens, dtype=np.float64)
    cdef double* intens_dst = <double*> np.PyArray_DATA(flat_intens)

    cdef np.ndarray flat_formula_vals = np.empty(total_formulas * NUM_ELEMENTS, dtype=np.int32)
    cdef F_DTYPE_t* fvals_dst = <F_DTYPE_t*> np.PyArray_DATA(flat_formula_vals)

    cdef np.ndarray flat_errors = np.empty(total_formulas, dtype=np.float64)
    cdef double* ferr_dst = <double*> np.PyArray_DATA(flat_errors)

    # Second pass: fill offsets and buffers
    cdef size_t frag_cursor = 0
    cdef size_t formula_cursor = 0
    cdef size_t mass_cursor = 0
    cdef size_t intens_cursor = 0
    cdef size_t n_mass_i = 0
    cdef size_t n_int_i = 0
    cdef size_t nf_pass2 = 0
    cdef size_t nfk = 0

    for si in range(n_specs):
        # masses/intensities (one per fragment kept)
        n_mass_i = all_results[si].masses.size()
        for fj in range(n_mass_i):
            masses_dst[mass_cursor + fj] = all_results[si].masses[fj]
        mass_cursor += n_mass_i

        n_int_i = all_results[si].intensities.size()
        for fj in range(n_int_i):
            intens_dst[intens_cursor + fj] = all_results[si].intensities[fj]
        intens_cursor += n_int_i

        # per-fragment formulas/errors
        nf_pass2 = all_results[si].fragment_formulas.size()
        for fj in range(nf_pass2):
            nfk = all_results[si].fragment_formulas[fj].size()
            # fill formulas (fixed-size lists) and errors for this fragment
            for fk in range(nfk):
                memcpy(
                    <void*>(fvals_dst + (formula_cursor + fk) * NUM_ELEMENTS),
                    <const void*> formula_data_const(all_results[si].fragment_formulas[fj][fk]),
                    FORMULA_NBYTES_C
                )
                ferr_dst[formula_cursor + fk] = all_results[si].fragment_errors_ppm[fj][fk]
            formula_cursor += nfk
            offs_formulas_v[frag_cursor + 1] = <np.int32_t>formula_cursor
            frag_cursor += 1

        offs_frags_v[si + 1] = <np.int32_t>frag_cursor

    # Build Arrow arrays
    value_masses = pa.array(flat_masses, type=pa.float64())
    value_intens = pa.array(flat_intens, type=pa.float64())
    offs_frags_arr = pa.array(offs_frags, type=pa.int32())
    masses_arr = pa.ListArray.from_arrays(offs_frags_arr, value_masses)
    intens_arr = pa.ListArray.from_arrays(offs_frags_arr, value_intens)

    # formulas nested: List (per spectrum) -> List (per fragment) -> FixedSizeList(NUM_ELEMENTS)
    formula_values_arr = pa.array(flat_formula_vals, type=pa.int32())
    fixed_formula_arr = pa.FixedSizeListArray.from_arrays(formula_values_arr, NUM_ELEMENTS)
    offs_formulas_arr = pa.array(offs_formulas, type=pa.int32())
    inner_frag_list = pa.ListArray.from_arrays(offs_formulas_arr, fixed_formula_arr)
    outer_spec_list_formulas = pa.ListArray.from_arrays(offs_frags_arr, inner_frag_list)

    # errors nested same shape as formulas
    error_values_arr = pa.array(flat_errors, type=pa.float64())
    inner_err_list = pa.ListArray.from_arrays(offs_formulas_arr, error_values_arr)
    outer_spec_list_errors = pa.ListArray.from_arrays(offs_frags_arr, inner_err_list)

    # Convert Arrow -> Polars Series and pack into a struct Series
    s_masses = pl.Series("normalized_masses", masses_arr)
    s_intensities = pl.Series("cleaned_intensities", intens_arr)
    s_formulas = pl.Series("fragment_formulas", outer_spec_list_formulas)
    s_errors = pl.Series("fragment_errors_ppm", outer_spec_list_errors)
    return pl.struct(
        s_masses,
        s_intensities,
        s_formulas,
        s_errors,
        eager=True
    )

def clean_spectra_known_precursor_parallel_verbose(
    precursor_formula_series: pl.Series,
    fragment_masses_series: pl.Series,
    fragment_intensities_series: pl.Series,
    tolerance_ppm: float = 5.0,
    ) -> pl.Series:
    cdef np.ndarray[np.int32_t, ndim=2, mode="c"] contig_precursors = np.ascontiguousarray(
        precursor_formula_series.to_numpy(), dtype=np.int32
    )
    cdef int n = <int>contig_precursors.shape[0]
    if contig_precursors.shape[1] != NUM_ELEMENTS:
        raise ValueError(f"Each precursor formula must have length {NUM_ELEMENTS} (got {contig_precursors.shape[1]}).")
    if fragment_masses_series.len() != n or fragment_intensities_series.len() != n:
        raise ValueError("fragment_masses_series and fragment_intensities_series lengths must match precursor_formula_series length.")
    if n == 0:
        s_masses = pl.Series("normalized_masses", [], dtype=pl.List(pl.Float64))
        s_intens = pl.Series("cleaned_intensities", [], dtype=pl.List(pl.Float64))
        s_frm = pl.Series("fragment_formulas", [], dtype=pl.List(pl.List(pl.Array(pl.Int32, NUM_ELEMENTS))))
        s_frm_str = pl.Series("fragment_formulas_str", [], dtype=pl.List(pl.List(pl.Utf8)))
        s_err = pl.Series("fragment_errors_ppm", [], dtype=pl.List(pl.List(pl.Float64)))
        return pl.struct(s_masses, s_intens, s_frm, s_frm_str, s_err, eager=True)

    cdef np.ndarray min_bounds = np.zeros(NUM_ELEMENTS, dtype=np.int32)
    cdef np.ndarray max_bounds = np.zeros(NUM_ELEMENTS, dtype=np.int32)
    cdef DecompositionParams params = _convert_params(
        tolerance_ppm, 0.0, 30.0,
        min_bounds,
        max_bounds,
    )

    frag_mass_lists = fragment_masses_series.to_list()
    frag_int_lists = fragment_intensities_series.to_list()

    cdef vector[CleanSpectrumWithKnownPrecursor_cpp] spectra_vec
    spectra_vec.reserve(n)

    cdef np.int32_t* prec_ptr = &contig_precursors[0, 0]
    cdef size_t formula_size_bytes = NUM_ELEMENTS * sizeof(F_DTYPE_t)
    cdef size_t i
    cdef Formula_cpp prec
    cdef CleanSpectrumWithKnownPrecursor_cpp s

    cdef np.ndarray[double, ndim=1, mode="c"] mass_contig
    cdef np.ndarray[double, ndim=1, mode="c"] inten_contig
    cdef double* mptr
    cdef double* iptr
    cdef Py_ssize_t mlen, ilen

    for i in range(n):
        memcpy(<void*>&prec[0], <const void*>(prec_ptr + i * NUM_ELEMENTS), formula_size_bytes)
        s.precursor_formula = prec

        seq_m = frag_mass_lists[i] if frag_mass_lists[i] is not None else []
        mass_contig = np.ascontiguousarray(seq_m, dtype=np.float64)
        mlen = mass_contig.shape[0]
        if mlen > 0:
            mptr = &mass_contig[0]
            s.fragment_masses.assign(mptr, mptr + mlen)
        else:
            s.fragment_masses.clear()

        seq_i = frag_int_lists[i] if frag_int_lists[i] is not None else []
        inten_contig = np.ascontiguousarray(seq_i, dtype=np.float64)
        ilen = inten_contig.shape[0]
        if ilen > 0:
            iptr = &inten_contig[0]
            s.fragment_intensities.assign(iptr, iptr + ilen)
        else:
            s.fragment_intensities.clear()

        spectra_vec.push_back(s)

    cdef vector[CleanedSpectrumResultVerbose_cpp] all_results
    all_results = MassDecomposer.clean_spectra_known_precursor_parallel_verbose(spectra_vec, params)

    cdef size_t n_specs = all_results.size()
    cdef size_t total_frags = 0
    cdef size_t total_formulas = 0
    cdef size_t total_masses = 0
    cdef size_t total_intens = 0
    cdef size_t total_chars = 0
    cdef size_t si, fj, fk

    for si in range(n_specs):
        total_frags += all_results[si].fragment_formulas.size()
        total_masses += all_results[si].masses.size()
        total_intens += all_results[si].intensities.size()
        for fj in range(all_results[si].fragment_formulas.size()):
            total_formulas += all_results[si].fragment_formulas[fj].size()
            for fk in range(all_results[si].fragment_formulas_strings[fj].size()):
                total_chars += all_results[si].fragment_formulas_strings[fj][fk].size()

    if total_frags > 2147483647:
        raise OverflowError("Number of fragments exceeds int32 limits for Arrow offsets")
    if total_formulas > 2147483647:
        raise OverflowError("Number of formulas exceeds int32 limits for Arrow offsets")
    if total_chars > 2147483647:
        raise OverflowError("Total string length exceeds int32 limits for Arrow offsets")

    cdef np.ndarray offs_frags = np.empty(n_specs + 1, dtype=np.int32)
    cdef np.int32_t[::1] offs_frags_v = offs_frags
    cdef np.ndarray offs_formulas = np.empty(total_frags + 1, dtype=np.int32)
    cdef np.int32_t[::1] offs_formulas_v = offs_formulas
    cdef np.ndarray flat_masses = np.empty(total_masses, dtype=np.float64)
    cdef double* masses_dst = <double*> np.PyArray_DATA(flat_masses)
    cdef np.ndarray flat_intens = np.empty(total_intens, dtype=np.float64)
    cdef double* intens_dst = <double*> np.PyArray_DATA(flat_intens)
    cdef np.ndarray flat_formula_vals = np.empty(total_formulas * NUM_ELEMENTS, dtype=np.int32)
    cdef F_DTYPE_t* fvals_dst = <F_DTYPE_t*> np.PyArray_DATA(flat_formula_vals)
    cdef np.ndarray flat_errors = np.empty(total_formulas, dtype=np.float64)
    cdef double* ferr_dst = <double*> np.PyArray_DATA(flat_errors)
    cdef np.ndarray string_offsets_array = np.empty(total_formulas + 1, dtype=np.int32)
    cdef np.int32_t[::1] string_offsets_view = string_offsets_array
    cdef np.ndarray string_data_array = np.empty(total_chars, dtype=np.uint8)
    cdef unsigned char* string_data_ptr = <unsigned char*> np.PyArray_DATA(string_data_array)

    cdef size_t frag_cursor = 0
    cdef size_t formula_cursor = 0
    cdef size_t mass_cursor = 0
    cdef size_t intens_cursor = 0
    cdef size_t char_cursor = 0
    cdef size_t n_mass_i = 0
    cdef size_t n_int_i = 0
    cdef size_t nf = 0
    cdef size_t nk = 0
    cdef size_t idx
    cdef string formula_str
    cdef size_t length

    offs_frags_v[0] = 0
    offs_formulas_v[0] = 0
    string_offsets_view[0] = 0

    for si in range(n_specs):
        n_mass_i = all_results[si].masses.size()
        for idx in range(n_mass_i):
            masses_dst[mass_cursor + idx] = all_results[si].masses[idx]
        mass_cursor += n_mass_i

        n_int_i = all_results[si].intensities.size()
        for idx in range(n_int_i):
            intens_dst[intens_cursor + idx] = all_results[si].intensities[idx]
        intens_cursor += n_int_i

        nf = all_results[si].fragment_formulas.size()
        for fj in range(nf):
            offs_formulas_v[frag_cursor] = <np.int32_t>formula_cursor
            nk = all_results[si].fragment_formulas[fj].size()
            for fk in range(nk):
                memcpy(
                    <void*>(fvals_dst + (formula_cursor + fk) * NUM_ELEMENTS),
                    <const void*> formula_data_const(all_results[si].fragment_formulas[fj][fk]),
                    FORMULA_NBYTES_C,
                )
                ferr_dst[formula_cursor + fk] = all_results[si].fragment_errors_ppm[fj][fk]
                formula_str = all_results[si].fragment_formulas_strings[fj][fk]
                length = formula_str.size()
                if length > 0:
                    memcpy(
                        <void*>(string_data_ptr + char_cursor),
                        <const void*> formula_str.c_str(),
                        length,
                    )
                char_cursor += length
                string_offsets_view[formula_cursor + fk + 1] = <np.int32_t>char_cursor
            formula_cursor += nk
            offs_formulas_v[frag_cursor + 1] = <np.int32_t>formula_cursor
            frag_cursor += 1
        offs_frags_v[si + 1] = <np.int32_t>frag_cursor

    string_offsets_view[formula_cursor] = <np.int32_t>char_cursor

    value_masses = pa.array(flat_masses, type=pa.float64())
    offs_frags_arr = pa.array(offs_frags, type=pa.int32())
    masses_arr = pa.ListArray.from_arrays(offs_frags_arr, value_masses)

    value_intens = pa.array(flat_intens, type=pa.float64())
    intens_arr = pa.ListArray.from_arrays(offs_frags_arr, value_intens)

    formula_values_arr = pa.array(flat_formula_vals, type=pa.int32())
    fixed_formula_arr = pa.FixedSizeListArray.from_arrays(formula_values_arr, NUM_ELEMENTS)
    offs_formulas_arr = pa.array(offs_formulas, type=pa.int32())
    inner_formula_list = pa.ListArray.from_arrays(offs_formulas_arr, fixed_formula_arr)
    outer_formula_list = pa.ListArray.from_arrays(offs_frags_arr, inner_formula_list)

    error_values_arr = pa.array(flat_errors, type=pa.float64())
    inner_error_list = pa.ListArray.from_arrays(offs_formulas_arr, error_values_arr)
    outer_error_list = pa.ListArray.from_arrays(offs_frags_arr, inner_error_list)

    string_offsets_buffer = pa.py_buffer(string_offsets_array)
    string_data_buffer = pa.py_buffer(string_data_array)
    string_values_array = pa.Array.from_buffers(pa.utf8(), total_formulas, [None, string_offsets_buffer, string_data_buffer])
    inner_string_list = pa.ListArray.from_arrays(offs_formulas_arr, string_values_array)
    outer_string_list = pa.ListArray.from_arrays(offs_frags_arr, inner_string_list)

    s_masses = pl.Series("normalized_masses", masses_arr)
    s_intensities = pl.Series("cleaned_intensities", intens_arr)
    s_formulas = pl.Series("fragment_formulas", outer_formula_list)
    s_formulas_str = pl.Series("fragment_formulas_str", outer_string_list)
    s_errors = pl.Series("fragment_errors_ppm", outer_error_list)
    return pl.struct(
        s_masses,
        s_intensities,
        s_formulas,
        s_formulas_str,
        s_errors,
        eager=True,
    )

def decompose_spectra_known_precursor_parallel_verbose(
    precursor_formula_series: pl.Series,
    fragment_masses_series: pl.Series,
    tolerance_ppm: float = 5.0,
    ) -> tuple[pl.Series, pl.Series]:
    cdef np.ndarray[np.int32_t, ndim=2, mode="c"] contig_precursors = np.ascontiguousarray(
        precursor_formula_series.to_numpy(), dtype=np.int32
    )

    cdef int n = <int>contig_precursors.shape[0]
    if n == 0:
        empty_formulas = pl.Series(
            "fragment_formulas",
            [],
            dtype=pl.List(pl.List(pl.Array(pl.Int32, NUM_ELEMENTS))),
        )
        empty_strings = pl.Series(
            "fragment_formulas_str",
            [],
            dtype=pl.List(pl.List(pl.Utf8)),
        )
        return empty_formulas, empty_strings
    if contig_precursors.shape[1] != NUM_ELEMENTS:
        raise ValueError(f"Each precursor formula must have length {NUM_ELEMENTS} (got {contig_precursors.shape[1]}).")
    if fragment_masses_series.len() != n:
        raise ValueError("fragment_masses_series length must match precursor_formula_series length.")

    cdef np.ndarray min_bounds = np.zeros(NUM_ELEMENTS, dtype=np.int32)
    cdef np.ndarray max_bounds = np.zeros(NUM_ELEMENTS, dtype=np.int32)
    cdef DecompositionParams params = _convert_params(
        tolerance_ppm, 0.0, 30.0,
        min_bounds,
        max_bounds,
    )

    cdef vector[SpectrumWithKnownPrecursor] spectra_vec
    spectra_vec.reserve(n)

    cdef np.int32_t* prec_ptr = &contig_precursors[0, 0]
    cdef size_t formula_size_bytes = NUM_ELEMENTS * sizeof(F_DTYPE_t)
    cdef size_t i
    cdef Formula_cpp prec

    frag_lists = fragment_masses_series.to_list()

    cdef SpectrumWithKnownPrecursor s
    cdef np.ndarray[double, ndim=1, mode="c"] frag_contig
    cdef double* fptr
    cdef Py_ssize_t flen

    for i in range(n):
        memcpy(<void*>&prec[0], <const void*>(prec_ptr + i * NUM_ELEMENTS), formula_size_bytes)
        s.precursor_formula = prec

        seq = frag_lists[i] if frag_lists[i] is not None else []
        frag_contig = np.ascontiguousarray(seq, dtype=np.float64)
        flen = frag_contig.shape[0]
        if flen > 0:
            fptr = &frag_contig[0]
            s.fragment_masses.assign(fptr, fptr + flen)
        else:
            s.fragment_masses.clear()

        spectra_vec.push_back(s)

    cdef vector[vector[vector[FormulaWithString]]] all_results
    all_results = MassDecomposer.decompose_spectra_known_precursor_parallel_verbose(spectra_vec, params)

    cdef size_t n_specs = all_results.size()
    cdef size_t total_fragments = 0
    cdef size_t total_formulas = 0
    cdef size_t total_chars = 0
    cdef size_t si, fj, fk

    for si in range(n_specs):
        total_fragments += all_results[si].size()
        for fj in range(all_results[si].size()):
            total_formulas += all_results[si][fj].size()
            for fk in range(all_results[si][fj].size()):
                total_chars += all_results[si][fj][fk].formula_string.size()

    if total_fragments > 2147483647:
        raise OverflowError("Number of fragments exceeds int32 limits for Arrow offsets")
    if total_formulas > 2147483647:
        raise OverflowError("Number of formulas exceeds int32 limits for Arrow offsets")
    if total_chars > 2147483647:
        raise OverflowError("Total string length exceeds int32 limits for Arrow offsets")

    cdef np.ndarray offs_specs = np.empty(n_specs + 1, dtype=np.int32)
    cdef np.int32_t[::1] offs_specs_view = offs_specs
    cdef np.ndarray offs_frags = np.empty(total_fragments + 1, dtype=np.int32)
    cdef np.int32_t[::1] offs_frags_view = offs_frags
    cdef np.ndarray flat_formulas = np.empty(total_formulas * NUM_ELEMENTS, dtype=np.int32)
    cdef F_DTYPE_t* dst_base = <F_DTYPE_t*> np.PyArray_DATA(flat_formulas)
    cdef np.ndarray string_offsets_array = np.empty(total_formulas + 1, dtype=np.int32)
    cdef np.int32_t[::1] string_offsets_view = string_offsets_array
    cdef np.ndarray string_data_array = np.empty(total_chars, dtype=np.uint8)
    cdef unsigned char* string_data_ptr = <unsigned char*> np.PyArray_DATA(string_data_array)

    cdef size_t frag_cursor = 0
    cdef size_t formula_cursor = 0
    cdef size_t char_cursor = 0
    cdef size_t nf = 0
    cdef size_t nk = 0
    cdef string formula_str
    cdef size_t length

    offs_specs_view[0] = 0
    offs_frags_view[0] = 0
    string_offsets_view[0] = 0

    for si in range(n_specs):
        nf = all_results[si].size()
        for fj in range(nf):
            offs_frags_view[frag_cursor] = <np.int32_t>formula_cursor
            nk = all_results[si][fj].size()
            if nk == 0:
                string_offsets_view[formula_cursor] = <np.int32_t>char_cursor
            for fk in range(nk):
                memcpy(
                    <void*>(dst_base + (formula_cursor + fk) * NUM_ELEMENTS),
                    <const void*> formula_data_const(all_results[si][fj][fk].formula),
                    FORMULA_NBYTES_C,
                )
                formula_str = all_results[si][fj][fk].formula_string
                length = formula_str.size()
                if length > 0:
                    memcpy(
                        <void*>(string_data_ptr + char_cursor),
                        <const void*> formula_str.c_str(),
                        length,
                    )
                char_cursor += length
                string_offsets_view[formula_cursor + fk + 1] = <np.int32_t>char_cursor
            formula_cursor += nk
            offs_frags_view[frag_cursor + 1] = <np.int32_t>formula_cursor
            frag_cursor += 1
        offs_specs_view[si + 1] = <np.int32_t>frag_cursor

    string_offsets_view[formula_cursor] = <np.int32_t>char_cursor

    offs_specs_arr = pa.array(offs_specs, type=pa.int32())
    offs_frags_arr = pa.array(offs_frags, type=pa.int32())

    value_formulas = pa.array(flat_formulas, type=pa.int32())
    fixed_formulas = pa.FixedSizeListArray.from_arrays(value_formulas, NUM_ELEMENTS)
    inner_formulas = pa.ListArray.from_arrays(offs_frags_arr, fixed_formulas)
    outer_formulas = pa.ListArray.from_arrays(offs_specs_arr, inner_formulas)

    string_offsets_buffer = pa.py_buffer(string_offsets_array)
    string_data_buffer = pa.py_buffer(string_data_array)
    string_values = pa.Array.from_buffers(pa.utf8(), total_formulas, [None, string_offsets_buffer, string_data_buffer])
    inner_strings = pa.ListArray.from_arrays(offs_frags_arr, string_values)
    outer_strings = pa.ListArray.from_arrays(offs_specs_arr, inner_strings)

    formula_series = pl.Series("fragment_formulas", outer_formulas)
    string_series = pl.Series("fragment_formulas_str", outer_strings)
    return formula_series, string_series

def clean_and_normalize_spectra_known_precursor_parallel(
    precursor_formula_series: pl.Series,   # pl.Array(int32, NUM_ELEMENTS)
    precursor_masses_series: pl.Series,    # Float64 per spectrum (observed)
    fragment_masses_series: pl.Series,     # list[float] per spectrum
    fragment_intensities_series: pl.Series,# list[float] per spectrum
    tolerance_ppm: float = 5.0,
    max_allowed_normalized_mass_error_ppm: float = 5.0
) -> pl.Series:
    """
    Normalizes fragment masses using a spectrum-level linear error model augmented by the precursor point.
    After normalization, drops fragments whose abs(normalized error ppm) > max_allowed_normalized_mass_error_ppm.
    Arrow-backed construction avoids per-row Python lists.
    """
    # Convert precursor formulas to 2D contiguous int32
    cdef np.ndarray[np.int32_t, ndim=2, mode="c"] contig_precursors = np.ascontiguousarray(
        precursor_formula_series.to_numpy(), dtype=np.int32
    )
    cdef int n = <int>contig_precursors.shape[0]
    if contig_precursors.shape[1] != NUM_ELEMENTS:
        raise ValueError(f"Each precursor formula must have length {NUM_ELEMENTS} (got {contig_precursors.shape[1]}).")
    if (fragment_masses_series.len() != n or
        fragment_intensities_series.len() != n or
        precursor_masses_series.len() != n):
        raise ValueError("All input series must have the same length.")
    if n == 0:
        s_masses = pl.Series("masses_normalized", [], dtype=pl.List(pl.Float64))
        s_intens = pl.Series("cleaned_intensities", [], dtype=pl.List(pl.Float64))
        s_frm = pl.Series("fragment_formulas", [], dtype=pl.List(pl.Array(pl.Int32, NUM_ELEMENTS)))
        s_err = pl.Series("fragment_errors_ppm", [], dtype=pl.List(pl.Float64))
        return pl.struct(s_masses, s_intens, s_frm, s_err, eager=True)

    # Params: bounds are ignored here; pass zeros; DBE range for fragments
    cdef np.ndarray min_bounds = np.zeros(NUM_ELEMENTS, dtype=np.int32)
    cdef np.ndarray max_bounds = np.zeros(NUM_ELEMENTS, dtype=np.int32)
    cdef DecompositionParams params = _convert_params(
        tolerance_ppm, 0.0, 30.0,
        min_bounds, max_bounds
    )

    # Prepare input vector<CleanSpectrumWithKnownPrecursor>
    cdef vector[CleanSpectrumWithKnownPrecursor_cpp] spectra_vec
    spectra_vec.reserve(n)

    frag_mass_lists = fragment_masses_series.to_list()
    frag_int_lists = fragment_intensities_series.to_list()
    prec_mass_list = precursor_masses_series.to_list()

    cdef np.int32_t* prec_ptr = &contig_precursors[0, 0]
    cdef size_t formula_size_bytes = NUM_ELEMENTS * sizeof(F_DTYPE_t)
    cdef size_t i
    cdef Formula_cpp prec
    cdef CleanSpectrumWithKnownPrecursor_cpp s

    cdef np.ndarray[double, ndim=1, mode="c"] mass_contig
    cdef np.ndarray[double, ndim=1, mode="c"] inten_contig
    cdef double* mptr
    cdef double* iptr
    cdef Py_ssize_t mlen, ilen

    for i in range(n):
        # Copy precursor row i
        memcpy(<void*>&prec[0], <const void*>(prec_ptr + i * NUM_ELEMENTS), formula_size_bytes)
        s.precursor_formula = prec

        # Observed precursor mass and ppm threshold
        s.precursor_mass = <double>(prec_mass_list[i] if prec_mass_list[i] is not None else 0.0)
        s.max_allowed_normalized_mass_error_ppm = <double>max_allowed_normalized_mass_error_ppm

        # Masses -> contiguous buffer
        seq_m = frag_mass_lists[i] if frag_mass_lists[i] is not None else []
        mass_contig = np.ascontiguousarray(seq_m, dtype=np.float64)
        mlen = mass_contig.shape[0]
        if mlen > 0:
            mptr = &mass_contig[0]
            s.fragment_masses.assign(mptr, mptr + mlen)
        else:
            s.fragment_masses.clear()

        # Intensities -> contiguous buffer
        seq_i = frag_int_lists[i] if frag_int_lists[i] is not None else []
        inten_contig = np.ascontiguousarray(seq_i, dtype=np.float64)
        ilen = inten_contig.shape[0]
        if ilen > 0:
            iptr = &inten_contig[0]
            s.fragment_intensities.assign(iptr, iptr + ilen)
        else:
            s.fragment_intensities.clear()

        spectra_vec.push_back(s)

    # Call C++ parallel cleaner + normalizer (single formula per fragment)
    cdef vector[CleanedAndNormalizedSpectrumResult_cpp] all_results
    all_results = MassDecomposer.clean_and_normalize_spectra_known_precursor_parallel(spectra_vec, params)

    # First pass: count kept fragments per spectrum (one formula per fragment)
    cdef size_t si, k
    cdef size_t n_specs = all_results.size()
    cdef size_t total_kept = 0
    for si in range(n_specs):
        total_kept += all_results[si].fragment_formulas.size()

    # Offsets per spectrum (shared by masses, intensities, formulas, errors)
    cdef np.ndarray offs_specs = np.empty(n_specs + 1, dtype=np.int32)
    cdef np.int32_t[::1] offs_specs_v = offs_specs
    offs_specs_v[0] = 0

    # Flat buffers
    cdef np.ndarray flat_masses_norm = np.empty(total_kept, dtype=np.float64)
    cdef double* mass_dst = <double*> np.PyArray_DATA(flat_masses_norm)

    cdef np.ndarray flat_intens = np.empty(total_kept, dtype=np.float64)
    cdef double* intens_dst = <double*> np.PyArray_DATA(flat_intens)

    cdef np.ndarray flat_formula_vals = np.empty(total_kept * NUM_ELEMENTS, dtype=np.int32)
    cdef F_DTYPE_t* fvals_dst = <F_DTYPE_t*> np.PyArray_DATA(flat_formula_vals)

    cdef np.ndarray flat_errors = np.empty(total_kept, dtype=np.float64)
    cdef double* ferr_dst = <double*> np.PyArray_DATA(flat_errors)

    # Fill buffers
    cdef size_t cursor = 0
    cdef size_t cnt = 0
    for si in range(n_specs):
        cnt = all_results[si].fragment_formulas.size()
        # masses normalized
        for k in range(cnt):
            mass_dst[cursor + k] = all_results[si].masses_normalized[k]
        # intensities
        for k in range(cnt):
            intens_dst[cursor + k] = all_results[si].intensities[k]
        # single formula per kept fragment
        for k in range(cnt):
            memcpy(
                <void*>(fvals_dst + (cursor + k) * NUM_ELEMENTS),
                <const void*> formula_data_const(all_results[si].fragment_formulas[k]),
                FORMULA_NBYTES_C
            )
            ferr_dst[cursor + k] = all_results[si].fragment_errors_ppm[k]
        cursor += cnt
        offs_specs_v[si + 1] = <np.int32_t>cursor

    # Build Arrow arrays
    offs_specs_arr = pa.array(offs_specs, type=pa.int32())

    val_masses = pa.array(flat_masses_norm, type=pa.float64())
    masses_arr = pa.ListArray.from_arrays(offs_specs_arr, val_masses)

    val_intens = pa.array(flat_intens, type=pa.float64())
    intens_arr = pa.ListArray.from_arrays(offs_specs_arr, val_intens)

    val_formulas = pa.array(flat_formula_vals, type=pa.int32())
    fixed_formulas = pa.FixedSizeListArray.from_arrays(val_formulas, NUM_ELEMENTS)
    formulas_arr = pa.ListArray.from_arrays(offs_specs_arr, fixed_formulas)

    val_errors = pa.array(flat_errors, type=pa.float64())
    errors_arr = pa.ListArray.from_arrays(offs_specs_arr, val_errors)

    # Convert Arrow -> Polars Series and pack into a struct Series
    s_masses = pl.Series("masses_normalized", masses_arr)
    s_intensities = pl.Series("cleaned_intensities", intens_arr)
    s_formulas = pl.Series("fragment_formulas", formulas_arr)
    s_errors = pl.Series("fragment_errors_ppm", errors_arr)
    return pl.struct(
        s_masses,
        s_intensities,
        s_formulas,
        s_errors,
        eager=True
    )

def clean_and_normalize_spectra_known_precursor_parallel_verbose(
    precursor_formula_series: pl.Series,
    precursor_masses_series: pl.Series,
    fragment_masses_series: pl.Series,
    fragment_intensities_series: pl.Series,
    *,
    tolerance_ppm: float = 5.0,
    max_allowed_normalized_mass_error_ppm: float = 5.0,
) -> pl.Series:
    """
    Verbose variant returning both numeric outputs and string representations for kept fragment formulas.
    """
    cdef np.ndarray[np.int32_t, ndim=2, mode="c"] contig_precursors = np.ascontiguousarray(
        precursor_formula_series.to_numpy(), dtype=np.int32
    )
    cdef int n = <int>contig_precursors.shape[0]
    if contig_precursors.shape[1] != NUM_ELEMENTS:
        raise ValueError(f"Each precursor formula must have length {NUM_ELEMENTS} (got {contig_precursors.shape[1]}).")
    if (
        fragment_masses_series.len() != n
        or fragment_intensities_series.len() != n
        or precursor_masses_series.len() != n
    ):
        raise ValueError("All input series must have the same length.")
    if n == 0:
        s_masses = pl.Series("masses_normalized", [], dtype=pl.List(pl.Float64))
        s_intens = pl.Series("cleaned_intensities", [], dtype=pl.List(pl.Float64))
        s_frm = pl.Series("fragment_formulas", [], dtype=pl.List(pl.Array(pl.Int32, NUM_ELEMENTS)))
        s_frm_str = pl.Series("fragment_formulas_str", [], dtype=pl.List(pl.Utf8))
        s_err = pl.Series("fragment_errors_ppm", [], dtype=pl.List(pl.Float64))
        return pl.struct(s_masses, s_intens, s_frm, s_frm_str, s_err, eager=True)

    cdef np.ndarray min_bounds = np.zeros(NUM_ELEMENTS, dtype=np.int32)
    cdef np.ndarray max_bounds = np.zeros(NUM_ELEMENTS, dtype=np.int32)
    cdef DecompositionParams params = _convert_params(
        tolerance_ppm,
        0.0,
        30.0,
        min_bounds,
        max_bounds,
    )

    frag_mass_lists = fragment_masses_series.to_list()
    frag_int_lists = fragment_intensities_series.to_list()
    prec_mass_list = precursor_masses_series.to_list()

    cdef vector[CleanSpectrumWithKnownPrecursor_cpp] spectra_vec
    spectra_vec.reserve(n)

    cdef np.int32_t* prec_ptr = &contig_precursors[0, 0]
    cdef size_t formula_size_bytes = NUM_ELEMENTS * sizeof(F_DTYPE_t)
    cdef size_t i
    cdef Formula_cpp prec
    cdef CleanSpectrumWithKnownPrecursor_cpp s

    cdef np.ndarray[double, ndim=1, mode="c"] mass_contig
    cdef np.ndarray[double, ndim=1, mode="c"] inten_contig
    cdef double* mptr
    cdef double* iptr
    cdef Py_ssize_t mlen
    cdef Py_ssize_t ilen

    for i in range(n):
        memcpy(<void*>&prec[0], <const void*>(prec_ptr + i * NUM_ELEMENTS), formula_size_bytes)
        s.precursor_formula = prec
        s.precursor_mass = <double>(prec_mass_list[i] if prec_mass_list[i] is not None else 0.0)
        s.max_allowed_normalized_mass_error_ppm = <double>max_allowed_normalized_mass_error_ppm

        seq_m = frag_mass_lists[i] if frag_mass_lists[i] is not None else []
        mass_contig = np.ascontiguousarray(seq_m, dtype=np.float64)
        mlen = mass_contig.shape[0]
        if mlen > 0:
            mptr = &mass_contig[0]
            s.fragment_masses.assign(mptr, mptr + mlen)
        else:
            s.fragment_masses.clear()

        seq_i = frag_int_lists[i] if frag_int_lists[i] is not None else []
        inten_contig = np.ascontiguousarray(seq_i, dtype=np.float64)
        ilen = inten_contig.shape[0]
        if ilen > 0:
            iptr = &inten_contig[0]
            s.fragment_intensities.assign(iptr, iptr + ilen)
        else:
            s.fragment_intensities.clear()

        spectra_vec.push_back(s)

    cdef vector[CleanedAndNormalizedSpectrumResultVerbose_cpp] all_results
    all_results = MassDecomposer.clean_and_normalize_spectra_known_precursor_parallel_verbose(spectra_vec, params)

    cdef size_t n_specs = all_results.size()
    cdef size_t total_kept = 0
    cdef size_t total_chars = 0
    cdef size_t si
    cdef size_t k

    for si in range(n_specs):
        total_kept += all_results[si].fragment_formulas.size()
        for k in range(all_results[si].fragment_formulas_strings.size()):
            total_chars += all_results[si].fragment_formulas_strings[k].size()

    if total_kept > 2147483647:
        raise OverflowError("Number of kept fragments exceeds int32 limits for Arrow offsets")
    if total_chars > 2147483647:
        raise OverflowError("Total string length exceeds int32 limits for Arrow offsets")

    cdef np.ndarray offs_specs = np.empty(n_specs + 1, dtype=np.int32)
    cdef np.int32_t[::1] offs_specs_v = offs_specs
    cdef np.ndarray flat_masses_norm = np.empty(total_kept, dtype=np.float64)
    cdef double* mass_dst = <double*> np.PyArray_DATA(flat_masses_norm)
    cdef np.ndarray flat_intens = np.empty(total_kept, dtype=np.float64)
    cdef double* intens_dst = <double*> np.PyArray_DATA(flat_intens)
    cdef np.ndarray flat_formula_vals = np.empty(max(total_kept, 1) * NUM_ELEMENTS, dtype=np.int32)
    cdef F_DTYPE_t* fvals_dst = <F_DTYPE_t*> np.PyArray_DATA(flat_formula_vals)
    cdef np.ndarray flat_errors = np.empty(total_kept, dtype=np.float64)
    cdef double* ferr_dst = <double*> np.PyArray_DATA(flat_errors)
    cdef np.ndarray string_offsets_array = np.empty(total_kept + 1, dtype=np.int32)
    cdef np.int32_t[::1] string_offsets_view = string_offsets_array
    cdef np.ndarray string_data_array = np.empty(total_chars, dtype=np.uint8)
    cdef unsigned char* string_data_ptr = <unsigned char*> np.PyArray_DATA(string_data_array)

    offs_specs_v[0] = 0
    string_offsets_view[0] = 0

    cdef size_t cursor = 0
    cdef size_t char_cursor = 0
    cdef string formula_str
    cdef size_t length
    cdef size_t cnt
    for si in range(n_specs):
        cnt = all_results[si].fragment_formulas.size()
        for k in range(cnt):
            mass_dst[cursor + k] = all_results[si].masses_normalized[k]
            intens_dst[cursor + k] = all_results[si].intensities[k]
            memcpy(
                <void*>(fvals_dst + (cursor + k) * NUM_ELEMENTS),
                <const void*> formula_data_const(all_results[si].fragment_formulas[k]),
                FORMULA_NBYTES_C,
            )
            ferr_dst[cursor + k] = all_results[si].fragment_errors_ppm[k]

            formula_str = all_results[si].fragment_formulas_strings[k]
            length = formula_str.size()
            if length > 0:
                memcpy(
                    <void*>(string_data_ptr + char_cursor),
                    <const void*> formula_str.c_str(),
                    length,
                )
            char_cursor += length
            string_offsets_view[cursor + k + 1] = <np.int32_t>char_cursor

        cursor += cnt
        offs_specs_v[si + 1] = <np.int32_t>cursor

    if total_kept == 0:
        flat_formula_vals = np.empty(0, dtype=np.int32)

    string_offsets_view[cursor] = <np.int32_t>char_cursor

    offs_specs_arr = pa.array(offs_specs, type=pa.int32())

    val_masses = pa.array(flat_masses_norm, type=pa.float64())
    masses_arr = pa.ListArray.from_arrays(offs_specs_arr, val_masses)

    val_intens = pa.array(flat_intens, type=pa.float64())
    intens_arr = pa.ListArray.from_arrays(offs_specs_arr, val_intens)

    val_formulas = pa.array(flat_formula_vals, type=pa.int32())
    fixed_formulas = pa.FixedSizeListArray.from_arrays(val_formulas, NUM_ELEMENTS)
    formulas_arr = pa.ListArray.from_arrays(offs_specs_arr, fixed_formulas)

    val_errors = pa.array(flat_errors, type=pa.float64())
    errors_arr = pa.ListArray.from_arrays(offs_specs_arr, val_errors)

    string_offsets_buffer = pa.py_buffer(string_offsets_array)
    string_data_buffer = pa.py_buffer(string_data_array)
    string_values = pa.Array.from_buffers(pa.utf8(), total_kept, [None, string_offsets_buffer, string_data_buffer])
    string_list = pa.ListArray.from_arrays(offs_specs_arr, string_values)

    s_masses = pl.Series("masses_normalized", masses_arr)
    s_intens = pl.Series("cleaned_intensities", intens_arr)
    s_frm = pl.Series("fragment_formulas", formulas_arr)
    s_frm_str = pl.Series("fragment_formulas_str", string_list)
    s_err = pl.Series("fragment_errors_ppm", errors_arr)
    return pl.struct(s_masses, s_intens, s_frm, s_frm_str, s_err, eager=True)
