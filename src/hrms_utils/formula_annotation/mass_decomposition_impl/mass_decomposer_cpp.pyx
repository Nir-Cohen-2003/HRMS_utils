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

    # Declarations for cleaning API (nested types in C++ are aliased here)
    cdef cppclass CleanSpectrumWithKnownPrecursor_cpp "MassDecomposer::CleanSpectrumWithKnownPrecursor":
        Formula_cpp precursor_formula
        vector[double] fragment_masses
        vector[double] fragment_intensities

    cdef cppclass CleanedSpectrumResult_cpp "MassDecomposer::CleanedSpectrumResult":
        vector[double] masses
        vector[double] intensities
        vector[vector[Formula_cpp]] fragment_formulas
        vector[vector[double]] fragment_errors_ppm

    # New: result for single-formula-per-fragment, normalized masses
    cdef cppclass CleanedAndNormalizedSpectrumResult_cpp "MassDecomposer::CleanedAndNormalizedSpectrumResult":
        vector[double] masses_normalized
        vector[double] intensities
        vector[Formula_cpp] fragment_formulas
        vector[double] fragment_errors_ppm

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
        @staticmethod
        vector[CleanedSpectrumResult_cpp] clean_spectra_known_precursor_parallel(const vector[CleanSpectrumWithKnownPrecursor_cpp]&, const DecompositionParams&)
        @staticmethod
        vector[CleanedAndNormalizedSpectrumResult_cpp] clean_and_normalize_spectra_known_precursor_parallel(const vector[CleanSpectrumWithKnownPrecursor_cpp]&, const DecompositionParams&)
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


def decompose_spectra_known_precursor_parallel(
    precursor_formula_series: pl.Series,  # series of 1D arrays shape (NUM_ELEMENTS,), dtype=int32 (pl.Array)
    fragment_masses_series: pl.Series,    # series of lists[float], variable length per spectrum
    tolerance_ppm: float = 5.0,
    max_results: int = 100000
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
        tolerance_ppm, 0.0, 30.0,  # dbe range for fragments with known precursor
        max_results,
        min_bounds, max_bounds
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
    max_results: int = 100000
) -> pl.Series:
    """
    Parallel cleaner with known precursor.
    Returns a Series[Struct] with fields:
      - masses: List[Float64]
      - intensities: List[Float64]
      - fragment_formulas: List[List[Array(Int32, NUM_ELEMENTS)]]
      - fragment_errors_ppm: List[List[Float64]]
    """
    # Convert precursor formulas to 2D contiguous int32
    cdef np.ndarray[np.int32_t, ndim=2, mode="c"] contig_precursors = np.ascontiguousarray(
        precursor_formula_series.to_numpy(), dtype=np.int32
    )
    cdef int n = <int>contig_precursors.shape[0]
    if n == 0:
        return pl.Series(
            "cleaned",
            [],
            dtype=pl.Struct(
                {
                    "masses": pl.List(pl.Float64),
                    "intensities": pl.List(pl.Float64),
                    "fragment_formulas": pl.List(pl.List(pl.Array(pl.Int32, NUM_ELEMENTS))),
                    "fragment_errors_ppm": pl.List(pl.List(pl.Float64)),
                }
            ),
        )
    if contig_precursors.shape[1] != NUM_ELEMENTS:
        raise ValueError(f"Each precursor formula must have length {NUM_ELEMENTS} (got {contig_precursors.shape[1]}).")
    if fragment_masses_series.len() != n or fragment_intensities_series.len() != n:
        raise ValueError("fragment_masses_series and fragment_intensities_series lengths must match precursor_formula_series length.")

    # Params: set DBE bounds for fragments; bounds ignored; pass zeros.
    cdef np.ndarray min_bounds = np.zeros(NUM_ELEMENTS, dtype=np.int32)
    cdef np.ndarray max_bounds = np.zeros(NUM_ELEMENTS, dtype=np.int32)
    cdef DecompositionParams params = _convert_params(
        tolerance_ppm, 0.0, 30.0,
        max_results,
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

    # Build field-wise Python containers (avoid dicts)
    masses_col = []
    intensities_col = []
    frag_formulas_col = []
    frag_errors_col = []

    cdef size_t si, fj, fk
    cdef size_t n_specs = all_results.size()
    for si in range(n_specs):
        # Masses and intensities
        masses_col.append([all_results[si].masses[j] for j in range(all_results[si].masses.size())])
        intensities_col.append([all_results[si].intensities[j] for j in range(all_results[si].intensities.size())])

        # Fragment formulas: List[List[Array(int32, NUM_ELEMENTS)]]
        spec_frag_formulas = []
        for fj in range(all_results[si].fragment_formulas.size()):
            frag_list_py = []
            for fk in range(all_results[si].fragment_formulas[fj].size()):
                frag_list_py.append(_convert_formula_to_array(all_results[si].fragment_formulas[fj][fk]))
            spec_frag_formulas.append(frag_list_py)
        frag_formulas_col.append(spec_frag_formulas)

        # Fragment errors: List[List[float]]
        spec_frag_errors = []
        for fj in range(all_results[si].fragment_errors_ppm.size()):
            spec_frag_errors.append(
                [all_results[si].fragment_errors_ppm[fj][k] for k in range(all_results[si].fragment_errors_ppm[fj].size())]
            )
        frag_errors_col.append(spec_frag_errors)

    # Create child Series with explicit dtypes
    s_masses = pl.Series("masses", masses_col, dtype=pl.List(pl.Float64))
    s_intensities = pl.Series("intensities", intensities_col, dtype=pl.List(pl.Float64))
    s_formulas = pl.Series(
        "fragment_formulas",
        frag_formulas_col,
        dtype=pl.List(pl.List(pl.Array(pl.Int32, NUM_ELEMENTS))),
    )
    s_errors = pl.Series("fragment_errors_ppm", frag_errors_col, dtype=pl.List(pl.List(pl.Float64)))

    # Pack into a Struct via Arrow to avoid dict overhead
    child_arrays = [
        s_masses.to_arrow(),
        s_intensities.to_arrow(),
        s_formulas.to_arrow(),
        s_errors.to_arrow(),
    ]
    struct_array = pa.StructArray.from_arrays(
        child_arrays,
        ["masses", "intensities", "fragment_formulas", "fragment_errors_ppm"],
    )

    return pl.Series(
        "cleaned",
        struct_array,
        dtype=pl.Struct(
            {
                "masses": pl.List(pl.Float64),
                "intensities": pl.List(pl.Float64),
                "fragment_formulas": pl.List(pl.List(pl.Array(pl.Int32, NUM_ELEMENTS))),
                "fragment_errors_ppm": pl.List(pl.List(pl.Float64)),
            }
        ),
        strict=False,
    )

def clean_and_normalize_spectra_known_precursor_parallel(
    precursor_formula_series: pl.Series,   # pl.Array(int32, NUM_ELEMENTS)
    fragment_masses_series: pl.Series,     # list[float] per spectrum
    fragment_intensities_series: pl.Series,# list[float] per spectrum
    tolerance_ppm: float = 5.0,
    max_results: int = 100000
) -> pl.Series:
    """
    Parallel cleaner that selects a single formula per fragment and normalizes masses
    using the spectrum-level mean error. Returns a Series[Struct] with fields:
      - masses_normalized: List[Float64]
      - intensities: List[Float64]
      - fragment_formulas: List[Array(Int32, NUM_ELEMENTS)]  # one formula per kept fragment
      - fragment_errors_ppm: List[Float64]                   # after normalization
    Note: This has one less nesting level than the previous cleaner.
    """
    # Convert precursor formulas to 2D contiguous int32
    cdef np.ndarray[np.int32_t, ndim=2, mode="c"] contig_precursors = np.ascontiguousarray(
        precursor_formula_series.to_numpy(), dtype=np.int32
    )
    cdef int n = <int>contig_precursors.shape[0]
    if n == 0:
        return pl.Series(
            "cleaned_normalized",
            [],
            dtype=pl.Struct(
                {
                    "masses_normalized": pl.List(pl.Float64),
                    "intensities": pl.List(pl.Float64),
                    "fragment_formulas": pl.List(pl.Array(pl.Int32, NUM_ELEMENTS)),
                    "fragment_errors_ppm": pl.List(pl.Float64),
                }
            ),
        )
    if contig_precursors.shape[1] != NUM_ELEMENTS:
        raise ValueError(f"Each precursor formula must have length {NUM_ELEMENTS} (got {contig_precursors.shape[1]}).")
    if fragment_masses_series.len() != n or fragment_intensities_series.len() != n:
        raise ValueError("fragment_masses_series and fragment_intensities_series lengths must match precursor_formula_series length.")

    # Params: bounds are ignored here; pass zeros; DBE range for fragments
    cdef np.ndarray min_bounds = np.zeros(NUM_ELEMENTS, dtype=np.int32)
    cdef np.ndarray max_bounds = np.zeros(NUM_ELEMENTS, dtype=np.int32)
    cdef DecompositionParams params = _convert_params(
        tolerance_ppm, 0.0, 30.0,
        max_results,
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

    # Build field-wise Python containers matching the reduced nesting
    masses_norm_col = []
    intensities_col = []
    frag_formulas_col = []
    frag_errors_col = []

    cdef size_t si, k
    cdef size_t n_specs = all_results.size()
    for si in range(n_specs):
        # Masses (normalized) and intensities
        masses_norm_col.append([all_results[si].masses_normalized[j] for j in range(all_results[si].masses_normalized.size())])
        intensities_col.append([all_results[si].intensities[j] for j in range(all_results[si].intensities.size())])

        # Fragment formulas: List[Array(int32, NUM_ELEMENTS)] (one per kept fragment)
        spec_formulas = []
        for k in range(all_results[si].fragment_formulas.size()):
            spec_formulas.append(_convert_formula_to_array(all_results[si].fragment_formulas[k]))
        frag_formulas_col.append(spec_formulas)

        # Fragment errors after normalization: List[Float64]
        frag_errors_col.append([all_results[si].fragment_errors_ppm[j] for j in range(all_results[si].fragment_errors_ppm.size())])

    # Create child Series with explicit dtypes
    s_masses = pl.Series("masses_normalized", masses_norm_col, dtype=pl.List(pl.Float64))
    s_intensities = pl.Series("intensities", intensities_col, dtype=pl.List(pl.Float64))
    s_formulas = pl.Series(
        "fragment_formulas",
        frag_formulas_col,
        dtype=pl.List(pl.Array(pl.Int32, NUM_ELEMENTS)),
    )
    s_errors = pl.Series("fragment_errors_ppm", frag_errors_col, dtype=pl.List(pl.Float64))

    # Pack into a Struct via Arrow for consistent schema
    child_arrays = [
        s_masses.to_arrow(),
        s_intensities.to_arrow(),
        s_formulas.to_arrow(),
        s_errors.to_arrow(),
    ]
    struct_array = pa.StructArray.from_arrays(
        child_arrays,
        ["masses_normalized", "intensities", "fragment_formulas", "fragment_errors_ppm"],
    )

    return pl.Series(
        "cleaned_normalized",
        struct_array,
        dtype=pl.Struct(
            {
                "masses_normalized": pl.List(pl.Float64),
                "intensities": pl.List(pl.Float64),
                "fragment_formulas": pl.List(pl.Array(pl.Int32, NUM_ELEMENTS)),
                "fragment_errors_ppm": pl.List(pl.Float64),
            }
        ),
        strict=False,
    )