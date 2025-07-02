# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Cython wrapper for the C++ mass decomposition implementation with OpenMP parallelization.
"""
cimport cython
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.pair cimport pair
from typing import List, Dict, Tuple

# C++ declarations from the header file
cdef extern from "mass_decomposer_common.hpp":
    cdef struct Element:
        string symbol
        double mass
        int min_count
        int max_count
    
    ctypedef unordered_map[string, int] Formula
    
    cdef struct Spectrum:
        double precursor_mass
        vector[double] fragment_masses
    
    cdef struct SpectrumWithBounds:
        double precursor_mass
        vector[double] fragment_masses
        vector[Element] precursor_bounds

    cdef struct SpectrumWithKnownPrecursor:
        Formula precursor_formula
        vector[double] fragment_masses
    
    cdef struct SpectrumDecomposition:
        Formula precursor
        vector[vector[Formula]] fragments
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
        string strategy
    
    cdef cppclass MassDecomposer:
        MassDecomposer(const vector[Element]&, const string&)
        vector[Formula] decompose(double, const DecompositionParams&)
        vector[vector[Formula]] decompose_parallel(const vector[double]&, const DecompositionParams&)
        vector[vector[Formula]] decompose_masses_parallel_per_bounds(const vector[double]&, const vector[vector[Element]]&, const DecompositionParams&)
        ProperSpectrumResults decompose_spectrum(double, const vector[double]&, const DecompositionParams&)
        vector[ProperSpectrumResults] decompose_spectra_parallel(const vector[Spectrum]&, const DecompositionParams&)
        vector[ProperSpectrumResults] decompose_spectra_parallel_per_bounds(const vector[SpectrumWithBounds]&, const DecompositionParams&)
        vector[vector[Formula]] decompose_spectrum_known_precursor(const Formula&, const vector[double]&, const DecompositionParams&)
        vector[vector[vector[Formula]]] decompose_spectra_known_precursor_parallel(const vector[SpectrumWithKnownPrecursor]&, const DecompositionParams&)
        vector[pair[string, int]] formula_to_pairs(const Formula&) const


# Helper functions for converting Python objects to C++ and vice-versa

cdef vector[Element] _convert_element_bounds(dict element_bounds):
    """Convert Python element bounds to C++ Element vector."""
    cdef vector[Element] elements
    cdef Element elem
    for symbol, (min_count, max_count) in element_bounds.items():
        elem.symbol = symbol.encode('utf-8')
        elem.min_count = min_count
        elem.max_count = max_count
        elements.push_back(elem)
    return elements

cdef DecompositionParams _convert_params(double tolerance_ppm, double min_dbe, double max_dbe,
                                        double max_hetero_ratio, int max_results, str strategy):
    """Convert Python parameters to C++ DecompositionParams."""
    cdef DecompositionParams params
    params.tolerance_ppm = tolerance_ppm
    params.min_dbe = min_dbe
    params.max_dbe = max_dbe
    params.max_hetero_ratio = max_hetero_ratio
    params.max_results = max_results
    params.strategy = strategy.encode('utf-8')
    return params

cdef dict _convert_formula_result(const Formula& formula, MassDecomposer* decomposer):
    """Convert C++ Formula to Python dict using helper function."""
    result = {}
    cdef vector[pair[string, int]] pairs = decomposer.formula_to_pairs(formula)
    for p in pairs:
        result[p.first.decode('utf-8')] = p.second
    return result

cdef Formula _convert_formula_to_cpp(dict py_formula):
    """Convert Python dict to C++ Formula."""
    cdef Formula cpp_formula
    for key, value in py_formula.items():
        cpp_formula[key.encode('utf-8')] = value
    return cpp_formula

# Public Python functions

def decompose_mass(
    target_mass: float,
    element_bounds: dict,
    strategy: str = "recursive",
    tolerance_ppm: float = 5.0,
    min_dbe: float = 0.0,
    max_dbe: float = 40.0,
    max_hetero_ratio: float = 100.0,
    max_results: int = 100000
) -> list:
    cdef vector[Element] elements = _convert_element_bounds(element_bounds)
    cdef DecompositionParams params = _convert_params(tolerance_ppm, min_dbe, max_dbe,
                                                     max_hetero_ratio, max_results, strategy)
    cdef MassDecomposer* decomposer = new MassDecomposer(elements, strategy.encode('utf-8'))
    cdef vector[Formula] results
    
    try:
        results = decomposer.decompose(target_mass, params)
        python_results = [_convert_formula_result(res, decomposer) for res in results]
        return python_results
    finally:
        del decomposer

def decompose_mass_parallel(
    target_masses: list,
    element_bounds: dict,
    strategy: str = "recursive",
    tolerance_ppm: float = 5.0,
    min_dbe: float = 0.0,
    max_dbe: float = 40.0,
    max_hetero_ratio: float = 100.0,
    max_results: int = 100000
) -> list:
    if not target_masses:
        return []
    
    cdef vector[Element] elements = _convert_element_bounds(element_bounds)
    cdef DecompositionParams params = _convert_params(tolerance_ppm, min_dbe, max_dbe,
                                                     max_hetero_ratio, max_results, strategy)
    cdef vector[double] masses_vec = target_masses
    cdef MassDecomposer* decomposer = new MassDecomposer(elements, strategy.encode('utf-8'))
    cdef vector[vector[Formula]] all_results
    
    try:
        all_results = decomposer.decompose_parallel(masses_vec, params)
        python_results = [[_convert_formula_result(res, decomposer) for res in mass_results] for mass_results in all_results]
        return python_results
    finally:
        del decomposer

def decompose_mass_parallel_per_bounds(
    target_masses: list,
    per_mass_bounds: list,
    strategy: str = "recursive",
    tolerance_ppm: float = 5.0,
    min_dbe: float = 0.0,
    max_dbe: float = 40.0,
    max_hetero_ratio: float = 100.0,
    max_results: int = 100000
) -> list:
    if not target_masses:
        return []
    if len(target_masses) != len(per_mass_bounds):
        raise ValueError("Length of target_masses and per_mass_bounds must be equal.")

    cdef DecompositionParams params = _convert_params(tolerance_ppm, min_dbe, max_dbe,
                                                     max_hetero_ratio, max_results, strategy)
    cdef vector[double] masses_vec = target_masses
    cdef vector[vector[Element]] bounds_vec
    for bounds_dict in per_mass_bounds:
        bounds_vec.push_back(_convert_element_bounds(bounds_dict))

    # A dummy decomposer to access helper methods, not used for decomposition itself
    cdef vector[Element] dummy_elements
    cdef MassDecomposer* decomposer = new MassDecomposer(dummy_elements, strategy.encode('utf-8'))
    cdef vector[vector[Formula]] all_results

    try:
        all_results = decomposer.decompose_masses_parallel_per_bounds(masses_vec, bounds_vec, params)
        python_results = [[_convert_formula_result(res, decomposer) for res in mass_results] for mass_results in all_results]
        return python_results
    finally:
        del decomposer

def decompose_spectrum(
    precursor_mass: float,
    fragment_masses: list,
    element_bounds: dict,
    strategy: str = "recursive",
    tolerance_ppm: float = 5.0,
    min_dbe: float = 0.0,
    max_dbe: float = 40.0,
    max_hetero_ratio: float = 100.0,
    max_results: int = 100000
) -> list:
    if not fragment_masses:
        return []
    cdef vector[Element] elements = _convert_element_bounds(element_bounds)
    cdef vector[double] frag_masses_vec = fragment_masses
    cdef DecompositionParams params = _convert_params(tolerance_ppm, min_dbe, max_dbe,
                                                     max_hetero_ratio, max_results, strategy)
    cdef MassDecomposer* decomposer = new MassDecomposer(elements, strategy.encode('utf-8'))
    cdef ProperSpectrumResults cpp_results
    
    try:
        cpp_results = decomposer.decompose_spectrum(precursor_mass, frag_masses_vec, params)
        python_results = []
        for decomp in cpp_results.decompositions:
            py_decomp = {
                'precursor': _convert_formula_result(decomp.precursor, decomposer),
                'precursor_mass': decomp.precursor_mass,
                'precursor_error_ppm': decomp.precursor_error_ppm,
                'fragments': [[_convert_formula_result(f, decomposer) for f in frag_list] for frag_list in decomp.fragments],
                'fragment_masses': decomp.fragment_masses,
                'fragment_errors_ppm': decomp.fragment_errors_ppm
            }
            python_results.append(py_decomp)
        return python_results
    finally:
        del decomposer

def decompose_spectra_parallel(
    spectra_data: list,
    element_bounds: dict,
    strategy: str = "recursive",
    tolerance_ppm: float = 5.0,
    min_dbe: float = 0.0,
    max_dbe: float = 40.0,
    max_hetero_ratio: float = 100.0,
    max_results: int = 100000
) -> list:
    if not spectra_data:
        return []
    cdef vector[Element] elements = _convert_element_bounds(element_bounds)
    cdef DecompositionParams params = _convert_params(tolerance_ppm, min_dbe, max_dbe,
                                                     max_hetero_ratio, max_results, strategy)
    cdef vector[Spectrum] spectra_vec
    cdef Spectrum s
    for spec_data in spectra_data:
        s.precursor_mass = spec_data['precursor_mass']
        s.fragment_masses = spec_data['fragment_masses']
        spectra_vec.push_back(s)

    cdef MassDecomposer* decomposer = new MassDecomposer(elements, strategy.encode('utf-8'))
    cdef vector[ProperSpectrumResults] all_cpp_results
    try:
        all_cpp_results = decomposer.decompose_spectra_parallel(spectra_vec, params)
        all_python_results = []
        for cpp_results in all_cpp_results:
            python_results = []
            for decomp in cpp_results.decompositions:
                py_decomp = {
                    'precursor': _convert_formula_result(decomp.precursor, decomposer),
                    'precursor_mass': decomp.precursor_mass,
                    'precursor_error_ppm': decomp.precursor_error_ppm,
                    'fragments': [[_convert_formula_result(f, decomposer) for f in frag_list] for frag_list in decomp.fragments],
                    'fragment_masses': decomp.fragment_masses,
                    'fragment_errors_ppm': decomp.fragment_errors_ppm
                }
                python_results.append(py_decomp)
            all_python_results.append(python_results)
        return all_python_results
    finally:
        del decomposer

def decompose_spectra_parallel_per_bounds(
    spectra_data: list,
    per_spectrum_bounds: list,
    strategy: str = "recursive",
    tolerance_ppm: float = 5.0,
    min_dbe: float = 0.0,
    max_dbe: float = 40.0,
    max_hetero_ratio: float = 100.0,
    max_results: int = 100000
) -> list:
    if not spectra_data:
        return []
    if len(spectra_data) != len(per_spectrum_bounds):
        raise ValueError("Length of spectra_data and per_spectrum_bounds must be equal.")

    cdef DecompositionParams params = _convert_params(tolerance_ppm, min_dbe, max_dbe,
                                                     max_hetero_ratio, max_results, strategy)
    cdef vector[SpectrumWithBounds] spectra_vec
    cdef SpectrumWithBounds s
    for i in range(len(spectra_data)):
        s.precursor_mass = spectra_data[i]['precursor_mass']
        s.fragment_masses = spectra_data[i]['fragment_masses']
        s.precursor_bounds = _convert_element_bounds(per_spectrum_bounds[i])
        spectra_vec.push_back(s)

    # A dummy decomposer to access helper methods, not used for decomposition itself
    cdef vector[Element] dummy_elements
    cdef MassDecomposer* decomposer = new MassDecomposer(dummy_elements, strategy.encode('utf-8'))
    cdef vector[ProperSpectrumResults] all_cpp_results
    try:
        all_cpp_results = decomposer.decompose_spectra_parallel_per_bounds(spectra_vec, params)
        all_python_results = []
        for cpp_results in all_cpp_results:
            python_results = []
            for decomp in cpp_results.decompositions:
                py_decomp = {
                    'precursor': _convert_formula_result(decomp.precursor, decomposer),
                    'precursor_mass': decomp.precursor_mass,
                    'precursor_error_ppm': decomp.precursor_error_ppm,
                    'fragments': [[_convert_formula_result(f, decomposer) for f in frag_list] for frag_list in decomp.fragments],
                    'fragment_masses': decomp.fragment_masses,
                    'fragment_errors_ppm': decomp.fragment_errors_ppm
                }
                python_results.append(py_decomp)
            all_python_results.append(python_results)
        return all_python_results
    finally:
        del decomposer

def decompose_spectrum_known_precursor(
    precursor_formula: dict,
    fragment_masses: list,
    element_bounds: dict,
    strategy: str = "recursive",
    tolerance_ppm: float = 5.0,
    min_dbe: float = 0.0,
    max_dbe: float = 100.0,
    max_hetero_ratio: float = 100.0,
    max_results: int = 100000
) -> list:
    cdef vector[Element] elements = _convert_element_bounds(element_bounds)
    cdef DecompositionParams params = _convert_params(tolerance_ppm, min_dbe, max_dbe, max_hetero_ratio, max_results, strategy)
    cdef Formula cpp_precursor = _convert_formula_to_cpp(precursor_formula)
    cdef vector[double] frag_masses_vec = fragment_masses
    cdef MassDecomposer* decomposer = new MassDecomposer(elements, strategy.encode('utf-8'))
    cdef vector[vector[Formula]] results

    try:
        results = decomposer.decompose_spectrum_known_precursor(cpp_precursor, frag_masses_vec, params)
        return [[_convert_formula_result(f, decomposer) for f in res] for res in results]
    finally:
        del decomposer

def decompose_spectra_known_precursor_parallel(
    spectra_data: list,
    element_bounds: dict,
    strategy: str = "recursive",
    tolerance_ppm: float = 5.0,
    min_dbe: float = 0.0,
    max_dbe: float = 100.0,
    max_hetero_ratio: float = 100.0,
    max_results: int = 100000
) -> list:
    if not spectra_data:
        return []
    cdef vector[Element] elements = _convert_element_bounds(element_bounds)
    cdef DecompositionParams params = _convert_params(tolerance_ppm,min_dbe, max_dbe , max_hetero_ratio, max_results, strategy)
    cdef vector[SpectrumWithKnownPrecursor] spectra_vec
    cdef SpectrumWithKnownPrecursor s
    for spec_data in spectra_data:
        s.precursor_formula = _convert_formula_to_cpp(spec_data['precursor_formula'])
        s.fragment_masses = spec_data['fragment_masses']
        spectra_vec.push_back(s)

    cdef MassDecomposer* decomposer = new MassDecomposer(elements, strategy.encode('utf-8'))
    cdef vector[vector[vector[Formula]]] all_results
    try:
        all_results = decomposer.decompose_spectra_known_precursor_parallel(spectra_vec, params)
        return [[[_convert_formula_result(f, decomposer) for f in res] for res in spec_res] for spec_res in all_results]
    finally:
        del decomposer