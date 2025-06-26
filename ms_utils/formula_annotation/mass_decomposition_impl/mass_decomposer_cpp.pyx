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

# C++ declarations
cdef extern from "mass_decomposer_core.hpp":
    cdef struct Element:
        string symbol
        double mass
        int min_count
        int max_count
    
    ctypedef unordered_map[string, int] Formula
    
    cdef struct Spectrum:
        double precursor_mass
        vector[double] fragment_masses
    
    cdef struct SpectrumDecomposition:
        Formula precursor
        vector[vector[Formula]] fragments
        double precursor_mass
        double precursor_error_ppm
        vector[vector[double]] fragment_masses
        vector[vector[double]] fragment_errors_ppm
    
    cdef struct SpectrumResults:
        vector[Formula] precursor_results
        vector[vector[Formula]] fragment_results
    
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
        MassDecomposer(const vector[Element]& elements, const string& strategy)
        vector[Formula] decompose(double target_mass, const DecompositionParams& params)
        vector[vector[Formula]] decompose_parallel(const vector[double]& target_masses, 
                                                  const DecompositionParams& params)
        ProperSpectrumResults decompose_spectrum(double precursor_mass,
                                           const vector[double]& fragment_masses,
                                           const DecompositionParams& params)
        vector[ProperSpectrumResults] decompose_spectra_parallel(const vector[Spectrum]& spectra,
                                                          const DecompositionParams& params)
        vector[pair[string, int]] formula_to_pairs(const Formula& formula)

# Standard atomic masses
ATOMIC_MASSES = {
    'C': 12.0000000, 'H': 1.0078250, 'O': 15.9949146, 'N': 14.0030740,
    'P': 30.9737620, 'S': 31.9720718, 'F': 18.9984032, 'Cl': 34.9688527,
    'Br': 78.9183376, 'I': 126.9044719, 'Si': 27.9769271, 'Na': 22.9897693,
    'K': 38.9637069, 'Ca': 39.9625912, 'Mg': 23.9850423, 'Fe': 55.9349421,
    'Zn': 63.9291466, 'Se': 79.9165218, 'B': 11.0093054, 'Al': 26.9815386
}

cdef vector[Element] _convert_element_bounds(dict element_bounds):
    """Convert Python element bounds to C++ Element vector."""
    cdef vector[Element] elements
    cdef Element elem
    
    for symbol, (min_count, max_count) in element_bounds.items():
        elem.symbol = symbol.encode('utf-8')
        elem.mass = ATOMIC_MASSES.get(symbol, 0.0)
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
    cdef size_t i
    
    for i in range(pairs.size()):
        symbol = pairs[i].first.decode('utf-8')
        count = pairs[i].second
        result[symbol] = count
    
    return result

def decompose_mass(double target_mass, dict element_bounds, 
                   str strategy="money_changing", double tolerance_ppm=5.0,
                   double min_dbe=0.0, double max_dbe=40.0, 
                   double max_hetero_ratio=1000.0, int max_results=10000):
    """
    Decompose a mass into possible molecular formulas using C++ implementation.
    
    Args:
        target_mass: Target mass to decompose
        element_bounds: Dictionary of element bounds {element: (min, max)}
        strategy: "recursive" or "money_changing"
        tolerance_ppm: Mass tolerance in ppm
        min_dbe: Minimum double bond equivalents
        max_dbe: Maximum double bond equivalents
        max_hetero_ratio: Maximum heteroatom to carbon ratio
        max_results: Maximum number of results to return
        
    Returns:
        List of formula dictionaries
    """
    cdef vector[Element] elements
    cdef DecompositionParams params
    cdef MassDecomposer* decomposer
    cdef vector[Formula] results
    cdef size_t i
    
    elements = _convert_element_bounds(element_bounds)
    params = _convert_params(tolerance_ppm, min_dbe, max_dbe,
                           max_hetero_ratio, max_results, strategy)
    
    decomposer = new MassDecomposer(elements, strategy.encode('utf-8'))
    results = decomposer.decompose(target_mass, params)
    
    try:
        # Convert results to Python
        python_results = []
        for i in range(results.size()):
            python_results.append(_convert_formula_result(results[i], decomposer))
        
        return python_results
    finally:
        del decomposer

def decompose_mass_parallel(list target_masses, dict element_bounds,
                           str strategy="money_changing", double tolerance_ppm=5.0,
                           double min_dbe=0.0, double max_dbe=40.0,
                           double max_hetero_ratio=1000.0, int max_results=10000):
    """
    Decompose multiple masses in parallel using C++ OpenMP implementation.
    
    Args:
        target_masses: List of target masses to decompose
        element_bounds: Dictionary of element bounds {element: (min, max)}
        strategy: "recursive" or "money_changing"
        tolerance_ppm: Mass tolerance in ppm
        min_dbe: Minimum double bond equivalents
        max_dbe: Maximum double bond equivalents
        max_hetero_ratio: Maximum heteroatom to carbon ratio
        max_results: Maximum number of results to return per mass
        
    Returns:
        List of lists of formula dictionaries
    """
    if not target_masses:
        return []
    
    cdef vector[Element] elements
    cdef DecompositionParams params
    cdef vector[double] masses_vec
    cdef MassDecomposer* decomposer
    cdef vector[vector[Formula]] all_results
    cdef size_t i, j
    
    elements = _convert_element_bounds(element_bounds)
    params = _convert_params(tolerance_ppm, min_dbe, max_dbe,
                           max_hetero_ratio, max_results, strategy)
    
    # Convert target masses to C++ vector
    for mass in target_masses:
        masses_vec.push_back(mass)
    
    decomposer = new MassDecomposer(elements, strategy.encode('utf-8'))
    all_results = decomposer.decompose_parallel(masses_vec, params)
    
    try:
        # Convert results to Python
        python_results = []
        for i in range(all_results.size()):
            mass_results = []
            for j in range(all_results[i].size()):
                mass_results.append(_convert_formula_result(all_results[i][j], decomposer))
            python_results.append(mass_results)
        
        return python_results
    finally:
        del decomposer

def decompose_spectrum(double precursor_mass, list fragment_masses,
                     dict element_bounds, str strategy="money_changing",
                     double tolerance_ppm=5.0, double min_dbe=0.0,
                     double max_dbe=40.0, double max_hetero_ratio=1000.0,
                     int max_results=10000):
    """
    Decompose a spectrum ensuring fragments are subsets of precursor formulas.
    Returns a list of precursor-fragment combinations.
    """
    if not fragment_masses:
        return []
    cdef vector[Element] elements
    cdef vector[double] frag_masses_vec
    cdef DecompositionParams params
    cdef MassDecomposer* decomposer
    cdef ProperSpectrumResults cpp_results
    cdef size_t i, j, k
    elements = _convert_element_bounds(element_bounds)
    params = _convert_params(tolerance_ppm, min_dbe, max_dbe,
                           max_hetero_ratio, max_results, strategy)
    for frag_mass in fragment_masses:
        frag_masses_vec.push_back(frag_mass)
    decomposer = new MassDecomposer(elements, strategy.encode('utf-8'))
    try:
        cpp_results = decomposer.decompose_spectrum(precursor_mass, frag_masses_vec, params)
        python_results = []
        for i in range(cpp_results.decompositions.size()):
            decomp = cpp_results.decompositions[i]
            precursor_dict = _convert_formula_result(decomp.precursor, decomposer)
            fragment_lists = []
            fragment_mass_lists = []
            fragment_error_lists = []
            for j in range(decomp.fragments.size()):
                frag_formulas = []
                for k in range(decomp.fragments[j].size()):
                    frag_formulas.append(_convert_formula_result(decomp.fragments[j][k], decomposer))
                fragment_lists.append(frag_formulas)
                mass_list = []
                error_list = []
                for k in range(decomp.fragment_masses[j].size()):
                    mass_list.append(decomp.fragment_masses[j][k])
                for k in range(decomp.fragment_errors_ppm[j].size()):
                    error_list.append(decomp.fragment_errors_ppm[j][k])
                fragment_mass_lists.append(mass_list)
                fragment_error_lists.append(error_list)
            result_dict = {
                'precursor': precursor_dict,
                'fragments': fragment_lists,
                'precursor_mass': decomp.precursor_mass,
                'precursor_error_ppm': decomp.precursor_error_ppm,
                'fragment_masses': fragment_mass_lists,
                'fragment_errors_ppm': fragment_error_lists
            }
            python_results.append(result_dict)
        return python_results
    finally:
        del decomposer

def decompose_spectra_parallel(list spectra_data, dict element_bounds,
                             str strategy="money_changing", double tolerance_ppm=5.0,
                             double min_dbe=0.0, double max_dbe=40.0,
                             double max_hetero_ratio=1000.0, int max_results=10000):
    """
    Decompose multiple spectra in parallel ensuring fragments are subsets of precursor formulas.
    Returns a list of spectrum results, each in the same format as decompose_spectrum.
    """
    if not spectra_data:
        return []
    cdef vector[Element] elements
    cdef DecompositionParams params
    cdef vector[Spectrum] spectra_vec
    cdef MassDecomposer* decomposer
    cdef vector[ProperSpectrumResults] all_results
    cdef size_t i, j, k, l
    cdef Spectrum spectrum
    elements = _convert_element_bounds(element_bounds)
    params = _convert_params(tolerance_ppm, min_dbe, max_dbe,
                           max_hetero_ratio, max_results, strategy)
    for spectrum_data in spectra_data:
        precursor_mass, fragment_masses = spectrum_data
        spectrum.precursor_mass = precursor_mass
        spectrum.fragment_masses.clear()
        for frag_mass in fragment_masses:
            spectrum.fragment_masses.push_back(frag_mass)
        spectra_vec.push_back(spectrum)
    decomposer = new MassDecomposer(elements, strategy.encode('utf-8'))
    try:
        all_results = decomposer.decompose_spectra_parallel(spectra_vec, params)
        python_results = []
        for i in range(all_results.size()):
            spectrum_results = []
            for j in range(all_results[i].decompositions.size()):
                decomp = all_results[i].decompositions[j]
                precursor_dict = _convert_formula_result(decomp.precursor, decomposer)
                fragment_lists = []
                fragment_mass_lists = []
                fragment_error_lists = []
                for k in range(decomp.fragments.size()):
                    frag_formulas = []
                    for l in range(decomp.fragments[k].size()):
                        frag_formulas.append(_convert_formula_result(decomp.fragments[k][l], decomposer))
                    fragment_lists.append(frag_formulas)
                    mass_list = []
                    error_list = []
                    for l in range(decomp.fragment_masses[k].size()):
                        mass_list.append(decomp.fragment_masses[k][l])
                    for l in range(decomp.fragment_errors_ppm[k].size()):
                        error_list.append(decomp.fragment_errors_ppm[k][l])
                    fragment_mass_lists.append(mass_list)
                    fragment_error_lists.append(error_list)
                result_dict = {
                    'precursor': precursor_dict,
                    'fragments': fragment_lists,
                    'precursor_mass': decomp.precursor_mass,
                    'precursor_error_ppm': decomp.precursor_error_ppm,
                    'fragment_masses': fragment_mass_lists,
                    'fragment_errors_ppm': fragment_error_lists
                }
                spectrum_results.append(result_dict)
            python_results.append(spectrum_results)
        return python_results
    finally:
        del decomposer
