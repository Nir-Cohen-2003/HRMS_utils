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
        vector[vector[Formula]] decompose_spectrum_known_precursor(const Formula& precursor_formula,
                                                                  const vector[double]& fragment_masses,
                                                                  const DecompositionParams& params)
        vector[vector[vector[Formula]]] decompose_spectra_known_precursor_parallel(const vector[SpectrumWithKnownPrecursor]& spectra,
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

def decompose_mass(
    target_mass: float,
    element_bounds: dict,
    strategy: str = "money_changing",
    tolerance_ppm: float = 5.0,
    min_dbe: float = 0.0,
    max_dbe: float = 40.0,
    max_hetero_ratio: float = 100.0,
    max_results: int = 100000
) -> list:
    """
    Decompose a single target mass into all possible molecular formulas within the specified constraints.

    Args:
        target_mass (float): The mass to decompose.
        element_bounds (dict[str, tuple[int, int]]): Dictionary mapping element symbols to (min_count, max_count) bounds.
        strategy (str, optional): Decomposition algorithm, "recursive" or "money_changing". Default is "recursive".
        tolerance_ppm (float, optional): Allowed mass error in ppm. Default is 5.0.
        min_dbe (float, optional): Minimum double bond equivalent (DBE) allowed. Default is 0.0.
        max_dbe (float, optional): Maximum DBE allowed. Default is 40.0.
        max_hetero_ratio (float, optional): Maximum allowed ratio of heteroatoms to carbon. Default is 100.0.
        max_results (int, optional): Maximum number of formulas to return. Default is 100000.

    Returns:
        list[dict[str, int]]: List of formulas as dictionaries mapping element symbols to counts.
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

def decompose_mass_parallel(
    target_masses: list,
    element_bounds: dict,
    strategy: str = "money_changing",
    tolerance_ppm: float = 5.0,
    min_dbe: float = 0.0,
    max_dbe: float = 40.0,
    max_hetero_ratio: float = 100.0,
    max_results: int = 100000
) -> list:
    """
    Decompose multiple target masses in parallel, each with the same element bounds and constraints.

    Args:
        target_masses (list[float]): List of masses to decompose.
        element_bounds (dict[str, tuple[int, int]]): See above.
        strategy (str, optional): See above.
        tolerance_ppm (float, optional): See above.
        min_dbe (float, optional): See above.
        max_dbe (float, optional): See above.
        max_hetero_ratio (float, optional): See above.
        max_results (int, optional): See above.

    Returns:
        list[list[dict[str, int]]]: List of results for each mass, each being a list of formulas.
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

def decompose_spectrum(
    precursor_mass: float,
    fragment_masses: list,
    element_bounds: dict,
    strategy: str = "money_changing",
    tolerance_ppm: float = 5.0,
    min_dbe: float = 0.0,
    max_dbe: float = 40.0,
    max_hetero_ratio: float = 100.0,
    max_results: int = 100000
) -> list:
    """
    Decompose a spectrum: find all possible precursor formulas for a given precursor mass, and for each, decompose the fragment masses as subsets of the precursor.

    Args:
        precursor_mass (float): The precursor ion mass.
        fragment_masses (list[float]): List of fragment masses to decompose.
        element_bounds (dict[str, tuple[int, int]]): See above.
        strategy (str, optional): See above.
        tolerance_ppm (float, optional): See above.
        min_dbe (float, optional): See above.
        max_dbe (float, optional): See above.
        max_hetero_ratio (float, optional): See above.
        max_results (int, optional): See above.

    Returns:
        list[dict]: List of precursor/fragment decomposition results. Each dict contains:
            - 'precursor': dict of precursor formula,
            - 'fragments': list of lists of fragment formulas,
            - 'precursor_mass': float,
            - 'precursor_error_ppm': float,
            - 'fragment_masses': list of lists of floats,
            - 'fragment_errors_ppm': list of lists of floats.
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

def decompose_spectra_parallel(
    spectra_data: list,
    element_bounds: dict,
    strategy: str = "money_changing",
    tolerance_ppm: float = 5.0,
    min_dbe: float = 0.0,
    max_dbe: float = 40.0,
    max_hetero_ratio: float = 100.0,
    max_results: int = 100000
) -> list:
    """
    Decompose multiple spectra in parallel. Each spectrum is defined by a precursor mass and a list of fragment masses.

    Args:
        spectra_data (list[tuple[float, list[float]]]): List of spectra, each as (precursor_mass, fragment_masses).
        element_bounds (dict[str, tuple[int, int]]): See above.
        strategy (str, optional): See above.
        tolerance_ppm (float, optional): See above.
        min_dbe (float, optional): See above.
        max_dbe (float, optional): See above.
        max_hetero_ratio (float, optional): See above.
        max_results (int, optional): See above.

    Returns:
        list[list[dict]]: List of results for each spectrum (see decompose_spectrum for result structure).
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

def decompose_spectrum_known_precursor(
    precursor_formula: dict,
    fragment_masses: list,
    element_bounds: dict,
    strategy: str = "money_changing",
    tolerance_ppm: float = 5.0,
    min_dbe: float = 0.0,
    max_dbe: float = 40.0,
    max_hetero_ratio: float = 100.0,
    max_results: int = 100000
) -> list:
    """
    Decompose fragment masses for a spectrum where the precursor formula is already known.

    Args:
        precursor_formula (dict[str, int]): The known precursor formula as a dictionary.
        fragment_masses (list[float]): List of fragment masses to decompose.
        element_bounds (dict[str, tuple[int, int]]): See above.
        strategy (str, optional): See above.
        tolerance_ppm (float, optional): See above.
        min_dbe (float, optional): See above.
        max_dbe (float, optional): See above.
        max_hetero_ratio (float, optional): See above.
        max_results (int, optional): See above.

    Returns:
        list[list[dict[str, int]]]: List of fragment results, one per fragment mass, each a list of formulas.
    """
    if not fragment_masses:
        return []
    
    cdef vector[Element] elements
    cdef DecompositionParams params
    cdef Formula cpp_precursor_formula
    cdef vector[double] cpp_fragment_masses
    cdef MassDecomposer* decomposer
    cdef vector[vector[Formula]] results
    cdef size_t i, j
    
    # Convert inputs
    elements = _convert_element_bounds(element_bounds)
    params = _convert_params(tolerance_ppm, min_dbe, max_dbe,
                           max_hetero_ratio, max_results, strategy)
    
    # Convert precursor formula
    for element, count in precursor_formula.items():
        cpp_precursor_formula[element.encode('utf-8')] = count
    
    # Convert fragment masses
    for mass in fragment_masses:
        cpp_fragment_masses.push_back(mass)
    
    decomposer = new MassDecomposer(elements, strategy.encode('utf-8'))
    try:
        results = decomposer.decompose_spectrum_known_precursor(cpp_precursor_formula, cpp_fragment_masses, params)
        
        # Convert results back to Python
        python_results = []
        for i in range(results.size()):
            fragment_formulas = []
            for j in range(results[i].size()):
                fragment_formulas.append(_convert_formula_result(results[i][j], decomposer))
            python_results.append(fragment_formulas)
        
        return python_results
    finally:
        del decomposer


def decompose_spectra_known_precursor_parallel(
    spectra_data: list,
    element_bounds: dict,
    strategy: str = "money_changing",
    tolerance_ppm: float = 5.0,
    min_dbe: float = 0.0,
    max_dbe: float = 40.0,
    max_hetero_ratio: float = 100.0,
    max_results: int = 100000
) -> list:
    """
    Decompose multiple spectra in parallel, where each spectrum has a known precursor formula and a list of fragment masses.

    Args:
        spectra_data (list[tuple[dict[str, int], list[float]]]): List of spectra, each as (precursor_formula, fragment_masses).
        element_bounds (dict[str, tuple[int, int]]): See above.
        strategy (str, optional): See above.
        tolerance_ppm (float, optional): See above.
        min_dbe (float, optional): See above.
        max_dbe (float, optional): See above.
        max_hetero_ratio (float, optional): See above.
        max_results (int, optional): See above.

    Returns:
        list[list[list[dict[str, int]]]]: List of results for each spectrum; for each spectrum, a list of fragment results (as in decompose_spectrum_known_precursor).
    """
    if not spectra_data:
        return []
    
    cdef vector[Element] elements
    cdef DecompositionParams params
    cdef vector[SpectrumWithKnownPrecursor] spectra_vec
    cdef MassDecomposer* decomposer
    cdef vector[vector[vector[Formula]]] all_results
    cdef size_t i, j, k
    cdef SpectrumWithKnownPrecursor spectrum
    
    # Convert inputs
    elements = _convert_element_bounds(element_bounds)
    params = _convert_params(tolerance_ppm, min_dbe, max_dbe,
                           max_hetero_ratio, max_results, strategy)
    
    # Convert spectra data
    for spectrum_data in spectra_data:
        precursor_formula, fragment_masses = spectrum_data
        
        # Clear and set precursor formula
        spectrum.precursor_formula.clear()
        for element, count in precursor_formula.items():
            spectrum.precursor_formula[element.encode('utf-8')] = count
        
        # Clear and set fragment masses
        spectrum.fragment_masses.clear()
        for frag_mass in fragment_masses:
            spectrum.fragment_masses.push_back(frag_mass)
        
        spectra_vec.push_back(spectrum)
    decomposer = new MassDecomposer(elements, strategy.encode('utf-8'))
    try:
        all_results = decomposer.decompose_spectra_known_precursor_parallel(spectra_vec, params)
        
        # Convert results back to Python
        python_results = []
        for i in range(all_results.size()):
            spectrum_results = []
            for j in range(all_results[i].size()):
                fragment_formulas = []
                for k in range(all_results[i][j].size()):
                    fragment_formulas.append(_convert_formula_result(all_results[i][j][k], decomposer))
                spectrum_results.append(fragment_formulas)
            python_results.append(spectrum_results)
        
        return python_results
    finally:
        del decomposer
