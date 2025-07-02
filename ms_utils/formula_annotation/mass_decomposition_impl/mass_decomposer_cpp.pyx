# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Cython wrapper for the C++ mass decomposition implementation with OpenMP parallelization.
"""
cimport cython
from libcpp.vector cimport vector
from libcpp.string cimport string
from typing import List, Dict, Tuple
import numpy as np
cimport numpy as np

cdef extern from "mass_decomposer_core.hpp":
    ctypedef vector[int] FormulaArray
    ctypedef vector[vector[int]] BoundsArray
    cdef struct Spectrum:
        double precursor_mass
        vector[double] fragment_masses
    cdef struct SpectrumWithBounds:
        double precursor_mass
        vector[double] fragment_masses
        BoundsArray bounds
    cdef struct SpectrumWithKnownPrecursor:
        FormulaArray precursor_formula
        vector[double] fragment_masses
    cdef struct DecompositionParams:
        double tolerance_ppm
        double min_dbe
        double max_dbe
        double max_hetero_ratio
        int max_results
        string strategy
    cdef cppclass MassDecomposer:
        MassDecomposer()
        vector[FormulaArray] decompose_mass(double, const BoundsArray&, const DecompositionParams&)
        vector[vector[FormulaArray]] decompose_spectrum(double, const vector[double]&, const BoundsArray&, const DecompositionParams&)
        vector[vector[FormulaArray]] decompose_spectrum_known_precursor(const FormulaArray&, const vector[double]&, const DecompositionParams&)
        vector[vector[FormulaArray]] decompose_mass_parallel(const vector[double]&, const BoundsArray&, const DecompositionParams&)
        vector[vector[FormulaArray]] decompose_mass_parallel(const vector[double]&, const vector[BoundsArray]&, const DecompositionParams&)
        vector[vector[vector[FormulaArray]]] decompose_spectrum_parallel(const vector[Spectrum]&, const BoundsArray&, const DecompositionParams&)
        vector[vector[vector[FormulaArray]]] decompose_spectrum_parallel(const vector[SpectrumWithBounds]&, const DecompositionParams&)
        vector[vector[vector[FormulaArray]]] decompose_spectrum_known_precursor_parallel(const vector[SpectrumWithKnownPrecursor]&, const DecompositionParams&)
        vector[string] get_element_order() const

cdef _convert_params(kwargs):
    cdef DecompositionParams params
    params.mass_accuracy_ppm = kwargs.get("mass_accuracy_ppm", 5.0)
    params.min_dbe = kwargs.get("min_dbe", 0.0)
    params.max_dbe = kwargs.get("max_dbe", 40.0)
    params.max_hetero_ratio = kwargs.get("max_hetero_ratio", 100.0)
    params.max_results = kwargs.get("max_results", 100000)
    params.strategy = kwargs.get("strategy", "recursive").encode('utf-8')
    return params

cdef _np_to_vecint(np.ndarray arr):
    cdef vector[int] v
    for x in arr: v.push_back(int(x))
    return v

cdef _np_to_boundsarray(np.ndarray arr):
    cdef BoundsArray bounds
    for i in range(arr.shape[0]): bounds.push_back(_np_to_vecint(arr[i]))
    return bounds

cdef _vecint_to_np(const vector[int]& v):
    cdef np.ndarray arr = np.zeros(v.size(), dtype=np.int32)
    for i in range(v.size()): arr[i] = v[i]
    return arr

def _formulaarraylist_to_py(const vector[FormulaArray]& formulas):
    return [_vecint_to_np(f) for f in formulas]

def _formulaarraylist2_to_py(const vector[vector[FormulaArray]]& formulas):
    return [_formulaarraylist_to_py(group) for group in formulas]

def _formulaarraylist3_to_py(const vector[vector[vector[FormulaArray]]]& formulas):
    return [_formulaarraylist2_to_py(spec) for spec in formulas]

def get_element_order():
    cdef MassDecomposer* decomposer = new MassDecomposer()
    try:
        return [s.decode('utf-8') for s in decomposer.get_element_order()]
    finally:
        del decomposer

def decompose_mass(target_mass, bounds, **kwargs):
    cdef MassDecomposer* decomposer = new MassDecomposer()
    cdef DecompositionParams params = _convert_params(kwargs)
    cdef BoundsArray bounds_arr = _np_to_boundsarray(bounds)
    try:
        return _formulaarraylist_to_py(decomposer.decompose_mass(target_mass, bounds_arr, params))
    finally:
        del decomposer

def decompose_spectrum(precursor_mass, fragment_masses, bounds, **kwargs):
    cdef MassDecomposer* decomposer = new MassDecomposer()
    cdef DecompositionParams params = _convert_params(kwargs)
    cdef BoundsArray bounds_arr = _np_to_boundsarray(bounds)
    cdef vector[double] frags = [f for f in fragment_masses]
    try:
        return _formulaarraylist2_to_py(decomposer.decompose_spectrum(precursor_mass, frags, bounds_arr, params))
    finally:
        del decomposer

def decompose_spectrum_known_precursor(known_precursor, fragment_masses, **kwargs):
    cdef MassDecomposer* decomposer = new MassDecomposer()
    cdef DecompositionParams params = _convert_params(kwargs)
    cdef FormulaArray precursor_arr = _np_to_vecint(known_precursor)
    cdef vector[double] frags = [f for f in fragment_masses]
    try:
        return _formulaarraylist2_to_py(decomposer.decompose_spectrum_known_precursor(precursor_arr, frags, params))
    finally:
        del decomposer

def decompose_mass_parallel(target_masses, bounds, **kwargs):
    cdef MassDecomposer* decomposer = new MassDecomposer()
    cdef DecompositionParams params = _convert_params(kwargs)
    cdef vector[double] masses_vec = [m for m in target_masses]
    cdef vector[BoundsArray] bounds_per_mass
    cdef BoundsArray bounds_arr

    try:
        if isinstance(bounds, list):
            # Per-mass bounds
            for b in bounds:
                bounds_per_mass.push_back(_np_to_boundsarray(b))
            return _formulaarraylist2_to_py(decomposer.decompose_mass_parallel(masses_vec, bounds_per_mass, params))
        else:
            # Single set of bounds for all masses
            bounds_arr = _np_to_boundsarray(bounds)
            return _formulaarraylist2_to_py(decomposer.decompose_mass_parallel(masses_vec, bounds_arr, params))
    finally:
        del decomposer

def decompose_spectrum_parallel(spectra, bounds, **kwargs):
    cdef MassDecomposer* decomposer = new MassDecomposer()
    cdef DecompositionParams params = _convert_params(kwargs)
    cdef vector[Spectrum] spectra_vec
    cdef BoundsArray bounds_arr
    cdef vector[SpectrumWithBounds] spectra_with_bounds
    cdef Spectrum spec
    cdef SpectrumWithBounds swb

    try:
        if isinstance(bounds, np.ndarray):
            # Single set of bounds for all spectra
            bounds_arr = _np_to_boundsarray(bounds)
            for s in spectra:
                spec.precursor_mass = s[0]
                spec.fragment_masses = [f for f in s[1]]
                spectra_vec.push_back(spec)
            return _formulaarraylist3_to_py(decomposer.decompose_spectrum_parallel(spectra_vec, bounds_arr, params))
        else:
            # Per-spectrum bounds
            for i, s in enumerate(spectra):
                swb.precursor_mass = s[0]
                swb.fragment_masses = [f for f in s[1]]
                swb.bounds = _np_to_boundsarray(bounds[i])
                spectra_with_bounds.push_back(swb)
            return _formulaarraylist3_to_py(decomposer.decompose_spectrum_parallel(spectra_with_bounds, params))
    finally:
        del decomposer

def decompose_spectrum_known_precursor_parallel(spectra, **kwargs):
    cdef MassDecomposer* decomposer = new MassDecomposer()
    cdef DecompositionParams params = _convert_params(kwargs)
    cdef vector[SpectrumWithKnownPrecursor] spectra_vec
    cdef SpectrumWithKnownPrecursor skp
    for s in spectra:
        skp.precursor_formula = _np_to_vecint(s[0])
        skp.fragment_masses = [f for f in s[1]]
        spectra_vec.push_back(skp)
    try:
        return _formulaarraylist3_to_py(decomposer.decompose_spectrum_known_precursor_parallel(spectra_vec, params))
    finally:
        del decomposer
