"""
Python interface for spectrum decomposition functions.
"""

try:
    from .spectrum_decomposer import (
        decompose_spectrum_cython,
        decompose_given_formula_spectrum_cython,
        SpectrumDecomposer
    )
except ImportError:
    # Fallback if Cython module is not compiled
    print("Warning: Cython spectrum decomposer not available. Please compile with: python setup.py build_ext --inplace")
    
    def decompose_spectrum_cython(*args, **kwargs):
        raise ImportError("Cython spectrum decomposer not compiled")
    
    def decompose_given_formula_spectrum_cython(*args, **kwargs):
        raise ImportError("Cython spectrum decomposer not compiled")
    
    class SpectrumDecomposer:
        def __init__(self, *args, **kwargs):
            raise ImportError("Cython spectrum decomposer not compiled")

# Convenience aliases
decompose_spectrum = decompose_spectrum_cython
decompose_given_formula_spectrum = decompose_given_formula_spectrum_cython

__all__ = [
    'decompose_spectrum_cython',
    'decompose_given_formula_spectrum_cython',
    'decompose_spectrum',
    'decompose_given_formula_spectrum',
    'SpectrumDecomposer'
]
