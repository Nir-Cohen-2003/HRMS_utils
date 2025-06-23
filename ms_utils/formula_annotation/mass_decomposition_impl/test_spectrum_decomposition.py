#!/usr/bin/env python3
"""
Test script for the spectrum decomposition functionality in sirius_decomposer.pyx

This script demonstrates how to use the new spectrum decomposition functions.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from sirius_decomposer import (
        decompose_spectrum_cython, 
        decompose_given_formula_spectrum_cython
    )
    print("Successfully imported spectrum decomposition functions!")
except ImportError as e:
    print(f"Import error: {e}")
    print("Please compile the Cython module first by running:")
    print("python setup.py build_ext --inplace")
    sys.exit(1)

def test_spectrum_decomposition():
    """Test the complete spectrum decomposition functionality."""
    
    # Define element bounds (typical for organic compounds)
    element_bounds = {
        'C': (0, 50),
        'H': (0, 100),
        'N': (0, 10),
        'O': (0, 20),
        'P': (0, 5),
        'S': (0, 5)
    }
    
    # Example: precursor mass and fragment masses
    precursor_mass = 180.0634  # Example: glucose C6H12O6
    # Note: Including precursor mass in fragments (molecular ion peak case)
    fragment_masses = [180.0634, 162.0528, 144.0422, 126.0317, 108.0211]  # Example fragments

    print("\\nTesting spectrum decomposition:")
    print(f"Precursor mass: {precursor_mass}")
    print(f"Fragment masses: {fragment_masses}")
    
    # Perform spectrum decomposition
    results = decompose_spectrum_cython(
        precursor_mass, fragment_masses, element_bounds, 
        tolerance_ppm=5.0, max_results=1000
    )
    
    print(f"\\nFound {len(results)} possible precursor formulas:")
    
    for i, result in enumerate(results[:5]):  # Show first 5 results
        print(f"\\nResult {i+1}:")
        print(f"  Precursor formula: {result['precursor_formula']}")
        
        for j, fragment_decomps in enumerate(result['fragment_decompositions']):
            print(f"  Fragment {j+1} (mass {fragment_masses[j]:.4f}): {len(fragment_decomps)} possibilities")
            if fragment_decomps:
                print(f"    Examples: {fragment_decomps[:3]}")  # Show first 3


def test_given_formula_spectrum():
    """Test decomposition with a known precursor formula."""
    
    # Define element bounds
    element_bounds = {
        'C': (0, 50),
        'H': (0, 100),
        'N': (0, 10),
        'O': (0, 20),
        'P': (0, 5),
        'S': (0, 5)
    }
    
    # Known precursor formula (glucose)
    given_formula = {'C': 6, 'H': 12, 'O': 6}
    fragment_masses = [180.0634, 162.0528, 144.0422, 126.0317, 108.0211]

    print("\\nTesting given formula spectrum decomposition:")
    print(f"Given formula: {given_formula}")
    print(f"Fragment masses: {fragment_masses}")
    
    # Decompose fragments for the given formula
    fragment_results = decompose_given_formula_spectrum_cython(
        given_formula, fragment_masses, element_bounds,
        tolerance_ppm=5.0, max_results=1000
    )
    
    print("\\nFragment decomposition results:")
    for i, fragment_decomps in enumerate(fragment_results):
        print(f"Fragment {i+1} (mass {fragment_masses[i]:.4f}): {len(fragment_decomps)} possibilities")
        if fragment_decomps:
            print(f"  Examples: {fragment_decomps[:5]}")  # Show first 5


def test_fragment_equals_precursor():
    """Test specific case where fragment mass equals precursor mass."""
    
    # Define element bounds
    element_bounds = {
        'C': (0, 20),
        'H': (0, 40),
        'N': (0, 5),
        'O': (0, 15)
    }
    
    # Known precursor formula (glucose)
    given_formula = {'C': 6, 'H': 12, 'O': 6}
    precursor_mass = 180.0634
    
    # Test case: fragment mass equals precursor mass (molecular ion)
    fragment_masses = [180.0634]  # Same as precursor
    
    print("\nTesting fragment mass equal to precursor mass:")
    print(f"Given formula: {given_formula}")
    print(f"Precursor mass: {precursor_mass}")
    print(f"Fragment mass: {fragment_masses[0]} (equals precursor)")
    
    # Decompose fragments for the given formula
    fragment_results = decompose_given_formula_spectrum_cython(
        given_formula, fragment_masses, element_bounds,
        tolerance_ppm=5.0, max_results=1000
    )
    
    print("\nFragment decomposition results:")
    for i, fragment_decomps in enumerate(fragment_results):
        print(f"Fragment {i+1} (mass {fragment_masses[i]:.4f}): {len(fragment_decomps)} possibilities")
        if fragment_decomps:
            print(f"  Found formulas: {fragment_decomps}")
            # Check if the original formula is among the results
            if given_formula in fragment_decomps:
                print("  ✓ Original precursor formula correctly found in fragments!")
            else:
                print("  ✗ Original precursor formula NOT found in fragments!")


def calculate_mass(formula):
    """Calculate exact mass of a formula for verification."""
    masses = {
        'C': 12.000000, 'H': 1.007825, 'N': 14.003074, 'O': 15.994915,
        'P': 30.973762, 'S': 31.972072
    }
    return sum(count * masses.get(element, 0) for element, count in formula.items())


if __name__ == "__main__":
    print("Testing Cython Spectrum Decomposition Functions")
    print("=" * 50)
    
    # Test the spectrum decomposition functions
    test_spectrum_decomposition()
    test_given_formula_spectrum()
    test_fragment_equals_precursor()
    
    print("\\nTesting completed!")
