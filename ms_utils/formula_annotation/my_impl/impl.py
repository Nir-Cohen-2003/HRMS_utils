"""
SIRIUS Mass Decomposition Algorithm Implementation
Based on the paper: "SIRIUS: decomposing isotope patterns for metabolite identification"
https://arxiv.org/pdf/1307.7805

This module provides a high-performance implementation of the mass decomposition
algorithm used in SIRIUS for finding all possible molecular formulas within a given mass tolerance.

For maximum performance, compile the Cython version using the provided setup.py
"""

from typing import List, Dict, Tuple, Optional
from base_data import ATOMIC_MASSES
from python_impl import SiriusMassDecomposer, decompose_mass_fast, add_chemical_constraints
try:
    from sirius_decomposer import cython_decompose_mass
except ImportError:
    def cython_decompose_mass(*args, **kwargs):
        raise NotImplementedError("Cython version not available")

# Standard atomic masses (most abundant isotopes)

def benchmark_algorithms():
    """
    Benchmark different mass decomposition algorithms.
    """
    import time
    
    # Test case: 2 * (C17H19N3O)
    target_mass = 285.136493 *2
    element_bounds = {
        'C': (0, 50),
        'H': (0, 100), 
        'N': (0, 20),
        'O': (0, 40),
        'S': (0, 6),
        'P': (0, 5),
        'Na': (0, 1),
        'F': (0, 40),
        'Cl': (0, 0),
        'Br': (0, 0),   
        'I': (0, 4),
    }
    
    print(f"Benchmarking mass decomposition for mass {target_mass}")
    print(f"Element bounds: {element_bounds}")
    print("-" * 60)
    
    # Test recursive algorithm
    start_time = time.time()
    recursive_decomposer = SiriusMassDecomposer(element_bounds, target_mass, tolerance_ppm=5.0)
    recursive_results = recursive_decomposer.decompose()
    recursive_time = time.time() - start_time
    
    print("Recursive algorithm:")
    print(f"  Time: {recursive_time:.3f} seconds")
    print(f"  Results: {len(recursive_results)} formulas")
    
    # Test iterative algorithm
    start_time = time.time()
    iterative_results = decompose_mass_fast(target_mass, element_bounds, tolerance_ppm=5.0)
    iterative_time = time.time() - start_time
    
    print("Iterative algorithm:")
    print(f"  Time: {iterative_time:.3f} seconds")
    print(f"  Results: {len(iterative_results)} formulas")
    
    # Test Cython algorithm (if available)
    try:
        start_time = time.time()
        cython_results = cython_decompose_mass(target_mass, element_bounds, tolerance_ppm=5.0)
        cython_time = time.time() - start_time
        
        print("Cython algorithm:")
        print(f"  Time: {cython_time:.3f} seconds")
        print(f"  Results: {len(cython_results)} formulas")
        cython_available = True
    except NotImplementedError:
        print("Cython algorithm: Not compiled (run 'python setup.py build_ext --inplace')")
        cython_results = []
        cython_time = None
        cython_available = False
    except Exception as e:
        print(f"Cython algorithm: Error occurred - {type(e).__name__}: {e}")
        print("  This may indicate a bug in the Cython implementation")
        cython_results = []
        cython_time = None
        cython_available = False
    
    # Verify results are the same
    recursive_set = {frozenset(formula.items()) for formula in recursive_results}
    iterative_set = {frozenset(formula.items()) for formula in iterative_results}
    if cython_available:
        cython_set = {frozenset(formula.items()) for formula in cython_results}
    
    if recursive_set == iterative_set:
        print("✓ Recursive and iterative algorithms produce identical results")
        speedup = recursive_time / iterative_time if iterative_time > 0 else float('inf')
        print(f"  Iterative speedup: {speedup:.1f}x")
        
        if cython_available:
            cython_set = {frozenset(formula.items()) for formula in cython_results}
            if recursive_set == cython_set:
                print("✓ Cython algorithm produces identical results")
                cython_speedup = recursive_time / cython_time if cython_time > 0 else float('inf')
                iter_speedup = iterative_time / cython_time if cython_time > 0 else float('inf')
                print(f"  Cython speedup vs recursive: {cython_speedup:.1f}x")
                print(f"  Cython speedup vs iterative: {iter_speedup:.1f}x")
            else:
                print("✗ Cython results differ!")
    else:
        print("✗ Results differ!")
        print(f"  Recursive only: {len(recursive_set - iterative_set)}")
        print(f"  Iterative only: {len(iterative_set - recursive_set)}")
    
    print("\nFirst 10 results:")
    for i, formula in enumerate(iterative_results[:10]):
        mass = sum(ATOMIC_MASSES[elem] * count for elem, count in formula.items())
        error_ppm = abs(mass - target_mass) / target_mass * 1e6
        print(f"  {i+1:2d}: {formula} (mass: {mass:.4f}, error: {error_ppm:.1f} ppm)")

# Example usage and testing
if __name__ == "__main__":
    print("SIRIUS Mass Decomposition Algorithm")
    print("=" * 50)
    
    # Run benchmark
    benchmark_algorithms()
    
    print("\n" + "=" * 50)
    print("Chemical constraint filtering example:")
    
    # Example with chemical constraints
    target_mass = 285.136493 *2
    element_bounds = {
        'C': (0, 20),
        'H': (0, 50), 
        'N': (0, 10),
        'O': (0, 20),
        'S': (0, 3),
        'P': (0, 5),
        'Na': (0, 1),
        'F': (0, 20),
        'Cl': (0, 0),
        'Br': (0, 0),   
        'I': (0, 2),
    }
    
    formulas = decompose_mass_fast(target_mass, element_bounds, tolerance_ppm=5.0)
    print(f"\nFound {len(formulas)} total formulas")
    
    # Apply chemical constraints
    filtered = add_chemical_constraints(formulas, min_dbe=0, max_dbe=20, max_hetero_ratio=2.0)
    print(f"After chemical filtering: {len(filtered)} formulas")
    
    print("\nTop 5 chemically valid formulas:")
    for i, formula in enumerate(filtered[:5]):
        mass = sum(ATOMIC_MASSES[elem] * count for elem, count in formula.items())
        
        # Calculate DBE
        c = formula.get('C', 0)
        h = formula.get('H', 0)
        n = formula.get('N', 0)
        dbe = c + 1 - (h + n) / 2.0 if c > 0 else 0
        
        print(f"  {i+1}: {formula} (mass: {mass:.4f}, DBE: {dbe:.1f})")
    
    # print("\nTo compile the ultra-fast Cython version:")
    # print("  1. pip install cython numpy")
    # print("  2. python setup.py build_ext --inplace")
    # print("  3. Import: from sirius_decomposer import cython_decompose_mass")