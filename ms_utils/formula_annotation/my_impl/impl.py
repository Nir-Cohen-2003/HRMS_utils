"""
SIRIUS Mass Decomposition Algorithm Implementation
Based on the paper: "SIRIUS: decomposing isotope patterns for metabolite identification"
https://arxiv.org/pdf/1307.7805

This module provides a high-performance implementation of the mass decomposition
algorithm used in SIRIUS for finding all possible molecular formulas within a given mass tolerance.

For maximum performance, compile the Cython version using the provided setup.py
"""

from typing import List, Dict, Tuple, Optional
try:
    from sirius_decomposer import cython_decompose_mass
except ImportError:
    def cython_decompose_mass(*args, **kwargs):
        raise NotImplementedError("Cython version not available")

# Standard atomic masses (most abundant isotopes)
ATOMIC_MASSES = {
    'C': 12.0000000,
    'H': 1.0078250,
    'N': 14.0030740,
    'O': 15.9949146,
    'P': 30.9737620,
    'S': 31.9720718,
    'F': 18.9984032,
    'Cl': 34.9688527,
    'Br': 78.9183376,
    'I': 126.9044719,
    'Si': 27.9769271,
    'Na': 22.9897693,
    'K': 38.9637069,
    'Ca': 39.9625912,
    'Mg': 23.9850423,
    'Fe': 55.9349421,
    'Zn': 63.9291466,
    'Se': 79.9165218,
    'B': 11.0093054,
    'Al': 26.9815386
}

def add_chemical_constraints(formulas: List[Dict[str, int]], 
                           min_dbe: Optional[float] = None,
                           max_dbe: Optional[float] = None,
                           max_hetero_ratio: Optional[float] = None) -> List[Dict[str, int]]:
    """
    Apply additional chemical constraints to filter formulas.
    
    Args:
        formulas: List of molecular formulas
        min_dbe: Minimum double bond equivalents
        max_dbe: Maximum double bond equivalents
        max_hetero_ratio: Maximum ratio of heteroatoms to carbons
    
    Returns:
        Filtered list of formulas
    """
    filtered = []
    
    for formula in formulas:
        # Calculate DBE (Double Bond Equivalents)
        c = formula.get('C', 0)
        h = formula.get('H', 0)
        n = formula.get('N', 0)
        p = formula.get('P', 0)
        x = sum(formula.get(halogen, 0) for halogen in ['F', 'Cl', 'Br', 'I'])
        
        if c == 0:
            continue  # Skip formulas without carbon
        
        dbe = c + 1 - (h - x + n + p) / 2.0
        
        # Apply DBE constraints
        if min_dbe is not None and dbe < min_dbe:
            continue
        if max_dbe is not None and dbe > max_dbe:
            continue
        
        # Apply heteroatom ratio constraint
        if max_hetero_ratio is not None:
            heteroatoms = sum(count for elem, count in formula.items() if elem not in ['C', 'H'])
            if c > 0 and heteroatoms / c > max_hetero_ratio:
                continue
        
        filtered.append(formula)
    
    return filtered

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