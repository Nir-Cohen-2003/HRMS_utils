"""
SIRIUS Mass Decomposition Algorithm Implementation
Based on the paper: "SIRIUS: decomposing isotope patterns for metabolite identification"
https://arxiv.org/pdf/1307.7805

This module provides a high-performance implementation of the mass decomposition
algorithm used in SIRIUS for finding all possible molecular formulas within a given mass tolerance.

For maximum performance, compile the Cython version using the provided setup.py
"""

import concurrent.futures
from base_data import ATOMIC_MASSES
from python_impl import SiriusMassDecomposer, decompose_mass_fast, add_chemical_constraints
try:
    from sirius_decomposer import cython_decompose_mass
except ImportError:
    def cython_decompose_mass(*args, **kwargs):
        raise NotImplementedError("Cython version not available")

# Standard atomic masses (most abundant isotopes)

def run_cython_decomposition(target_mass, element_bounds, tolerance_ppm=5.0, min_dbe=0, max_dbe=40):
    """Helper function to run a single Cython decomposition - used for parallel processing."""
    return cython_decompose_mass(target_mass, element_bounds, tolerance_ppm=tolerance_ppm, 
                               min_dbe=float(min_dbe), max_dbe=float(max_dbe))

def benchmark_algorithms():
    """
    Benchmark different mass decomposition algorithms with DBE constraints.
    """
    import time
    
    # Test case: 2 * (C17H19N3O)
    target_mass = 285.136493 * 1
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
    
    # DBE constraints
    min_dbe = 0
    max_dbe = 40
    
    print(f"Benchmarking mass decomposition for mass {target_mass}")
    print(f"Element bounds: {element_bounds}")
    print(f"DBE constraints: {min_dbe} <= DBE <= {max_dbe}")
    print("-" * 60)
    
    # Test recursive algorithm without constraints
    start_time = time.time()
    recursive_decomposer = SiriusMassDecomposer(element_bounds, target_mass, tolerance_ppm=5.0)
    recursive_results_raw = recursive_decomposer.decompose()
    recursive_time_raw = time.time() - start_time
    
    # Apply constraints using Python function
    start_time = time.time()
    recursive_results = add_chemical_constraints(recursive_results_raw, min_dbe=min_dbe, max_dbe=max_dbe)
    recursive_constraint_time = time.time() - start_time
    recursive_time = recursive_time_raw + recursive_constraint_time
    
    print("Recursive algorithm (Python constraints):")
    print(f"  Decomposition time: {recursive_time_raw:.3f} seconds")
    print(f"  Constraint time: {recursive_constraint_time:.3f} seconds")
    print(f"  Total time: {recursive_time:.3f} seconds")
    print(f"  Results before constraints: {len(recursive_results_raw)} formulas")
    print(f"  Results after constraints: {len(recursive_results)} formulas")
    
    # Test iterative algorithm without constraints
    start_time = time.time()
    iterative_results_raw = decompose_mass_fast(target_mass, element_bounds, tolerance_ppm=5.0)
    iterative_time_raw = time.time() - start_time
    
    # Apply constraints using Python function
    start_time = time.time()
    iterative_results = add_chemical_constraints(iterative_results_raw, min_dbe=min_dbe, max_dbe=max_dbe)
    iterative_constraint_time = time.time() - start_time
    iterative_time = iterative_time_raw + iterative_constraint_time
    
    print("Iterative algorithm (Python constraints):")
    print(f"  Decomposition time: {iterative_time_raw:.3f} seconds")
    print(f"  Constraint time: {iterative_constraint_time:.3f} seconds")
    print(f"  Total time: {iterative_time:.3f} seconds")
    print(f"  Results before constraints: {len(iterative_results_raw)} formulas")
    print(f"  Results after constraints: {len(iterative_results)} formulas")
    
    # Test Cython algorithm (if available) with constraints during enumeration
    # Run 200 parallel instances to test performance under load
    try:
        print("Cython algorithm (200 parallel instances):")
        start_time = time.time()
        
        # Run 200 parallel instances of the same decomposition
        with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
            futures = []
            for i in range(200):
                future = executor.submit(run_cython_decomposition, target_mass, element_bounds, 
                                       tolerance_ppm=5.0, min_dbe=min_dbe, max_dbe=max_dbe)
                futures.append(future)
            
            # Collect all results
            cython_results_list = []
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                cython_results_list.append(result)
        
        cython_time = time.time() - start_time
        
        # Use the first result for comparison (all should be identical)
        cython_results = cython_results_list[0] if cython_results_list else []
        
        print(f"  Time for 200 parallel runs: {cython_time:.3f} seconds")
        print(f"  Average time per run: {cython_time/200:.3f} seconds")
        print(f"  Results per run: {len(cython_results)} formulas")
        print(f"  Total completed runs: {len(cython_results_list)}")
        
        # Verify all results are identical
        if len(cython_results_list) > 1:
            first_result_set = {frozenset(formula.items()) for formula in cython_results_list[0]}
            all_identical = all(
                {frozenset(formula.items()) for formula in result} == first_result_set 
                for result in cython_results_list[1:]
            )
            print(f"  All parallel results identical: {'✓' if all_identical else '✗'}")
        
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
    
    print("\nPerformance Comparison:")
    print("-" * 60)
    
    if recursive_set == iterative_set:
        print("✓ Recursive and iterative algorithms produce identical results")
        speedup = recursive_time / iterative_time if iterative_time > 0 else float('inf')
        print(f"  Iterative speedup: {speedup:.1f}x")
        
        if cython_available:
            if recursive_set == cython_set:
                print("✓ Cython algorithm produces identical results")
                # Compare single-run performance (average cython time vs single python runs)
                avg_cython_time = cython_time / 200
                cython_speedup = recursive_time / avg_cython_time if avg_cython_time > 0 else float('inf')
                iter_speedup = iterative_time / avg_cython_time if avg_cython_time > 0 else float('inf')
                print(f"  Cython speedup vs recursive (single run): {cython_speedup:.1f}x")
                print(f"  Cython speedup vs iterative (single run): {iter_speedup:.1f}x")
                print(f"  Parallel processing throughput: {200/cython_time:.1f} decompositions/second")
                
                # Calculate efficiency gains from constraint integration
                total_python_time = max(recursive_time, iterative_time)
                constraint_overhead = max(recursive_constraint_time, iterative_constraint_time)
                efficiency_gain = (total_python_time - avg_cython_time) / total_python_time * 100
                print(f"  Constraint integration efficiency gain: {efficiency_gain:.1f}%")
                print(f"  Python constraint overhead: {constraint_overhead:.3f}s ({constraint_overhead/total_python_time*100:.1f}%)")
            else:
                print("✗ Cython results differ!")
                print(f"  Common results: {len(recursive_set & cython_set)}")
                print(f"  Python only: {len(recursive_set - cython_set)}")
                print(f"  Cython only: {len(cython_set - recursive_set)}")
    else:
        print("✗ Results differ!")
        print(f"  Recursive only: {len(recursive_set - iterative_set)}")
        print(f"  Iterative only: {len(iterative_set - recursive_set)}")
    
    # Show constraint filtering statistics
    print("\nConstraint Filtering Statistics:")
    print(f"  Before constraints: {len(recursive_results_raw)} formulas")
    print(f"  After constraints: {len(recursive_results)} formulas")
    filter_ratio = (len(recursive_results_raw) - len(recursive_results)) / len(recursive_results_raw) * 100
    print(f"  Filtered out: {filter_ratio:.1f}%")
    
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
    