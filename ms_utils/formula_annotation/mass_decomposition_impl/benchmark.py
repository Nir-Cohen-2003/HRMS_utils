"""
SIRIUS Mass Decomposition Algorithm Implementation
Based on the paper: "SIRIUS: decomposing isotope patterns for metabolite identification"
https://arxiv.org/pdf/1307.7805

This module provides a high-performance implementation of the mass decomposition
algorithm used in SIRIUS for finding all possible molecular formulas within a given mass tolerance.

For maximum performance, compile the Cython version using the provided setup.py
"""

from base_data import ATOMIC_MASSES
from python_impl import SiriusMassDecomposer, decompose_mass_fast, add_chemical_constraints

# Try to import new C++ implementation
try:
    from mass_decomposer_cpp import decompose_mass, decompose_mass_parallel, decompose_spectrum_parallel
    cpp_available = True
except ImportError:
    cpp_available = False
    def decompose_mass(*args, **kwargs):
        raise NotImplementedError("C++ decomposer not available")
    def decompose_mass_parallel(*args, **kwargs):
        raise NotImplementedError("C++ decomposer not available")
    def decompose_spectrum_parallel(*args, **kwargs):
        raise NotImplementedError("C++ decomposer not available")

# Standard atomic masses (most abundant isotopes)

def run_unified_decomposition(target_mass, element_bounds, strategy="recursive", tolerance_ppm=5.0, min_dbe=0, max_dbe=40):
    """Helper function to run a single unified decomposition - used for parallel processing."""
    return decompose_mass(target_mass, element_bounds, strategy=strategy, tolerance_ppm=tolerance_ppm, 
                         min_dbe=float(min_dbe), max_dbe=float(max_dbe))

def benchmark_algorithms(num_parallel_runs=2000):
    """
    Benchmark different mass decomposition algorithms with new unified interface.
    
    Args:
        num_parallel_runs (int): Number of parallel runs to test scaling performance.
    """
    import time
    
    # Test case: C17H19N3O
    target_mass = 285.136493
    tolerance_ppm = 5.0
    element_bounds = {
        'C': (0, 50), 'H': (0, 100), 'N': (0, 20), 'O': (0, 40),
        'S': (0, 6), 'P': (0, 5), 'Na': (0, 0), 'F': (0, 40),
        'Cl': (0, 0), 'Br': (0, 0), 'I': (0, 4),
    }
    
    min_dbe = 0
    max_dbe = 40
    
    print(f"Benchmarking mass decomposition for mass {target_mass}")
    print(f"Tolerance: {tolerance_ppm} ppm")
    print(f"Element bounds: {element_bounds}")
    print(f"DBE constraints: {min_dbe} <= DBE <= {max_dbe}")
    print("-" * 60)
    
    # Initialize variables for better scope handling
    recursive_cpp_results = []
    money_changing_cpp_results = []
    recursive_cpp_time = 0
    money_changing_cpp_time = 0
    
    # Test recursive algorithm without constraints
    start_time = time.time()
    recursive_decomposer = SiriusMassDecomposer(element_bounds, target_mass, tolerance_ppm=tolerance_ppm)
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
    iterative_results_raw = decompose_mass_fast(target_mass, element_bounds, tolerance_ppm=tolerance_ppm)
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
    
    # Test advanced algorithms (C++ only)
    if cpp_available:
        print("Advanced algorithms:")
        print("  C++ OpenMP implementations:")
        
        # Test recursive C++ strategy
        try:
            start_time = time.time()
            recursive_cpp_results = decompose_mass(target_mass, element_bounds, 
                                                  strategy="recursive", tolerance_ppm=tolerance_ppm, 
                                                  min_dbe=min_dbe, max_dbe=max_dbe, max_hetero_ratio=1000.0)
            recursive_cpp_time = time.time() - start_time
            print(f"    Recursive C++: {recursive_cpp_time:.3f} seconds")
            print(f"    Results: {len(recursive_cpp_results)} formulas")
        except Exception as e:
            print(f"    Recursive C++: Error - {e}")
            recursive_cpp_results = []
        
        # Test money-changing C++ strategy
        try:
            start_time = time.time()
            money_changing_cpp_results = decompose_mass(target_mass, element_bounds,
                                                       strategy="money_changing", tolerance_ppm=tolerance_ppm, 
                                                       min_dbe=min_dbe, max_dbe=max_dbe, max_hetero_ratio=1000.0)
            money_changing_cpp_time = time.time() - start_time
            print(f"    Money-changing C++: {money_changing_cpp_time:.3f} seconds")
            print(f"    Results: {len(money_changing_cpp_results)} formulas")
        except Exception as e:
            print(f"    Money-changing C++: Error - {e}")
            money_changing_cpp_results = []
        
        # Test parallel processing with C++ implementation
        try:
            print(f"C++ OpenMP parallel processing ({num_parallel_runs} masses):")
            target_masses = [target_mass] * num_parallel_runs
            
            # Parallel recursive C++
            start_time = time.time()
            parallel_recursive_cpp_results = decompose_mass_parallel(
                target_masses, element_bounds, strategy="recursive", tolerance_ppm=tolerance_ppm,
                min_dbe=min_dbe, max_dbe=max_dbe, max_hetero_ratio=1000.0)
            parallel_recursive_cpp_time = time.time() - start_time
            print(f"  Parallel recursive C++: {parallel_recursive_cpp_time:.3f} seconds")
            print(f"  Average per mass: {parallel_recursive_cpp_time/num_parallel_runs:.6f} seconds")
            print(f"  Throughput: {num_parallel_runs/parallel_recursive_cpp_time:.1f} decompositions/second")
            
            # Parallel money-changing C++
            start_time = time.time()
            parallel_money_cpp_results = decompose_mass_parallel(
                target_masses, element_bounds, strategy="money_changing", tolerance_ppm=tolerance_ppm,
                min_dbe=min_dbe, max_dbe=max_dbe, max_hetero_ratio=1000.0)
            parallel_money_cpp_time = time.time() - start_time
            print(f"  Parallel money-changing C++: {parallel_money_cpp_time:.3f} seconds")
            print(f"  Average per mass: {parallel_money_cpp_time/num_parallel_runs:.6f} seconds")
            print(f"  Throughput: {num_parallel_runs/parallel_money_cpp_time:.1f} decompositions/second")
            
            # Verify parallel results consistency
            first_recursive_cpp = parallel_recursive_cpp_results[0] if parallel_recursive_cpp_results else []
            first_money_cpp = parallel_money_cpp_results[0] if parallel_money_cpp_results else []
            
            if first_recursive_cpp and first_money_cpp:
                recursive_cpp_set = {frozenset(f.items()) for f in first_recursive_cpp}
                money_cpp_set = {frozenset(f.items()) for f in first_money_cpp}
                if recursive_cpp_set == money_cpp_set:
                    print("  ✓ C++ parallel strategies produce identical results")
                else:
                    print("  ✗ C++ parallel strategies produce different results")
                    print(f"    Recursive: {len(first_recursive_cpp)} results, Money-changing: {len(first_money_cpp)} results")
                    common = recursive_cpp_set & money_cpp_set
                    print(f"    Common results: {len(common)}")
                    
        except Exception as e:
            print(f"  C++ parallel processing: Error - {e}")
        
        # Test spectrum decomposition with C++
        try:
            print("C++ spectrum decomposition test:")
            fragment_masses = [target_mass * 0.8, target_mass * 0.6, target_mass * 0.4]
            
            start_time = time.time()
            spectrum_cpp_results = decompose_spectrum_parallel(
                target_mass, fragment_masses, element_bounds,
                strategy="money_changing", tolerance_ppm=tolerance_ppm,
                min_dbe=min_dbe, max_dbe=max_dbe, max_hetero_ratio=1000.0)
            spectrum_cpp_time = time.time() - start_time
            
            print(f"  C++ spectrum decomposition: {spectrum_cpp_time:.3f} seconds")
            print(f"  Precursor candidates: {len(spectrum_cpp_results['precursor'])}")
            print(f"  Fragment sets: {len(spectrum_cpp_results['fragments'])}")
                
        except Exception as e:
            print(f"  C++ spectrum decomposition: Error - {e}")
            
    else:
        print("Advanced algorithms: Not compiled (run 'python setup.py build_ext --inplace')")
    
    # Helper to print detailed formula differences
    def print_formula_differences(set1, set2, name1, name2, target_mass):
        def format_formula(f_set):
            formula = dict(f_set)
            mass = sum(ATOMIC_MASSES[elem] * count for elem, count in formula.items())
            error_ppm = abs(mass - target_mass) / target_mass * 1e6
            return f"{str(formula)} (mass: {mass:.6f}, error: {error_ppm:.2f} ppm)"

        diff1 = set1 - set2
        if diff1:
            print(f"  {name1} only ({len(diff1)}):")
            for f_set in sorted(list(diff1), key=lambda x: tuple(sorted(x))):
                print(f"    {format_formula(f_set)}")

        diff2 = set2 - set1
        if diff2:
            print(f"  {name2} only ({len(diff2)}):")
            for f_set in sorted(list(diff2), key=lambda x: tuple(sorted(x))):
                print(f"    {format_formula(f_set)}")

    # Verify results are the same
    recursive_set = {frozenset(formula.items()) for formula in recursive_results}
    iterative_set = {frozenset(formula.items()) for formula in iterative_results}
    
    if cpp_available:
        if recursive_cpp_results:
            recursive_cpp_set = {frozenset(formula.items()) for formula in recursive_cpp_results}
        if money_changing_cpp_results:
            money_changing_cpp_set = {frozenset(formula.items()) for formula in money_changing_cpp_results}
    
    print("\nPerformance Comparison:")
    print("-" * 60)
    
    if recursive_set == iterative_set:
        print("✓ Recursive and iterative algorithms produce identical results")
        speedup = recursive_time / iterative_time if iterative_time > 0 else float('inf')
        print(f"  Iterative speedup: {speedup:.1f}x")
        
        # Compare with C++ implementations
        if cpp_available:
            # Compare with recursive C++
            if recursive_cpp_results and recursive_set == recursive_cpp_set:
                print("✓ Recursive C++ produces identical results")
                cpp_speedup = recursive_time / recursive_cpp_time if recursive_cpp_time > 0 else float('inf')
                iter_cpp_speedup = iterative_time / recursive_cpp_time if recursive_cpp_time > 0 else float('inf')
                print(f"  Recursive C++ speedup vs recursive Python: {cpp_speedup:.1f}x")
                print(f"  Recursive C++ speedup vs iterative Python: {iter_cpp_speedup:.1f}x")
            elif recursive_cpp_results:
                print("✗ Recursive C++ results differ!")
                print(f"  Common results: {len(recursive_set & recursive_cpp_set)}")
                print_formula_differences(recursive_set, recursive_cpp_set, "Python", "Recursive C++", target_mass)
            
            # Compare with money-changing C++
            if money_changing_cpp_results and recursive_set == money_changing_cpp_set:
                print("✓ Money-changing C++ produces identical results")
                money_cpp_speedup = recursive_time / money_changing_cpp_time if money_changing_cpp_time > 0 else float('inf')
                print(f"  Money-changing C++ speedup vs recursive Python: {money_cpp_speedup:.1f}x")
            elif money_changing_cpp_results:
                print("✗ Money-changing C++ results differ!")
                print(f"  Common results: {len(recursive_set & money_changing_cpp_set)}")
                print_formula_differences(recursive_set, money_changing_cpp_set, "Python", "Money-changing C++", target_mass)
    else:
        print("✗ Results differ!")
        print_formula_differences(recursive_set, iterative_set, "Recursive", "Iterative", target_mass)
    
    # Store results for file output
    all_results = {
        "Recursive Python": recursive_results,
        "Iterative Python": iterative_results,
    }
    
    if cpp_available:
        if recursive_cpp_results:
            all_results["Recursive C++"] = recursive_cpp_results
        if money_changing_cpp_results:
            all_results["Money-changing C++"] = money_changing_cpp_results

    # Write results to a file for detailed comparison
    def write_results_to_file(filename, results_dict, target_mass):
        with open(filename, 'w') as f:
            f.write("Mass Decomposition Benchmark Results\n")
            f.write(f"Target Mass: {target_mass:.6f}\n")
            f.write("="*40 + "\n\n")
            for algo, results in results_dict.items():
                f.write(f"--- {algo} ({len(results)} results) ---\n")
                # Sort formulas for consistent ordering
                # Convert formula dict to a frozenset of items to handle duplicates, then to a list of dicts
                unique_formulas = [dict(fs) for fs in {frozenset(f.items()) for f in results}]
                # Sort by a canonical representation
                sorted_results = sorted(unique_formulas, key=lambda x: tuple(sorted(x.items())))
                for formula in sorted_results:
                    mass = sum(ATOMIC_MASSES[elem] * count for elem, count in formula.items())
                    error_ppm = abs(mass - target_mass) / target_mass * 1e6
                    f.write(f"{str(formula):<50} mass: {mass:<18.6f} error: {error_ppm:.2f} ppm\n")
                f.write("\n")
    
    write_results_to_file("benchmark_results.txt", all_results, target_mass)
    print("\nResults have been written to benchmark_results.txt for detailed comparison.")
    
    # Show constraint filtering statistics
    print("\nConstraint Filtering Statistics:")
    print(f"  Before constraints: {len(recursive_results_raw)} formulas")
    print(f"  After constraints: {len(recursive_results)} formulas")
    filter_ratio = (len(recursive_results_raw) - len(recursive_results)) / len(recursive_results_raw) * 100 if recursive_results_raw else 0
    print(f"  Filtered out: {filter_ratio:.1f}%")
    
    print("\nFirst 10 results (from iterative Python):")
    for i, formula in enumerate(iterative_results[:10]):
        mass = sum(ATOMIC_MASSES[elem] * count for elem, count in formula.items())
        error_ppm = abs(mass - target_mass) / target_mass * 1e6
        print(f"  {i+1:2d}: {formula} (mass: {mass:.4f}, error: {error_ppm:.1f} ppm)")

# Example usage and testing
if __name__ == "__main__":
    print("SIRIUS Mass Decomposition Algorithm")
    print("=" * 50)
    
    # Run benchmark
    benchmark_algorithms(200000)
