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
    from mass_decomposer_cpp import (decompose_mass, decompose_mass_parallel, 
                                    decompose_spectrum, decompose_spectra_parallel,
                                    decompose_spectrum_known_precursor, 
                                    decompose_spectra_known_precursor_parallel)
    cpp_available = True
except ImportError:
    cpp_available = False
    def decompose_mass(*args, **kwargs):
        raise NotImplementedError("C++ decomposer not available")
    def decompose_mass_parallel(*args, **kwargs):
        raise NotImplementedError("C++ decomposer not available")
    def decompose_spectrum(*args, **kwargs):
        raise NotImplementedError("C++ decomposer not available")
    def decompose_spectra_parallel(*args, **kwargs):
        raise NotImplementedError("C++ decomposer not available")
    def decompose_spectrum_known_precursor(*args, **kwargs):
        raise NotImplementedError("C++ decomposer not available")
    def decompose_spectra_known_precursor_parallel(*args, **kwargs):
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
    target_mass = 281.152812
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
        
        # Initialize fragment_masses outside try block
        fragment_masses = [target_mass * 0.8, target_mass * 0.6, target_mass * 0.4]
        
        # Test spectrum decomposition with C++
        try:
            print("C++ spectrum decomposition test:")
            
            # Test single spectrum (backward compatibility)
            start_time = time.time()
            spectrum_cpp_results = decompose_spectrum(
                target_mass, fragment_masses, element_bounds,
                strategy="money_changing", tolerance_ppm=tolerance_ppm,
                min_dbe=min_dbe, max_dbe=max_dbe, max_hetero_ratio=1000.0)
            spectrum_cpp_time = time.time() - start_time
            
            print(f"  C++ single spectrum decomposition: {spectrum_cpp_time:.3f} seconds")
            
            # Handle new format - spectrum_cpp_results is now a list of decompositions
            if spectrum_cpp_results:
                total_precursors = len(spectrum_cpp_results)
                total_fragment_sets = sum(len(result['fragments']) for result in spectrum_cpp_results)
                print(f"  Precursor candidates: {total_precursors}")
                print(f"  Total fragment sets: {total_fragment_sets}")
            else:
                print("  Precursor candidates: 0")
                print("  Total fragment sets: 0")
            
            # Test multiple spectra in parallel
            num_spectra = num_parallel_runs
            print(f"C++ multi-spectrum parallel processing ({num_spectra} spectra):")
            
            # Create test spectra with va
            spectra_data = [
                (281.152812, [281.152812, 241.121512, 237.090212, 221.095297, 93.070425]),
            ]

            # Test proper spectrum decomposition
            print("C++ proper spectrum decomposition test:")
            proper_fragment_masses = [263.142247, 220.125200, 78.046950]
            
            start_time = time.time()
            proper_spectrum_results = decompose_spectrum(
                target_mass, proper_fragment_masses, element_bounds,
                strategy="money_changing", tolerance_ppm=tolerance_ppm,
                min_dbe=min_dbe, max_dbe=max_dbe, max_hetero_ratio=1000.0)
            proper_spectrum_time = time.time() - start_time
            
            print(f"  C++ spectrum decomposition: {proper_spectrum_time:.3f} seconds")
            print(f"  Precursor-fragment combinations: {len(proper_spectrum_results)}")
            
            if proper_spectrum_results:
                sample_proper = proper_spectrum_results[0]
                print(f"  Sample result - Precursor: {sample_proper['precursor']}")
                print(f"  Sample result - Precursor mass: {sample_proper['precursor_mass']:.6f} "
                      f"(error: {sample_proper['precursor_error_ppm']:.2f} ppm)")
                for i, frag_formulas in enumerate(sample_proper['fragments']):
                    print(f"  Sample result - Fragment {i+1}: {len(frag_formulas)} possible formulas")
                    if frag_formulas:
                        print(f"    Best: {frag_formulas[0]} "
                              f"(mass: {sample_proper['fragment_masses'][i][0]:.6f}, "
                              f"error: {sample_proper['fragment_errors_ppm'][i][0]:.2f} ppm)")



            # multiplate the test spectrum for parallel processing
            spectra_data = spectra_data * num_spectra
            
            
            start_time = time.time()
            multi_spectrum_results = decompose_spectra_parallel(
                spectra_data, element_bounds, strategy="money_changing", 
                tolerance_ppm=tolerance_ppm, min_dbe=min_dbe, max_dbe=max_dbe, 
                max_hetero_ratio=1000.0)
            multi_spectrum_time = time.time() - start_time
            
            print(f"  Multi-spectrum parallel: {multi_spectrum_time:.3f} seconds")
            print(f"  Average per spectrum: {multi_spectrum_time/num_spectra:.6f} seconds")
            print(f"  Throughput: {num_spectra/multi_spectrum_time:.1f} spectra/second")
            print(f"  Processed spectra: {len(multi_spectrum_results)}")
            
            # Show sample results
            if multi_spectrum_results:
                sample_result = multi_spectrum_results[0]  # Get first spectrum's results
                if isinstance(sample_result, list) and sample_result:
                    # New format: each spectrum returns a list of decomposition results
                    total_precursors = len(sample_result)
                    total_fragment_sets = sum(len(decomp['fragments']) for decomp in sample_result)
                    print(f"  Sample result - Precursor candidates: {total_precursors}")
                    print(f"  Sample result - Total fragment sets: {total_fragment_sets}")
                elif isinstance(sample_result, dict) and 'precursor' in sample_result:
                    # Old format: each spectrum returns a dictionary
                    print(f"  Sample result - Precursor: {len(sample_result['precursor'])} formulas")
                    print(f"  Sample result - Fragments: {[len(frag) for frag in sample_result['fragments']]}")
                else:
                    print(f"  Sample result - Unexpected format: {type(sample_result)}")
            else:
                print("  No results returned")
            
            # Compare single vs batch processing efficiency
            single_spectrum_time_estimate = spectrum_cpp_time * num_spectra
            speedup = single_spectrum_time_estimate / multi_spectrum_time if multi_spectrum_time > 0 else float('inf')
            print(f"  Batch processing speedup: {speedup:.1f}x")
            
            
            
            
        except Exception as e:
            print(f"  C++ spectrum decomposition: Error - {e}")
            import traceback
            traceback.print_exc()
            spectrum_cpp_results = None
            proper_spectrum_results = None
            
    else:
        print("Advanced algorithms: Not compiled (run 'python setup.py build_ext --inplace')")
        spectrum_cpp_results = None
        proper_spectrum_results = None

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
    def write_results_to_file(filename, results_dict, target_mass, fragment_masses=None, spectrum_results=None, proper_spectrum_results=None):
        with open(filename, 'w') as f:
            f.write("Mass Decomposition Benchmark Results\n")
            f.write(f"Target Mass: {target_mass:.6f}\n")
            f.write("="*60 + "\n\n")
            
            # Write mass decomposition results
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
            
            # Write spectrum decomposition results
            if spectrum_results:
                f.write("="*60 + "\n")
                f.write("SPECTRUM DECOMPOSITION RESULTS\n")
                f.write("="*60 + "\n\n")
                
                try:
                    precursor_formulas = spectrum_results['precursor']
                    fragment_sets = spectrum_results['fragments']
                    
                    f.write(f"Precursor Mass: {target_mass:.6f}\n")
                    f.write(f"Fragment Masses: {fragment_masses}\n")
                    f.write(f"Total Precursor Candidates: {len(precursor_formulas)}\n")
                    f.write(f"Number of Fragment Masses: {len(fragment_masses)}\n")
                    f.write(f"Fragment Sets Structure Length: {len(fragment_sets)}\n\n")
                    
                    # Debug information
                    f.write(f"DEBUG: Spectrum results structure:\n")
                    f.write(f"  Type of spectrum_results: {type(spectrum_results)}\n")
                    f.write(f"  Keys: {list(spectrum_results.keys()) if hasattr(spectrum_results, 'keys') else 'N/A'}\n")
                    f.write(f"  Type of precursor_formulas: {type(precursor_formulas)}\n")
                    f.write(f"  Type of fragment_sets: {type(fragment_sets)}\n")
                    if fragment_sets:
                        f.write(f"  Fragment sets length: {len(fragment_sets)}\n")
                        for idx, frag_set in enumerate(fragment_sets):
                            f.write(f"  Fragment set {idx}: type={type(frag_set)}, length={len(frag_set) if hasattr(frag_set, '__len__') else 'N/A'}\n")
                    f.write("\n")
                    
                    # The fragment_sets appears to be organized by fragment mass, not by precursor
                    # So we have len(fragment_masses) fragment sets, each containing formulas for that fragment
                    
                    # First, write the fragment decomposition results organized by fragment mass
                    f.write("FRAGMENT DECOMPOSITION BY MASS:\n")
                    f.write("-" * 40 + "\n")
                    for j, fragment_mass in enumerate(fragment_masses):
                        f.write(f"\nFragment Mass {j+1}: {fragment_mass:.6f}\n")
                        if j < len(fragment_sets):
                            fragment_formulas = fragment_sets[j]
                            f.write(f"  Found {len(fragment_formulas)} possible formulas:\n")
                            
                            if isinstance(fragment_formulas, (list, tuple)):
                                # Sort fragment formulas for consistent output
                                sorted_fragments = []
                                for fragment in fragment_formulas:
                                    if isinstance(fragment, dict):
                                        sorted_fragments.append(fragment)
                                    else:
                                        f.write(f"    {str(fragment):<30} (unexpected format: {type(fragment)})\n")
                                        continue
                                
                                # Sort the valid dictionary fragments
                                sorted_fragments = sorted(sorted_fragments, key=lambda x: tuple(sorted(x.items())))
                                
                                for fragment in sorted_fragments:
                                    fragment_calc_mass = sum(ATOMIC_MASSES[elem] * count for elem, count in fragment.items())
                                    fragment_error = abs(fragment_calc_mass - fragment_mass) / fragment_mass * 1e6 if fragment_mass > 0 else 0.0
                                    f.write(f"    {str(fragment):<35} mass: {fragment_calc_mass:<12.6f} error: {fragment_error:.2f} ppm\n")
                            else:
                                f.write(f"    Unexpected fragment set format: {type(fragment_formulas)}\n")
                        else:
                            f.write("  No fragment data available\n")
                    
                    f.write("\n" + "="*60 + "\n")
                    f.write("PRECURSOR FORMULAS:\n")
                    f.write("-" * 40 + "\n")
                    
                    # Write precursor formulas
                    for i, precursor in enumerate(precursor_formulas):
                        # Handle different precursor formats
                        if isinstance(precursor, dict):
                            precursor_mass = sum(ATOMIC_MASSES[elem] * count for elem, count in precursor.items())
                            precursor_error = abs(precursor_mass - target_mass) / target_mass * 1e6
                            precursor_str = str(precursor)
                        else:
                            # If precursor is not a dict, try to convert or handle as string
                            precursor_str = str(precursor)
                            precursor_mass = target_mass  # Fallback
                            precursor_error = 0.0
                        
                        f.write(f"\nPrecursor {i+1}: {precursor_str}\n")
                        f.write(f"  Mass: {precursor_mass:.6f}, Error: {precursor_error:.2f} ppm\n")
                        
                        # Note: In this structure, fragments are not linked to specific precursors
                        # Each fragment mass has its own set of possible formulas independent of precursor
                        f.write(f"  Note: Fragment formulas are listed separately by mass above\n")
                
                except Exception as e:
                    f.write(f"Error writing spectrum results: {str(e)}\n")
                    f.write(f"Spectrum results type: {type(spectrum_results)}\n")
                    f.write(f"Spectrum results content: {str(spectrum_results)[:500]}...\n")
                    import traceback
                    f.write(f"Traceback: {traceback.format_exc()}\n")
            
            # Write proper spectrum decomposition results  
            if proper_spectrum_results:
                f.write("\n" + "="*60 + "\n")
                f.write("PROPER SPECTRUM DECOMPOSITION RESULTS\n")
                f.write("="*60 + "\n\n")
                f.write("This shows proper spectrum decomposition where fragments are guaranteed\n")
                f.write("to be elemental subsets of their corresponding precursor formulas.\n\n")
                
                try:
                    f.write(f"Total precursor-fragment combinations: {len(proper_spectrum_results)}\n\n")
                    
                    for i, decomp in enumerate(proper_spectrum_results[:10]):  # Show first 10
                        f.write(f"Combination {i+1}:\n")
                        f.write(f"  Precursor: {decomp['precursor']}\n")
                        f.write(f"  Precursor mass: {decomp['precursor_mass']:.6f} "
                               f"(error: {decomp['precursor_error_ppm']:.2f} ppm)\n")
                        
                        f.write("  Fragment decompositions:\n")
                        for j, frag_formulas in enumerate(decomp['fragments']):
                            if frag_formulas:
                                f.write(f"    Fragment {j+1}: {len(frag_formulas)} possible formulas\n")
                                # Show best 3 formulas for each fragment
                                for k, frag_formula in enumerate(frag_formulas[:3]):
                                    mass = decomp['fragment_masses'][j][k]
                                    error = decomp['fragment_errors_ppm'][j][k]
                                    f.write(f"      {frag_formula} "
                                           f"(mass: {mass:.6f}, error: {error:.2f} ppm)\n")
                                if len(frag_formulas) > 3:
                                    f.write(f"      ... and {len(frag_formulas)-3} more\n")
                            else:
                                f.write(f"    Fragment {j+1}: No valid formulas found\n")
                        f.write("\n")
                        
                    if len(proper_spectrum_results) > 10:
                        f.write(f"... and {len(proper_spectrum_results)-10} more combinations\n")
                
                except Exception as e:
                    f.write(f"Error writing proper spectrum results: {str(e)}\n")
                    f.write(f"Proper spectrum results type: {type(proper_spectrum_results)}\n")
                    f.write(f"Proper spectrum results content: {str(proper_spectrum_results)[:500]}...\n")
                    import traceback
                    f.write(f"Traceback: {traceback.format_exc()}\n")
    
    write_results_to_file("benchmark_results.txt", all_results, target_mass, fragment_masses, spectrum_cpp_results, proper_spectrum_results)
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

def benchmark_known_precursor_decomposition(num_parallel_runs=1000):
    """
    Benchmark spectrum decomposition with known precursor formulas.
    
    Args:
        num_parallel_runs (int): Number of parallel runs to test scaling performance.
    """
    import time
    
    print("\n" + "="*80)
    print("KNOWN PRECURSOR SPECTRUM DECOMPOSITION BENCHMARK")
    print("="*80)
    
    # Test case: Known precursor formula C17H19N3O
    known_precursor = {'C': 17, 'H': 19, 'N': 3, 'O': 1}
    fragment_masses = [263.142247, 220.125200, 194.109550, 167.081475, 78.046950]
    
    # Standard element bounds (for creating the decomposer)
    element_bounds = {
        'C': (0, 100), 'H': (0, 200), 'O': (0, 50), 'N': (0, 50),
        'P': (0, 20), 'S': (0, 20), 'F': (0, 20), 'Cl': (0, 20),
        'Br': (0, 10), 'I': (0, 10)
    }
    
    tolerance_ppm = 5.0
    min_dbe = 0.0
    max_dbe = 40.0
    max_hetero_ratio = 100.0
    max_results = 100000
    
    print(f"Test case:")
    print(f"  Known precursor formula: {known_precursor}")
    print(f"  Fragment masses: {fragment_masses}")
    print(f"  Tolerance: {tolerance_ppm} ppm")
    print(f"  DBE range: {min_dbe} - {max_dbe}")
    print(f"  Max results per fragment: {max_results}")
    
    if cpp_available:
        print("\nTesting C++ known precursor decomposition:")
        
        try:
            # Test single spectrum with known precursor
            print("Single spectrum with known precursor:")
            start_time = time.time()
            single_results = decompose_spectrum_known_precursor(
                known_precursor, fragment_masses, element_bounds,
                strategy="money_changing", tolerance_ppm=tolerance_ppm,
                min_dbe=min_dbe, max_dbe=max_dbe, max_hetero_ratio=max_hetero_ratio,
                max_results=max_results)
            single_time = time.time() - start_time
            
            print(f"  Single known precursor decomposition: {single_time:.3f} seconds")
            print(f"  Fragment results: {[len(frag_results) for frag_results in single_results]}")
            
            # Show sample results
            if single_results:
                for i, frag_results in enumerate(single_results[:3]):  # Show first 3 fragments
                    print(f"  Fragment {i+1} ({fragment_masses[i]:.6f} Da): {len(frag_results)} formulas")
                    if frag_results:
                        # Calculate masses for top 3 formulas
                        from base_data import ATOMIC_MASSES
                        for j, formula in enumerate(frag_results[:3]):
                            calc_mass = sum(ATOMIC_MASSES[elem] * count for elem, count in formula.items())
                            error_ppm = abs(calc_mass - fragment_masses[i]) / fragment_masses[i] * 1e6
                            print(f"    {formula} (mass: {calc_mass:.6f}, error: {error_ppm:.2f} ppm)")
            
            # Test parallel processing with multiple spectra (different known precursors)
            print(f"\nParallel processing ({num_parallel_runs} spectra with different known precursors):")
            
            # Create test data with different known precursor formulas
            test_precursors = [
                {'C': 17, 'H': 19, 'N': 3, 'O': 1},  # C17H19N3O
                {'C': 15, 'H': 17, 'N': 2, 'O': 2},  # C15H17N2O2  
                {'C': 20, 'H': 22, 'N': 2, 'O': 3},  # C20H22N2O3
                {'C': 12, 'H': 15, 'N': 1, 'O': 4},  # C12H15NO4
                {'C': 18, 'H': 20, 'N': 4, 'O': 1},  # C18H20N4O
            ]
            
            # Create parallel test data
            parallel_spectra_data = []
            for i in range(num_parallel_runs):
                # Cycle through different precursor formulas
                precursor = test_precursors[i % len(test_precursors)]
                # Use the same fragment masses for simplicity
                parallel_spectra_data.append((precursor, fragment_masses))
            
            start_time = time.time()
            parallel_results = decompose_spectra_known_precursor_parallel(
                parallel_spectra_data, element_bounds,
                strategy="money_changing", tolerance_ppm=tolerance_ppm,
                min_dbe=min_dbe, max_dbe=max_dbe, max_hetero_ratio=max_hetero_ratio,
                max_results=max_results)
            parallel_time = time.time() - start_time
            
            print(f"  Parallel known precursor decomposition: {parallel_time:.3f} seconds")
            print(f"  Average per spectrum: {parallel_time/num_parallel_runs:.6f} seconds")
            print(f"  Throughput: {num_parallel_runs/parallel_time:.1f} spectra/second")
            print(f"  Processed spectra: {len(parallel_results)}")
            
            # Show sample parallel results
            if parallel_results:
                sample_result = parallel_results[0]
                print(f"  Sample result - Fragment results: {[len(frag_results) for frag_results in sample_result]}")
                
                # Compare with original known precursor
                if len(parallel_results) > 1:
                    different_precursor_result = parallel_results[1]
                    print(f"  Different precursor result - Fragment results: {[len(frag_results) for frag_results in different_precursor_result]}")
            
            # Compare single vs parallel efficiency
            single_time_estimate = single_time * num_parallel_runs
            speedup = single_time_estimate / parallel_time if parallel_time > 0 else float('inf')
            print(f"  Parallel processing speedup: {speedup:.1f}x")
            
            # Compare strategies
            print("\nComparing strategies for known precursor decomposition:")
            
            # Recursive strategy
            start_time = time.time()
            recursive_results = decompose_spectrum_known_precursor(
                known_precursor, fragment_masses, element_bounds,
                strategy="recursive", tolerance_ppm=tolerance_ppm,
                min_dbe=min_dbe, max_dbe=max_dbe, max_hetero_ratio=max_hetero_ratio,
                max_results=max_results)
            recursive_time = time.time() - start_time
            
            print(f"  Recursive strategy: {recursive_time:.3f} seconds")
            print(f"  Fragment results: {[len(frag_results) for frag_results in recursive_results]}")
            
            # Money-changing strategy (already done above)
            print(f"  Money-changing strategy: {single_time:.3f} seconds") 
            print(f"  Fragment results: {[len(frag_results) for frag_results in single_results]}")
            
            # Verify both strategies give same results
            if recursive_results and single_results:
                strategies_match = True
                for i, (rec_frags, mc_frags) in enumerate(zip(recursive_results, single_results)):
                    rec_set = {frozenset(f.items()) for f in rec_frags}
                    mc_set = {frozenset(f.items()) for f in mc_frags}
                    if rec_set != mc_set:
                        strategies_match = False
                        print(f"  ✗ Fragment {i+1} results differ between strategies")
                        print(f"    Recursive: {len(rec_frags)} formulas, Money-changing: {len(mc_frags)} formulas")
                        break
                
                if strategies_match:
                    print("  ✓ Both strategies produce identical results")
                    
        except Exception as e:
            print(f"  Known precursor decomposition error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("C++ implementation not available. Please compile with 'python setup.py build_ext --inplace'")


# Example usage and testing
if __name__ == "__main__":
    print("SIRIUS Mass Decomposition Algorithm")
    print("=" * 50)
    
    # Run benchmark
    benchmark_algorithms(20000)
    benchmark_known_precursor_decomposition(10000)
