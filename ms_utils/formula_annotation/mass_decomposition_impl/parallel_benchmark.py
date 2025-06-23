"""
Enhanced Benchmark for Parallel Cython Decomposer

This benchmark demonstrates the performance improvements of parallelizing 
the Cython decomposer across multiple CPU cores for batch processing.
"""

import time
import os
from typing import List
from base_data import ATOMIC_MASSES

# Import Cython implementations
try:
    from sirius_decomposer import cython_decompose_mass
    from parallel_cython_decomposer import (
        parallel_decompose_masses,
        parallel_decompose_single_mass_multiple_times
    )
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False
    print("Warning: Cython implementations not available. Run 'python setup.py build_ext --inplace'")


def generate_test_masses(n_masses: int = 10) -> List[float]:
    """Generate a list of test masses for benchmarking."""
    base_masses = [
        285.136493,  # C17H19N3O
        342.198745,  # Larger molecule
        198.087623,  # Medium molecule
        456.234567,  # Large molecule
        123.045678,  # Small molecule
        567.289345,  # Very large molecule
        89.012345,   # Very small molecule
        234.145678,  # Another medium molecule
        389.156789,  # Another large molecule
        145.067890,  # Another small molecule
    ]
    
    # Extend the list by cycling through base masses
    test_masses = []
    for i in range(n_masses):
        test_masses.append(base_masses[i % len(base_masses)])
    
    return test_masses


def benchmark_parallel_vs_sequential(n_masses: int = 20, 
                                    n_processes: int = None,
                                    show_progress: bool = True):
    """
    Benchmark parallel vs sequential execution of Cython decomposer.
    
    Args:
        n_masses: Number of different masses to decompose
        n_processes: Number of processes to use (None = all cores)
        show_progress: Whether to show progress during execution
    """
    if not CYTHON_AVAILABLE:
        print("Cython not available. Cannot run parallel benchmark.")
        return
    
    print("Parallel vs Sequential Benchmark")
    print("Sequential processing: first 50 masses only")
    print(f"Parallel processing: all {n_masses} masses")
    print("Comparison metric: average time per mass")
    print(f"CPU cores available: {os.cpu_count()}")
    print(f"Processes to use: {n_processes or os.cpu_count()}")
    print("=" * 50)
    
    # Generate test data
    target_masses = generate_test_masses(n_masses)
    element_bounds = {
        'C': (0, 50),
        'H': (0, 100), 
        'N': (0, 20),
        'O': (0, 40),
        'S': (0, 6),
        'P': (0, 5),
        'Na': (0, 1),
        'F': (0, 20),
        'Cl': (0, 10),
        'Br': (0, 5),   
        'I': (0, 4),
    }
    
    # Parameters
    tolerance_ppm = 5.0
    min_dbe = 0.0
    max_dbe = 40.0
    
    print(f"Element bounds: {element_bounds}")
    print(f"DBE constraints: {min_dbe} <= DBE <= {max_dbe}")
    print(f"Tolerance: {tolerance_ppm} ppm")
    print("-" * 50)
    
    # Progress callback
    def progress_callback(completed, total):
        if show_progress:
            percent = (completed / total) * 100
            print(f"  Progress: {completed}/{total} ({percent:.1f}%)")
     # 1. Sequential execution (limited to first 20 masses)
    sequential_masses = min(20, n_masses)
    sequential_target_masses = target_masses[:sequential_masses]
    
    print(f"1. Sequential Execution (first {sequential_masses} masses):")
    start_time = time.time()
    sequential_results = []
    
    for i, mass in enumerate(sequential_target_masses):
        result = cython_decompose_mass(
            target_mass=mass,
            element_bounds=element_bounds,
            tolerance_ppm=tolerance_ppm,
            min_dbe=min_dbe,
            max_dbe=max_dbe
        )
        sequential_results.append(result)

    sequential_time = time.time() - start_time
    sequential_avg_time = sequential_time / sequential_masses
    print(f"  Time: {sequential_time:.3f} seconds")
    print(f"  Average per mass: {sequential_avg_time:.3f} seconds")
    
    # 2. Parallel execution using ParallelCythonDecomposer (all masses)
    print(f"\n2. Parallel Execution (all {n_masses} masses):")
    start_time = time.time()
    
    parallel_results = parallel_decompose_masses(
        target_masses=target_masses,
        element_bounds=element_bounds,
        tolerance_ppm=tolerance_ppm,
        min_dbe=min_dbe,
        max_dbe=max_dbe,
        n_processes=n_processes,
    )
    
    parallel_time = time.time() - start_time
    parallel_avg_time = parallel_time / n_masses
    print(f"  Time: {parallel_time:.3f} seconds")
    print(f"  Average per mass: {parallel_avg_time:.3f} seconds")
    
    # 3. Calculate performance metrics (comparing average times per mass)
    print("\n3. Performance Analysis (Average Time Per Mass):")
    print("-" * 40)
    
    # Compare average times per mass
    avg_time_ratio = sequential_avg_time / parallel_avg_time if parallel_avg_time > 0 else float('inf')
    efficiency = avg_time_ratio / (n_processes or os.cpu_count()) * 100
    throughput_sequential = 1.0 / sequential_avg_time  # masses per second
    throughput_parallel = 1.0 / parallel_avg_time     # masses per second
    
    print(f"  Sequential avg time per mass: {sequential_avg_time:.3f} seconds")
    print(f"  Parallel avg time per mass: {parallel_avg_time:.3f} seconds")
    print(f"  Average time improvement: {avg_time_ratio:.2f}x faster per mass")
    print(f"  Efficiency: {efficiency:.1f}%")
    print(f"  Sequential throughput: {throughput_sequential:.1f} masses/second")
    print(f"  Parallel throughput: {throughput_parallel:.1f} masses/second")
    
    # 4. Verify results are identical (for overlapping masses)
    print("\n4. Result Verification (first 50 masses):")
    results_match = True
    total_formulas_sequential = 0
    total_formulas_parallel = 0
    
    # Only compare results for the masses that were processed sequentially
    comparison_count = min(len(sequential_results), len(parallel_results))
    
    for i in range(comparison_count):
        seq_result = sequential_results[i]
        par_result = parallel_results[i]
        
        seq_set = {frozenset(formula.items()) for formula in seq_result}
        par_set = {frozenset(formula.items()) for formula in par_result}
        
        total_formulas_sequential += len(seq_result)
        total_formulas_parallel += len(par_result)
        
        if seq_set != par_set:
            print(f"  ✗ Results differ for mass {target_masses[i]}")
            print(f"    Sequential: {len(seq_result)} formulas")
            print(f"    Parallel: {len(par_result)} formulas")
            results_match = False
    
    if results_match:
        print(f"  ✓ All results identical for first {comparison_count} masses")
        print(f"  Total formulas found (sequential): {total_formulas_sequential}")
        print(f"  Total formulas found (parallel, first {comparison_count}): {total_formulas_parallel}")
    else:
        print("  ✗ Some results differ")
    
    # 5. Show sample results
    print("\n5. Sample Results:")
    print("-" * 20)
    for i in range(min(3, len(target_masses))):
        mass = target_masses[i]
        result = parallel_results[i]
        print(f"  Mass {mass:.6f}: {len(result)} formulas")
        
        if result:
            # Show first formula
            first_formula = result[0]
            calc_mass = sum(ATOMIC_MASSES[elem] * count for elem, count in first_formula.items())
            error_ppm = abs(calc_mass - mass) / mass * 1e6
            print(f"    Example: {first_formula}")
            print(f"    Calculated mass: {calc_mass:.6f} (error: {error_ppm:.1f} ppm)")


def benchmark_scaling(max_masses: int = 100, step: int = 10):
    """
    Benchmark how performance scales with the number of masses.
    
    Args:
        max_masses: Maximum number of masses to test
        step: Step size for testing
    """
    if not CYTHON_AVAILABLE:
        print("Cython not available. Cannot run scaling benchmark.")
        return
    
    print("Scaling Benchmark")
    print(f"Testing from {step} to {max_masses} masses (step: {step})")
    print("=" * 50)
    
    element_bounds = {
        'C': (0, 30),
        'H': (0, 60), 
        'N': (0, 10),
        'O': (0, 20),
        'S': (0, 3),
    }
    
    results = []
    
    for n_masses in range(step, max_masses + 1, step):
        target_masses = generate_test_masses(n_masses)
        
        # Sequential
        start_time = time.time()
        for mass in target_masses:
            cython_decompose_mass(
                target_mass=mass,
                element_bounds=element_bounds,
                tolerance_ppm=5.0,
                min_dbe=0.0,
                max_dbe=30.0
            )
        sequential_time = time.time() - start_time
        
        # Parallel
        start_time = time.time()
        parallel_decompose_masses(
            target_masses=target_masses,
            element_bounds=element_bounds,
            tolerance_ppm=5.0,
            min_dbe=0.0,
            max_dbe=30.0
        )
        parallel_time = time.time() - start_time
        
        speedup = sequential_time / parallel_time if parallel_time > 0 else float('inf')
        efficiency = speedup / os.cpu_count() * 100
        
        results.append({
            'n_masses': n_masses,
            'sequential_time': sequential_time,
            'parallel_time': parallel_time,
            'speedup': speedup,
            'efficiency': efficiency
        })
        
        print(f"  {n_masses:3d} masses: {parallel_time:.2f}s parallel, {sequential_time:.2f}s sequential, "
              f"{speedup:.1f}x speedup, {efficiency:.1f}% efficiency")
    
    # Summary
    print("\nScaling Summary:")
    best_speedup = max(results, key=lambda x: x['speedup'])
    best_efficiency = max(results, key=lambda x: x['efficiency'])
    
    print(f"  Best speedup: {best_speedup['speedup']:.1f}x at {best_speedup['n_masses']} masses")
    print(f"  Best efficiency: {best_efficiency['efficiency']:.1f}% at {best_efficiency['n_masses']} masses")


def benchmark_repeated_decomposition(n_repeats: int = 100):
    """
    Benchmark repeated decomposition of the same mass (like the original benchmark).
    
    This is useful for comparing with the existing benchmark results.
    
    Args:
        n_repeats: Number of times to repeat the same decomposition
    """
    if not CYTHON_AVAILABLE:
        print("Cython not available. Cannot run repeated decomposition benchmark.")
        return
    
    print("Repeated Decomposition Benchmark")
    print(f"Repeating the same decomposition {n_repeats} times")
    print("=" * 50)
    
    # Test case from original benchmark
    target_mass = 285.136493
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
    
    min_dbe = 0
    max_dbe = 40
    tolerance_ppm = 5.0
    
    print(f"Target mass: {target_mass}")
    print(f"Element bounds: {element_bounds}")
    print(f"DBE constraints: {min_dbe} <= DBE <= {max_dbe}")
    print("-" * 50)
    
    # Use the parallel function for repeated decomposition
    start_time = time.time()
    results = parallel_decompose_single_mass_multiple_times(
        target_mass=target_mass,
        element_bounds=element_bounds,
        n_runs=n_repeats,
        tolerance_ppm=tolerance_ppm,
        min_dbe=float(min_dbe),
        max_dbe=float(max_dbe)
    )
    parallel_time = time.time() - start_time
    
    print(f"Parallel execution ({n_repeats} runs): {parallel_time:.3f} seconds")
    print(f"Average time per run: {parallel_time/n_repeats:.3f} seconds")
    print(f"Throughput: {n_repeats/parallel_time:.1f} decompositions/second")
    
    if results:
        print(f"Results per run: {len(results[0])} formulas")
        
        # Verify all results are identical
        first_result_set = {frozenset(formula.items()) for formula in results[0]}
        all_identical = all(
            {frozenset(formula.items()) for formula in result} == first_result_set 
            for result in results[1:]
        )
        print(f"All results identical: {'✓' if all_identical else '✗'}")


# Main execution
if __name__ == "__main__":
    import sys
    
    print("Enhanced Parallel Cython Decomposer Benchmark")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"CPU cores: {os.cpu_count()}")
    print()
    
    if not CYTHON_AVAILABLE:
        print("ERROR: Cython decomposer not available!")
        print("Please run: python setup.py build_ext --inplace")
        sys.exit(1)
    
    # Parse command line arguments
    
    # Default: run all benchmarks
    print("1. Parallel vs Sequential (20 sequential vs 48000 parallel masses)")
    benchmark_parallel_vs_sequential(n_masses=48000)
    
    print("\n" + "="*60 + "\n")
    
    # print("2. Scaling Test (10-50 masses)")
    # benchmark_scaling(max_masses=50, step=10)
    
    # print("\n" + "="*60 + "\n")

    # print("3. Repeated Decomposition (1000 runs)")
    # benchmark_repeated_decomposition(n_repeats=10000)
