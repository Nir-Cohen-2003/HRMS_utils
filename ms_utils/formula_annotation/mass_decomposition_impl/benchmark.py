"""
Benchmark script for the mass_decomposer_cpp library.

This script tests the performance of all decomposition functions and strategies,
and writes the results to benchmark_results.txt.
"""

import time
import numpy as np

try:
    from mass_decomposer_cpp import (
        get_element_order,
        decompose_mass,
        decompose_spectrum,
        decompose_spectrum_known_precursor,
        decompose_mass_parallel,
        decompose_spectrum_parallel,
        decompose_spectrum_known_precursor_parallel
    )
    cpp_available = True
    print("C++ implementation available.")
except ImportError as e:
    cpp_available = False
    print(f"Import error: {e}")
    import importlib.util
    spec = importlib.util.find_spec("mass_decomposer_cpp")
    print(f"Module spec: {spec}")

try:
    ELEMENT_ORDER = get_element_order() if cpp_available else []
except Exception as e:
    cpp_available = False
    ELEMENT_ORDER = []
    print(f"Error getting element order: {e}")
    print("C++ implementation not available. Please compile with 'python setup.py build_ext --inplace'")
def formula_to_array(formula: dict) -> np.ndarray:
    """Converts a formula dictionary to a numpy array based on the element order."""
    arr = np.zeros(len(ELEMENT_ORDER), dtype=np.int32)
    for i, element in enumerate(ELEMENT_ORDER):
        arr[i] = formula.get(element, 0)
    return arr

def array_to_formula(arr: np.ndarray) -> dict:
    """Converts a numpy array to a formula dictionary."""
    return {ELEMENT_ORDER[i]: count for i, count in enumerate(arr) if count > 0}

def run_benchmarks(num_parallel: int = 10000):
    """Runs all benchmarks and writes the results to a file."""
    if not cpp_available:
        print("C++ implementation not available. Please compile with 'python setup.py build_ext --inplace'")
        return

    with open("benchmark_results.txt", "w") as f:
        f.write("Mass Decomposition Benchmark Results\n")
        f.write("="*60 + "\n\n")
        print("Running benchmarks...")
        # --- Test Data ---
        target_mass = 281.152812
        bounds = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # Min counts
            [100, 1, 50, 20, 6, 5, 1, 1, 4, 5, 1, 1, 1, 1, 1]  # Max counts
        ], dtype=np.int32)
        fragment_masses = [263.142247, 220.125200, 78.046950]
        known_precursor = formula_to_array({'C': 17, 'H': 19, 'N': 3, 'O': 1})
        mass_accuracy_ppm = 5.0
        print("Test data initialized.")
        # --- Single Mass Decomposition ---
        f.write("--- Single Mass Decomposition ---\n")
        print("--- Single Mass Decomposition ---")
        for strategy in ["recursive", "money_changing"]:
            print(f"Running strategy: {strategy}")
            start_time = time.time()
            try:
                results = decompose_mass(target_mass, bounds, strategy=strategy, mass_accuracy_ppm=mass_accuracy_ppm)
                end_time = time.time()
                print(f"Strategy {strategy} completed in {end_time - start_time:.4f}s, found {len(results)} results.")
                f.write(f"Strategy: {strategy}\n")
                f.write(f"  Time: {end_time - start_time:.4f}s\n")
                f.write(f"  Results: {len(results)}\n\n")
            except Exception as e:
                print(f"ERROR in strategy {strategy}: {e}")
                f.write(f"Strategy: {strategy}\n")
                f.write(f"  ERROR: {e}\n\n")
        print("Single Mass Decomposition benchmarks completed.")
        # --- Spectrum Decomposition (Unknown Precursor) ---
        f.write("--- Spectrum Decomposition (Unknown Precursor) ---\n")
        print("--- Spectrum Decomposition (Unknown Precursor) ---")
        for strategy in ["recursive", "money_changing"]:
            print(f"Running strategy: {strategy}")
            start_time = time.time()
            try:
                results = decompose_spectrum(target_mass, fragment_masses, bounds, strategy=strategy, mass_accuracy_ppm=mass_accuracy_ppm)
                end_time = time.time()
                print(f"Strategy {strategy} completed in {end_time - start_time:.4f}s, found {len(results)} results.")
                f.write(f"Strategy: {strategy}\n")
                f.write(f"  Time: {end_time - start_time:.4f}s\n")
                f.write(f"  Precursor Candidates: {len(results)}\n\n")
            except Exception as e:
                print(f"ERROR in strategy {strategy}: {e}")
                f.write(f"Strategy: {strategy}\n")
                f.write(f"  ERROR: {e}\n\n")
        print("Spectrum Decomposition (Unknown Precursor) benchmarks completed.")
        # --- Spectrum Decomposition (Known Precursor) ---
        f.write("--- Spectrum Decomposition (Known Precursor) ---\n")
        print("--- Spectrum Decomposition (Known Precursor) ---")
        for strategy in ["recursive", "money_changing"]:
            print(f"Running strategy: {strategy}")
            start_time = time.time()
            try:
                results = decompose_spectrum_known_precursor(known_precursor, fragment_masses, strategy=strategy, mass_accuracy_ppm=mass_accuracy_ppm)
                end_time = time.time()
                print(f"Strategy {strategy} completed in {end_time - start_time:.4f}s, found {len(results)} results.")
                f.write(f"Strategy: {strategy}\n")
                f.write(f"  Time: {end_time - start_time:.4f}s\n")
                f.write(f"  Fragment Results: {[len(fr) for fr in results]}\n\n")
            except Exception as e:
                print(f"ERROR in strategy {strategy}: {e}")
                f.write(f"Strategy: {strategy}\n")
                f.write(f"  ERROR: {e}\n\n")
        print("Spectrum Decomposition (Known Precursor) benchmarks completed.")
        # --- Parallel Mass Decomposition ---
        f.write("--- Parallel Mass Decomposition ---\n")
        print("--- Parallel Mass Decomposition ---")
        target_masses = [target_mass] * num_parallel
        for strategy in ["recursive", "money_changing"]:
            # Uniform bounds
            start_time = time.time()
            results = decompose_mass_parallel(target_masses, bounds, strategy=strategy, mass_accuracy_ppm=mass_accuracy_ppm)
            end_time = time.time()
            f.write(f"Strategy: {strategy} (Uniform Bounds)\n")
            f.write(f"  Time: {end_time - start_time:.4f}s\n")
            f.write(f"time per mass: {(end_time - start_time) / num_parallel:.6f}s\n")
            f.write(f"  Results per mass: {[len(r) for r in results[:5]]}...\n\n")

            # Per-mass bounds
            bounds_per_mass = [bounds] * num_parallel
            start_time = time.time()
            results = decompose_mass_parallel(target_masses, bounds_per_mass, strategy=strategy, mass_accuracy_ppm=mass_accuracy_ppm)
            end_time = time.time()
            f.write(f"Strategy: {strategy} (Per-Mass Bounds)\n")
            f.write(f"  Time: {end_time - start_time:.4f}s\n")
            f.write(f"time per mass: {(end_time - start_time) / num_parallel:.6f}s\n")
            f.write(f"  Results per mass: {[len(r) for r in results[:5]]}...\n\n")


        # --- Parallel Spectrum Decomposition (Unknown Precursor) ---
        f.write("--- Parallel Spectrum Decomposition (Unknown Precursor) ---\n")
        spectra = [(target_mass, fragment_masses)] * num_parallel
        for strategy in ["recursive", "money_changing"]:
            # Uniform bounds
            start_time = time.time()
            results = decompose_spectrum_parallel(spectra, bounds, strategy=strategy, mass_accuracy_ppm=mass_accuracy_ppm)
            end_time = time.time()
            f.write(f"Strategy: {strategy} (Uniform Bounds)\n")
            f.write(f"  Time: {end_time - start_time:.4f}s\n")
            f.write(f"time per spectrum: {(end_time - start_time) / len(spectra):.6f}s\n")
            f.write(f"  Precursor Candidates per spectrum: {[len(r) for r in results[:5]]}...\n\n")

            # Per-spectrum bounds
            bounds_per_spectrum = [bounds] * num_parallel
            start_time = time.time()
            results = decompose_spectrum_parallel(spectra, bounds_per_spectrum, strategy=strategy, mass_accuracy_ppm=mass_accuracy_ppm)
            end_time = time.time()
            f.write(f"Strategy: {strategy} (Per-Spectrum Bounds)\n")
            f.write(f"  Time: {end_time - start_time:.4f}s\n")
            f.write(f"time per spectrum: {(end_time - start_time) / len(spectra):.6f}s\n")
            f.write(f"  Precursor Candidates per spectrum: {[len(r) for r in results[:5]]}...\n\n")

        # --- Parallel Spectrum Decomposition (Known Precursor) ---
        f.write("--- Parallel Spectrum Decomposition (Known Precursor) ---\n")
        spectra_known = [(known_precursor, fragment_masses)] * num_parallel
        for strategy in ["recursive", "money_changing"]:
            start_time = time.time()
            results = decompose_spectrum_known_precursor_parallel(spectra_known, strategy=strategy, mass_accuracy_ppm=mass_accuracy_ppm)
            end_time = time.time()
            f.write(f"Strategy: {strategy}\n")
            f.write(f"  Time: {end_time - start_time:.4f}s\n")
            f.write(f"time per spectrum: {(end_time - start_time) / len(spectra_known):.6f}s\n")
            f.write(f"  Fragment Results per spectrum: {[len(fr) for fr in results[0]]}...\n\n")
    print("Benchmark results written to benchmark_results.txt")

if __name__ == "__main__":
    print("Starting benchmark...")
    run_benchmarks()
    print("Benchmark complete. Results written to benchmark_results.txt")