import time
import numpy as np
import mass_decomposer_cpp

def setup_bounds():
    """
    Sets up the min and max bounds arrays based on the element order
    from the Cython module.
    """
    bounds_dict = {
        'C': (0, 50), 'H': (0, 100), 'N': (0, 20), 'O': (0, 40),
        'S': (0, 6), 'P': (0, 5), 'Na': (0, 0), 'F': (0, 40),
        'Cl': (0, 0), 'Br': (0, 0), 'I': (0, 4),
    }

    element_info = mass_decomposer_cpp.get_element_info()
    element_order = element_info['order']
    num_elements = element_info['count']

    min_bounds = np.zeros(num_elements, dtype=np.int32)
    max_bounds = np.zeros(num_elements, dtype=np.int32)

    for i, element in enumerate(element_order):
        if element in bounds_dict:
            min_val, max_val = bounds_dict[element]
            min_bounds[i] = min_val
            max_bounds[i] = max_val
        # Other elements will remain at their default (0, 0) bounds

    return min_bounds, max_bounds, element_order

def formula_dict_to_array(formula_dict, element_order):
    """Converts a formula dictionary to a numpy array."""
    num_elements = len(element_order)
    formula_array = np.zeros(num_elements, dtype=np.int32)
    for i, element in enumerate(element_order):
        if element in formula_dict:
            formula_array[i] = formula_dict[element]
    return formula_array

def run_mass_decomposition_tests(mass, min_bounds, max_bounds, num_parallel=1000):
    """Tests for single, parallel (uniform), and parallel (per-bounds) mass decomposition."""
    print("\n--- Testing Mass Decomposition ---")
    
    # --- Single Mass Decomposition ---
    print("\n[1] Testing `decompose_mass` (single-threaded)...")
    start_time = time.perf_counter()
    results_single = mass_decomposer_cpp.decompose_mass(
        target_mass=mass,
        min_bounds=min_bounds,
        max_bounds=max_bounds,
        tolerance_ppm=5.0
    )
    end_time = time.perf_counter()
    print(f"Found {len(results_single)} formulas in {end_time - start_time:.4f} seconds.")
    if results_single:
        print(f"First result: {results_single[0]}")

    # --- Parallel Mass Decomposition (same bounds) ---
    print("\n[2] Testing `decompose_mass_parallel` (multi-threaded, uniform bounds)...")
    masses_list = [mass] * num_parallel 
    start_time = time.perf_counter()
    results_parallel = mass_decomposer_cpp.decompose_mass_parallel(
        target_masses=masses_list,
        min_bounds=min_bounds,
        max_bounds=max_bounds,
        tolerance_ppm=5.0
    )
    end_time = time.perf_counter()
    print(f"Decomposed {len(masses_list)} masses in {end_time - start_time:.4f} seconds.")
    if results_parallel:
        print(f"Found {len(results_parallel[0])} formulas for the first mass.")

    # --- Parallel Mass Decomposition (per-bounds) ---
    print("\n[2b] Testing `decompose_mass_parallel_per_bounds` (multi-threaded, per-bounds)...")
    per_mass_bounds = [(min_bounds.copy(), max_bounds.copy()) for _ in range(num_parallel)]
    start_time = time.perf_counter()
    results_per_bounds = mass_decomposer_cpp.decompose_mass_parallel_per_bounds(
        target_masses=masses_list,
        per_mass_bounds=per_mass_bounds,
        tolerance_ppm=5.0
    )
    end_time = time.perf_counter()
    print(f"Decomposed {len(masses_list)} masses (per-bounds) in {end_time - start_time:.4f} seconds.")
    if results_per_bounds:
        print(f"Found {len(results_per_bounds[0])} formulas for the first mass (per-bounds).")
    # Optional: check correctness
    if results_parallel and results_per_bounds:
        if np.all([np.array_equal(results_parallel[i][0], results_per_bounds[i][0]) for i in range(len(results_parallel)) if results_parallel[i] and results_per_bounds[i]]):
            print("Uniform and per-bounds results match for first formula of each mass.")
        else:
            print("Warning: Uniform and per-bounds results differ.")

    return results_single

def run_spectrum_decomposition_tests(precursor_mass, fragments, min_bounds, max_bounds, num_parallel=1000):
    """Tests for single, parallel (uniform), and parallel (per-bounds) spectrum decomposition."""
    print("\n--- Testing Spectrum Decomposition ---")

    # --- Single Spectrum Decomposition ---
    print("\n[3] Testing `decompose_spectrum` (single-threaded)...")
    start_time = time.perf_counter()
    results_single = mass_decomposer_cpp.decompose_spectrum(
        precursor_mass=precursor_mass,
        fragment_masses=fragments,
        min_bounds=min_bounds,
        max_bounds=max_bounds,
        tolerance_ppm=5.0
    )
    end_time = time.perf_counter()
    print(f"Found {len(results_single)} precursor decompositions in {end_time - start_time:.4f} seconds.")
    if results_single:
        print(f"First result precursor: {results_single[0]['precursor']}")

    # --- Parallel Spectrum Decomposition (uniform bounds) ---
    print("\n[4] Testing `decompose_spectra_parallel` (multi-threaded, uniform bounds)...")
    spectra_list = [{'precursor_mass': precursor_mass, 'fragment_masses': fragments}] * num_parallel
    start_time = time.perf_counter()
    results_parallel = mass_decomposer_cpp.decompose_spectra_parallel(
        spectra_data=spectra_list,
        min_bounds=min_bounds,
        max_bounds=max_bounds,
        tolerance_ppm=5.0
    )
    end_time = time.perf_counter()
    print(f"Decomposed {len(spectra_list)} spectra in {end_time - start_time:.4f} seconds.")
    if results_parallel:
        print(f"Found {len(results_parallel[0])} decompositions for the first spectrum.")

    # --- Parallel Spectrum Decomposition (per-bounds) ---
    print("\n[4b] Testing `decompose_spectra_parallel_per_bounds` (multi-threaded, per-bounds)...")
    spectra_per_bounds = [
        {
            'precursor_mass': precursor_mass,
            'fragment_masses': fragments,
            'min_bounds': min_bounds.copy(),
            'max_bounds': max_bounds.copy()
        }
        for _ in range(num_parallel)
    ]
    start_time = time.perf_counter()
    results_per_bounds = mass_decomposer_cpp.decompose_spectra_parallel_per_bounds(
        spectra_data=spectra_per_bounds,
        tolerance_ppm=5.0
    )
    end_time = time.perf_counter()
    print(f"Decomposed {len(spectra_per_bounds)} spectra (per-bounds) in {end_time - start_time:.4f} seconds.")
    if results_per_bounds:
        print(f"Found {len(results_per_bounds[0])} decompositions for the first spectrum (per-bounds).")
    # Optional: check correctness
    if results_parallel and results_per_bounds:
        if results_parallel[0] and results_per_bounds[0]:
            if np.array_equal(results_parallel[0][0]['precursor'], results_per_bounds[0][0]['precursor']):
                print("Uniform and per-bounds results match for first precursor formula.")
            else:
                print("Warning: Uniform and per-bounds results differ.")

def run_known_precursor_tests(precursor_formula_arr, fragments, min_bounds, max_bounds, num_parallel=1000):
    """Tests for known precursor decomposition."""
    if precursor_formula_arr is None:
        print("\n--- Skipping Known Precursor Tests (no valid precursor found) ---")
        return
        
    print("\n--- Testing Known Precursor Spectrum Decomposition ---")

    # --- Single Known Precursor Decomposition ---
    print("\n[5] Testing `decompose_spectrum_known_precursor` (single-threaded)...")
    start_time = time.perf_counter()
    results_single = mass_decomposer_cpp.decompose_spectrum_known_precursor(
        precursor_formula=precursor_formula_arr,
        fragment_masses=fragments,
        min_bounds=min_bounds,
        max_bounds=max_bounds, # max_bounds is used for fragment bounds
        tolerance_ppm=5.0
    )
    end_time = time.perf_counter()
    print(f"Found explanations for {len(results_single)} fragments in {end_time - start_time:.4f} seconds.")
    if results_single:
        print(f"First fragment explained by {len(results_single[0])} formulas.")

    # --- Parallel Known Precursor Decomposition ---
    print("\n[6] Testing `decompose_spectra_known_precursor_parallel` (multi-threaded)...")
    spectra_list = [{'precursor_formula': precursor_formula_arr, 'fragment_masses': fragments}] * num_parallel
    start_time = time.perf_counter()
    results_parallel = mass_decomposer_cpp.decompose_spectra_known_precursor_parallel(
        spectra_data=spectra_list,
        min_bounds=min_bounds,
        max_bounds=max_bounds,
        tolerance_ppm=5.0
    )
    end_time = time.perf_counter()
    print(f"Decomposed {len(spectra_list)} spectra in {end_time - start_time:.4f} seconds.")
    if results_parallel:
        print(f"Found explanations for {len(results_parallel[0])} fragments in the first spectrum.")

if __name__ == "__main__":
    # --- Configuration ---
    TARGET_MASS = 281.152812
    FRAGMENT_MASSES = [281.152812, 241.121512, 237.090212, 221.095297, 93.070425]
    num_parallel = 10000  # Number of parallel tests to run
    print("Setting up element bounds...")
    min_b, max_b, element_order = setup_bounds()
    print(f"Element order: {element_order}")

    # --- Run Tests ---
    mass_results = run_mass_decomposition_tests(TARGET_MASS, min_b, max_b, num_parallel=num_parallel)
    run_spectrum_decomposition_tests(TARGET_MASS, FRAGMENT_MASSES, min_b, max_b, num_parallel=num_parallel)
    # Use the first valid formula from mass decomposition for the known precursor test
    first_formula = mass_results[0] if mass_results else None
    run_known_precursor_tests(first_formula, FRAGMENT_MASSES, min_b, max_b, num_parallel=num_parallel)