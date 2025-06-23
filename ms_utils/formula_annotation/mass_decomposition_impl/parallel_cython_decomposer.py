"""
Parallel Cython Mass Decomposer

This module provides parallel execution of the Cython SIRIUS mass decomposition algorithm
across all available CPU cores. Each decomposition task is small, so we parallelize
at the task level rather than within individual decompositions.
"""

import multiprocessing as mp
from typing import List, Dict, Tuple, Optional, Union, Callable
import os

try:
    from sirius_decomposer import cython_decompose_mass
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False
    
    def cython_decompose_mass(*args, **kwargs):
        raise NotImplementedError("Cython version not available. Run 'python setup.py build_ext --inplace'")


class ParallelCythonDecomposer:
    """
    Parallel wrapper for the Cython SIRIUS mass decomposer.
    
    This class manages parallel execution of mass decompositions across all available CPU cores.
    Each decomposition is handled as a separate task, making it ideal for batch processing
    of multiple masses.
    """
    
    def __init__(self, n_processes: Optional[int] = None, chunksize: Optional[int] = None):
        """
        Initialize the parallel decomposer.
        
        Args:
            n_processes: Number of processes to use. If None, uses all available CPU cores.
            chunksize: Number of tasks to send to each worker at once. If None, auto-calculated.
        """
        self.n_processes = n_processes or os.cpu_count()
        self.chunksize = chunksize
        
        if not CYTHON_AVAILABLE:
            raise ImportError("Cython decomposer not available. Run 'python setup.py build_ext --inplace'")
    
    def decompose_single(self, 
                        target_mass: float,
                        element_bounds: Dict[str, Tuple[int, int]],
                        tolerance_ppm: float = 5.0,
                        max_results: int = 1000000,
                        min_dbe: Optional[float] = None,
                        max_dbe: Optional[float] = None,
                        max_hetero_ratio: Optional[float] = None) -> List[Dict[str, int]]:
        """
        Decompose a single mass using the Cython implementation.
        
        This is just a wrapper around the Cython function for consistency.
        """
        return cython_decompose_mass(
            target_mass=target_mass,
            element_bounds=element_bounds,
            tolerance_ppm=tolerance_ppm,
            max_results=max_results,
            min_dbe=min_dbe,
            max_dbe=max_dbe,
            max_hetero_ratio=max_hetero_ratio
        )
    
    def decompose_batch(self,
                       target_masses: List[float],
                       element_bounds: Union[Dict[str, Tuple[int, int]], List[Dict[str, Tuple[int, int]]]],
                       tolerance_ppm: Union[float, List[float]] = 5.0,
                       max_results: Union[int, List[int]] = 1000000,
                       min_dbe: Union[Optional[float], List[Optional[float]]] = None,
                       max_dbe: Union[Optional[float], List[Optional[float]]] = None,
                       max_hetero_ratio: Union[Optional[float], List[Optional[float]]] = None,
                       progress_callback: Optional[Callable[[int, int], None]] = None) -> List[List[Dict[str, int]]]:
        """
        Decompose multiple masses in parallel across all CPU cores.
        
        Args:
            target_masses: List of target masses to decompose
            element_bounds: Element bounds for each mass (single dict for all, or list of dicts)
            tolerance_ppm: Tolerance for each mass (single value for all, or list of values)
            max_results: Max results for each mass (single value for all, or list of values)
            min_dbe: Min DBE constraint for each mass (single value for all, or list of values)
            max_dbe: Max DBE constraint for each mass (single value for all, or list of values)
            max_hetero_ratio: Max heteroatom ratio for each mass (single value for all, or list of values)
            progress_callback: Optional callback function(completed, total) for progress tracking
            
        Returns:
            List of results, where each element is a list of formulas for the corresponding mass
        """
        n_masses = len(target_masses)
        
        # Normalize parameters to lists
        element_bounds_list = self._normalize_parameter(element_bounds, n_masses)
        tolerance_ppm_list = self._normalize_parameter(tolerance_ppm, n_masses)
        max_results_list = self._normalize_parameter(max_results, n_masses)
        min_dbe_list = self._normalize_parameter(min_dbe, n_masses)
        max_dbe_list = self._normalize_parameter(max_dbe, n_masses)
        max_hetero_ratio_list = self._normalize_parameter(max_hetero_ratio, n_masses)
        
        # Create task parameters
        tasks = []
        for i in range(n_masses):
            task = {
                'target_mass': target_masses[i],
                'element_bounds': element_bounds_list[i],
                'tolerance_ppm': tolerance_ppm_list[i],
                'max_results': max_results_list[i],
                'min_dbe': min_dbe_list[i],
                'max_dbe': max_dbe_list[i],
                'max_hetero_ratio': max_hetero_ratio_list[i]
            }
            tasks.append(task)
        
        # Calculate optimal chunksize if not provided
        chunksize = self.chunksize
        if chunksize is None:
            chunksize = max(1, n_masses // (self.n_processes * 4))
        
        # Execute in parallel
        with mp.Pool(processes=self.n_processes) as pool:
            if progress_callback:
                # Use imap for progress tracking
                results = []
                for i, result in enumerate(pool.imap(_decompose_task_wrapper, tasks, chunksize=chunksize)):
                    results.append(result)
                    progress_callback(i + 1, n_masses)
                return results
            else:
                # Use map for maximum performance
                return pool.map(_decompose_task_wrapper, tasks, chunksize=chunksize)
    
    def decompose_masses_with_common_bounds(self,
                                          target_masses: List[float],
                                          element_bounds: Dict[str, Tuple[int, int]],
                                          tolerance_ppm: float = 5.0,
                                          max_results: int = 1000000,
                                          min_dbe: Optional[float] = None,
                                          max_dbe: Optional[float] = None,
                                          max_hetero_ratio: Optional[float] = None,
                                          progress_callback: Optional[Callable[[int, int], None]] = None) -> List[List[Dict[str, int]]]:
        """
        Convenience method for decomposing multiple masses with the same parameters.
        
        This is more efficient than decompose_batch when all masses use the same constraints.
        """
        return self.decompose_batch(
            target_masses=target_masses,
            element_bounds=element_bounds,
            tolerance_ppm=tolerance_ppm,
            max_results=max_results,
            min_dbe=min_dbe,
            max_dbe=max_dbe,
            max_hetero_ratio=max_hetero_ratio,
            progress_callback=progress_callback
        )
    
    @staticmethod
    def _normalize_parameter(param, n_masses):
        """Normalize a parameter to a list of the correct length."""
        if isinstance(param, list):
            if len(param) != n_masses:
                raise ValueError(f"Parameter list length ({len(param)}) doesn't match number of masses ({n_masses})")
            return param
        else:
            return [param] * n_masses


def _decompose_task_wrapper(task_params: Dict) -> List[Dict[str, int]]:
    """
    Wrapper function for parallel execution of decomposition tasks.
    
    This function is defined at module level to ensure it can be pickled
    for multiprocessing.
    """
    return cython_decompose_mass(**task_params)


# Convenience functions for common use cases

def parallel_decompose_masses(target_masses: List[float],
                            element_bounds: Dict[str, Tuple[int, int]],
                            tolerance_ppm: float = 5.0,
                            max_results: int = 1000000,
                            min_dbe: Optional[float] = None,
                            max_dbe: Optional[float] = None,
                            max_hetero_ratio: Optional[float] = None,
                            n_processes: Optional[int] = None,
                            progress_callback: Optional[Callable[[int, int], None]] = None) -> List[List[Dict[str, int]]]:
    """
    Convenience function to decompose multiple masses in parallel.
    
    Args:
        target_masses: List of masses to decompose
        element_bounds: Element bounds to use for all masses
        tolerance_ppm: Mass tolerance in ppm
        max_results: Maximum results per mass
        min_dbe: Minimum double bond equivalents (None = no constraint)
        max_dbe: Maximum double bond equivalents (None = no constraint)
        max_hetero_ratio: Maximum heteroatom to carbon ratio (None = no constraint)
        n_processes: Number of processes (None = all CPU cores)
        progress_callback: Optional callback function(completed, total)
        
    Returns:
        List of results for each mass
    """
    decomposer = ParallelCythonDecomposer(n_processes=n_processes)
    return decomposer.decompose_masses_with_common_bounds(
        target_masses=target_masses,
        element_bounds=element_bounds,
        tolerance_ppm=tolerance_ppm,
        max_results=max_results,
        min_dbe=min_dbe,
        max_dbe=max_dbe,
        max_hetero_ratio=max_hetero_ratio,
        progress_callback=progress_callback
    )


def parallel_decompose_single_mass_multiple_times(target_mass: float,
                                                element_bounds: Dict[str, Tuple[int, int]],
                                                n_runs: int,
                                                tolerance_ppm: float = 5.0,
                                                max_results: int = 1000000,
                                                min_dbe: Optional[float] = None,
                                                max_dbe: Optional[float] = None,
                                                max_hetero_ratio: Optional[float] = None,
                                                n_processes: Optional[int] = None) -> List[List[Dict[str, int]]]:
    """
    Run the same decomposition multiple times in parallel (useful for benchmarking).
    
    Args:
        target_mass: Mass to decompose
        element_bounds: Element bounds
        n_runs: Number of times to run the decomposition
        tolerance_ppm: Mass tolerance in ppm
        max_results: Maximum results
        min_dbe: Minimum double bond equivalents (None = no constraint)
        max_dbe: Maximum double bond equivalents (None = no constraint)
        max_hetero_ratio: Maximum heteroatom to carbon ratio (None = no constraint)
        n_processes: Number of processes (None = all CPU cores)
        
    Returns:
        List of results for each run (should all be identical)
    """
    target_masses = [target_mass] * n_runs
    return parallel_decompose_masses(
        target_masses=target_masses,
        element_bounds=element_bounds,
        tolerance_ppm=tolerance_ppm,
        max_results=max_results,
        min_dbe=min_dbe,
        max_dbe=max_dbe,
        max_hetero_ratio=max_hetero_ratio,
        n_processes=n_processes
    )


# Example usage and demonstration
if __name__ == "__main__":
    import time
    from base_data import ATOMIC_MASSES
    
    print("Parallel Cython Decomposer Demo")
    print("=" * 40)
    
    # Test data
    element_bounds = {
        'C': (0, 50),
        'H': (0, 100), 
        'N': (0, 20),
        'O': (0, 40),
        'S': (0, 6),
        'P': (0, 5),
    }
    
    # Test with different masses
    test_masses = [
        285.136493,  # Example mass 1
        342.198745,  # Example mass 2  
        198.087623,  # Example mass 3
        456.234567,  # Example mass 4
        123.045678,  # Example mass 5
    ]
    
    print(f"Testing with {len(test_masses)} different masses")
    print(f"Using {os.cpu_count()} CPU cores")
    print(f"Element bounds: {element_bounds}")
    print("-" * 40)
    
    # Progress callback
    def progress(completed, total):
        percent = (completed / total) * 100
        print(f"Progress: {completed}/{total} ({percent:.1f}%)")
    
    # Run parallel decomposition
    start_time = time.time()
    results = parallel_decompose_masses(
        target_masses=test_masses,
        element_bounds=element_bounds,
        tolerance_ppm=5.0,
        min_dbe=0.0,
        max_dbe=40.0,
        progress_callback=progress
    )
    parallel_time = time.time() - start_time
    
    print(f"\nParallel decomposition completed in {parallel_time:.3f} seconds")
    print(f"Average time per mass: {parallel_time/len(test_masses):.3f} seconds")
    
    # Show results summary
    print("\nResults summary:")
    for i, (mass, result) in enumerate(zip(test_masses, results)):
        print(f"  Mass {mass:.6f}: {len(result)} formulas found")
        if result:
            # Show first formula as example
            first_formula = result[0]
            calc_mass = sum(ATOMIC_MASSES[elem] * count for elem, count in first_formula.items())
            error_ppm = abs(calc_mass - mass) / mass * 1e6
            print(f"    Example: {first_formula} (error: {error_ppm:.1f} ppm)")
    
    # Compare with sequential execution
    print("\nComparing with sequential execution...")
    start_time = time.time()
    sequential_results = []
    for mass in test_masses:
        result = cython_decompose_mass(
            target_mass=mass,
            element_bounds=element_bounds,
            tolerance_ppm=5.0,
            min_dbe=0.0,
            max_dbe=40.0
        )
        sequential_results.append(result)
    sequential_time = time.time() - start_time
    
    print(f"Sequential execution time: {sequential_time:.3f} seconds")
    speedup = sequential_time / parallel_time
    print(f"Parallel speedup: {speedup:.1f}x")
    
    # Verify results are identical
    results_match = all(
        {frozenset(formula.items()) for formula in par_result} == 
        {frozenset(formula.items()) for formula in seq_result}
        for par_result, seq_result in zip(results, sequential_results)
    )
    print(f"Results identical: {'✓' if results_match else '✗'}")
