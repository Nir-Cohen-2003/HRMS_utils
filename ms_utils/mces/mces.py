import numpy as np
import polars as pl
import os
from multiprocessing import cpu_count
from joblib import Parallel, delayed, parallel_backend
from typing import List, Optional, Generator, Iterable
from itertools import batched, chain
from contextlib import contextmanager
import sys
from .par import _calculate_bounds_batch, _calculate_exact_batch, _calculate_distinct_batch
import time

PROFILE = True  # Set to True to enable profiling

def calculate_mces_distances(
        smiles_list1: List[str], smiles_list2: Optional[List[str]] = None,
        n_jobs: int = -1, symmetric: bool = False, batch_size: int = 20, threshold: int = -1, solver: str = "GUROBI") -> np.typing.NDArray[np.int64]:
    """
    Efficiently computes exact MCES distances between all pairs of molecules.
    
    Parameters
    ----------
    smiles_list1 : List[str]
        List of SMILES strings for the first set of molecules
    smiles_list2 : Optional[List[str]]
        List of SMILES strings for the second set of molecules.
        If None and symmetric=True, will compare molecules within smiles_list1.
    n_jobs : int
        Number of parallel jobs to run. -1 means use all available cores.
    symmetric : bool
        If True, optimize for comparing molecules within the same list.
    batch_size : int
        Number of pairs to process in each parallel job to reduce overhead.
        
    Returns
    -------
    np.ndarray
        Matrix where element [i,j] is the exact MCES distance between molecules i and j
    """
    if PROFILE:
        total_start = time.perf_counter()
        print("\n=== Profiling calculate_mces_distances ===")
    
    if n_jobs == -1:
        n_jobs: int = cpu_count()

    # Handle symmetric case
    if symmetric:
        smiles_list2: List[str] = smiles_list1
    elif smiles_list2 is None:
        raise ValueError("smiles_list2 must be provided when symmetric=False")
    
    if PROFILE:
        setup_start = time.perf_counter()
    
    # Generate appropriate pairs
    if symmetric:
        all_pairs: List[tuple[int, int]] = [(i, j) for i in range(len(smiles_list1)) for j in range(i+1, len(smiles_list2))]
    else:
        all_pairs: List[tuple[int, int]] = [(i, j) for i in range(len(smiles_list1)) for j in range(len(smiles_list2))]
    
    # Create batches of pairs to process together
    batches = batched(all_pairs, batch_size)
    
    # Initialize distance matrix with infinity
    distance_matrix = np.full((len(smiles_list1), len(smiles_list2)), np.inf)
    if symmetric:
        # Set diagonal to 0 for symmetric case
        np.fill_diagonal(distance_matrix, 0)
    
    if PROFILE:
        setup_time = time.perf_counter() - setup_start
        print(f"Setup time: {setup_time:.3f}s")
        bounds_start = time.perf_counter()
    
    # Use persistent worker pool for all parallel operations
    with parallel_backend('loky', n_jobs=n_jobs):
        # Calculate filter2 bounds for batches of pairs
        batch_results = Parallel(batch_size="auto")(
            delayed(_calculate_bounds_batch)(smiles_list1, smiles_list2, batch) for batch in batches
        )
        
        # Flatten batch results
        bounds_results = [item for sublist in batch_results for item in sublist]
        
        if PROFILE:
            bounds_time = time.perf_counter() - bounds_start
            print(f"Bounds calculation time: {bounds_time:.3f}s")
            update_start = time.perf_counter()
        
        # Use bounds as initial estimates
        if symmetric:
            for i, j, bound in bounds_results:
                distance_matrix[i, j] = bound
                distance_matrix[j, i] = bound
        else:
            for i, j, bound in bounds_results:
                distance_matrix[i, j] = bound
        
        if PROFILE:
            update_time = time.perf_counter() - update_start
            print(f"Bounds update time: {update_time:.3f}s")
            exact_start = time.perf_counter()
        
        # Filter pairs that need exact calculation based on bounds and threshold
        if threshold > 0:
            pairs_needing_exact = [(i, j) for i, j, bound in bounds_results if bound < threshold]
            if PROFILE:
                print(f"Pairs needing exact calculation: {len(pairs_needing_exact)} out of {len(bounds_results)}")
            
            if len(pairs_needing_exact) > 0:
                # Create batches for exact computation
                exact_batches = batched(pairs_needing_exact, batch_size)
                
                # Calculate exact distances in batches
                exact_batch_results = Parallel(batch_size="auto")(
                    delayed(_calculate_exact_batch)(smiles_list1, smiles_list2, batch, threshold, solver) for batch in exact_batches
                )
            else:
                exact_batch_results = []
        else:
            # Calculate exact distances for all pairs if no threshold
            exact_batch_results = Parallel(batch_size="auto")(
                delayed(_calculate_exact_batch)(smiles_list1, smiles_list2, batch, threshold, solver) for batch in batches
            )
        
        # Flatten batch results
        exact_results = [item for sublist in exact_batch_results for item in sublist]
        
        if PROFILE:
            exact_time = time.perf_counter() - exact_start
            print(f"Exact calculation time: {exact_time:.3f}s")
            final_update_start = time.perf_counter()
        
        # Update distance matrix with exact results
        for i, j, distance in exact_results:
            distance_matrix[i, j] = distance
            if symmetric:
                distance_matrix[j, i] = distance
    
    if PROFILE:
        final_update_time = time.perf_counter() - final_update_start
        total_time = time.perf_counter() - total_start
        print(f"Final update time: {final_update_time:.3f}s")
        print(f"Total time: {total_time:.3f}s")
        print("=" * 40)
    
    return distance_matrix

def are_close_mols(smiles_list1: List[str], smiles_list2: Optional[List[str]] = None, 
                  n_jobs: int = -1, symmetric: bool = False, batch_size: int = 20, solver: str = "GUROBI") -> np.ndarray:
    """
    Efficiently computes whether each pair of molecules has an MCES distance of 1 or lower.
    
    Parameters
    ----------
    smiles_list1 : list
        List of SMILES strings for the first set of molecules
    smiles_list2 : list or None
        List of SMILES strings for the second set of molecules.
        If None and symmetric=True, will compare molecules within smiles_list1.
    n_jobs : int
        Number of parallel jobs to run. -1 means use all available cores.
    symmetric : bool
        If True, optimize for comparing molecules within the same list.
    batch_size : int
        Number of pairs to process in each parallel job to reduce overhead.
        
    Returns
    -------
    numpy.ndarray
        Boolean matrix where element [i,j] is True if molecules i and j have MCES distance ≤ 1
    """
    if PROFILE:
        total_start = time.perf_counter()
        print(f"\n=== Profiling are_close_mols ===")
    
    if n_jobs == -1:
        n_jobs = cpu_count()

    # Handle symmetric case
    if symmetric:
        smiles_list2 = smiles_list1
    elif smiles_list2 is None:
        raise ValueError("smiles_list2 must be provided when symmetric=False")
    
    if PROFILE:
        setup_start = time.perf_counter()
    
    # Generate appropriate pairs - directly using indices to avoid preloading graphs
    if symmetric:
        all_pairs = [(i, j) for i in range(len(smiles_list1)) for j in range(i+1, len(smiles_list2))]
    else:
        all_pairs = [(i, j) for i in range(len(smiles_list1)) for j in range(len(smiles_list2))]
    
    # Create batches of pairs to process together
    batches = batched(all_pairs, batch_size)
    
    if PROFILE:
        setup_time = time.perf_counter() - setup_start
        print(f"Setup time: {setup_time:.3f}s")
        bounds_start = time.perf_counter()
    
    # Use persistent worker pool for all parallel operations
    with parallel_backend('loky', n_jobs=n_jobs):
        # Calculate filter2 bounds for batches of pairs
        batch_results = Parallel(batch_size="auto")(
            delayed(_calculate_bounds_batch)(smiles_list1, smiles_list2, batch) for batch in batches
        )
        
        # Flatten batch results
        bounds_results = list(chain(*batch_results))
        
        if PROFILE:
            bounds_time = time.perf_counter() - bounds_start
            print(f"Bounds calculation time: {bounds_time:.3f}s")
            filter_start = time.perf_counter()
        
        # Initialize result matrix
        if symmetric:
            result_matrix = np.eye(len(smiles_list1), dtype=bool)
        else:
            result_matrix = np.zeros((len(smiles_list1), len(smiles_list2)), dtype=bool)
        
        # Only perform expensive ILP computation on potential matches
        pairs_needing_ilp = [(i, j) for i, j, bound in bounds_results if bound < 2]
        
        if PROFILE:
            filter_time = time.perf_counter() - filter_start
            print(f"Filtering time: {filter_time:.3f}s")
            print(f"Pairs needing ILP: {len(pairs_needing_ilp)} out of {len(bounds_results)}")
        
        if len(pairs_needing_ilp) > 0:
            if PROFILE:
                exact_start = time.perf_counter()
            
            # Create batches for ILP computation
            ilp_batches = batched(pairs_needing_ilp, batch_size)
            
            # Calculate exact results in batches - pass SMILES lists instead of precomputed graphs
            exact_batch_results = Parallel(batch_size="auto")(
                delayed(_calculate_exact_batch)(smiles_list1, smiles_list2, batch, 2, solver) for batch in ilp_batches
            )
            
            # Flatten batch results
            exact_results = list(chain(*exact_batch_results))
            
            if PROFILE:
                exact_time = time.perf_counter() - exact_start
                print(f"Exact calculation time: {exact_time:.3f}s")
                update_start = time.perf_counter()
            
            # Update result matrix
            if symmetric:
                for i, j, distance in exact_results:
                    result_matrix[i, j] = False if distance is None else distance <= 1
                    result_matrix[j, i] = result_matrix[i, j]
            else:
                for i, j, distance in exact_results:
                    result_matrix[i, j] = False if distance is None else distance <= 1
            
            if PROFILE:
                update_time = time.perf_counter() - update_start
                print(f"Matrix update time: {update_time:.3f}s")
    
    if PROFILE:
        total_time = time.perf_counter() - total_start
        print(f"Total time: {total_time:.3f}s")
        print("=" * 40)
    
    return result_matrix

def are_very_distinct(smiles_list1: List[str], smiles_list2: Optional[List[str]] = None,
                     n_jobs: int = -1, symmetric: bool = False, batch_size: int = 20, solver: str = "GUROBI", use_solver: bool = True, distinction_threshold: int = 10) -> np.ndarray:
    """
    Efficiently computes whether each pair of molecules has an MCES distance greater than 10.
    
    Parameters as in are_close_mols.
    if we set use_solver=False, we will not use the solver to compute the distances, but rather use a fast filter2 bound
     and return the results it gives, even if they are "worse", meaning 2 molecules that might be distinct (mces>10) will not be detected as such.
    """
    if PROFILE:
        total_start = time.perf_counter()
        print(f"\n=== Profiling are_very_distinct ===")
    
    if n_jobs == -1:
        n_jobs = cpu_count()
    
    # Handle symmetric case
    if symmetric:
        smiles_list2 = smiles_list1
    elif smiles_list2 is None:
        raise ValueError("smiles_list2 must be provided when symmetric=False")
    
    if PROFILE:
        setup_start = time.perf_counter()
    
    # Generate appropriate pairs - directly using indices
    if symmetric:
        all_pairs = [(i, j) for i in range(len(smiles_list1)) for j in range(i+1, len(smiles_list2))]
    else:
        all_pairs = [(i, j) for i in range(len(smiles_list1)) for j in range(len(smiles_list2))]
    
    # Create batches of pairs to process together
    batches = [all_pairs[i:i+batch_size] for i in range(0, len(all_pairs), batch_size)]
    
    if PROFILE:
        setup_time = time.perf_counter() - setup_start
        print(f"Setup time: {setup_time:.3f}s")
        bounds_start = time.perf_counter()
    
    # Use persistent worker pool for all parallel operations
    with parallel_backend('loky', n_jobs=n_jobs):
        # Calculate filter2 bounds for batches of pairs
        batch_results = Parallel(batch_size="auto")(
            delayed(_calculate_bounds_batch)(smiles_list1, smiles_list2, batch) for batch in batches
        )
        
        # Flatten batch results
        bounds_results = [item for sublist in batch_results for item in sublist]
        
        if PROFILE:
            bounds_time = time.perf_counter() - bounds_start
            print(f"Bounds calculation time: {bounds_time:.3f}s")
            filter_start = time.perf_counter()
        
        # Initialize result matrix
        result_matrix = np.zeros((len(smiles_list1), len(smiles_list2)), dtype=bool)
        
        # Use NumPy vectorization:
        indices = np.array([(i, j) for i, j, bound in bounds_results if bound > distinction_threshold])
        if indices.size > 0:
            result_matrix[indices[:, 0], indices[:, 1]] = True
            if symmetric:
                result_matrix[indices[:, 1], indices[:, 0]] = True
        if not use_solver: # then we don't do the exact mces calc
            if PROFILE:
                filter_time = time.perf_counter() - filter_start
                print(f"Filtering time: {filter_time:.3f}s")
                print(f"Pairs needing ILP: {len(bounds_results)} out of {len(all_pairs)}")
            return result_matrix
        # Only perform expensive ILP computation on potential non-distinct pairs
        bounds_array = np.array(bounds_results)
        mask = bounds_array[:, 2] <= distinction_threshold
        pairs_needing_ilp = bounds_array[mask, :2].astype(int).tolist()
        
        if PROFILE:
            filter_time = time.perf_counter() - filter_start
            print(f"Filtering time: {filter_time:.3f}s")
            print(f"Pairs needing ILP: {len(pairs_needing_ilp)} out of {len(bounds_results)}")
        
        if len(pairs_needing_ilp) > 0:
            if PROFILE:
                exact_start = time.perf_counter()
            
            # Create batches for ILP computation
            ilp_batches = [pairs_needing_ilp[i:i+batch_size] for i in range(0, len(pairs_needing_ilp), batch_size)]
            
            # Process exact calculations in batches - create a custom function for distinct calculations
            exact_batch_results = Parallel(batch_size="auto")(
                delayed(_calculate_distinct_batch)(smiles_list1, smiles_list2, batch, solver) for batch in ilp_batches
            )
            
            # Flatten batch results
            exact_results = [item for sublist in exact_batch_results for item in sublist]
            
            if PROFILE:
                exact_time = time.perf_counter() - exact_start
                print(f"Exact calculation time: {exact_time:.3f}s")
                update_start = time.perf_counter()
            
            # Update result matrix
            for i, j, is_distinct in exact_results:
                result_matrix[i, j] = is_distinct
                if symmetric:
                    result_matrix[j, i] = is_distinct
            
            if PROFILE:
                update_time = time.perf_counter() - update_start
                print(f"Matrix update time: {update_time:.3f}s")
    
    if PROFILE:
        total_time = time.perf_counter() - total_start
        print(f"Total time: {total_time:.3f}s")
        print("=" * 40)
    
    return result_matrix


@contextmanager
def suppress_output() -> Generator[None, None, None]:
    """Suppress stdout and stderr output for both terminal and notebook environments"""
    # Save the original stdout/stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    # Create dummy streams to redirect output
    # Using io.StringIO captures output in memory, os.devnull discards it.
    # os.devnull is generally better for pure suppression.
    devnull_w = open(os.devnull, 'w')
    # For notebooks, redirecting sys streams is usually sufficient.
    # File descriptor redirection can be problematic in notebooks.
    sys.stdout = devnull_w
    sys.stderr = devnull_w

    try:
        yield
    finally:
        # Restore original streams
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        # Close the dummy stream
        devnull_w.close()


if __name__ == "__main__":
    from time import perf_counter
    nist_smiles: List[str] = pl.scan_parquet('/home/analytit_admin/dev/MS_encoder/data/NIST_prepared_labeled.parquet').sort('ExactMass').select('CanonicalSMILES').unique(maintain_order=True).collect().slice(offset=10000, length=2048).to_series().to_list()
    smiles1 : List[str] = nist_smiles[:1023]
    smiles2 : List[str] = nist_smiles[1024:]
    # smiles1 = nist_smiles

    # Example usage
    start = perf_counter()
    # with suppress_output():
    result_matrix= are_very_distinct(smiles1, symmetric=True,batch_size=200,solver="GUROBI")
    print(f"Time taken: {perf_counter() - start:.2f} seconds")
    # start = time()
    # with suppress_output():
    #     result_matrix = are_very_distinct(smiles1,symmetric=False)
    # print(f"Time taken: {time() - start:.2f} seconds")
    # start = time()
    # with suppress_output():
    #     result_matrix = calculate_mces_distances(smiles1, smiles2)
    # print(f"Time taken: {time() - start:.2f} seconds")

    # print(result_matrix)