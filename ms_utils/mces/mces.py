import numpy as np
import polars as pl
import os
from multiprocessing import cpu_count
from joblib import Parallel, delayed, parallel_backend
from typing import List, Optional, Generator
from itertools import batched, chain
from contextlib import contextmanager
from time import time
import sys
from ms_utils.mces.par import _calculate_bounds_batch, _calculate_exact_batch, _calculate_distinct_batch

def calculate_mces_distances(smiles_list1: List[str], smiles_list2: Optional[List[str]] = None,
                           n_jobs: int = -1, symmetric: bool = False, batch_size: int = 20, threshold:int=-1) -> np.ndarray:
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
    if n_jobs == -1:
        n_jobs = cpu_count()

    # Handle symmetric case
    if symmetric:
        smiles_list2 = smiles_list1
    elif smiles_list2 is None:
        raise ValueError("smiles_list2 must be provided when symmetric=False")
    
    # Generate appropriate pairs
    if symmetric:
        all_pairs = [(i, j) for i in range(len(smiles_list1)) for j in range(i+1, len(smiles_list2))]
    else:
        all_pairs = [(i, j) for i in range(len(smiles_list1)) for j in range(len(smiles_list2))]
    
    # Create batches of pairs to process together
    batches = batched(all_pairs, batch_size)
    
    # Initialize distance matrix with infinity
    distance_matrix = np.full((len(smiles_list1), len(smiles_list2)), np.inf)
    if symmetric:
        # Set diagonal to 0 for symmetric case
        np.fill_diagonal(distance_matrix, 0)
    
    # Use persistent worker pool for all parallel operations
    with parallel_backend('loky', n_jobs=n_jobs):
        # Calculate filter2 bounds for batches of pairs
        batch_results = Parallel(batch_size="auto")(
            delayed(_calculate_bounds_batch)(smiles_list1, smiles_list2, batch) for batch in batches
        )
        
        # Flatten batch results
        bounds_results = [item for sublist in batch_results for item in sublist]
        
        # Use bounds as initial estimates
        for i, j, bound in bounds_results:
            distance_matrix[i, j] = bound
            if symmetric:
                distance_matrix[j, i] = bound
        
        # Calculate exact distances in batches
        exact_batch_results = Parallel(batch_size="auto")(
            delayed(_calculate_exact_batch)(smiles_list1, smiles_list2, batch,threshold) for batch in batches
        )
        
        # Flatten batch results
        exact_results = [item for sublist in exact_batch_results for item in sublist]
        
        # Update distance matrix with exact results
        for i, j, distance in exact_results:
            distance_matrix[i, j] = distance
            if symmetric:
                distance_matrix[j, i] = distance
    
    return distance_matrix

def are_close_mols(smiles_list1: List[str], smiles_list2: Optional[List[str]] = None, 
                  n_jobs: int = -1, symmetric: bool = False, batch_size: int = 20) -> np.ndarray:
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
    if n_jobs == -1:
        n_jobs = cpu_count()

    # Handle symmetric case
    if symmetric:
        smiles_list2 = smiles_list1
    elif smiles_list2 is None:
        raise ValueError("smiles_list2 must be provided when symmetric=False")
    
    # Generate appropriate pairs - directly using indices to avoid preloading graphs
    if symmetric:
        all_pairs = [(i, j) for i in range(len(smiles_list1)) for j in range(i+1, len(smiles_list2))]
    else:
        all_pairs = [(i, j) for i in range(len(smiles_list1)) for j in range(len(smiles_list2))]
    
    # Create batches of pairs to process together
    batches = batched(all_pairs, batch_size)
    
    # Use persistent worker pool for all parallel operations
    with parallel_backend('loky', n_jobs=n_jobs):
        # Calculate filter2 bounds for batches of pairs
        batch_results = Parallel(batch_size="auto")(
            delayed(_calculate_bounds_batch)(smiles_list1, smiles_list2, batch) for batch in batches
        )
        
        # Flatten batch results
        bounds_results = list(chain(*batch_results))
        
        # Initialize result matrix
        if symmetric:
            result_matrix = np.eye(len(smiles_list1), dtype=bool)
        else:
            result_matrix = np.zeros((len(smiles_list1), len(smiles_list2)), dtype=bool)
        
        # Only perform expensive ILP computation on potential matches
        pairs_needing_ilp = [(i, j) for i, j, bound in bounds_results if bound < 2]
        
        if len(pairs_needing_ilp) > 0:
            # Create batches for ILP computation
            ilp_batches = batched(pairs_needing_ilp, batch_size)
            
            # Calculate exact results in batches - pass SMILES lists instead of precomputed graphs
            exact_batch_results = Parallel(batch_size="auto")(
                delayed(_calculate_exact_batch)(smiles_list1, smiles_list2, batch, 2) for batch in ilp_batches
            )
            
            # Flatten batch results
            exact_results = list(chain(*exact_batch_results))
            
            # Update result matrix
            if symmetric:
                for i, j, distance in exact_results:
                    result_matrix[i, j] = False if distance is None else distance <= 1
                    result_matrix[j, i] = result_matrix[i, j]
            else:
                for i, j, distance in exact_results:
                    result_matrix[i, j] = False if distance is None else distance <= 1
    
    return result_matrix

def are_very_distinct(smiles_list1: List[str], smiles_list2: Optional[List[str]] = None,
                     n_jobs: int = -1, symmetric: bool = False, batch_size: int = 20) -> np.ndarray:
    """
    Efficiently computes whether each pair of molecules has an MCES distance greater than 10.
    
    Parameters as in are_close_mols.
    """
    if n_jobs == -1:
        n_jobs = cpu_count()
    
    # Handle symmetric case
    if symmetric:
        smiles_list2 = smiles_list1
    elif smiles_list2 is None:
        raise ValueError("smiles_list2 must be provided when symmetric=False")
    
    # Generate appropriate pairs - directly using indices
    if symmetric:
        all_pairs = [(i, j) for i in range(len(smiles_list1)) for j in range(i+1, len(smiles_list2))]
    else:
        all_pairs = [(i, j) for i in range(len(smiles_list1)) for j in range(len(smiles_list2))]
    
    # Create batches of pairs to process together
    batches = [all_pairs[i:i+batch_size] for i in range(0, len(all_pairs), batch_size)]
    
    # Use persistent worker pool for all parallel operations
    with parallel_backend('loky', n_jobs=n_jobs):
        # Calculate filter2 bounds for batches of pairs
        batch_results = Parallel(batch_size="auto")(
            delayed(_calculate_bounds_batch)(smiles_list1, smiles_list2, batch) for batch in batches
        )
        
        # Flatten batch results
        bounds_results = [item for sublist in batch_results for item in sublist]
        
        # Initialize result matrix
        result_matrix = np.zeros((len(smiles_list1), len(smiles_list2)), dtype=bool)
        
        # Use NumPy vectorization:
        indices = np.array([(i, j) for i, j, bound in bounds_results if bound > 10])
        if indices.size > 0:
            result_matrix[indices[:, 0], indices[:, 1]] = True
            if symmetric:
                result_matrix[indices[:, 1], indices[:, 0]] = True
        
        # Only perform expensive ILP computation on potential non-distinct pairs
        bounds_array = np.array(bounds_results)
        mask = bounds_array[:, 2] <= 10
        pairs_needing_ilp = bounds_array[mask, :2].astype(int).tolist()
        
        if len(pairs_needing_ilp) > 0:
            # Create batches for ILP computation
            ilp_batches = [pairs_needing_ilp[i:i+batch_size] for i in range(0, len(pairs_needing_ilp), batch_size)]
            
            # Process exact calculations in batches - create a custom function for distinct calculations

            
            exact_batch_results = Parallel(batch_size="auto")(
                delayed(_calculate_distinct_batch)(smiles_list1, smiles_list2, batch) for batch in ilp_batches
            )
            
            # Flatten batch results
            exact_results = [item for sublist in exact_batch_results for item in sublist]
            
            # Update result matrix
            for i, j, is_distinct in exact_results:
                result_matrix[i, j] = is_distinct
                if symmetric:
                    result_matrix[j, i] = is_distinct
    
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
    nist_smiles = pl.scan_parquet('/home/analytit_admin/dev/MS_encoder/data/NIST_prepared_labeled.parquet').sort('ExactMass').select('CanonicalSMILES').unique(maintain_order=True).collect().slice(offset=10000,length=2048).to_series().to_list()
    smiles1 = nist_smiles[:1023]
    smiles2 = nist_smiles[1024:]
    # smiles1 = nist_smiles

    # Example usage
    start = time()
    with suppress_output():
        result_matrix = are_close_mols(smiles1, symmetric=True,batch_size=20)
    print(f"Time taken: {time() - start:.2f} seconds")
    # start = time()
    # with suppress_output():
    #     result_matrix = are_very_distinct(smiles1,symmetric=False)
    # print(f"Time taken: {time() - start:.2f} seconds")
    # start = time()
    # with suppress_output():
    #     result_matrix = calculate_mces_distances(smiles1, smiles2)
    # print(f"Time taken: {time() - start:.2f} seconds")

    # print(result_matrix)