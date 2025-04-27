import os
import numpy as np
import polars as pl
from joblib import Memory
from ms_utils.mces.lib import construct_graph, filter2, MCES_ILP
from typing import List, Tuple, Any, Generator
from contextlib import contextmanager
import sys
# Keep the memory cache configuration
memory = Memory("./cachedir", verbose=0)

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

# This function now provides the only caching layer
@memory.cache
def _cached_construct_graph(smiles: str) -> Any:
    return construct_graph(smiles)

# Process bounds in batches to reduce overhead
def _calculate_bounds_batch(smiles_list1: List[str], smiles_list2: List[str], batch_pairs: List[Tuple[int, int]]) -> List[Tuple[int, int, float]]:
    with suppress_output():
        results = []
        for i, j in batch_pairs:
            # Load graphs on-demand
            g1 = _cached_construct_graph(smiles_list1[i])
            g2 = _cached_construct_graph(smiles_list2[j])
            bound = filter2(g1, g2)
            results.append((i, j, bound))
    return results

# Process exact calculations in batches
def _calculate_exact_batch(smiles_list1: List[str], smiles_list2: List[str], batch_pairs: List[Tuple[int, int]], threshold: int) -> List[Tuple[int, int, int]]:
    '''this function is used to calculate the exact MCES distance in batches.
    if the distance is greater than the threshold, it will return the threshold value.
    if the distance is less than the threshold, it will return the distance value.
    if the computation fails, it will return None.
    The function will return a list of tuples, where each tuple contains the indices of the two molecules and the distance value.
    '''
    with suppress_output():
        results = []
        for i, j in batch_pairs:
            # Load graphs on-demand
            g1 = _cached_construct_graph(smiles_list1[i])
            g2 = _cached_construct_graph(smiles_list2[j])
            try:
                distance, _ = MCES_ILP(g1, g2, threshold=threshold)
            except Exception("GurobiError") as e:
                # Handle GurobiError gracefully
                print(f"GurobiError for pair ({i}, {j}): {e}")
                distance = None
            results.append((i, j, distance))
    return results

def _calculate_distinct_batch(smiles_list1, smiles_list2, batch_pairs):
    with suppress_output():
        results = []
        for i, j in batch_pairs:
            # Load graphs on-demand
            g1 = _cached_construct_graph(smiles_list1[i])
            g2 = _cached_construct_graph(smiles_list2[j])
            distance, _ = MCES_ILP(g1, g2, threshold=10)
            is_distinct = distance > 10
            results.append((i, j, is_distinct))
    return results


def calculate_mces_distance_pair(smiles1: str, smiles2: str, threshold:int=10) -> float:
    """
    Calculate the exact MCES distance between two molecules.
    
    Parameters
    ----------
    smiles1 : str
        SMILES string of first molecule
    smiles2 : str
        SMILES string of second molecule
        
    Returns
    -------
    float
        The MCES distance between the two molecules
    """
    g1 = _cached_construct_graph(smiles1)
    g2 = _cached_construct_graph(smiles2)
    
    # Calculate the lower bound first
    bound = filter2(g1, g2)
    if bound > threshold:
        return bound
    
    # If we need exact distance, run the ILP
    distance, _ = MCES_ILP(g1, g2,threshold=threshold)
    
    return distance


def are_close_mol_pair(smiles1: str, smiles2: str) -> bool:
    """
    Check if two molecules have an MCES distance of 1 or lower.
    
    Parameters
    ----------
    smiles1 : str
        SMILES string of first molecule
    smiles2 : str
        SMILES string of second molecule
        
    Returns
    -------
    bool
        True if the molecules have an MCES distance ≤ 1, False otherwise
    """
    g1 = _cached_construct_graph(smiles1)
    g2 = _cached_construct_graph(smiles2)
    
    # Calculate the lower bound first
    bound = filter2(g1, g2)
    
    # If the bound is already > 1, we know they're not close
    if bound > 1:
        return False
    
    # Otherwise calculate the exact distance with threshold=1
    distance, _ = MCES_ILP(g1, g2, threshold=1)
    
    return distance <= 1


def are_very_distinct_pair(smiles1: str, smiles2: str) -> bool:
    """
    Check if two molecules have an MCES distance greater than 10.
    
    Parameters
    ----------
    smiles1 : str
        SMILES string of first molecule
    smiles2 : str
        SMILES string of second molecule
        
    Returns
    -------
    bool
        True if the molecules have an MCES distance > 10, False otherwise
    """
    g1 = _cached_construct_graph(smiles1)
    g2 = _cached_construct_graph(smiles2)
    
    # Calculate the lower bound first
    bound = filter2(g1, g2)
    
    # If the bound is already > 10, we know they're very distinct
    if bound > 10:
        return True
    
    # Otherwise calculate the exact distance with threshold=10
    distance, _ = MCES_ILP(g1, g2, threshold=10)
    
    return distance > 10

