import os
from joblib import Memory
from .lib import MCES_ILP
from .graph_construction import construct_graph
from .bounds import filter2_batch, filter2
from typing import List, Tuple, Any, Generator
import networkx as nx
from contextlib import contextmanager
import sys
# Keep the memory cache configuration
memory = Memory(os.path.join(os.path.dirname(__file__), "__pycache__"), verbose=0)

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
def _cached_construct_graph(smiles: str) -> nx.Graph:
    return construct_graph(smiles)

# Process bounds in batches to reduce overhead using optimized batch processing
def _calculate_bounds_batch(smiles_list1: List[str], smiles_list2: List[str], batch_pairs: List[Tuple[int, int]]) -> List[Tuple[int, int, float]]:
    
    
    with suppress_output():
        # Extract unique molecule indices from batch_pairs
        indices1 = sorted(set(i for i, j in batch_pairs))
        indices2 = sorted(set(j for i, j in batch_pairs))
        
        # Build graphs for the unique molecules only
        graphs1: List[nx.Graph] = [_cached_construct_graph(smiles_list1[i]) for i in indices1]
        # Use batch processing to calculate all distances at once
        if smiles_list1 is smiles_list2 and indices1 == indices2:
            # Same molecule list case - use symmetric batch processing
            distance_matrix = filter2_batch(graphs1)
        else:
            # Different molecule lists case - use asymmetric batch processing
            graphs2: List[nx.Graph] = [_cached_construct_graph(smiles_list2[j]) for j in indices2]
            distance_matrix = filter2_batch(graphs1, graphs2)
        
        # Create index mappings for fast lookup
        idx1_to_pos = {idx: pos for pos, idx in enumerate(indices1)}
        idx2_to_pos = {idx: pos for pos, idx in enumerate(indices2)}
        
        
        # Extract results for the requested pairs
        results = []
        for i, j in batch_pairs:
            pos1 = idx1_to_pos[i]
            pos2 = idx2_to_pos[j]
            bound = distance_matrix[pos1, pos2]
            results.append((i, j, bound))
    
    return results

# Process exact calculations in batches
def _calculate_exact_batch(smiles_list1: List[str], smiles_list2: List[str], batch_pairs: List[Tuple[int, int]], threshold: int, solver="default") -> List[Tuple[int, int, int]]:
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
            g1: nx.Graph = _cached_construct_graph(smiles_list1[i])
            g2: nx.Graph = _cached_construct_graph(smiles_list2[j])
            try:
                distance, _ = MCES_ILP(g1, g2, threshold=threshold, solver=solver)
            except Exception as e:
                # Handle errors gracefully (including GurobiError)
                if "Gurobi" in str(type(e)) or "gurobi" in str(e).lower():
                    print(f"GurobiError for pair ({i}, {j}): {e}")
                else:
                    print(f"Error for pair ({i}, {j}): {e}")
                distance = None
            results.append((i, j, distance))
    return results

def _calculate_distinct_batch(
        smiles_list1: List[str],
        smiles_list2: List[str],
        batch_pairs: List[Tuple[int, int]],
        solver: str = "default"
    ) -> List[Tuple[int, int, bool]]:
        with suppress_output():
            results: List[Tuple[int, int, bool]] = []
            for i, j in batch_pairs:
                # Load graphs on-demand
                g1 = _cached_construct_graph(smiles_list1[i])
                g2 = _cached_construct_graph(smiles_list2[j])
                distance, _ = MCES_ILP(g1, g2, threshold=10, solver=solver)
                is_distinct: bool = distance > 10
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

