"""
Example usage of fast C++ MCES bounds calculation.
"""

import numpy as np
from ms_utils.mces.fast_mol_filter.fast_mol_filter.calculator import calculate_distances_symmetric, filter2

def calculate_mces_lower_bound(smiles_list):
    """
    Calculate MCES lower bound using symmetric distance matrix.
    
    Args:
        smiles_list (list): List of SMILES strings
        
    Returns:
        dict: Dictionary containing various bounds and metrics
    """
    n = len(smiles_list)
    if n < 2:
        return {"error": "Need at least 2 molecules"}
    
    # Method 1: Get full symmetric distance matrix
    print(f"Calculating distance matrix for {n} molecules...")
    distance_matrix = calculate_distances_symmetric(smiles_list)
    
    # Print the full distance matrix
    print("\nFull Distance Matrix:")
    print("=" * 50)
    print(distance_matrix)
    print("=" * 50)
    
    # Extract lower triangle (unique pairs) for MCES bound
    triu_indices = np.triu_indices(n, k=1)
    pairwise_distances = distance_matrix[triu_indices]
    
    # Calculate bounds
    min_pairwise_distance = np.min(pairwise_distances)
    max_pairwise_distance = np.max(pairwise_distances)
    mean_pairwise_distance = np.mean(pairwise_distances)
    total_distance_sum = np.sum(pairwise_distances)
    
    # Method 2: Direct total cost calculation (should match sum above)
    print("Calculating total cost directly...")
    total_cost_direct = filter2(smiles_list)
    
    return {
        "n_molecules": n,
        "n_pairs": len(pairwise_distances),
        "distance_matrix_shape": distance_matrix.shape,
        "distance_matrix": distance_matrix,  # Include matrix in return
        "min_pairwise_distance": min_pairwise_distance,
        "max_pairwise_distance": max_pairwise_distance,
        "mean_pairwise_distance": mean_pairwise_distance,
        "total_distance_sum": total_distance_sum,
        "total_cost_direct": total_cost_direct,
        "methods_match": np.isclose(total_distance_sum, total_cost_direct),
        "mces_lower_bound": min_pairwise_distance  # Conservative lower bound
    }

# Example usage
if __name__ == "__main__":
    # Sample SMILES strings for testing
    test_smiles = [
        "CCO",                    # ethanol
        "CCC",                    # propane
        "C1=CC=CC=C1",           # benzene
        "CC(=O)O",               # acetic acid
        "CC(C)O",                # isopropanol
        "C1=CC=C(C=C1)O",        # phenol
        "CCN",                   # ethylamine
        "C1CCCCC1"               # cyclohexane
    ]
    
    print("MCES Lower Bound Calculation Example")
    print("=" * 40)
    
    # Calculate bounds
    results = calculate_mces_lower_bound(test_smiles)
    
    # Display results
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    print("\nDistance Matrix (first 5x5):")
    if results["n_molecules"] > 0 and "distance_matrix_shape" in results:
        distance_matrix = calculate_distances_symmetric(test_smiles)
        print(distance_matrix[:min(5, len(test_smiles)), :min(5, len(test_smiles))])
    
    print(f"\nMCES Lower Bound: {results.get('mces_lower_bound', 'N/A')}")
    print(f"This means the maximum common edge subgraph")
    print(f"has at least {results.get('mces_lower_bound', 'N/A')} edges")