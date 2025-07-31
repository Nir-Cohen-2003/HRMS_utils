import polars as pl
import numpy as np
# from .mces import are_very_distinct, suppress_output
from typing import Tuple, List
import os
from scipy.sparse.csgraph import connected_components
import scipy.sparse as sp
import random
# from pickle import dump, load
# from .par import _calculate_bounds_batch
from .bounds import filter2_cpp
from .mces import calculate_mces_distances  # Assuming this function exists

### logic for dataset splitting:
# we take a list of smiles, and the requested fractions for validation and test sets.
# we calculate the mces (or approximate) between all pairs of molecules, and usign a threshold (distinction_threshold) we determine which molecules are distinct- below is non distinct, above is distinct.
# we then group the molecules into clusters, where each cluster is a set of molecules that are not distinct from each other- basically findign the "islands" of distinct molecules.
# we then map the clsuters into sets usign round-robin distribution, where we try to keep the validation and test sets as small as possible, while still having a good representation of the clusters. generally smalelr cluster will go to the validation and tests, which is a wanted behavior, as it will make them more "weird" and will help evaluate the model better.


# how is mces calculation done?
# 1. we caclualte lower bounds usign comaprison of the environmet of each atom.
# 2. then we try to split the data set usign the given fractions and distinction threshold.
# if this fails, we lower the distinction threshold and try again, until we reach a point where we can split the dataset.

def split_dataset_adaptive_threshold(
    dataset: list[str],
    validation_fraction=0.1,
    test_fraction=0.1,
    initial_distinction_threshold: int = 10,
    min_distinction_threshold: int = 2,
    threshold_step: int = -1,
    min_ratio: float = 0.7,
) -> Tuple[list[str], list[str], list[str], int]:
    """
    Split dataset using precomputed lower bounds, adaptively lowering the threshold if needed.
    Only the lower bounds are used (no exact MCES).
    """
    
    n = len(dataset)
    print(f"Calculating lower bounds matrix for {n} molecules using C++ implementation...")

    # Use C++ implementation for batch processing - much faster
    bounds_matrix = filter2_cpp(dataset)
    
    print("Lower bounds matrix calculated, starting adaptive threshold search...")

    thresholds = list(range(initial_distinction_threshold, min_distinction_threshold - 1, threshold_step))
    for attempt, threshold in enumerate(thresholds):
        print(f"Attempt {attempt+1}: Trying distinction_threshold={threshold} (lower bound only)")
        not_distinct = (bounds_matrix < threshold)
        rows, cols = np.where(not_distinct)
        edge_count = len(rows)
        not_distinct_sparse = sp.csr_matrix((np.ones(edge_count, dtype=np.int8), (rows, cols)), shape=(n, n))
        _, labels = connected_components(csgraph=not_distinct_sparse, directed=False, return_labels=True)
        unique_labels = np.unique(labels)
        cluster_list = [np.where(labels == label)[0].tolist() for label in unique_labels]
        random.shuffle(cluster_list)
        train_indices, validation_indices, test_indices = [], [], []
        validation_size = int(n * validation_fraction)
        test_size = int(n * test_fraction)
        large_cluster_threshold = max(validation_size, test_size) // 2
        large_clusters = [c for c in cluster_list if len(c) > large_cluster_threshold]
        small_clusters = [c for c in cluster_list if len(c) <= large_cluster_threshold]
        for cluster in large_clusters:
            train_indices.extend(cluster)
        current_set = 0
        for cluster in small_clusters:
            cluster_size = len(cluster)
            if current_set == 0 and len(test_indices) + cluster_size <= test_size * 1.1:
                test_indices.extend(cluster)
                current_set = 1
            elif current_set == 1 and len(validation_indices) + cluster_size <= validation_size * 1.1:
                validation_indices.extend(cluster)
                current_set = 2
            elif current_set == 2:
                train_indices.extend(cluster)
                current_set = 0
            else:
                if current_set == 0:
                    if len(validation_indices) + cluster_size <= validation_size * 1.1:
                        validation_indices.extend(cluster)
                        current_set = 2
                    else:
                        train_indices.extend(cluster)
                        current_set = 1
                elif current_set == 1:
                    if len(test_indices) + cluster_size <= test_size * 1.1:
                        test_indices.extend(cluster)
                        current_set = 2
                    else:
                        train_indices.extend(cluster)
                        current_set = 0
        train_set = [dataset[i] for i in train_indices]
        validation_set = [dataset[i] for i in validation_indices]
        test_set = [dataset[i] for i in test_indices]
        min_val = int(n * validation_fraction * min_ratio)
        min_test = int(n * test_fraction * min_ratio)
        if len(validation_set) >= min_val and len(test_set) >= min_test:
            print(f"Success with threshold {threshold}")
            return train_set, validation_set, test_set, threshold
        else:
            print(f"Split failed: validation ({len(validation_set)}) or test ({len(test_set)}) too small")
    raise RuntimeError("Could not split dataset with given parameters and thresholds.")


def find_critical_pairs_for_threshold_optimization(
    dataset: list[str],
    bounds_matrix: np.ndarray,
    current_threshold: int,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    min_ratio: float = 0.7,
    max_exact_calculations: int = 1000
) -> list[tuple[int, int]]:
    """
    Find pairs where calculating exact MCES might enable using a higher threshold.
    Returns pairs sorted by potential impact.
    """
    n = len(dataset)
    validation_size = int(n * validation_fraction)
    test_size = int(n * test_fraction)
    large_cluster_threshold = max(validation_size, test_size) // 2
    
    # Get current clustering
    not_distinct = (bounds_matrix < current_threshold)
    rows, cols = np.where(not_distinct)
    edge_count = len(rows)
    not_distinct_sparse = sp.csr_matrix((np.ones(edge_count, dtype=np.int8), (rows, cols)), shape=(n, n))
    _, labels = connected_components(csgraph=not_distinct_sparse, directed=False, return_labels=True)
    
    # Find clusters that are too large
    unique_labels, cluster_sizes = np.unique(labels, return_counts=True)
    problematic_clusters = unique_labels[cluster_sizes > large_cluster_threshold]
    
    if len(problematic_clusters) == 0:
        return []  # Current threshold already works
    
    critical_pairs = []
    
    for cluster_id in problematic_clusters:
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_size = len(cluster_indices)
        
        # Find pairs within this cluster where lower_bound < exact_distance might >= current_threshold
        # Focus on pairs with lower bounds close to threshold (most likely to cross it)
        for i in range(len(cluster_indices)):
            for j in range(i + 1, len(cluster_indices)):
                idx1, idx2 = cluster_indices[i], cluster_indices[j]
                lower_bound = bounds_matrix[idx1, idx2]
                
                # Only consider pairs where exact distance might exceed threshold
                if lower_bound < current_threshold and lower_bound >= current_threshold - 3:
                    # Calculate potential impact: how much splitting this cluster could help
                    impact_score = cluster_size * (current_threshold - lower_bound)
                    critical_pairs.append((idx1, idx2, impact_score, cluster_size))
    
    # Sort by impact score (higher is better)
    critical_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Return top pairs, limited by max_exact_calculations
    return [(pair[0], pair[1]) for pair in critical_pairs[:max_exact_calculations]]

def split_dataset_with_selective_exact_calculation(
    dataset: list[str],
    validation_fraction=0.1,
    test_fraction=0.1,
    initial_distinction_threshold: int = 10,
    min_distinction_threshold: int = 2,
    threshold_step: int = -1,
    min_ratio: float = 0.7,
    max_exact_calculations: int = 1000
) -> Tuple[list[str], list[str], list[str], int]:
    """
    Split dataset with strategic exact MCES calculations to enable higher thresholds.
    """
    
    n = len(dataset)
    print(f"Calculating lower bounds matrix for {n} molecules...")
    bounds_matrix = filter2_cpp(dataset)
    
    # Try to optimize by calculating strategic exact distances
    for target_threshold in range(initial_distinction_threshold, min_distinction_threshold - 1, threshold_step):
        print(f"Attempting threshold {target_threshold}")
        
        # First check if lower bounds alone work
        train_set, validation_set, test_set = try_split_with_threshold(
            dataset, bounds_matrix, target_threshold, validation_fraction, test_fraction, min_ratio
        )
        
        if train_set is not None:
            print(f"Success with threshold {target_threshold} using only lower bounds")
            return train_set, validation_set, test_set, target_threshold
        
        # If not, find critical pairs and calculate exact distances
        critical_pairs = find_critical_pairs_for_threshold_optimization(
            dataset, bounds_matrix, target_threshold, validation_fraction, 
            test_fraction, min_ratio, max_exact_calculations
        )
        
        if not critical_pairs:
            continue
            
        print(f"Calculating exact MCES for {len(critical_pairs)} critical pairs...")
        
        # Create enhanced matrix with exact calculations
        enhanced_matrix = bounds_matrix.copy()
        exact_count = 0
        
        for idx1, idx2 in critical_pairs:
            exact_distance = calculate_mces_distances(dataset[idx1], dataset[idx2])
            enhanced_matrix[idx1, idx2] = exact_distance
            enhanced_matrix[idx2, idx1] = exact_distance
            exact_count += 1
            
            # Early stopping: check if we've broken enough connections
            if exact_count % 100 == 0:
                train_set, validation_set, test_set = try_split_with_threshold(
                    dataset, enhanced_matrix, target_threshold, validation_fraction, test_fraction, min_ratio
                )
                if train_set is not None:
                    print(f"Success with threshold {target_threshold} after {exact_count} exact calculations")
                    return train_set, validation_set, test_set, target_threshold
        
        # Final attempt with all exact calculations
        train_set, validation_set, test_set = try_split_with_threshold(
            dataset, enhanced_matrix, target_threshold, validation_fraction, test_fraction, min_ratio
        )
        
        if train_set is not None:
            print(f"Success with threshold {target_threshold} after {len(critical_pairs)} exact calculations")
            return train_set, validation_set, test_set, target_threshold
    
    raise RuntimeError("Could not split dataset even with exact calculations")

def try_split_with_threshold(
    dataset: list[str], 
    distance_matrix: np.ndarray, 
    threshold: int,
    validation_fraction: float,
    test_fraction: float,
    min_ratio: float
) -> Tuple[list[str], list[str], list[str]] | Tuple[None, None, None]:
    """
    Attempt to split dataset with given threshold. Returns None if unsuccessful.
    """
    n = len(dataset)
    not_distinct = (distance_matrix < threshold)
    rows, cols = np.where(not_distinct)
    edge_count = len(rows)
    not_distinct_sparse = sp.csr_matrix((np.ones(edge_count, dtype=np.int8), (rows, cols)), shape=(n, n))
    _, labels = connected_components(csgraph=not_distinct_sparse, directed=False, return_labels=True)
    
    unique_labels = np.unique(labels)
    cluster_list = [np.where(labels == label)[0].tolist() for label in unique_labels]
    random.shuffle(cluster_list)
    
    # Apply the same splitting logic as original function
    train_indices, validation_indices, test_indices = [], [], []
    validation_size = int(n * validation_fraction)
    test_size = int(n * test_fraction)
    large_cluster_threshold = max(validation_size, test_size) // 2
    
    large_clusters = [c for c in cluster_list if len(c) > large_cluster_threshold]
    small_clusters = [c for c in cluster_list if len(c) <= large_cluster_threshold]
    
    for cluster in large_clusters:
        train_indices.extend(cluster)
    
    current_set = 0
    for cluster in small_clusters:
        cluster_size = len(cluster)
        if current_set == 0 and len(test_indices) + cluster_size <= test_size * 1.1:
            test_indices.extend(cluster)
            current_set = 1
        elif current_set == 1 and len(validation_indices) + cluster_size <= validation_size * 1.1:
            validation_indices.extend(cluster)
            current_set = 2
        elif current_set == 2:
            train_indices.extend(cluster)
            current_set = 0
        else:
            # Handle overflow cases...
            train_indices.extend(cluster)
    
    train_set = [dataset[i] for i in train_indices]
    validation_set = [dataset[i] for i in validation_indices]
    test_set = [dataset[i] for i in test_indices]
    
    min_val = int(n * validation_fraction * min_ratio)
    min_test = int(n * test_fraction * min_ratio)
    
    if len(validation_set) >= min_val and len(test_set) >= min_test:
        return train_set, validation_set, test_set
    else:
        return None, None, None


if __name__ == "__main__":
    from time import perf_counter
    nist_smiles: List[str] = pl.scan_parquet('/home/analytit_admin/dev/MS_encoder/data/NIST_prepared_labeled.parquet').select('CanonicalSMILES').unique(maintain_order=True).head(1000).collect().to_series().to_list()
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.join(os.path.dirname(__file__), "__pycache__"), exist_ok=True)

    start = perf_counter()
    train_set, validation_set, test_set, threshold = split_dataset_adaptive_threshold(
        nist_smiles,
        validation_fraction=0.1,
        test_fraction=0.1,
        initial_distinction_threshold=10,
        min_distinction_threshold=2,
        threshold_step=-1,
    )  # type: Tuple[List[str], List[str], List[str], int]
    end = perf_counter()
    adaptive_time = end - start
    # # now write the sets to files named train_set.parquet, validation_set.parquet, test_set.parquet
    # pl.DataFrame({"CanonicalSMILES": train_set}).write_parquet(os.path.join(os.path.dirname(__file__), "__pycache__", "train_set.parquet"))
    # pl.DataFrame({"CanonicalSMILES": validation_set}).write_parquet(os.path.join(os.path.dirname(__file__), "__pycache__", "validation_set.parquet"))
    # pl.DataFrame({"CanonicalSMILES": test_set}).write_parquet(os.path.join(os.path.dirname(__file__), "__pycache__", "test_set.parquet"))
    print(f"Training set size: {len(train_set)}")
    print(f"Validation set size: {len(validation_set)}")
    print(f"Test set size: {len(test_set)}")
    print(f"Total size: {len(nist_smiles)}")
    print(f"Time taken: {adaptive_time:.2f} seconds")

    # now use the selective exact calculation method
    start = perf_counter()
    train_set, validation_set, test_set, threshold = split_dataset_with_selective_exact_calculation(
        nist_smiles,
        validation_fraction=0.1,
        test_fraction=0.1,
        initial_distinction_threshold=10,
        min_distinction_threshold=2,
        threshold_step=-1,
        max_exact_calculations=1000,
    )  # type: Tuple[List[str], List[str], List[str], int]
    end = perf_counter()
    selective_time = end - start
    # # now write the sets to files named train_set.parquet, validation_set.parquet, test_set.parquet
    # pl.DataFrame({"CanonicalSMILES": train_set}).write_parquet(os.path.join(os.path.dirname(__file__), "__pycache__", "train_set.parquet"))
    # pl.DataFrame({"CanonicalSMILES": validation_set}).write_parquet(os.path.join      
    # (os.path.dirname(__file__), "__pycache__", "validation_set.parquet"))
    # pl.DataFrame({"CanonicalSMILES": test_set}).write_parquet(os.path.join(os.path.dirname(__file__), "__pycache__", "test_set.parquet"))
    print(f"Training set size: {len(train_set)}")
    print(f"Validation set size: {len(validation_set)}")
    print(f"Test set size: {len(test_set)}")
    print(f"Total size: {len(nist_smiles)}")
    print(f"Time taken (selective): {selective_time:.2f} seconds")