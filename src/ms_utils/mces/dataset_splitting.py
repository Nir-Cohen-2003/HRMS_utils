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
from .mces import calculate_mces_distances, exact_mces_for_list_of_pairs  # Assuming this function exists
from multiprocessing import cpu_count

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
    mces_matrix_save_path: str | None = None
) -> Tuple[list[str], list[str], list[str], int]:
    """
    Split dataset using precomputed lower bounds, adaptively lowering the threshold if needed.
    Only the lower bounds are used (no exact MCES).
    """
    
    n = len(dataset)
    print(f"Calculating lower bounds matrix for {n} molecules using C++ implementation...")

    # Use C++ implementation for batch processing - much faster
    bounds_matrix = filter2_cpp(dataset.copy())
    if mces_matrix_save_path is not None:
        np.save(mces_matrix_save_path, bounds_matrix)
    if n < 20:
        # then we print the matrix to see if it looks correct
        print("Lower bounds matrix:")
        print(bounds_matrix)
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
            print(f"10 largest cluster size: {sorted([len(c) for c in cluster_list], reverse=True)[:10]}")
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
        
        # Focus on pairs most likely to exceed threshold when calculated exactly
        # Prioritize pairs with bounds closer to threshold (more likely to cross)
        cluster_pairs = []
        for i in range(len(cluster_indices)):
            for j in range(i + 1, len(cluster_indices)):
                idx1, idx2 = cluster_indices[i], cluster_indices[j]
                lower_bound = bounds_matrix[idx1, idx2]
                
                # Only consider pairs where exact distance might exceed threshold
                # Use tighter range - pairs too far from threshold are less likely to help
                if current_threshold - 2 <= lower_bound < current_threshold:
                    # Higher score for bounds closer to threshold
                    proximity_to_threshold = current_threshold - lower_bound
                    impact_score = cluster_size * (1.0 / (proximity_to_threshold + 0.1))
                    cluster_pairs.append((idx1, idx2, impact_score, cluster_size))
        
        # Sort cluster pairs by impact and take top ones
        cluster_pairs.sort(key=lambda x: x[2], reverse=True)
        # Limit pairs per cluster to avoid spending all budget on one cluster
        max_pairs_per_cluster = min(len(cluster_pairs), max_exact_calculations // len(problematic_clusters))
        critical_pairs.extend(cluster_pairs[:max_pairs_per_cluster])
    
    # Sort all pairs by impact score (higher is better)
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
    max_exact_calculations: int = 1000,
    mces_matrix_save_path: str | None = None
) -> Tuple[list[str], list[str], list[str], int]:
    """
    Split dataset with strategic exact MCES calculations to enable higher thresholds.
    Uses entire budget to achieve highest possible threshold.
    """
    
    n = len(dataset)
    print(f"Calculating lower bounds matrix for {n} molecules...")
    bounds_matrix = filter2_cpp(dataset)
    if mces_matrix_save_path is not None:
        np.save(mces_matrix_save_path, bounds_matrix)
    thresholds = list(range(initial_distinction_threshold, min_distinction_threshold - 1, threshold_step))
    remaining_calculations = max_exact_calculations
    
    # Try thresholds from highest to lowest
    for threshold_idx, target_threshold in enumerate(thresholds):
        print(f"Attempting threshold {target_threshold} (remaining budget: {remaining_calculations})")
        
        # First check if lower bounds alone work
        train_set, validation_set, test_set = try_split_with_threshold(
            dataset, bounds_matrix, target_threshold, validation_fraction, test_fraction, min_ratio
        )
        
        if train_set is not None:
            print(f"Success with threshold {target_threshold} using only lower bounds")
            return train_set, validation_set, test_set, target_threshold
        
        # If we have budget, try to improve with exact calculations
        if remaining_calculations > 0:
            # Allocate budget strategically - more for higher thresholds
            if threshold_idx == 0:  # Highest threshold gets most budget
                threshold_budget = min(remaining_calculations, max_exact_calculations // 2)
            elif threshold_idx == 1:  # Second highest gets good budget
                threshold_budget = min(remaining_calculations, max_exact_calculations // 3)
            else:  # Lower thresholds get remaining budget divided by remaining thresholds
                remaining_thresholds = len(thresholds) - threshold_idx
                threshold_budget = min(remaining_calculations // remaining_thresholds, remaining_calculations)
            
            # Find critical pairs and calculate exact distances
            critical_pairs = find_critical_pairs_for_threshold_optimization(
                dataset, bounds_matrix, target_threshold, validation_fraction, 
                test_fraction, min_ratio, threshold_budget
            )
            
            if critical_pairs and threshold_budget > 0:
                # Limit pairs to budget
                limited_pairs = critical_pairs[:threshold_budget]
                print(f"Calculating exact MCES for {len(limited_pairs)} critical pairs (budget: {threshold_budget})...")
                
                # Calculate batch size for the internal batching
                batch_size = max(20, len(limited_pairs) // (3 * cpu_count()))
                
                # Let exact_mces_for_list_of_pairs handle all the batching
                exact_results = exact_mces_for_list_of_pairs(
                    dataset, dataset, limited_pairs,
                    threshold=target_threshold, solver="GUROBI", batch_size=batch_size
                )
                
                # Create enhanced matrix with exact calculations
                enhanced_matrix = bounds_matrix.copy()
                successful_calcs = 0
                
                # Update matrix with exact results
                for idx1, idx2, exact_distance in exact_results:
                    if exact_distance is not None:
                        enhanced_matrix[idx1, idx2] = exact_distance
                        enhanced_matrix[idx2, idx1] = exact_distance
                        successful_calcs += 1
                if mces_matrix_save_path is not None:
                    np.save(mces_matrix_save_path, enhanced_matrix)
                remaining_calculations -= len(limited_pairs)
                print(f"Used {successful_calcs} exact calculations on threshold {target_threshold}")
                
                # Check if exact calculations enabled this threshold
                train_set, validation_set, test_set = try_split_with_threshold(
                    dataset, enhanced_matrix, target_threshold, validation_fraction, test_fraction, min_ratio
                )
                
                if train_set is not None:
                    print(f"Success with threshold {target_threshold} after {successful_calcs} exact calculations")
                    return train_set, validation_set, test_set, target_threshold
                
                # Update bounds_matrix for next iteration (carry forward the exact calculations)
                bounds_matrix = enhanced_matrix
            
            print(f"Threshold {target_threshold} failed even with exact calculations")
        else:
            print(f"No budget remaining for threshold {target_threshold}")
    
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


def split_dataset_brute_force_exact(
    dataset: list[str],
    validation_fraction=0.1,
    test_fraction=0.1,
    initial_distinction_threshold: int = 10,
    min_distinction_threshold: int = 2,
    threshold_step: int = -1,
    min_ratio: float = 0.7,
    max_exact_calculations: int | None = None,
    mces_matrix_save_path: str | None = None
) -> Tuple[list[str], list[str], list[str], int]:
    """
    Brute force version: calculate exact MCES for all pairs below threshold.
    If max_exact_calculations is set, prioritize pairs with highest bounds (closest to threshold).
    """
    
    n = len(dataset)
    print(f"Calculating lower bounds matrix for {n} molecules using C++ implementation...")

    # Use C++ implementation for batch processing
    bounds_matrix = filter2_cpp(dataset.copy())
    if mces_matrix_save_path is not None:
        np.save(mces_matrix_save_path, bounds_matrix)
    
    if n < 20:
        print("Lower bounds matrix:")
        print(bounds_matrix)
    
    print("Lower bounds matrix calculated, starting brute force exact calculation...")

    thresholds = list(range(initial_distinction_threshold, min_distinction_threshold - 1, threshold_step))
    calculated_pairs = set()  # Track pairs we've already calculated exactly
    
    for attempt, threshold in enumerate(thresholds):
        print(f"Attempt {attempt+1}: Trying distinction_threshold={threshold}")
        
        # First try with just lower bounds
        train_set, validation_set, test_set = try_split_with_threshold(
            dataset, bounds_matrix, threshold, validation_fraction, test_fraction, min_ratio
        )
        
        if train_set is not None:
            print(f"Success with threshold {threshold} using only lower bounds")
            return train_set, validation_set, test_set, threshold
        
        # Find all pairs where bound < threshold AND we haven't calculated them exactly yet
        below_threshold_mask = bounds_matrix < threshold
        # Only consider upper triangle to avoid duplicates
        upper_triangle_mask = np.triu(np.ones_like(bounds_matrix, dtype=bool), k=1)
        candidate_mask = below_threshold_mask & upper_triangle_mask
        
        candidate_pairs = []
        rows, cols = np.where(candidate_mask)
        for i, j in zip(rows, cols):
            # Skip if we've already calculated this pair exactly
            if (i, j) in calculated_pairs or (j, i) in calculated_pairs:
                continue
            bound_value = bounds_matrix[i, j]
            candidate_pairs.append((i, j, bound_value))
        
        total_candidates = len(candidate_pairs)
        already_calculated = len(rows) - total_candidates
        print(f"Found {total_candidates} pairs with bounds below threshold {threshold}")
        print(f"Skipped {already_calculated} pairs already calculated exactly")
        
        if total_candidates == 0:
            print(f"No new pairs to calculate for threshold {threshold}, continuing to next threshold")
            continue
        
        # Sort by bound value (highest first - closest to threshold)
        candidate_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Limit to max_exact_calculations if specified
        if max_exact_calculations is not None and total_candidates > max_exact_calculations:
            pairs_to_calculate = candidate_pairs[:max_exact_calculations]
            print(f"Limiting to top {max_exact_calculations} pairs (highest bounds)")
        else:
            pairs_to_calculate = candidate_pairs
        
        if len(pairs_to_calculate) == 0:
            print(f"No pairs to calculate for threshold {threshold}")
            continue
        
        print(f"Calculating exact MCES for {len(pairs_to_calculate)} pairs...")
        print(f"Bound range: {pairs_to_calculate[-1][2]} to {pairs_to_calculate[0][2]}")
        
        # Prepare pairs for exact calculation (remove bound values)
        exact_pairs = [(pair[0], pair[1]) for pair in pairs_to_calculate]
        
        # Calculate batch size for the internal batching
        batch_size = max(20, len(exact_pairs) // (3 * cpu_count()))
        
        # Calculate exact MCES distances
        exact_results = exact_mces_for_list_of_pairs(
            dataset, dataset, exact_pairs,
            threshold=threshold, solver="GUROBI", batch_size=batch_size
        )
        
        # Create enhanced matrix with exact calculations
        enhanced_matrix = bounds_matrix.copy()
        successful_calcs = 0
        
        # Update matrix with exact results and track calculated pairs
        for idx1, idx2, exact_distance in exact_results:
            if exact_distance is not None:
                enhanced_matrix[idx1, idx2] = exact_distance
                enhanced_matrix[idx2, idx1] = exact_distance
                calculated_pairs.add((idx1, idx2))
                successful_calcs += 1
        
        print(f"Successfully calculated exact MCES for {successful_calcs} pairs")
        
        if mces_matrix_save_path is not None:
            # Save the enhanced matrix with exact calculations
            enhanced_save_path = mces_matrix_save_path.replace('.npy', f'_enhanced_t{threshold}.npy')
            np.save(enhanced_save_path, enhanced_matrix)
            print(f"Saved enhanced matrix to {enhanced_save_path}")
        
        # Try splitting with enhanced matrix
        train_set, validation_set, test_set = try_split_with_threshold(
            dataset, enhanced_matrix, threshold, validation_fraction, test_fraction, min_ratio
        )
        
        if train_set is not None:
            print(f"Success with threshold {threshold} after {successful_calcs} exact calculations")
            return train_set, validation_set, test_set, threshold
        else:
            print(f"Split still failed with threshold {threshold} even after exact calculations")
            print(f"Validation set size: {len(validation_set) if validation_set else 0}, "
                  f"Test set size: {len(test_set) if test_set else 0}")
            
            # Carry forward exact calculations to next iteration
            bounds_matrix = enhanced_matrix
    
    raise RuntimeError("Could not split dataset with given parameters and thresholds, even with brute force exact calculations.")


if __name__ == "__main__":
    from time import perf_counter
    from ..rdkit.mol import sanitize_smiles_polars

    nist_smiles: List[str] = pl.scan_parquet('/home/analytit_admin/dev/MS_encoder/data/NIST_prepared_labeled.parquet').select('CanonicalSMILES').unique(maintain_order=True).with_columns(
            pl.col("CanonicalSMILES").map_batches(
                function=sanitize_smiles_polars,
                return_dtype=pl.String,
            )
        ).filter(
            pl.col("CanonicalSMILES").is_not_null(),
            pl.col("CanonicalSMILES").ne("")
        ).head(2000).collect().to_series().to_list()
    if any(smile=="=" for smile in nist_smiles):
        raise ValueError("Invalid SMILES found in dataset. Please check the input data.")
    # Create data directory if it doesn't exist
    os.makedirs(os.path.join(os.path.dirname(__file__), "__pycache__"), exist_ok=True)

    start = perf_counter()
    train_set, validation_set, test_set, threshold_adaptive = split_dataset_adaptive_threshold(
        nist_smiles.copy(),
        validation_fraction=0.1,
        test_fraction=0.1,
        initial_distinction_threshold=10,
        min_distinction_threshold=0,
        threshold_step=-1,
        # mces_matrix_save_path=os.path.join(os.path.dirname(__file__), "__pycache__", "mces_matrix.npy")
        
    )  # type: Tuple[List[str], List[str], List[str], int]
    end = perf_counter()
    adaptive_time = end - start
    # now write the sets to files named train_set.parquet, validation_set.parquet, test_set.parquet
    # pl.DataFrame({"CanonicalSMILES": train_set}).write_parquet(os.path.join(os.path.dirname(__file__), "__pycache__", "train_set.parquet"))
    # pl.DataFrame({"CanonicalSMILES": validation_set}).write_parquet(os.path.join(os.path.dirname(__file__), "__pycache__", "validation_set.parquet"))
    # pl.DataFrame({"CanonicalSMILES": test_set}).write_parquet(os.path.join(os.path.dirname(__file__), "__pycache__", "test_set.parquet"))
    print(f"Training set size: {len(train_set)}")
    print(f"Validation set size: {len(validation_set)}")
    print(f"Test set size: {len(test_set)}")
    print(f"Total size: {len(nist_smiles)}")
    print(f"Time taken: {adaptive_time:.2f} seconds")

    # # now use the selective exact calculation method
    # start = perf_counter()
    # train_set, validation_set, test_set, threshold_selective_exact = split_dataset_with_selective_exact_calculation(
    #     nist_smiles.copy(),
    #     validation_fraction=0.1,
    #     test_fraction=0.1,
    #     initial_distinction_threshold=10,
    #     min_distinction_threshold=0,
    #     threshold_step=-1,
    #     max_exact_calculations=10_000,
    # )  # type: Tuple[List[str], List[str], List[str], int]
    # end = perf_counter()
    # selective_time = end - start

    # # # now write them to similar files, but add _with_exact to the names
    # # pl.DataFrame({"CanonicalSMILES": train_set}).write_parquet(os.path.join(os.path.dirname(__file__), "__pycache__", "train_set_with_exact.parquet"))
    # # pl.DataFrame({"CanonicalSMILES": validation_set}).write_parquet(os.path.join(os.path.dirname(__file__), "__pycache__", "validation_set_with_exact.parquet"))
    # # pl.DataFrame({"CanonicalSMILES": test_set}).write_parquet(os.path.join(os.path.dirname(__file__), "__pycache__", "test_set_with_exact.parquet"))

    # print(f"Training set size: {len(train_set)}")
    # print(f"Validation set size: {len(validation_set)}")
    # print(f"Test set size: {len(test_set)}")
    # print(f"Total size: {len(nist_smiles)}")
    # print(f"Time taken (selective): {selective_time:.2f} seconds")
    
    # print(f"Adaptive threshold: {threshold}")
    # print(f"Selective exact calculation threshold: {threshold}")
    # print(f"Adaptive time: {adaptive_time:.2f} seconds")
    # print(f"Selective exact calculation time: {selective_time:.2f} seconds")
    # print(f"Speedup of forgoing exact calculations: {selective_time / adaptive_time:.2f}x")


    # now use the brute force exact calculation method
    start = perf_counter()
    train_set, validation_set, test_set, threshold_brute_force = split_dataset_brute_force_exact(
        nist_smiles.copy(),
        validation_fraction=0.1,
        test_fraction=0.1,
        initial_distinction_threshold=10,
        min_distinction_threshold=0,
        threshold_step=-1,
        max_exact_calculations=30_000,
    )  # type: Tuple[List[str], List[str], List[str], int]
    end = perf_counter()
    brute_force_time = end - start

    # # now write them to similar files, but add _brute_force to the names
    # pl.DataFrame({"CanonicalSMILES": train_set}).write_parquet(os.path.join(os.path.dirname(__file__), "__pycache__", "train_set_brute_force.parquet"))
    # pl.DataFrame({"CanonicalSMILES": validation_set}).write_parquet(os.path.join(os.path.dirname(__file__), "__pycache__", "validation_set_brute_force.parquet"))
    # pl.DataFrame({"CanonicalSMILES": test_set}).write_parquet(os.path.join(os.path.dirname(__file__), "__pycache__", "test_set_brute_force.parquet"))

    print(f"\n=== BRUTE FORCE EXACT RESULTS ===")
    print(f"Training set size: {len(train_set)}")
    print(f"Validation set size: {len(validation_set)}")
    print(f"Test set size: {len(test_set)}")
    print(f"Total size: {len(nist_smiles)}")
    print(f"Time taken (brute force): {brute_force_time:.2f} seconds")
    
    print(f"\n=== COMPARISON ===")
    print(f"Adaptive threshold: {threshold_adaptive}")
    # print(f"Selective exact calculation threshold: {threshold_selective_exact}")
    print(f"Brute force exact calculation threshold: {threshold_brute_force}")
    print(f"Adaptive time: {adaptive_time:.2f} seconds")
    # print(f"Selective exact calculation time: {selective_time:.2f} seconds")
    print(f"Brute force exact calculation time: {brute_force_time:.2f} seconds")
    # print(f"Selective vs Adaptive speedup: {selective_time / adaptive_time:.2f}x")
    print(f"Brute force vs Adaptive is lower by: {brute_force_time / adaptive_time:.2f}x")
    # print(f"Selective vs Brute force speedup: {selective_time / brute_force_time:.2f}x")