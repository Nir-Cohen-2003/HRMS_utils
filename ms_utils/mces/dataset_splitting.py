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
import networkx as nx
from collections import defaultdict
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
    Uses graph-theoretic analysis to identify critical structural connections.
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
    
    # Sort problematic clusters by size (largest first) - focus on biggest blockers
    cluster_priorities = [(cluster_id, cluster_sizes[np.where(unique_labels == cluster_id)[0][0]]) 
                         for cluster_id in problematic_clusters]
    cluster_priorities.sort(key=lambda x: x[1], reverse=True)
    
    all_critical_pairs = []
    
    for cluster_id, cluster_size in cluster_priorities:
        cluster_indices = np.where(labels == cluster_id)[0]
        
        # Build subgraph for this cluster
        cluster_graph = _build_cluster_subgraph(cluster_indices, bounds_matrix, current_threshold)
        
        # Find structurally critical edges using multiple strategies
        critical_edges = _find_structural_bottlenecks(cluster_graph, cluster_indices, bounds_matrix, current_threshold)
        
        # Score each critical edge
        cluster_pairs = []
        for (idx1, idx2) in critical_edges:
            lower_bound = bounds_matrix[idx1, idx2]
            
            # Only consider pairs where exact distance might exceed threshold
            if current_threshold - 2 <= lower_bound < current_threshold:
                proximity_to_threshold = current_threshold - lower_bound
                # Higher weight for larger clusters and edges closer to threshold
                impact_score = cluster_size * (1.0 / (proximity_to_threshold + 0.1))
                cluster_pairs.append((idx1, idx2, impact_score, cluster_size))
        
        # For the largest cluster, allow using most of the budget
        if cluster_id == cluster_priorities[0][0]:  # Largest cluster
            max_pairs_this_cluster = min(len(cluster_pairs), max_exact_calculations // 2)
        else:
            # Distribute remaining budget among other clusters
            remaining_clusters = len(cluster_priorities) - 1
            max_pairs_this_cluster = min(len(cluster_pairs), 
                                       max_exact_calculations // (2 * remaining_clusters) if remaining_clusters > 0 else 0)
        
        # Sort by impact score and take top pairs
        cluster_pairs.sort(key=lambda x: x[2], reverse=True)
        all_critical_pairs.extend(cluster_pairs[:max_pairs_this_cluster])
    
    # Sort all pairs by impact score (higher is better)
    all_critical_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Return top pairs, limited by max_exact_calculations
    return [(pair[0], pair[1]) for pair in all_critical_pairs[:max_exact_calculations]]


def _build_cluster_subgraph(cluster_indices: np.ndarray, bounds_matrix: np.ndarray, threshold: int) -> nx.Graph:
    """Build NetworkX graph for cluster analysis."""
    G = nx.Graph()
    G.add_nodes_from(cluster_indices)
    
    # Add edges for connections within the cluster (below threshold)
    for i in range(len(cluster_indices)):
        for j in range(i + 1, len(cluster_indices)):
            idx1, idx2 = cluster_indices[i], cluster_indices[j]
            if bounds_matrix[idx1, idx2] < threshold:
                # Store the bound value as edge weight
                G.add_edge(idx1, idx2, weight=bounds_matrix[idx1, idx2])
    
    return G


def _find_structural_bottlenecks(graph: nx.Graph, cluster_indices: np.ndarray, 
                                bounds_matrix: np.ndarray, threshold: int) -> set[tuple[int, int]]:
    """
    Identify structurally critical edges using multiple graph-theoretic approaches.
    """
    critical_edges = set()
    
    # Strategy 1: Bridge edges (articulation edges)
    # These are edges whose removal would split the cluster
    bridges = list(nx.bridges(graph))
    for u, v in bridges:
        critical_edges.add((min(u, v), max(u, v)))
    
    # Strategy 2: High betweenness centrality edges
    # Edges that lie on many shortest paths between nodes
    try:
        edge_betweenness = nx.edge_betweenness_centrality(graph, k=min(50, len(cluster_indices)))
        # Take top 20% of edges by betweenness centrality
        sorted_edges = sorted(edge_betweenness.items(), key=lambda x: x[1], reverse=True)
        top_count = max(1, len(sorted_edges) // 5)
        for (u, v), _ in sorted_edges[:top_count]:
            critical_edges.add((min(u, v), max(u, v)))
    except:
        # Fallback if betweenness calculation fails
        pass
    
    # Strategy 3: Minimum cut edges (for very large clusters)
    if len(cluster_indices) > 100:
        try:
            # Find edges in minimum cuts that would create balanced partitions
            target_partition_size = len(cluster_indices) // 2
            nodes = list(graph.nodes())
            
            # Try a few different source-sink pairs to find good cuts
            for i in range(0, min(10, len(nodes)), max(1, len(nodes) // 10)):
                for j in range(i + target_partition_size, min(i + target_partition_size + 10, len(nodes))):
                    try:
                        cut_value, (partition1, partition2) = nx.minimum_cut(graph, nodes[i], nodes[j])
                        # If this creates reasonably balanced partitions
                        if 0.3 <= len(partition1) / len(nodes) <= 0.7:
                            # Find edges in the cut
                            cut_edges = nx.edge_boundary(graph, partition1, partition2)
                            for u, v in cut_edges:
                                critical_edges.add((min(u, v), max(u, v)))
                            break  # Found a good cut, stop searching
                    except:
                        continue
        except:
            pass
    
    # Strategy 4: Edges with highest weight (closest to threshold)
    # These are most likely to cross the threshold when calculated exactly
    cluster_edges = []
    for i in range(len(cluster_indices)):
        for j in range(i + 1, len(cluster_indices)):
            idx1, idx2 = cluster_indices[i], cluster_indices[j]
            if bounds_matrix[idx1, idx2] < threshold:
                cluster_edges.append(((idx1, idx2), bounds_matrix[idx1, idx2]))
    
    # Sort by weight (higher = closer to threshold) and take top candidates
    cluster_edges.sort(key=lambda x: x[1], reverse=True)
    top_by_weight = min(len(cluster_edges) // 10, 50)  # Top 10% or 50, whichever is smaller
    for (u, v), _ in cluster_edges[:top_by_weight]:
        critical_edges.add((min(u, v), max(u, v)))
    
    # Strategy 5: High degree nodes' edges
    # Nodes with many connections are often structural hubs
    degrees = dict(graph.degree())
    high_degree_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
    top_hubs = [node for node, degree in high_degree_nodes[:max(1, len(cluster_indices) // 10)]]
    
    for hub in top_hubs:
        for neighbor in graph.neighbors(hub):
            u, v = min(hub, neighbor), max(hub, neighbor)
            critical_edges.add((u, v))
    
    return critical_edges
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
    Uses entire budget and retests higher thresholds after each round of calculations.
    """
    
    n = len(dataset)
    print(f"Calculating lower bounds matrix for {n} molecules...")
    bounds_matrix = filter2_cpp(dataset)
    if mces_matrix_save_path is not None:
        np.save(mces_matrix_save_path, bounds_matrix)
    
    thresholds = list(range(initial_distinction_threshold, min_distinction_threshold - 1, threshold_step))
    remaining_calculations = max_exact_calculations
    enhanced_matrix = bounds_matrix.copy()
    
    # Keep track of calculations per round
    calculation_rounds = []
    
    # First, find initial achievable threshold with bounds only
    best_achievable_threshold = None
    best_split = None
    
    for target_threshold in thresholds:
        train_set, validation_set, test_set = try_split_with_threshold(
            dataset, enhanced_matrix, target_threshold, validation_fraction, test_fraction, min_ratio
        )
        
        if train_set is not None:
            best_achievable_threshold = target_threshold
            best_split = (train_set, validation_set, test_set)
            break
    
    if best_achievable_threshold is None:
        raise RuntimeError("Could not split dataset even at lowest threshold")
    
    print(f"Initial achievable threshold with bounds only: {best_achievable_threshold}")
    
    while remaining_calculations > 0:
        # If we're already at the highest threshold, we're done
        if best_achievable_threshold == thresholds[0]:
            print(f"Already achieved highest threshold {best_achievable_threshold}")
            break
        
        # Find the next higher threshold to target
        target_threshold = None
        for thresh in thresholds:
            if thresh > best_achievable_threshold:
                target_threshold = thresh
                break
        
        if target_threshold is None:
            print(f"No higher threshold to target from {best_achievable_threshold}")
            break
        
        # Allocate budget for this round - be more aggressive with remaining budget
        if remaining_calculations >= 1000:
            round_budget = min(2000, remaining_calculations // 2)  # Use larger chunks
        elif remaining_calculations >= 500:
            round_budget = min(1000, remaining_calculations)
        else:
            round_budget = remaining_calculations  # Use all remaining budget
        
        print(f"Targeting threshold {target_threshold} (current best: {best_achievable_threshold}, budget: {round_budget})")
        
        # Find critical pairs for the target threshold
        critical_pairs = find_critical_pairs_for_threshold_optimization(
            dataset, enhanced_matrix, target_threshold, validation_fraction, 
            test_fraction, min_ratio, round_budget
        )
        
        if not critical_pairs:
            print(f"No critical pairs found for threshold {target_threshold}")
            break
        
        # Use the allocated budget
        pairs_to_calculate = critical_pairs[:round_budget]
        print(f"Calculating exact MCES for {len(pairs_to_calculate)} critical pairs...")
        
        # Calculate batch size for the internal batching
        batch_size = max(20, len(pairs_to_calculate) // (3 * cpu_count()))
        
        # Calculate exact distances
        exact_results = exact_mces_for_list_of_pairs(
            dataset, dataset, pairs_to_calculate,
            threshold=target_threshold, solver="GUROBI", batch_size=batch_size
        )
        
        # Update matrix with exact results
        successful_calcs = 0
        for idx1, idx2, exact_distance in exact_results:
            if exact_distance is not None:
                original_bound = bounds_matrix[idx1, idx2]
                
                # CRITICAL CHECK: Lower bound must never exceed exact distance
                if original_bound > exact_distance:
                    raise RuntimeError(
                        f"ALGORITHM ERROR: Lower bound ({original_bound}) exceeds exact distance ({exact_distance}) "
                        f"for pair ({idx1}, {idx2}). This indicates a fundamental bug in the lower bound calculation algorithm. "
                        f"Lower bounds must always be ≤ exact distances by definition."
                    )
                
                # Update with exact distance (which should always be ≥ lower bound)
                enhanced_matrix[idx1, idx2] = exact_distance
                enhanced_matrix[idx2, idx1] = exact_distance
                successful_calcs += 1
        
        remaining_calculations -= len(pairs_to_calculate)
        calculation_rounds.append((target_threshold, len(pairs_to_calculate), successful_calcs))
        
        print(f"Used {successful_calcs}/{len(pairs_to_calculate)} exact calculations (remaining budget: {remaining_calculations})")
        
        # Recheck what threshold is now achievable
        for target_threshold in thresholds:
            train_set, validation_set, test_set = try_split_with_threshold(
                dataset, enhanced_matrix, target_threshold, validation_fraction, test_fraction, min_ratio
            )
            
            if train_set is not None:
                best_achievable_threshold = target_threshold
                best_split = (train_set, validation_set, test_set)
                break
        
        # Save enhanced matrix periodically
        if mces_matrix_save_path is not None:
            np.save(mces_matrix_save_path, enhanced_matrix)
    
    # Final check - find the highest achievable threshold
    final_threshold = None
    final_split = None
    
    for target_threshold in thresholds:
        train_set, validation_set, test_set = try_split_with_threshold(
            dataset, enhanced_matrix, target_threshold, validation_fraction, test_fraction, min_ratio
        )
        
        if train_set is not None:
            final_threshold = target_threshold
            final_split = (train_set, validation_set, test_set)
            break
    
    if final_split is None:
        raise RuntimeError("Could not split dataset even after all exact calculations")
    
    # Print summary
    total_used = max_exact_calculations - remaining_calculations
    print(f"\nFinal results:")
    print(f"Achieved threshold: {final_threshold}")
    print(f"Total exact calculations used: {total_used}/{max_exact_calculations}")
    print(f"Calculation rounds: {len(calculation_rounds)}")
    for i, (thresh, attempted, successful) in enumerate(calculation_rounds):
        print(f"  Round {i+1}: Targeting threshold {thresh}, {successful}/{attempted} successful calculations")
    
    return final_split[0], final_split[1], final_split[2], final_threshold
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
        elif current_set == 1 and len(validation_indices) + cluster_size <= validation_fraction * 1.1:
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
    train_set, validation_set, test_set, threshold = split_dataset_adaptive_threshold(
        nist_smiles.copy(),
        validation_fraction=0.1,
        test_fraction=0.1,
        initial_distinction_threshold=10,
        min_distinction_threshold=0,
        threshold_step=-1,
        mces_matrix_save_path=os.path.join(os.path.dirname(__file__), "__pycache__", "mces_matrix.npy")
        
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

    # now use the selective exact calculation method
    start = perf_counter()
    train_set, validation_set, test_set, threshold = split_dataset_with_selective_exact_calculation(
        nist_smiles.copy(),
        validation_fraction=0.1,
        test_fraction=0.1,
        initial_distinction_threshold=10,
        min_distinction_threshold=0,
        threshold_step=-1,
        max_exact_calculations=10_000,
    )  # type: Tuple[List[str], List[str], List[str], int]
    end = perf_counter()
    selective_time = end - start

    # # now write them to similar files, but add _with_exact to the names
    # pl.DataFrame({"CanonicalSMILES": train_set}).write_parquet(os.path.join(os.path.dirname(__file__), "__pycache__", "train_set_with_exact.parquet"))
    # pl.DataFrame({"CanonicalSMILES": validation_set}).write_parquet(os.path.join(os.path.dirname(__file__), "__pycache__", "validation_set_with_exact.parquet"))
    # pl.DataFrame({"CanonicalSMILES": test_set}).write_parquet(os.path.join(os.path.dirname(__file__), "__pycache__", "test_set_with_exact.parquet"))

    print(f"Training set size: {len(train_set)}")
    print(f"Validation set size: {len(validation_set)}")
    print(f"Test set size: {len(test_set)}")
    print(f"Total size: {len(nist_smiles)}")
    print(f"Time taken (selective): {selective_time:.2f} seconds")
    
    # print(f"Adaptive threshold: {threshold}")
    # print(f"Selective exact calculation threshold: {threshold}")
    # print(f"Adaptive time: {adaptive_time:.2f} seconds")
    # print(f"Selective exact calculation time: {selective_time:.2f} seconds")
    # print(f"Speedup of forgoing exact calculations: {selective_time / adaptive_time:.2f}x")