import polars as pl
import numpy as np
from .mces import are_very_distinct, suppress_output
from typing import Tuple, List
import os
from scipy.sparse.csgraph import connected_components
import scipy.sparse as sp
import random
from pickle import dump, load
from .par import _calculate_bounds_batch
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

### logic for dataset splitting:
# we take a list of smiles, and the requested fractions for validation and test sets.
# we calculate the mces (or approximate) between all pairs of molecules, and usign a threshold (distinction_threshold) we determine which molecules are distinct- below is non distinct, above is distinct.
# we then group the molecules into clusters, where each cluster is a set of molecules that are not distinct from each other- basically findign the "islands" of distinct molecules.
# we then map the clsuters into sets usign round-robin distribution, where we try to keep the validation and test sets as small as possible, while still having a good representation of the clusters. generally smalelr cluster will go to the validation and tests, which is a wanted behavior, as it will make them more "weird" and will help evaluate the model better.



def split_dataset(dataset: list[str], validation_fraction=0.1, test_fraction=0.1, 
                  batch_size=100, checkpoint_path="data/split_checkpoint.pkl",distinction_threshold:int=10,use_solver:bool=True) -> Tuple[list[str], list[str], list[str]]:
    """Split the dataset into training, validation and testing datasets.
    Args:
        dataset (list[str]): List of SMILES strings.
        validation_fraction (float): Fraction of data to use for validation.
        test_fraction (float): Fraction of data to use for testing.
        batch_size (int): Size of each batch for processing.
        checkpoint_path (str): Path to save the checkpoint.
        distinction_threshold (int): Threshold for distinguishing molecules as distinct.
        use_solver (bool): Whether to use the solver for MCES calculations, or stick to the lower bound of the mces and use that. usign the solver is more accurate, but slower.
    Returns:
        Tuple[list[str], list[str], list[str]]: Tuple of training, validation and testing datasets.
    """
    
    n = len(dataset)
    
    # Create batches of molecules
    num_batches = (n + batch_size - 1) // batch_size
    batches = [dataset[i*batch_size:min((i+1)*batch_size, n)] for i in range(num_batches)]
    
    print(f"Building sparse similarity graph for {n} molecules using batched processing...")
    
    # Check for existing checkpoint
    start_i = 0
    start_j = 0
    edge_count = 0
    rows = None
    cols = None

    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint at {checkpoint_path}, loading...")
        try:
            checkpoint = load(open(checkpoint_path, 'rb'))
            rows = checkpoint['rows']
            cols = checkpoint['cols']
            edge_count = checkpoint['edge_count']
            start_i = checkpoint['i']
            start_j = checkpoint['j']
            
            # Validate checkpoint compatibility with current dataset
            if edge_count > 0:
                max_row_idx = rows[:edge_count].max() if edge_count > 0 else 0
                max_col_idx = cols[:edge_count].max() if edge_count > 0 else 0
                if max_row_idx >= n or max_col_idx >= n:
                    print(f"Checkpoint incompatible: max indices ({max_row_idx}, {max_col_idx}) exceed dataset size {n}")
                    print("Starting from scratch")
                    start_i = 0
                    start_j = 0
                    rows = None
                    cols = None
                    edge_count = 0
                else:
                    print(f"Resuming from batch {start_i+1}, comparison {start_j+1}")
            else:
                print(f"Resuming from batch {start_i+1}, comparison {start_j+1}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting from scratch")
            start_i = 0
            start_j = 0
            rows = None
            cols = None
            edge_count = 0
    
    # Initialize arrays if not loaded from checkpoint
    if rows is None:
        # Pre-allocate lists to avoid resizing
        estimated_edges = min(n * 10, n * n // 10)  # Conservative estimate
        rows = np.zeros(estimated_edges, dtype=np.int32)
        cols = np.zeros(estimated_edges, dtype=np.int32)
        edge_count = 0
    
    # Process batch by batch
    for i in range(start_i, num_batches):
        batch_i = batches[i]
        base_idx_i = i * batch_size
        
        # Process the internal batch if we're starting a new batch
        if start_j == 0:
            print(f"Processing batch {i+1}/{num_batches} internally (size: {len(batch_i)})")
            
            with suppress_output():
                same_batch_result = are_very_distinct(batch_i, symmetric=True,use_solver=use_solver,distinction_threshold=distinction_threshold)
            
            local_indices = np.where(~same_batch_result)
            local_x, local_y = local_indices[0], local_indices[1]
            
            upper_mask = local_x < local_y
            local_x, local_y = local_x[upper_mask], local_y[upper_mask]
            
            global_x = base_idx_i + local_x
            global_y = base_idx_i + local_y
            
            if edge_count + len(global_x) * 2 >= len(rows):
                new_size = max(len(rows) * 2, edge_count + len(global_x) * 2 + 1000)
                rows = np.resize(rows, new_size)
                cols = np.resize(cols, new_size)
            
            rows[edge_count:edge_count+len(global_x)] = global_x
            cols[edge_count:edge_count+len(global_x)] = global_y
            edge_count += len(global_x)
            
            rows[edge_count:edge_count+len(global_x)] = global_y
            cols[edge_count:edge_count+len(global_x)] = global_x
            edge_count += len(global_x)
            
            # Save checkpoint after processing internal batch
            checkpoint = {
                'rows': rows,
                'cols': cols,
                'edge_count': edge_count,
                'i': i,
                'j': 0
            }
            dump(checkpoint, open(checkpoint_path, 'wb'))
            print(f"Saved checkpoint after batch {i+1} internal processing")
        
        # Compare with other batches
        for j in range(max(i + 1, start_j), num_batches):
            batch_j = batches[j]
            base_idx_j = j * batch_size
            
            
            with suppress_output():
                cross_batch_result = are_very_distinct(batch_i, batch_j, symmetric=False)
             

            local_indices = np.where(~cross_batch_result)
            local_x, local_y = local_indices[0], local_indices[1]
            
            global_x = base_idx_i + local_x
            global_y = base_idx_j + local_y
            
            if edge_count + len(global_x) * 2 >= len(rows):
                new_size = max(len(rows) * 2, edge_count + len(global_x) * 2 + 1000)
                rows = np.resize(rows, new_size)
                cols = np.resize(cols, new_size)
            
            rows[edge_count:edge_count+len(global_x)] = global_x
            cols[edge_count:edge_count+len(global_x)] = global_y
            edge_count += len(global_x)
            
            rows[edge_count:edge_count+len(global_x)] = global_y
            cols[edge_count:edge_count+len(global_x)] = global_x
            edge_count += len(global_x)
            
            # Save checkpoint after each batch comparison
            checkpoint = {
                'rows': rows,
                'cols': cols,
                'edge_count': edge_count,
                'i': i,
                'j': j + 1  # Save the next j to process
            }
            dump(checkpoint, open(checkpoint_path, 'wb'))
            print(f"Saved checkpoint after comparing batch {i+1} with {j+1}")
        
        # Reset j counter for next batch
        start_j = 0
    
    # Trim arrays to actual size
    rows = rows[:edge_count]
    cols = cols[:edge_count]
    
    # Create sparse adjacency matrix 
    print("Building final sparse adjacency matrix...")
    not_distinct_sparse = sp.csr_matrix((np.ones(edge_count, dtype=np.int8), (rows, cols)), shape=(n, n))
    
    # Save matrix as checkpoint
    checkpoint = {
        'matrix': not_distinct_sparse,
        'dataset': dataset,
        'validation_fraction': validation_fraction,
        'test_fraction': test_fraction
    }
    dump(checkpoint, open(checkpoint_path.replace('.pkl', '_matrix.pkl'), 'wb'))
    print("Saved sparse matrix checkpoint")
    
    # Find connected components
    print("Finding connected components...")
    _, labels = connected_components(csgraph=not_distinct_sparse, directed=False, return_labels=True)
    
    # Use numpy for grouping indices by cluster
    unique_labels = np.unique(labels)
    cluster_list = [np.where(labels == label)[0].tolist() for label in unique_labels]
    
    # Shuffle clusters
    random.shuffle(cluster_list)
    
    # Calculate target sizes
    train_indices = []
    validation_indices = []
    test_indices = []

    # Calculate target sizes
    validation_size = int(n * validation_fraction)
    test_size = int(n * test_fraction)

    # Separate large clusters that should go to training
    # Use a threshold based on target sizes to identify "large" clusters
    large_cluster_threshold = max(validation_size, test_size) // 2
    large_clusters = [cluster for cluster in cluster_list if len(cluster) > large_cluster_threshold]
    small_clusters = [cluster for cluster in cluster_list if len(cluster) <= large_cluster_threshold]

    # Put large clusters in training set
    for cluster in large_clusters:
        train_indices.extend(cluster)

    # Round-robin distribution for smaller clusters
    current_set = 0  # 0: test, 1: validation, 2: training
    for cluster in small_clusters:
        cluster_size = len(cluster)
        
        if current_set == 0 and len(test_indices) + cluster_size <= test_size * 1.1:  # Allow 10% overflow
            test_indices.extend(cluster)
            current_set = 1
        elif current_set == 1 and len(validation_indices) + cluster_size <= validation_size * 1.1:  # Allow 10% overflow
            validation_indices.extend(cluster)
            current_set = 2
        elif current_set == 2:
            train_indices.extend(cluster)
            current_set = 0
        else:
            # If current set would overflow too much, try next set
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
    
    # Create the final datasets
    train_set = [dataset[i] for i in train_indices]
    validation_set = [dataset[i] for i in validation_indices]
    test_set = [dataset[i] for i in test_indices]
    
    max_cluster_size = max(len(cluster) for cluster in cluster_list)
    avg_cluster_size = n / len(cluster_list)
    min_cluster_size = min(len(cluster) for cluster in cluster_list)
    print(f"Max cluster size: {max_cluster_size}, Avg cluster size: {avg_cluster_size:.2f}, Min cluster size: {min_cluster_size}, num clusters: {len(cluster_list)}")
    # Remove checkpoint file if process completed successfully
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
    return train_set, validation_set, test_set

# how is mces calculation done?
# 1. we caclualte lower bounds usign comaprison of the environmet of each atom.
# 2. then we try to split the data set usign the given fractions and distinction threshold.
# if this fails, we lower the distinction threshold and try again, until we reach a point where we can split the dataset.


def _process_batch_bounds(args):
    """Helper function for ProcessPoolExecutor"""
    batch_i, batch_j, batch_pairs, base_idx_i, base_idx_j = args
    bounds_results = _calculate_bounds_batch(batch_i, batch_j, batch_pairs)
    # Convert to global indices
    global_results = []
    for local_i, local_j, bound in bounds_results:
        global_i = base_idx_i + local_i
        global_j = base_idx_j + local_j
        global_results.append((global_i, global_j, bound))
    return global_results

def split_dataset_adaptive_threshold(
    dataset: list[str],
    validation_fraction=0.1,
    test_fraction=0.1,
    batch_size=100,
    checkpoint_path="data/split_checkpoint_adaptive.pkl",
    initial_distinction_threshold: int = 10,
    min_distinction_threshold: int = 2,
    threshold_step: int = -1,
    min_ratio: float = 0.7,
    max_attempts: int = 10,
    n_workers: int = -1
) -> Tuple[list[str], list[str], list[str], int]:
    """
    Split dataset using precomputed lower bounds, adaptively lowering the threshold if needed.
    Only the lower bounds are used (no exact MCES).
    """

    n_workers = cpu_count()
    
    n = len(dataset)
    print("Calculating lower bounds matrix once...")

    # Create batches of molecules
    num_batches = (n + batch_size - 1) // batch_size
    batches = [dataset[i*batch_size:min((i+1)*batch_size, n)] for i in range(num_batches)]
    
    # Initialize the bounds matrix
    bounds_matrix = np.zeros((n, n), dtype=float)
    
    # Prepare all batch comparison tasks
    tasks = []
    
    # Internal batch comparisons (symmetric)
    for i in range(num_batches):
        batch_i = batches[i]
        base_idx_i = i * batch_size
        batch_pairs_internal = [(x, y) for x in range(len(batch_i)) for y in range(x, len(batch_i))]
        tasks.append((batch_i, batch_i, batch_pairs_internal, base_idx_i, base_idx_i))
    
    # Cross-batch comparisons
    for i in range(num_batches):
        for j in range(i + 1, num_batches):
            batch_i = batches[i]
            batch_j = batches[j]
            base_idx_i = i * batch_size
            base_idx_j = j * batch_size
            batch_pairs_cross = [(x, y) for x in range(len(batch_i)) for y in range(len(batch_j))]
            tasks.append((batch_i, batch_j, batch_pairs_cross, base_idx_i, base_idx_j))
    
    # Process all tasks in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_process_batch_bounds, tasks))
    
    # Fill the bounds matrix with results
    for task_results in results:
        for global_i, global_j, bound in task_results:
            bounds_matrix[global_i, global_j] = bound
            bounds_matrix[global_j, global_i] = bound  # symmetry

    np.fill_diagonal(bounds_matrix, 0)

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


if __name__ == "__main__":
    from time import perf_counter
    nist_smiles: List[str] = pl.scan_parquet('/home/analytit_admin/dev/MS_encoder/data/NIST_prepared_labeled.parquet').select('CanonicalSMILES').unique(maintain_order=True).collect().slice(offset=10000, length=1024).to_series().to_list()
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.join(os.path.dirname(__file__), "__pycache__"), exist_ok=True)

    start = perf_counter()
    train_set, validation_set, test_set, threshold = split_dataset_adaptive_threshold(
        nist_smiles,
        validation_fraction=0.1,
        test_fraction=0.1,
        batch_size=100,
        checkpoint_path=os.path.join(os.path.dirname(__file__), "__pycache__", "split_checkpoint_adaptive.pkl"),
        initial_distinction_threshold=10,
        min_distinction_threshold=2,
        threshold_step=-1,
        max_attempts=10
    )  # type: Tuple[List[str], List[str], List[str], int]
    end = perf_counter()
    adaptive_time = end - start
    print(f"Training set size: {len(train_set)}")
    print(f"Validation set size: {len(validation_set)}")
    print(f"Test set size: {len(test_set)}")
    print(f"Total size: {len(nist_smiles)}")
    print(f"Time taken: {adaptive_time:.2f} seconds")


    # now we do the same using the non adaptive, with the determined threshold
    start_time = perf_counter()
    train_set, validation_set, test_set = split_dataset(
        nist_smiles,
        validation_fraction=0.1,
        test_fraction=0.1,
        batch_size=100, 
        checkpoint_path=os.path.join(os.path.dirname(__file__), "__pycache__", "split_checkpoint.pkl"),
        distinction_threshold=threshold,
        use_solver=True 
    )  # type: Tuple[List[str], List[str], List[str]]
    end_time = perf_counter()
    non_adaptive_time = end_time - start_time
    print(f"Training set size: {len(train_set)}")
    print(f"Validation set size: {len(validation_set)}")
    print(f"Test set size: {len(test_set)}")
    print(f"Time taken: {non_adaptive_time:.2f} seconds")
    # and print the ratio of the two
    print(f"Adaptive time: {adaptive_time:.2f} seconds")
    print(f"Non-adaptive time: {non_adaptive_time:.2f} seconds")
    print(f"Speedup: {non_adaptive_time / adaptive_time:.2f}x")