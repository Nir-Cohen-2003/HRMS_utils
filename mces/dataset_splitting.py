import polars as pl
import numpy as np
from mces import are_very_distinct, suppress_output
from typing import Tuple
from time import time
import os
from scipy.sparse.csgraph import connected_components
import scipy.sparse as sp
import random
from pickle import dump, load

def split_dataset(dataset: list[str], validation_fraction=0.1, test_fraction=0.1, 
                  batch_size=100, checkpoint_path="data/split_checkpoint.pkl") -> Tuple[list[str], list[str], list[str]]:
    """Split the dataset into training, validation and testing datasets.
    Args:
        dataset (list[str]): List of SMILES strings.
        validation_fraction (float): Fraction of data to use for validation.
        test_fraction (float): Fraction of data to use for testing.
        batch_size (int): Size of each batch for processing.
        checkpoint_path (str): Path to save the checkpoint.
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
            print(f"Resuming from batch {start_i+1}, comparison {start_j+1}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting from scratch")
            start_i = 0
            start_j = 0
    
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
                same_batch_result = are_very_distinct(batch_i, symmetric=True)
            
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
            
            print(f"Comparing batch {i+1} with batch {j+1}/{num_batches} (sizes: {len(batch_i)}, {len(batch_j)})")
            start = time()
            
            with suppress_output():
                cross_batch_result = are_very_distinct(batch_i, batch_j, symmetric=False)
             
            print(f"compared batch {i+1} with batch {j+1}")
            print(f"Time taken: {time() - start:.2f} seconds")

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
    
    # Trim arrays to actual size used
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
    validation_size = int(n * validation_fraction)
    test_size = int(n * test_fraction)
    
    # Distribute clusters
    train_indices = []
    validation_indices = []
    test_indices = []
    
    for cluster in cluster_list:
        cluster_size = len(cluster)
        
        if len(validation_indices) < validation_size and (
            len(validation_indices) + cluster_size <= validation_size or
            abs(len(validation_indices) + cluster_size - validation_size) < 
            abs(len(validation_indices) - validation_size)
        ):
            validation_indices.extend(cluster)
        elif len(test_indices) < test_size and (
            len(test_indices) + cluster_size <= test_size or
            abs(len(test_indices) + cluster_size - test_size) < 
            abs(len(test_indices) - test_size)
        ):
            test_indices.extend(cluster)
        else:
            train_indices.extend(cluster)
    
    # Create the final datasets
    train_set = [dataset[i] for i in train_indices]
    validation_set = [dataset[i] for i in validation_indices]
    test_set = [dataset[i] for i in test_indices]
    
    # Remove checkpoint file if process completed successfully
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
    return train_set, validation_set, test_set


if __name__ == "__main__":
    dataset = pl.read_parquet('data/NIST_prepared.parquet').select('CanonicalSMILES').unique().to_series().to_list() #TODO: add sorting to ensure reproducibility
    start = time()
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    train_set, validation_set, test_set = split_dataset(dataset, batch_size=5000, checkpoint_path='data/split_checkpoint.pkl')
    
    dump(train_set, open('data/train_set_smiles.pickle', 'wb'))
    dump(validation_set, open('data/validation_set_smiles.pickle', 'wb'))
    dump(test_set, open('data/test_set_smiles.pickle', 'wb'))
    
    print(f"Training set size: {len(train_set)}")
    print(f"Validation set size: {len(validation_set)}")
    print(f"Test set size: {len(test_set)}")
    print(f"Total size: {len(dataset)}")
    print(f"Time taken: {time() - start:.2f} seconds")