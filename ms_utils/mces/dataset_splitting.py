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


if __name__ == "__main__":
    from time import perf_counter
    nist_smiles: List[str] = pl.scan_parquet('/home/analytit_admin/dev/MS_encoder/data/NIST_prepared_labeled.parquet').select('CanonicalSMILES').unique(maintain_order=True).collect().to_series().to_list()
    
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
    # now write the sets to files named train_set.parquet, validation_set.parquet, test_set.parquet
    pl.DataFrame({"CanonicalSMILES": train_set}).write_parquet(os.path.join(os.path.dirname(__file__), "__pycache__", "train_set.parquet"))
    pl.DataFrame({"CanonicalSMILES": validation_set}).write_parquet(os.path.join(os.path.dirname(__file__), "__pycache__", "validation_set.parquet"))
    pl.DataFrame({"CanonicalSMILES": test_set}).write_parquet(os.path.join(os.path.dirname(__file__), "__pycache__", "test_set.parquet"))
    print(f"Training set size: {len(train_set)}")
    print(f"Validation set size: {len(validation_set)}")
    print(f"Test set size: {len(test_set)}")
    print(f"Total size: {len(nist_smiles)}")
    print(f"Time taken: {adaptive_time:.2f} seconds")
