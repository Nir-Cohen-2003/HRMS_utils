"""
Standalone test for fragmentation tree builder core logic.
Does not require hrms_utils to be installed.
"""

import numpy as np

# We need to test the numba functions. Since the module imports hrms_utils,
# we'll copy the core numba functions here for testing.

from numba import njit, prange


@njit(cache=True, fastmath=True)
def _is_superset(parent: np.ndarray, child: np.ndarray) -> bool:
    """Check if parent formula is a strict superset of child formula."""
    n = parent.shape[0]
    all_ge = True
    any_gt = False
    for i in range(n):
        if parent[i] < child[i]:
            all_ge = False
            break
        if parent[i] > child[i]:
            any_gt = True
    return all_ge and any_gt


@njit(cache=True, fastmath=True, parallel=True)
def _build_superset_matrix(formulas: np.ndarray) -> np.ndarray:
    """Build a boolean matrix where M[i, j] = True if formula[i] is superset of formula[j]."""
    n = formulas.shape[0]
    matrix = np.zeros((n, n), dtype=np.bool_)
    for i in prange(n):
        for j in range(n):
            if i == j:
                continue
            matrix[i, j] = _is_superset(formulas[i], formulas[j])
    return matrix


@njit(cache=True, fastmath=True)
def _truncate_msn(
    superset_matrix: np.ndarray,
    fragment_indices: np.ndarray,
    msn_precursor_idx: int,
) -> np.ndarray:
    """Truncate parent edges for fragments in an MSn spectrum."""
    n_all = superset_matrix.shape[0]
    for frag_idx in fragment_indices:
        for parent_idx in range(n_all):
            if not superset_matrix[parent_idx, frag_idx]:
                continue
            # The MSn precursor is always allowed as a parent
            if parent_idx == msn_precursor_idx:
                continue
            if not superset_matrix[msn_precursor_idx, parent_idx]:
                superset_matrix[parent_idx, frag_idx] = False
    return superset_matrix


@njit(cache=True, fastmath=True)
def _compute_edge_weights(superset_matrix: np.ndarray) -> np.ndarray:
    """Compute edge weights so incoming edges for each node sum to 1."""
    n = superset_matrix.shape[0]
    weights = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        parent_count = 0
        for i in range(n):
            if superset_matrix[i, j]:
                parent_count += 1
        if parent_count > 0:
            w = 1.0 / parent_count
            for i in range(n):
                if superset_matrix[i, j]:
                    weights[i, j] = w
    return weights


@njit(cache=True, fastmath=True)
def _drop_orphans(superset_matrix: np.ndarray, precursor_idx: int) -> np.ndarray:
    """Drop fragments with zero parents (except the molecular precursor)."""
    n = superset_matrix.shape[0]
    keep = np.ones(n, dtype=np.bool_)
    for j in range(n):
        if j == precursor_idx:
            continue
        has_parent = False
        for i in range(n):
            if superset_matrix[i, j]:
                has_parent = True
                break
        if not has_parent:
            keep[j] = False
    return keep


def test_is_superset():
    """Test the superset check."""
    # Element order: H, C, N, O, F, Na, P, S, Cl, K, Br, I, ...
    parent = np.array([12, 6, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)  # C6H12O
    child = np.array([8, 4, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)    # C4H8O
    not_child = np.array([12, 8, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)  # C8H12O

    assert _is_superset(parent, child)
    assert not _is_superset(child, parent)
    assert not _is_superset(parent, not_child)
    assert not _is_superset(parent, parent)  # strict superset
    print("test_is_superset PASSED")


def test_build_superset_matrix():
    """Test superset matrix construction."""
    # Element order: H, C, N, O, ...
    formulas = np.array([
        [12, 6, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # C6H12O
        [8, 4, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # C4H8O
        [4, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # C2H4O
        [12, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # C6H12
    ], dtype=np.int32)

    matrix = _build_superset_matrix(formulas)

    # C6H12O should be superset of C4H8O, C2H4O, C6H12
    assert matrix[0, 1]
    assert matrix[0, 2]
    assert matrix[0, 3]

    # C4H8O should be superset of C2H4O
    assert matrix[1, 2]

    # C6H12 should NOT be superset of C4H8O (missing O)
    assert not matrix[3, 1]

    print("test_build_superset_matrix PASSED")


def test_truncate_msn():
    """Test MSn truncation logic."""
    # Element order: H, C, N, O, ...
    # 5 fragments: precursor(A), B, C, D, E
    # A is superset of B, C, D, E
    # B is superset of D, E
    # C is superset of E
    formulas = np.array([
        [20, 10, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # A (precursor)
        [12, 6, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # B
        [10, 5, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # C
        [8, 4, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],    # D
        [6, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],    # E
    ], dtype=np.int32)

    matrix = _build_superset_matrix(formulas)

    # MSn spectrum with fragments D, E, and MSn precursor B
    # After truncation, D and E should only have B as parent (not A)
    frag_indices = np.array([3, 4], dtype=np.int32)  # D, E
    msn_precursor_idx = 1  # B

    truncated = _truncate_msn(matrix.copy(), frag_indices, msn_precursor_idx)

    # D should have B as parent, not A
    assert truncated[1, 3]  # B -> D
    assert not truncated[0, 3]  # A -> D (truncated)

    # E should have B and C as parents; C is a child of B (B is superset of C),
    # so C -> E is also kept
    assert truncated[1, 4]  # B -> E
    assert not truncated[0, 4]  # A -> E (truncated, A is not child of B)
    assert truncated[2, 4]  # C -> E (kept, C is child of B)

    print("test_truncate_msn PASSED")


def test_edge_weights():
    """Test edge weight computation."""
    # 3 fragments: A (precursor), B, C
    # A -> B, A -> C, B -> C
    matrix = np.array([
        [False, True, True],
        [False, False, True],
        [False, False, False],
    ], dtype=np.bool_)

    weights = _compute_edge_weights(matrix)

    # B has 1 parent (A), weight = 1.0
    assert weights[0, 1] == 1.0

    # C has 2 parents (A, B), weight = 0.5 each
    assert weights[0, 2] == 0.5
    assert weights[1, 2] == 0.5

    print("test_edge_weights PASSED")


def test_drop_orphans():
    """Test orphan dropping."""
    # 4 fragments: A (precursor), B (parent A), C (parent B), D (no parent)
    matrix = np.array([
        [False, True, False, False],
        [False, False, True, False],
        [False, False, False, False],
        [False, False, False, False],
    ], dtype=np.bool_)

    keep = _drop_orphans(matrix, 0)

    assert keep[0]  # precursor kept
    assert keep[1]  # B has parent A
    assert keep[2]  # C has parent B
    assert not keep[3]  # D is orphan, dropped

    print("test_drop_orphans PASSED")


def test_full_pipeline():
    """Test the full pipeline with a realistic example."""
    # Element order: H, C, N, O, ...
    # Molecular precursor: C6H12O6 (glucose-like)
    # MS2: fragments C4H8O4, C2H4O2
    # MS3 (precursor C4H8O4): fragment C2H4O2

    formulas = np.array([
        [12, 6, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # C6H12O6 (precursor)
        [8, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # C4H8O4
        [4, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # C2H4O2
    ], dtype=np.int32)

    # Build superset matrix
    matrix = _build_superset_matrix(formulas)

    # Check initial relationships
    assert matrix[0, 1]  # precursor -> C4H8O4
    assert matrix[0, 2]  # precursor -> C2H4O2
    assert matrix[1, 2]  # C4H8O4 -> C2H4O2

    # MS3 spectrum: fragments C2H4O2, MSn precursor is C4H8O4 (index 1)
    frag_indices = np.array([2], dtype=np.int32)
    msn_precursor_idx = 1

    # Truncate
    truncated = _truncate_msn(matrix.copy(), frag_indices, msn_precursor_idx)

    # C2H4O2 should only have C4H8O4 as parent now
    assert not truncated[0, 2]  # precursor -> C2H4O2 truncated
    assert truncated[1, 2]      # C4H8O4 -> C2H4O2 kept

    # Compute weights
    weights = _compute_edge_weights(truncated)

    # C4H8O4 has 1 parent (precursor), weight = 1.0
    assert weights[0, 1] == 1.0

    # C2H4O2 has 1 parent (C4H8O4), weight = 1.0
    assert weights[1, 2] == 1.0

    # Drop orphans
    keep = _drop_orphans(truncated, 0)
    assert keep[0] and keep[1] and keep[2]

    print("test_full_pipeline PASSED")


if __name__ == "__main__":
    test_is_superset()
    test_build_superset_matrix()
    test_truncate_msn()
    test_edge_weights()
    test_drop_orphans()
    test_full_pipeline()
    print("\nAll tests passed!")
