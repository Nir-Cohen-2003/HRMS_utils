"""
Test script for fragmentation tree builder.
Creates synthetic data and verifies the tree building logic.
"""

from pathlib import Path
import sys

# Ensure the workspace root is on path so imports work
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

import numpy as np
import polars as pl

from experiments.frag_trees.fragmentation_tree import (
    FragmentationTreeConfig,
    _build_superset_matrix,
    _compute_edge_weights,
    _drop_orphans_iterative,
    _formula_array_to_string,
    _is_superset,
    build_fragmentation_trees,
)


def test_is_superset():
    """Test the superset check."""
    # Element order: H, C, N, O, F, Na, P, S, Cl, K, Br, I, ...
    # Element order: H, C, N, O, F, Na, P, S, Cl, K, Br, I
    parent = np.array([12, 6, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)  # C6H12O
    child = np.array([8, 4, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)    # C4H8O
    not_child = np.array([12, 8, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)  # C8H12O

    assert _is_superset(parent, child)
    assert not _is_superset(child, parent)
    assert not _is_superset(parent, not_child)
    assert not _is_superset(parent, parent)  # strict superset
    print("test_is_superset PASSED")


def test_build_superset_matrix():
    """Test superset matrix construction."""
    # Element order: H, C, N, O, F, Na, P, S, Cl, K, Br, I
    formulas = np.array([
        [12, 6, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # C6H12O
        [8, 4, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],   # C4H8O
        [4, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],   # C2H4O
        [12, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # C6H12
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
    """Test orphan dropping using the iterative orphan dropper."""
    # 4 fragments: A (precursor), B (parent A), C (parent B), D (no parent)
    matrix = np.array([
        [False, True, False, False],
        [False, False, True, False],
        [False, False, False, False],
        [False, False, False, False],
    ], dtype=np.bool_)

    keep = _drop_orphans_iterative(matrix, 0)

    assert keep[0]  # precursor kept
    assert keep[1]  # B has parent A
    assert keep[2]  # C has parent B
    assert not keep[3]  # D is orphan, dropped

    print("test_drop_orphans PASSED")


def test_drop_orphans_iterative():
    """Test iterative orphan dropping."""
    # 4 fragments: A (precursor), B (parent A), C (parent B), D (parent C)
    # If we drop C, D should also be dropped
    matrix = np.array([
        [False, True, False, False],
        [False, False, True, False],
        [False, False, False, True],
        [False, False, False, False],
    ], dtype=np.bool_)

    # First, truncate B -> C (simulate MSn truncation removing B->C)
    matrix[1, 2] = False

    keep = _drop_orphans_iterative(matrix, 0)

    assert keep[0]  # precursor kept
    assert keep[1]  # B has parent A
    assert not keep[2]  # C is orphan (B->C truncated), dropped
    assert not keep[3]  # D's only parent C was dropped, so D is also dropped

    print("test_drop_orphans_iterative PASSED")


def test_formula_to_string():
    """Test formula array to string conversion."""
    # Element order: H, C, N, O, F, Na, P, S, Cl, K, Br, I
    formula = np.array([2, 6, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
    result = _formula_array_to_string(formula)
    assert result == "H2C6O", f"Expected H2C6O, got {result}"
    print("test_formula_to_string PASSED")


def _find_fragment_index(tree, formula_array: np.ndarray) -> int:
    """Find fragment index by formula array."""
    for i in range(tree.n_fragments):
        if np.array_equal(tree.fragment_formulas[i], formula_array):
            return i
    raise AssertionError(f"Fragment with formula {formula_array} not found in tree")


def test_build_tree_synthetic():
    """Test full tree building with synthetic data.

    Creates a simple MS2→MS3 case:
      - Molecular precursor: C6H12O6 ([M+H]+ = C6H13O6)
      - MS2 fragments: C4H8O4, C2H4O2
      - MS3 (precursor C4H8O4): fragment C2H4O2

    Expected edge structure (Rule 1 transitive reduction):
      [M+H]+ -> C4H8O4 (weight 1.0)
      C4H8O4 -> C2H4O2 (weight 1.0)
      [M+H]+ -> C2H4O2 (removed by Rule 1)
    """
    # Exact monoisotopic masses (12C: 12.000000, 1H: 1.007825, 16O: 15.994915)
    # C2H4O2: 60.02113, C4H8O4: 120.04226, C6H13O6: 181.07122
    frag2_formula = np.array([4, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)      # C2H4O2
    frag1_formula = np.array([8, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)      # C4H8O4
    ms2_precursor = np.array([13, 6, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)     # C6H13O6

    frag2_mz = 60.02113
    frag1_mz = 120.04226
    precursor_mz = 181.07122

    df = pl.DataFrame({
        "base_inchikey": ["TEST123", "TEST123"],
        "ion_mode": ["P", "P"],
        "precursor_type": ["[M+H]+", "[M+H]+"],
        "precursor_formula_array": [ms2_precursor.tolist(), ms2_precursor.tolist()],
        "precursor_mz": [precursor_mz, frag1_mz],
        "cleaned_normalized_mz": [
            [frag1_mz, frag2_mz],  # MS2: C4H8O4, C2H4O2
            [frag2_mz],             # MS3: C2H4O2
        ],
        "cleaned_normalized_intensity": [
            [100.0, 50.0],
            [75.0],
        ],
        "mslevel": [2, 3],
    })

    trees = build_fragmentation_trees(df)

    assert len(trees) == 1
    key = ("TEST123", "P", "[M+H]+")
    assert key in trees

    tree = trees[key]
    print(f"Tree has {tree.n_fragments} fragments")
    print(f"Fragment formulas: {tree.fragment_formulas_str}")

    # Should have 3 fragments: [M+H]+, C4H8O4, C2H4O2
    assert tree.n_fragments == 3, f"Expected 3 fragments, got {tree.n_fragments}"

    precursor_idx = tree.precursor_idx
    c4h8o4_idx = _find_fragment_index(tree, frag1_formula)
    c2h4o2_idx = _find_fragment_index(tree, frag2_formula)

    print(f"Precursor index: {precursor_idx}")
    print(f"C4H8O4 index: {c4h8o4_idx}")
    print(f"C2H4O2 index: {c2h4o2_idx}")

    # C4H8O4 should have 1 parent ([M+H]+)
    assert tree.edge_weights[precursor_idx, c4h8o4_idx] == 1.0, \
        f"Expected [M+H]+ -> C4H8O4 weight 1.0, got {tree.edge_weights[precursor_idx, c4h8o4_idx]}"

    # C2H4O2 should have 1 parent (C4H8O4) because Rule 1 (transitive reduction
    # with evidence) removes the direct [M+H]+ -> C2H4O2 edge
    assert tree.edge_weights[c4h8o4_idx, c2h4o2_idx] == 1.0, \
        f"Expected C4H8O4 -> C2H4O2 weight 1.0, got {tree.edge_weights[c4h8o4_idx, c2h4o2_idx]}"
    assert tree.edge_weights[precursor_idx, c2h4o2_idx] == 0.0, \
        f"Expected [M+H]+ -> C2H4O2 weight 0.0, got {tree.edge_weights[precursor_idx, c2h4o2_idx]}"

    print("test_build_tree_synthetic PASSED")


def test_build_tree_synthetic_with_config():
    """Test full tree building with synthetic data and explicit FragmentationTreeConfig.

    Same structure as test_build_tree_synthetic but passes a config explicitly.
    """
    frag2_formula = np.array([4, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)      # C2H4O2
    frag1_formula = np.array([8, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)      # C4H8O4
    ms2_precursor = np.array([13, 6, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)     # C6H13O6

    frag2_mz = 60.02113
    frag1_mz = 120.04226
    precursor_mz = 181.07122

    df = pl.DataFrame({
        "base_inchikey": ["TEST123", "TEST123"],
        "ion_mode": ["P", "P"],
        "precursor_type": ["[M+H]+", "[M+H]+"],
        "precursor_formula_array": [ms2_precursor.tolist(), ms2_precursor.tolist()],
        "precursor_mz": [precursor_mz, frag1_mz],
        "cleaned_normalized_mz": [
            [frag1_mz, frag2_mz],
            [frag2_mz],
        ],
        "cleaned_normalized_intensity": [
            [100.0, 50.0],
            [75.0],
        ],
        "mslevel": [2, 3],
    })

    config = FragmentationTreeConfig()
    trees = build_fragmentation_trees(df, config=config)

    assert len(trees) == 1
    key = ("TEST123", "P", "[M+H]+")
    assert key in trees

    tree = trees[key]
    print(f"Tree has {tree.n_fragments} fragments")
    print(f"Fragment formulas: {tree.fragment_formulas_str}")

    assert tree.n_fragments == 3, f"Expected 3 fragments, got {tree.n_fragments}"

    precursor_idx = tree.precursor_idx
    c4h8o4_idx = _find_fragment_index(tree, frag1_formula)
    c2h4o2_idx = _find_fragment_index(tree, frag2_formula)

    # C4H8O4 should have 1 parent ([M+H]+)
    assert tree.edge_weights[precursor_idx, c4h8o4_idx] == 1.0, \
        f"Expected [M+H]+ -> C4H8O4 weight 1.0, got {tree.edge_weights[precursor_idx, c4h8o4_idx]}"

    # C2H4O2 should have 1 parent (C4H8O4) via Rule 1 transitive reduction
    assert tree.edge_weights[c4h8o4_idx, c2h4o2_idx] == 1.0, \
        f"Expected C4H8O4 -> C2H4O2 weight 1.0, got {tree.edge_weights[c4h8o4_idx, c2h4o2_idx]}"
    assert tree.edge_weights[precursor_idx, c2h4o2_idx] == 0.0, \
        f"Expected [M+H]+ -> C2H4O2 weight 0.0, got {tree.edge_weights[precursor_idx, c2h4o2_idx]}"

    print("test_build_tree_synthetic_with_config PASSED")


if __name__ == "__main__":
    test_is_superset()
    test_build_superset_matrix()
    test_edge_weights()
    test_drop_orphans()
    test_drop_orphans_iterative()
    test_formula_to_string()
    test_build_tree_synthetic()
    test_build_tree_synthetic_with_config()
    print("\nAll tests passed!")
