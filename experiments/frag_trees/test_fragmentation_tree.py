"""
Test script for fragmentation tree builder.
Creates synthetic data and verifies the tree building logic.
"""

import numpy as np
import polars as pl

from experiments.frag_trees.fragmentation_tree import (
    _build_superset_matrix,
    _compute_edge_weights,
    _drop_orphans,
    _drop_orphans_iterative,
    _formula_array_to_string,
    _is_superset,
    _truncate_msn,
    build_fragmentation_trees,
)


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

    print("test_truncate_mSN PASSED")


def test_truncate_msn_precursor_not_truncated():
    """Test that the MSn precursor itself is NOT truncated."""
    # 3 fragments: A (molecular precursor), B (MS3 precursor), C (MS3 fragment)
    # A -> B, A -> C, B -> C
    formulas = np.array([
        [12, 6, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # A
        [8, 4, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],    # B
        [4, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],    # C
    ], dtype=np.int32)

    matrix = _build_superset_matrix(formulas)

    # MS3 spectrum: fragment C, precursor B
    # If we ONLY pass C as the fragment (not B), B should keep A as parent
    frag_indices = np.array([2], dtype=np.int32)  # Only C
    msn_precursor_idx = 1  # B

    truncated = _truncate_msn(matrix.copy(), frag_indices, msn_precursor_idx)

    # C should only have B as parent
    assert not truncated[0, 2]  # A -> C truncated
    assert truncated[1, 2]      # B -> C kept

    # B should STILL have A as parent (B was NOT in frag_indices)
    assert truncated[0, 1]  # A -> B kept

    print("test_truncate_msn_precursor_not_truncated PASSED")


def test_truncate_msn_precursor_as_peak():
    """Test that the MSn precursor is NOT truncated even if it appears as a peak."""
    # 3 fragments: A (molecular precursor), B (MS3 precursor), C (MS3 fragment)
    # A -> B, A -> C, B -> C
    formulas = np.array([
        [12, 6, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # A
        [8, 4, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],    # B
        [4, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],    # C
    ], dtype=np.int32)

    matrix = _build_superset_matrix(formulas)

    # MS3 spectrum: fragments B and C, precursor B
    # B appears as both a peak and the precursor - it should NOT be truncated
    frag_indices = np.array([1, 2], dtype=np.int32)  # B and C
    msn_precursor_idx = 1  # B

    truncated = _truncate_msn(matrix.copy(), frag_indices, msn_precursor_idx)

    # B should STILL have A as parent (B is the precursor, not truncated)
    assert truncated[0, 1]  # A -> B kept

    # C should only have B as parent
    assert not truncated[0, 2]  # A -> C truncated
    assert truncated[1, 2]      # B -> C kept

    print("test_truncate_msn_precursor_as_peak PASSED")


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


def test_msn_precursor_not_orphan():
    """Test that MSn precursors are NOT dropped as orphans.

    An MS3 precursor is a fragment from MS2. It may not appear as a fragment
    peak in any MS2 spectrum, but it IS a sub-formula of the molecular precursor,
    so the molecular precursor is its parent. It should NOT be dropped.
    """
    # 3 fragments: A (molecular precursor), B (MS3 precursor), C (MS3 fragment)
    # A -> B, A -> C, B -> C
    matrix = np.array([
        [False, True, True],
        [False, False, True],
        [False, False, False],
    ], dtype=np.bool_)

    # After MS3 truncation (C's parents must be children of B):
    # A -> C is truncated, only B -> C remains
    frag_indices = np.array([2], dtype=np.int32)
    msn_precursor_idx = 1
    truncated = _truncate_msn(matrix.copy(), frag_indices, msn_precursor_idx)

    # B should still have A as parent (B was not in frag_indices)
    assert truncated[0, 1]

    # Drop orphans with A as protected precursor
    keep = _drop_orphans_iterative(truncated, 0)

    assert keep[0]  # A kept
    assert keep[1]  # B has parent A, NOT an orphan
    assert keep[2]  # C has parent B, NOT an orphan

    print("test_msn_precursor_not_orphan PASSED")


def test_build_tree_synthetic():
    """Test full tree building with synthetic data."""
    # Create a synthetic DataFrame
    # Compound: C6H12O6 (glucose-like), adduct [M+H]+
    # MS2 spectrum: fragments C4H8O4, C2H4O2
    # MS3 spectrum (precursor C4H8O4): fragment C2H4O2

    # Element order: H, C, N, O, ...
    molecular_formula = [12, 6, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # C6H12O6
    ms2_precursor = [13, 6, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]     # [M+H]+ = C6H13O6
    frag1 = [8, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]              # C4H8O4
    frag2 = [4, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]              # C2H4O2
    ms3_precursor = [8, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]       # C4H8O4 (same as frag1)

    df = pl.DataFrame({
        "base_inchikey": ["TEST123", "TEST123"],
        "ion_mode": ["P", "P"],
        "precursor_type": ["[M+H]+", "[M+H]+"],
        "molecular_formula_array": [molecular_formula, molecular_formula],
        "precursor_formula_array": [ms2_precursor, ms3_precursor],
        "cleaned_fragment_formulas": [
            [frag1, frag2],
            [frag2],
        ],
        "cleaned_fragment_formulas_str": [
            ["C4H8O4", "C2H4O2"],
            ["C2H4O2"],
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
    assert tree.n_fragments == 3

    # After MS3 truncation:
    # C2H4O2 in MS3 should only have C4H8O4 as parent (not [M+H]+)
    # C2H4O2 in MS2 should have both [M+H]+ and C4H8O4 as parents
    # But since we combine all spectra and then truncate, the final graph should have:
    # [M+H]+ -> C4H8O4 (weight 1.0)
    # C4H8O4 -> C2H4O2 (weight 1.0, because MS3 truncated [M+H]+ -> C2H4O2)

    # Check edge weights
    precursor_idx = tree.precursor_idx
    print(f"Precursor index: {precursor_idx}")

    # Find indices
    c4h8o4_idx = tree.fragment_formulas_str.index("C4H8O4")
    c2h4o2_idx = tree.fragment_formulas_str.index("C2H4O2")

    # C4H8O4 should have 1 parent ([M+H]+)
    assert tree.edge_weights[precursor_idx, c4h8o4_idx] == 1.0

    # C2H4O2 should have 1 parent (C4H8O4) because MS3 truncated the [M+H]+ edge
    assert tree.edge_weights[c4h8o4_idx, c2h4o2_idx] == 1.0
    assert tree.edge_weights[precursor_idx, c2h4o2_idx] == 0.0

    print("test_build_tree_synthetic PASSED")


def test_formula_to_string():
    """Test formula array to string conversion."""
    # Element order: H, C, N, O, ... (first 4 elements)
    formula = np.array([2, 6, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
    result = _formula_array_to_string(formula)
    assert result == "H2C6O", f"Expected H2C6O, got {result}"
    print("test_formula_to_string PASSED")


if __name__ == "__main__":
    test_is_superset()
    test_build_superset_matrix()
    test_truncate_msn()
    test_truncate_msn_precursor_not_truncated()
    test_edge_weights()
    test_drop_orphans()
    test_drop_orphans_iterative()
    test_msn_precursor_not_orphan()
    test_formula_to_string()
    test_build_tree_synthetic()
    print("\nAll tests passed!")
