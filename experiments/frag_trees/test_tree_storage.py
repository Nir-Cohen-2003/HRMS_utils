"""
Round-trip + key-matching smoke tests for tree_storage.py.

Implements the full test specification from plans/approved-plan.md §6 Phase 3.

Tests:
    test_roundtrip_full_tree_preserves_all_fields
    test_roundtrip_ms2_tree_preserves_all_fields
    test_single_node_tree_roundtrip
    test_empty_edges_tree_roundtrip
    test_coo_extraction_matches_dense
    test_align_keys_pairs_correctly
    test_align_keys_raises_on_mismatch
    test_key_order_is_canonical
    test_ms2_one_spectrum_per_tree_invariant_asserted

Run via:
    pixi run -e experiments python -m pytest experiments/frag_trees/test_tree_storage.py -v
  or (standalone):
    pixi run -e experiments python experiments/frag_trees/test_tree_storage.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure imports work — both the workspace root (for experiments.xxx absolute
# imports) and the sibling directory (for ms2_tree_builder's bare
# "from fragmentation_tree import ..." relative import).
# ---------------------------------------------------------------------------
_workspace_root = Path(__file__).parent.parent.parent
for _p in [str(_workspace_root), str(_workspace_root / "experiments/frag_trees")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import tempfile
import unittest

import numpy as np
import polars as pl

from experiments.frag_trees.fragmentation_tree import (
    FragmentationTree,
    FragmentationTreeConfig,
    build_fragmentation_trees,
)
from experiments.frag_trees.ms2_tree_builder import build_ms2_trees
from experiments.frag_trees.tree_storage import (
    TreeKey,
    TreeStorageConfig,
    _dense_edge_weights_to_coo,
    align_keys,
    load_tree_arrays_npz,
    load_trees_npz,
    save_trees_npz,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_NUM_ELEMENTS: int = 12

# Path to the real-world fixture parquet
_PARQUET_PATH: Path = _workspace_root / "cladribine.parquet"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_trees_equal(
    original: FragmentationTree,
    loaded: FragmentationTree,
    label: str = "",
) -> None:
    """Assert all fields of two FragmentationTree instances are equal."""
    # Base fields
    assert original.base_inchikey == loaded.base_inchikey, (
        f"{label}: base_inchikey mismatch: "
        f"'{original.base_inchikey}' vs '{loaded.base_inchikey}'"
    )
    assert original.ion_mode == loaded.ion_mode, (
        f"{label}: ion_mode mismatch: "
        f"'{original.ion_mode}' vs '{loaded.ion_mode}'"
    )
    assert original.n_fragments == loaded.n_fragments, (
        f"{label}: n_fragments mismatch: "
        f"{original.n_fragments} vs {loaded.n_fragments}"
    )

    # Precursor formula
    np.testing.assert_array_equal(
        original.precursor_formula,
        loaded.precursor_formula,
        err_msg=f"{label}: precursor_formula mismatch",
    )

    # Fragment formulas
    np.testing.assert_array_equal(
        original.fragment_formulas,
        loaded.fragment_formulas,
        err_msg=f"{label}: fragment_formulas mismatch",
    )

    # Fragment formula strings
    assert original.fragment_formulas_str == loaded.fragment_formulas_str, (
        f"{label}: fragment_formulas_str mismatch"
    )

    # Edge weights (dense)
    np.testing.assert_array_equal(
        original.edge_weights,
        loaded.edge_weights,
        err_msg=f"{label}: edge_weights mismatch",
    )

    # Spectrum fragments (list of arrays)
    assert len(original.spectrum_fragments) == len(loaded.spectrum_fragments), (
        f"{label}: spectrum_fragments length mismatch: "
        f"{len(original.spectrum_fragments)} vs {len(loaded.spectrum_fragments)}"
    )
    for idx, (orig_arr, loaded_arr) in enumerate(
        zip(original.spectrum_fragments, loaded.spectrum_fragments)
    ):
        np.testing.assert_array_equal(
            orig_arr,
            loaded_arr,
            err_msg=f"{label}: spectrum_fragments[{idx}] mismatch",
        )

    # Spectrum mslevels
    np.testing.assert_array_equal(
        original.spectrum_mslevels,
        loaded.spectrum_mslevels,
        err_msg=f"{label}: spectrum_mslevels mismatch",
    )

    # Spectrum msn precursors
    assert original.spectrum_msn_precursors == loaded.spectrum_msn_precursors, (
        f"{label}: spectrum_msn_precursors mismatch"
    )

    # Fragment errors ppm
    np.testing.assert_array_equal(
        original.fragment_errors_ppm,
        loaded.fragment_errors_ppm,
        err_msg=f"{label}: fragment_errors_ppm mismatch",
    )


def _build_hand_tree(
    n_fragments: int,
    base_inchikey: str = "HAND",
    ion_mode: str = "P",
    precursor_type: str = "[M+H]+",
    has_edges: bool = True,
) -> tuple[tuple[str, str, str], FragmentationTree]:
    """Build a hand-crafted FragmentationTree for synthetic tests.

    Creates n_fragments fragments using controlled formulas. The first fragment
    is the molecular precursor (largest formula). If has_edges is True, creates
    a simple chain: 0 -> 1, 1 -> 2, etc. Otherwise, edge_weights is all-zero.

    Element order: H, C, N, O, F, Na, P, S, Cl, K, Br, I (12 elements).
    """
    # Build increasingly smaller formulas so the first is the precursor
    formulas_list: list[np.ndarray] = []
    for i in range(n_fragments):
        c = 6 - i
        h = 12 - 2 * i
        o = 6 - i
        # Ensure no negative or zero-element counts crash validation
        c = max(c, 2)
        h = max(h, 4)
        o = max(o, 1)
        arr = np.zeros(_NUM_ELEMENTS, dtype=np.int32)
        arr[0] = h  # H
        arr[1] = c  # C
        arr[2] = 0  # N
        arr[3] = o  # O
        formulas_list.append(arr)

    fragment_formulas = np.stack(formulas_list, axis=0)
    fragment_formulas_str = [f"Frag{i}" for i in range(n_fragments)]

    if has_edges:
        # Simple chain: 0->1, 1->2, ...
        ew = np.zeros((n_fragments, n_fragments), dtype=np.float64)
        for i in range(n_fragments - 1):
            ew[i, i + 1] = 1.0
        edge_weights = ew
    else:
        edge_weights = np.zeros((n_fragments, n_fragments), dtype=np.float64)

    key = (base_inchikey, ion_mode, precursor_type)
    tree = FragmentationTree(
        base_inchikey=base_inchikey,
        ion_mode=ion_mode,
        precursor_formula=formulas_list[0].copy(),
        fragment_formulas=fragment_formulas,
        fragment_formulas_str=fragment_formulas_str,
        edge_weights=edge_weights,
        spectrum_fragments=[np.arange(n_fragments, dtype=np.int32)],
        spectrum_mslevels=np.array([2], dtype=np.int32),
        spectrum_msn_precursors=[-1],
        fragment_errors_ppm=np.zeros(n_fragments, dtype=np.float64),
    )
    return key, tree


# ===========================================================================
# Tests
# ===========================================================================


def test_roundtrip_full_tree_preserves_all_fields() -> None:
    """Build a full tree from cladribine.parquet, save, load, assert all fields equal."""
    df = pl.read_parquet(_PARQUET_PATH)
    config = FragmentationTreeConfig()
    original_trees = build_fragmentation_trees(df, config)
    assert len(original_trees) > 0, "Expected at least one full tree"

    # Save and load
    storage_config = TreeStorageConfig(tree_type="full", include_formula_strings=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "full_trees.npz"
        save_trees_npz(original_trees, out_path, storage_config)
        loaded_trees = load_trees_npz(out_path)

    assert len(original_trees) == len(loaded_trees), (
        f"Tree count mismatch: {len(original_trees)} vs {len(loaded_trees)}"
    )
    assert set(original_trees.keys()) == set(loaded_trees.keys()), (
        f"Key set mismatch"
    )

    for key in original_trees:
        _assert_trees_equal(
            original_trees[key], loaded_trees[key],
            label=f"full tree {key}",
        )


def test_roundtrip_ms2_tree_preserves_all_fields() -> None:
    """Build an MS2 tree from cladribine.parquet, save, load, assert all fields equal.

    MS2 trees have trivial spectrum metadata synthesized on load — this test
    confirms the synthesis produces the same spectrum fields as the original.
    """
    df = pl.read_parquet(_PARQUET_PATH)
    config = FragmentationTreeConfig()
    original_trees = build_ms2_trees(df, config)
    assert len(original_trees) > 0, "Expected at least one MS2 tree"

    storage_config = TreeStorageConfig(tree_type="ms2", include_formula_strings=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "ms2_trees.npz"
        save_trees_npz(original_trees, out_path, storage_config)
        loaded_trees = load_trees_npz(out_path)

    assert len(original_trees) == len(loaded_trees), (
        f"Tree count mismatch: {len(original_trees)} vs {len(loaded_trees)}"
    )
    assert set(original_trees.keys()) == set(loaded_trees.keys()), (
        f"Key set mismatch"
    )

    for key in original_trees:
        _assert_trees_equal(
            original_trees[key], loaded_trees[key],
            label=f"ms2 tree {key}",
        )


def test_single_node_tree_roundtrip() -> None:
    """A tree with n=1 (precursor only) and e=0 round-trips correctly.

    Verifies:
      - node_features: (1, 12)
      - edge_index: (2, 0) empty
      - edge_weights: (0,) empty
      - fragment_errors_ppm: 1 value
      - CSR offsets: node span 1, edge span 0
    """
    key, tree = _build_hand_tree(
        n_fragments=1,
        base_inchikey="SINGLE_NODE",
        has_edges=False,
    )
    original_trees = {key: tree}

    storage_config = TreeStorageConfig(tree_type="ms2")
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "single_node.npz"
        save_trees_npz(original_trees, out_path, storage_config)

        # Check raw arrays
        arrays = load_tree_arrays_npz(out_path)
        assert arrays.node_features.shape == (1, _NUM_ELEMENTS), (
            f"node_features shape {arrays.node_features.shape} != (1, {_NUM_ELEMENTS})"
        )
        assert arrays.edge_index.shape == (2, 0), (
            f"edge_index shape {arrays.edge_index.shape} != (2, 0)"
        )
        assert arrays.edge_weights.shape == (0,), (
            f"edge_weights shape {arrays.edge_weights.shape} != (0,)"
        )
        assert arrays.fragment_errors_ppm.shape == (1,), (
            f"fragment_errors_ppm shape {arrays.fragment_errors_ppm.shape} != (1,)"
        )
        # CSR offsets: one tree, 1 node, 0 edges
        assert arrays.node_offsets[0] == 0
        assert arrays.node_offsets[1] == 1
        assert arrays.edge_offsets[0] == 0
        assert arrays.edge_offsets[1] == 0

        # Round-trip test
        loaded_trees = load_trees_npz(out_path)

    assert key in loaded_trees, "Loaded trees missing key"
    _assert_trees_equal(tree, loaded_trees[key], label="single_node")


def test_empty_edges_tree_roundtrip() -> None:
    """A multi-node tree (n>1) with no edges (e=0) round-trips correctly.

    Confirms the (2, 0) edge_index / (0,) edge_weights shapes with n>1,
    distinct from the single-node case. Verifies save-time range validation's
    ``size > 0`` guard does not spuriously fire on empty edges.
    """
    key, tree = _build_hand_tree(
        n_fragments=3,
        base_inchikey="EMPTY_EDGES",
        has_edges=False,
    )
    original_trees = {key: tree}

    storage_config = TreeStorageConfig(tree_type="ms2")
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "empty_edges.npz"
        save_trees_npz(original_trees, out_path, storage_config)

        arrays = load_tree_arrays_npz(out_path)
        assert arrays.node_features.shape[0] == 3, (
            f"Expected 3 nodes, got {arrays.node_features.shape[0]}"
        )
        assert arrays.edge_index.shape == (2, 0), (
            f"edge_index shape {arrays.edge_index.shape} != (2, 0)"
        )
        assert arrays.edge_weights.shape == (0,), (
            f"edge_weights shape {arrays.edge_weights.shape} != (0,)"
        )
        # CSR: one tree, 3 nodes, 0 edges
        assert arrays.node_offsets[0] == 0
        assert arrays.node_offsets[1] == 3
        assert arrays.edge_offsets[0] == 0
        assert arrays.edge_offsets[1] == 0, (
            f"Expected edge_offsets[1]=0, got {arrays.edge_offsets[1]}"
        )

        loaded_trees = load_trees_npz(out_path)

    assert key in loaded_trees, "Loaded trees missing key"
    _assert_trees_equal(tree, loaded_trees[key], label="empty_edges")


def test_coo_extraction_matches_dense() -> None:
    """Reconstructed dense matrix from COO equals the original edge_weights
    (exact float64 equality). Also asserts a known parent->child edge lands
    at [parent, child] (not [child, parent]).
    """
    rng = np.random.default_rng(42)
    n = 5

    # Build a known sparse edge weight matrix with varied weights
    ew = np.zeros((n, n), dtype=np.float64)
    # Parent -> child: 0->1, 0->2, 0->3, 1->2, 2->3, 2->4
    ew[0, 1] = 1.0
    ew[0, 2] = 0.5
    ew[0, 3] = 0.33
    ew[1, 2] = 1.0
    ew[2, 3] = 0.5
    ew[2, 4] = 0.25

    # Extract COO
    edge_index, edge_weights_flat = _dense_edge_weights_to_coo(ew)

    # Reconstruct dense
    ew_reconstructed = np.zeros((n, n), dtype=np.float64)
    for k in range(edge_index.shape[1]):
        r = edge_index[0, k]
        c = edge_index[1, k]
        ew_reconstructed[r, c] = edge_weights_flat[k]

    # Exact float64 equality
    np.testing.assert_array_equal(
        ew_reconstructed,
        ew,
        err_msg="Reconstructed dense matrix does not match original",
    )

    # Direction check: known parent->child edge 0->1 must be at [0, 1], not [1, 0]
    assert ew_reconstructed[0, 1] == 1.0, (
        "Parent->child edge 0->1 not found at [0, 1]"
    )
    assert ew_reconstructed[1, 0] == 0.0, (
        "Unexpected edge at [1, 0] — direction may be reversed"
    )
    assert ew_reconstructed[2, 3] == 0.5, (
        "Parent->child edge 2->3 not found at [2, 3]"
    )

    # Also verify that the all-zero case works (single-node fallback)
    zero_ew = np.zeros((1, 1), dtype=np.float64)
    zi, zw = _dense_edge_weights_to_coo(zero_ew)
    assert zi.shape == (2, 0), (
        f"Zero matrix edge_index shape {zi.shape} != (2, 0)"
    )
    assert zw.shape == (0,), (
        f"Zero matrix edge_weights shape {zw.shape} != (0,)"
    )


def test_align_keys_pairs_correctly() -> None:
    """Two lists with same keys in different orders pair correctly."""
    ms2_keys = [
        TreeKey("A", "P", "[M+H]+"),
        TreeKey("B", "N", "[M-H]-"),
        TreeKey("C", "P", "[M+Na]+"),
    ]
    full_keys = [
        TreeKey("C", "P", "[M+Na]+"),
        TreeKey("A", "P", "[M+H]+"),
        TreeKey("B", "N", "[M-H]-"),
    ]

    pairs = align_keys(ms2_keys, full_keys, fail_on_mismatch=True)

    # Expected: sorted by canonical order (base_inchikey, ion_mode, precursor_type)
    # A/P/[M+H]+: ms2_idx=0, full_idx=1
    # B/N/[M-H]-: ms2_idx=1, full_idx=2
    # C/P/[M+Na]+: ms2_idx=2, full_idx=0
    expected = [(0, 1), (1, 2), (2, 0)]
    assert pairs == expected, (
        f"Expected {expected}, got {pairs}"
    )

    # Also test with TreeKey instances reversed (full_keys not just reordered)
    ms2_keys_2 = [
        TreeKey("X", "P", "[M+H]+"),
        TreeKey("Y", "P", "[M+H]+"),
    ]
    full_keys_2 = [
        TreeKey("Y", "P", "[M+H]+"),
        TreeKey("X", "P", "[M+H]+"),
    ]
    pairs_2 = align_keys(ms2_keys_2, full_keys_2)
    # X/P/[M+H]+: ms2_idx=0, full_idx=1
    # Y/P/[M+H]+: ms2_idx=1, full_idx=0
    assert pairs_2 == [(0, 1), (1, 0)], (
        f"Expected [(0, 1), (1, 0)], got {pairs_2}"
    )


def test_align_keys_raises_on_mismatch() -> None:
    """Missing key in one file raises AssertionError (fail-fast)."""
    # MS2 has a key that full does not
    ms2_keys = [
        TreeKey("A", "P", "[M+H]+"),
        TreeKey("B", "N", "[M-H]-"),
    ]
    full_keys = [
        TreeKey("A", "P", "[M+H]+"),
    ]
    with _assert_raises(AssertionError):
        align_keys(ms2_keys, full_keys, fail_on_mismatch=True)

    # Full has a key that MS2 does not
    ms2_keys_2 = [
        TreeKey("A", "P", "[M+H]+"),
    ]
    full_keys_2 = [
        TreeKey("A", "P", "[M+H]+"),
        TreeKey("C", "P", "[M+Na]+"),
    ]
    with _assert_raises(AssertionError):
        align_keys(ms2_keys_2, full_keys_2, fail_on_mismatch=True)

    # Both have unique keys (no overlap)
    ms2_keys_3 = [
        TreeKey("A", "P", "[M+H]+"),
    ]
    full_keys_3 = [
        TreeKey("B", "N", "[M-H]-"),
    ]
    with _assert_raises(AssertionError):
        align_keys(ms2_keys_3, full_keys_3, fail_on_mismatch=True)

    # Same keys should NOT raise
    ms2_keys_4 = [
        TreeKey("A", "P", "[M+H]+"),
    ]
    full_keys_4 = [
        TreeKey("A", "P", "[M+H]+"),
    ]
    # Should not raise
    pairs = align_keys(ms2_keys_4, full_keys_4, fail_on_mismatch=True)
    assert pairs == [(0, 0)], f"Expected [(0, 0)], got {pairs}"

    # When fail_on_mismatch=False, mismatches are silently dropped
    pairs_silent = align_keys(
        [TreeKey("A", "P", "[M+H]+"), TreeKey("B", "N", "[M-H]-")],
        [TreeKey("A", "P", "[M+H]+")],
        fail_on_mismatch=False,
    )
    assert pairs_silent == [(0, 0)], (
        f"Expected [(0, 0)] with silent drop, got {pairs_silent}"
    )


def test_key_order_is_canonical() -> None:
    """Saved keys are sorted by (base_inchikey, ion_mode, precursor_type).

    Trees are provided in unsorted order; after save, the arrays must be
    sorted by the canonical key tuple.
    """
    # Build three trees with deliberately unsorted keys
    trees: dict[tuple[str, str, str], FragmentationTree] = {}
    keys_unsorted = [
        ("C", "P", "[M+Na]+"),
        ("A", "N", "[M-H]-"),
        ("A", "P", "[M+H]+"),
        ("B", "P", "[M+H]+"),
    ]
    for key in keys_unsorted:
        _, tree = _build_hand_tree(
            n_fragments=2, base_inchikey=key[0],
            ion_mode=key[1], precursor_type=key[2],
        )
        trees[key] = tree

    storage_config = TreeStorageConfig(tree_type="ms2")
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "sorted_keys.npz"
        save_trees_npz(trees, out_path, storage_config)
        arrays = load_tree_arrays_npz(out_path)

    n = len(keys_unsorted)
    base_inchikey_list = [str(arrays.base_inchikey[i]) for i in range(n)]
    ion_mode_list = [str(arrays.ion_mode[i]) for i in range(n)]
    precursor_type_list = [str(arrays.precursor_type[i]) for i in range(n)]

    # Build tuples from arrays
    saved_keys = list(zip(base_inchikey_list, ion_mode_list, precursor_type_list))

    # Expected sorted order
    expected_sorted = sorted(keys_unsorted, key=lambda x: (x[0], x[1], x[2]))

    assert saved_keys == expected_sorted, (
        f"Saved keys not in canonical order.\n"
        f"  Expected: {expected_sorted}\n"
        f"  Got:      {saved_keys}"
    )


def test_ms2_one_spectrum_per_tree_invariant_asserted() -> None:
    """``load_trees_npz`` on a well-formed MS2 file synthesizes exactly one
    spectrum per tree and the invariant assertion passes.  (A negative variant
    is not constructible from the public API since the MS2 builder always emits
    one spectrum; this test guards the load path.)
    """
    df = pl.read_parquet(_PARQUET_PATH)
    config = FragmentationTreeConfig()
    original_trees = build_ms2_trees(df, config)
    assert len(original_trees) > 0, "Expected at least one MS2 tree"

    storage_config = TreeStorageConfig(tree_type="ms2")
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "ms2_invariant.npz"
        save_trees_npz(original_trees, out_path, storage_config)

        # This should NOT raise because the file is well-formed
        loaded_trees = load_trees_npz(out_path)

    # Confirm that every loaded tree has exactly one spectrum
    for key, tree in loaded_trees.items():
        n_spectra = len(tree.spectrum_fragments)
        assert n_spectra == 1, (
            f"MS2 tree {key}: expected 1 spectrum, got {n_spectra}"
        )
        np.testing.assert_array_equal(
            tree.spectrum_mslevels, np.array([2], dtype=np.int32),
            err_msg=f"MS2 tree {key}: spectrum_mslevels mismatch",
        )
        assert tree.spectrum_msn_precursors == [-1], (
            f"MS2 tree {key}: spectrum_msn_precursors expected [-1], "
            f"got {tree.spectrum_msn_precursors}"
        )


# ---------------------------------------------------------------------------
# Context-manager helper for "assert raises" (no pytest dependency)
# ---------------------------------------------------------------------------


class _assert_raises:
    """Stand-in for ``pytest.raises`` so tests work without pytest installed."""

    def __init__(self, exc_type: type[BaseException]) -> None:
        self._exc_type = exc_type

    def __enter__(self) -> "_assert_raises":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> bool:
        if exc_type is None:
            raise AssertionError(
                f"Expected {self._exc_type.__name__} but no exception was raised"
            )
        if not issubclass(exc_type, self._exc_type):
            # Re-raise unexpected exceptions
            return False
        return True  # Suppress the expected exception


# ===========================================================================
# Main — standalone runner (works without pytest)
# ===========================================================================

if __name__ == "__main__":
    # Run all test functions in order
    test_fns = [
        ("test_roundtrip_full_tree_preserves_all_fields", test_roundtrip_full_tree_preserves_all_fields),
        ("test_roundtrip_ms2_tree_preserves_all_fields", test_roundtrip_ms2_tree_preserves_all_fields),
        ("test_single_node_tree_roundtrip", test_single_node_tree_roundtrip),
        ("test_empty_edges_tree_roundtrip", test_empty_edges_tree_roundtrip),
        ("test_coo_extraction_matches_dense", test_coo_extraction_matches_dense),
        ("test_align_keys_pairs_correctly", test_align_keys_pairs_correctly),
        ("test_align_keys_raises_on_mismatch", test_align_keys_raises_on_mismatch),
        ("test_key_order_is_canonical", test_key_order_is_canonical),
        ("test_ms2_one_spectrum_per_tree_invariant_asserted", test_ms2_one_spectrum_per_tree_invariant_asserted),
    ]

    n_pass = 0
    n_fail = 0
    for name, fn in test_fns:
        try:
            fn()
            print(f"  PASS  {name}")
            n_pass += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            import traceback
            traceback.print_exc()
            n_fail += 1

    print(f"\n{'=' * 40}")
    print(f"  {n_pass} passed, {n_fail} failed, {len(test_fns)} total")
    if n_fail > 0:
        sys.exit(1)
