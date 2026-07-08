"""
Disk-efficient storage format for fragmentation trees using collated NPZ.

Saves/loads collections of FragmentationTree objects to/from a single
compressed .npz file with CSR-offset indexing. See tree_format_spec.md
for the full format specification.

Dependencies:
    - numpy (only)
    - FragmentationTree from experiments.frag_trees.fragmentation_tree
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fragmentation_tree import FragmentationTree


# Number of elements in a formula array: H, C, N, O, F, Na, P, S, Cl, K, Br, I
_NUM_ELEMENTS: int = 12


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TreeStorageConfig:
    """Configuration for tree storage.

    Attributes:
        tree_type: "ms2" or "full".
        include_spectrum_metadata: If True (default), store per-spectrum MSn
            metadata for full trees. Only meaningful for tree_type == "full".
        include_formula_strings: If True (default), store fragment formula
            strings (debug/viz convenience). Derivable from node_features
            but stored for fast access.
        compress: If True (default), use np.savez_compressed.
        fail_on_key_mismatch: If True (default), raise when keys present
            in one file but missing in the other during align_keys.
    """
    tree_type: str
    include_spectrum_metadata: bool = True
    include_formula_strings: bool = True
    compress: bool = True
    fail_on_key_mismatch: bool = True


@dataclass(frozen=True)
class TreeKey:
    """Unique identifier for a fragmentation tree.

    Hashable (frozen=True) for dict/set use.
    """
    base_inchikey: str
    ion_mode: str
    precursor_type: str


@dataclass
class TreeArrays:
    """Raw collated arrays loaded from a tree NPZ file.

    No FragmentationTree reconstruction is performed — these are the
    flat concatenated arrays indexed by CSR offsets.
    """

    # Keys
    base_inchikey: np.ndarray            # (n_trees,) <U14
    ion_mode: np.ndarray                 # (n_trees,) <U1
    precursor_type: np.ndarray           # (n_trees,) <U16

    # Per-graph scalars
    precursor_formulas: np.ndarray       # (n_trees, _NUM_ELEMENTS) int32
    precursor_indices: np.ndarray        # (n_trees,) int32

    # CSR offsets
    node_offsets: np.ndarray             # (n_trees + 1,) int64
    edge_offsets: np.ndarray             # (n_trees + 1,) int64

    # Concatenated features
    node_features: np.ndarray            # (total_nodes, _NUM_ELEMENTS) int32
    fragment_errors_ppm: np.ndarray      # (total_nodes,) float64
    edge_index: np.ndarray               # (2, total_edges) int32
    edge_weights: np.ndarray             # (total_edges,) float64

    # Optional
    fragment_formulas_str: np.ndarray | None  # (total_nodes,) <U64

    # Full-tree only spectrum metadata
    spectrum_offsets: np.ndarray | None            # (n_trees + 1,) int64
    spectrum_fragments_offsets: np.ndarray | None   # (total_spectra + 1,) int64
    spectrum_fragments_flat: np.ndarray | None      # (total_fragment_refs,) int32
    spectrum_mslevels: np.ndarray | None            # (total_spectra,) int32
    spectrum_msn_precursors: np.ndarray | None      # (total_spectra,) int32


# ---------------------------------------------------------------------------
# Private helpers — standalone (no nested functions, per AGENTS.md)
# ---------------------------------------------------------------------------


def _dense_edge_weights_to_coo(
    edge_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert (n, n) dense edge weight matrix to sparse COO format.

    Uses np.nonzero, which on an all-zero (1, 1) matrix (single-node tree)
    correctly yields empty arrays — no special-casing needed.

    Args:
        edge_weights: Dense (n, n) float64 matrix.

    Returns:
        edge_index: shape (2, n_edges) int32 — row 0 = source (parent),
            row 1 = target (child).
        edge_weights_flat: shape (n_edges,) float64.
    """
    rows, cols = np.nonzero(edge_weights)
    edge_index = np.stack([rows, cols], axis=0).astype(np.int32)
    edge_weights_flat = edge_weights[rows, cols].astype(np.float64)
    return edge_index, edge_weights_flat


def _validate_edge_index(
    edge_index: np.ndarray,
    n_nodes: int,
    tree_key: tuple[str, str, str],
) -> None:
    """Validate that all edge_index entries are in [0, n_nodes).

    Uses a guard for empty edge_index (single-node or zero-edge trees)
    so that .min() / .max() on an empty array does not raise.

    Args:
        edge_index: shape (2, e) int32.
        n_nodes: Number of nodes in this graph.
        tree_key: The (base_inchikey, ion_mode, precursor_type) tuple.

    Raises:
        AssertionError: If any edge_index value is out of range.
    """
    if edge_index.size > 0:
        min_val = edge_index.min()
        max_val = edge_index.max()
        assert min_val >= 0 and max_val < n_nodes, (
            f"Tree {tree_key}: edge_index out of range [0, {n_nodes}); "
            f"got min={min_val}, max={max_val}"
        )


def _sorted_trees(
    trees: dict[tuple[str, str, str], FragmentationTree],
) -> list[tuple[tuple[str, str, str], FragmentationTree]]:
    """Return trees sorted by canonical ordering: (base_inchikey, ion_mode, precursor_type)."""
    return sorted(trees.items(), key=lambda item: item[0])


def _make_formula_strings_array(tree: FragmentationTree) -> np.ndarray:
    """Convert fragment_formulas_str list to a fixed-width <U64 array."""
    return np.array(tree.fragment_formulas_str, dtype="<U64")


def _make_ms2_spectrum_arrays(
    n_trees: int,
    node_offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthesize trivial spectrum metadata for MS2 trees.

    MS2 trees have exactly one spectrum covering all nodes of each tree.
    The spectrum_fragments_flat array stores LOCAL fragment indices per tree
    (0..n_nodes-1 for each tree), not global indices, so that the load-path
    reconstruction produces the same local-index spectrum_fragments that the
    original MS2 tree carried.

    Returns:
        spectrum_offsets: (n_trees + 1,) int64
        spectrum_fragments_offsets: (total_nodes + 1,) int64 — mirrors node_offsets
        spectrum_fragments_flat: (total_nodes,) int32 — per-tree local node indices
        spectrum_mslevels: (n_trees,) int32 — all 2
        spectrum_msn_precursors: (n_trees,) int32 — all -1
    """
    total_nodes = int(node_offsets[-1])
    spectrum_offsets = np.arange(n_trees + 1, dtype=np.int64)
    spectrum_fragments_offsets = node_offsets.copy()
    # Build per-tree local-index spans so each tree's spectrum has indices 0..n-1
    flat_parts: list[np.ndarray] = []
    for i in range(n_trees):
        n_nodes_i = int(node_offsets[i + 1] - node_offsets[i])
        flat_parts.append(np.arange(n_nodes_i, dtype=np.int32))
    spectrum_fragments_flat = np.concatenate(flat_parts) if flat_parts else np.array([], dtype=np.int32)
    spectrum_mslevels = np.full(n_trees, 2, dtype=np.int32)
    spectrum_msn_precursors = np.full(n_trees, -1, dtype=np.int32)
    return (
        spectrum_offsets,
        spectrum_fragments_offsets,
        spectrum_fragments_flat,
        spectrum_mslevels,
        spectrum_msn_precursors,
    )


def _collate_full_spectrum_arrays(
    trees_sorted: list[tuple[tuple[str, str, str], FragmentationTree]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collate per-spectrum MSn metadata from full trees into flat arrays.

    Builds the two-level CSR:
      - spectrum_offsets: per-tree cumulative spectrum counts.
      - spectrum_fragments_offsets: per-spectrum cumulative fragment-ref counts.
      - spectrum_fragments_flat: flattened per-spectrum fragment index lists.

    Returns:
        spectrum_offsets, spectrum_fragments_offsets, spectrum_fragments_flat,
        spectrum_mslevels, spectrum_msn_precursors.
    """
    spectrum_offsets_list: list[int] = [0]
    frag_offsets_list: list[int] = [0]
    frags_flat_list: list[int] = []
    mslevels_list: list[int] = []
    msn_precursors_list: list[int] = []

    for _key, tree in trees_sorted:
        n_spectra = len(tree.spectrum_fragments)
        spectrum_offsets_list.append(spectrum_offsets_list[-1] + n_spectra)

        for spec_idx in range(n_spectra):
            frags = tree.spectrum_fragments[spec_idx]
            frags_flat_list.extend(int(f) for f in frags)
            frag_offsets_list.append(frag_offsets_list[-1] + len(frags))
            mslevels_list.append(int(tree.spectrum_mslevels[spec_idx]))
            msn_precursors_list.append(int(tree.spectrum_msn_precursors[spec_idx]))

    return (
        np.array(spectrum_offsets_list, dtype=np.int64),
        np.array(frag_offsets_list, dtype=np.int64),
        np.array(frags_flat_list, dtype=np.int32),
        np.array(mslevels_list, dtype=np.int32),
        np.array(msn_precursors_list, dtype=np.int32),
    )


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_trees_npz(
    trees: dict[tuple[str, str, str], FragmentationTree],
    output_path: str | Path,
    config: TreeStorageConfig,
) -> None:
    """Collate trees and write a single compressed NPZ file.

    Trees are written in canonical order: sorted by
    (base_inchikey, ion_mode, precursor_type).

    Per-graph edge_index range validation is performed at save time:
    for each tree, after extracting its local COO edge_index, asserts
        edge_index.size == 0 or (edge_index.min() >= 0 and edge_index.max() < n_nodes_i)

    Args:
        trees: Dict mapping (base_inchikey, ion_mode, precursor_type) to
            FragmentationTree.
        output_path: Path for the output .npz file.
        config: Storage configuration.

    Raises:
        AssertionError: If any per-graph edge_index is out of range,
            if offsets do not start at 0, or if tree_type is invalid.
    """
    assert config.tree_type in ("ms2", "full"), (
        f"tree_type must be 'ms2' or 'full', got '{config.tree_type}'"
    )

    trees_sorted = _sorted_trees(trees)
    n_trees = len(trees_sorted)

    # Accumulators
    base_inchikey_list: list[str] = []
    ion_mode_list: list[str] = []
    precursor_type_list: list[str] = []
    precursor_formulas_list: list[np.ndarray] = []
    precursor_indices_list: list[int] = []

    node_offsets_list: list[int] = [0]
    edge_offsets_list: list[int] = [0]

    node_features_list: list[np.ndarray] = []
    fragment_errors_list: list[np.ndarray] = []
    edge_index_list: list[np.ndarray] = []
    edge_weights_list: list[np.ndarray] = []

    formula_strings_list: list[np.ndarray] = []

    for key, tree in trees_sorted:
        base_inchikey, ion_mode, precursor_type = key

        # Sanity: tree fields must match key
        assert tree.base_inchikey == base_inchikey, (
            f"Key base_inchikey '{base_inchikey}' does not match "
            f"tree.base_inchikey '{tree.base_inchikey}'"
        )
        assert tree.ion_mode == ion_mode, (
            f"Key ion_mode '{ion_mode}' does not match "
            f"tree.ion_mode '{tree.ion_mode}'"
        )

        base_inchikey_list.append(tree.base_inchikey)
        ion_mode_list.append(tree.ion_mode)
        precursor_type_list.append(precursor_type)

        # Precursor data
        precursor_formulas_list.append(tree.precursor_formula)
        precursor_indices_list.append(tree.precursor_idx)

        # Node data
        n_nodes = tree.n_fragments
        node_features_list.append(tree.fragment_formulas)
        fragment_errors_list.append(tree.fragment_errors_ppm)

        # Edge data: dense -> COO
        edge_index_i, edge_weights_i = _dense_edge_weights_to_coo(tree.edge_weights)

        # Per-graph range validation
        _validate_edge_index(edge_index_i, n_nodes, key)

        edge_index_list.append(edge_index_i)
        edge_weights_list.append(edge_weights_i)

        # Update CSR offsets
        node_offsets_list.append(node_offsets_list[-1] + n_nodes)
        edge_offsets_list.append(edge_offsets_list[-1] + edge_index_i.shape[1])

        # Optional formula strings
        if config.include_formula_strings:
            formula_strings_list.append(_make_formula_strings_array(tree))

    # --- Assemble flat arrays ---

    node_offsets_arr = np.array(node_offsets_list, dtype=np.int64)
    edge_offsets_arr = np.array(edge_offsets_list, dtype=np.int64)

    assert node_offsets_arr[0] == 0, (
        f"node_offsets must start at 0, got {node_offsets_arr[0]}"
    )
    assert edge_offsets_arr[0] == 0, (
        f"edge_offsets must start at 0, got {edge_offsets_arr[0]}"
    )

    # Concatenate per-graph arrays
    if n_trees > 0:
        node_features_arr = np.concatenate(node_features_list, axis=0)
        fragment_errors_arr = np.concatenate(fragment_errors_list, axis=0)
        precursor_formulas_arr = np.stack(precursor_formulas_list, axis=0)
    else:
        node_features_arr = np.zeros((0, _NUM_ELEMENTS), dtype=np.int32)
        fragment_errors_arr = np.array([], dtype=np.float64)
        precursor_formulas_arr = np.zeros((0, _NUM_ELEMENTS), dtype=np.int32)

    total_edges = edge_offsets_arr[-1]
    if n_trees > 0 and total_edges > 0:
        edge_index_arr = np.concatenate(edge_index_list, axis=1)
        edge_weights_arr = np.concatenate(edge_weights_list, axis=0)
    else:
        edge_index_arr = np.zeros((2, 0), dtype=np.int32)
        edge_weights_arr = np.array([], dtype=np.float64)

    # --- Build save dict ---

    save_dict: dict[str, Any] = {
        "format_version": np.int32(1),
        "tree_type": np.array(config.tree_type, dtype="<U4"),
        "base_inchikey": np.array(base_inchikey_list, dtype="<U14"),
        "ion_mode": np.array(ion_mode_list, dtype="<U1"),
        "precursor_type": np.array(precursor_type_list, dtype="<U16"),
        "precursor_formulas": precursor_formulas_arr,
        "precursor_indices": np.array(precursor_indices_list, dtype=np.int32),
        "node_offsets": node_offsets_arr,
        "edge_offsets": edge_offsets_arr,
        "node_features": node_features_arr,
        "fragment_errors_ppm": fragment_errors_arr,
        "edge_index": edge_index_arr,
        "edge_weights": edge_weights_arr,
    }

    # Optional formula strings
    if config.include_formula_strings:
        save_dict["fragment_formulas_str"] = (
            np.concatenate(formula_strings_list, axis=0)
            if n_trees > 0
            else np.array([], dtype="<U64")
        )

    # Full-tree spectrum metadata
    if config.tree_type == "full" and config.include_spectrum_metadata:
        (
            spec_offsets,
            spec_frag_offsets,
            spec_frags_flat,
            spec_mslevels,
            spec_msn_precursors,
        ) = _collate_full_spectrum_arrays(trees_sorted)
        save_dict["spectrum_offsets"] = spec_offsets
        save_dict["spectrum_fragments_offsets"] = spec_frag_offsets
        save_dict["spectrum_fragments_flat"] = spec_frags_flat
        save_dict["spectrum_mslevels"] = spec_mslevels
        save_dict["spectrum_msn_precursors"] = spec_msn_precursors

    # --- Write ---
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if config.compress:
        np.savez_compressed(output_path, **save_dict)
    else:
        np.savez(output_path, **save_dict)


# ---------------------------------------------------------------------------
# Load (raw arrays)
# ---------------------------------------------------------------------------


def load_tree_arrays_npz(
    input_path: str | Path,
) -> TreeArrays:
    """Load raw collated arrays from a tree NPZ file.

    No FragmentationTree reconstruction is performed. All shape/dtype
    invariants are validated on load.

    Args:
        input_path: Path to the .npz file.

    Returns:
        TreeArrays instance with the loaded arrays.

    Raises:
        AssertionError: If format_version is not 1, or any shape/dtype
            invariant is violated.
    """
    data = np.load(input_path, allow_pickle=False)

    # Validate format version
    format_version = int(data["format_version"])
    assert format_version == 1, (
        f"Unknown format_version {format_version}; expected 1"
    )

    tree_type = str(data["tree_type"])

    # Helper to get optional array
    def _get_opt(name: str) -> np.ndarray | None:
        return data[name] if name in data else None

    result = TreeArrays(
        base_inchikey=data["base_inchikey"],
        ion_mode=data["ion_mode"],
        precursor_type=data["precursor_type"],
        precursor_formulas=data["precursor_formulas"],
        precursor_indices=data["precursor_indices"],
        node_offsets=data["node_offsets"],
        edge_offsets=data["edge_offsets"],
        node_features=data["node_features"],
        fragment_errors_ppm=data["fragment_errors_ppm"],
        edge_index=data["edge_index"],
        edge_weights=data["edge_weights"],
        fragment_formulas_str=_get_opt("fragment_formulas_str"),
        spectrum_offsets=_get_opt("spectrum_offsets"),
        spectrum_fragments_offsets=_get_opt("spectrum_fragments_offsets"),
        spectrum_fragments_flat=_get_opt("spectrum_fragments_flat"),
        spectrum_mslevels=_get_opt("spectrum_mslevels"),
        spectrum_msn_precursors=_get_opt("spectrum_msn_precursors"),
    )

    # Validate shapes
    n_trees = len(result.base_inchikey)

    assert result.node_offsets.shape == (n_trees + 1,), (
        f"node_offsets shape {result.node_offsets.shape} != ({n_trees + 1},)"
    )
    assert result.edge_offsets.shape == (n_trees + 1,), (
        f"edge_offsets shape {result.edge_offsets.shape} != ({n_trees + 1},)"
    )
    assert result.node_offsets[0] == 0, (
        f"node_offsets[0] must be 0, got {result.node_offsets[0]}"
    )
    assert result.edge_offsets[0] == 0, (
        f"edge_offsets[0] must be 0, got {result.edge_offsets[0]}"
    )

    total_nodes = int(result.node_offsets[-1])
    total_edges = int(result.edge_offsets[-1])

    assert result.node_features.shape == (total_nodes, _NUM_ELEMENTS), (
        f"node_features shape {result.node_features.shape} != "
        f"({total_nodes}, {_NUM_ELEMENTS})"
    )
    assert result.fragment_errors_ppm.shape == (total_nodes,), (
        f"fragment_errors_ppm shape {result.fragment_errors_ppm.shape} != ({total_nodes},)"
    )
    assert result.edge_index.shape == (2, total_edges), (
        f"edge_index shape {result.edge_index.shape} != (2, {total_edges})"
    )
    assert result.edge_weights.shape == (total_edges,), (
        f"edge_weights shape {result.edge_weights.shape} != ({total_edges},)"
    )

    # Validate precursor arrays
    assert result.precursor_formulas.shape == (n_trees, _NUM_ELEMENTS), (
        f"precursor_formulas shape {result.precursor_formulas.shape} != "
        f"({n_trees}, {_NUM_ELEMENTS})"
    )
    assert result.precursor_indices.shape == (n_trees,), (
        f"precursor_indices shape {result.precursor_indices.shape} != ({n_trees},)"
    )

    # Validate formula strings array if present
    if result.fragment_formulas_str is not None:
        assert result.fragment_formulas_str.shape == (total_nodes,), (
            f"fragment_formulas_str shape {result.fragment_formulas_str.shape} != ({total_nodes},)"
        )

    # Validate spectrum arrays if present
    if result.spectrum_offsets is not None:
        assert result.spectrum_offsets.shape == (n_trees + 1,), (
            f"spectrum_offsets shape {result.spectrum_offsets.shape} != ({n_trees + 1},)"
        )
        total_spectra = int(result.spectrum_offsets[-1])
        assert result.spectrum_fragments_offsets is not None
        assert result.spectrum_fragments_offsets.shape == (total_spectra + 1,), (
            f"spectrum_fragments_offsets shape {result.spectrum_fragments_offsets.shape} != "
            f"({total_spectra + 1},)"
        )
        total_frag_refs = int(result.spectrum_fragments_offsets[-1])
        assert result.spectrum_fragments_flat is not None
        assert result.spectrum_fragments_flat.shape == (total_frag_refs,), (
            f"spectrum_fragments_flat shape {result.spectrum_fragments_flat.shape} != "
            f"({total_frag_refs},)"
        )
        assert result.spectrum_mslevels is not None
        assert result.spectrum_mslevels.shape == (total_spectra,), (
            f"spectrum_mslevels shape {result.spectrum_mslevels.shape} != ({total_spectra},)"
        )
        assert result.spectrum_msn_precursors is not None
        assert result.spectrum_msn_precursors.shape == (total_spectra,), (
            f"spectrum_msn_precursors shape {result.spectrum_msn_precursors.shape} != "
            f"({total_spectra},)"
        )

    return result


# ---------------------------------------------------------------------------
# Load (reconstruct FragmentationTree objects)
# ---------------------------------------------------------------------------


def load_trees_npz(
    input_path: str | Path,
) -> dict[tuple[str, str, str], FragmentationTree]:
    """Load and reconstruct FragmentationTree objects (lossless round-trip).

    For tree_type == "ms2": the file omits spectrum_* fields. This function
    synthesizes trivial spectrum metadata (one spectrum per tree covering
    all nodes, mslevel=2, msn_precursor=-1). This synthesis relies on the
    invariant that MS2 files contain EXACTLY one spectrum per tree; the
    function asserts this after synthesis.

    Args:
        input_path: Path to the .npz file.

    Returns:
        Dict mapping (base_inchikey, ion_mode, precursor_type) -> FragmentationTree.

    Raises:
        AssertionError: If the MS2 one-spectrum-per-tree invariant is violated,
            or if any loaded array invariant is violated.
    """
    arrays = load_tree_arrays_npz(input_path)

    # Determine tree_type from the file (re-read scalar to avoid carrying it in TreeArrays)
    data_for_tree_type = np.load(input_path, allow_pickle=False)
    tree_type = str(data_for_tree_type["tree_type"])
    data_for_tree_type.close()

    n_trees = len(arrays.base_inchikey)

    # --- Build spectrum metadata arrays ---
    if tree_type == "ms2":
        # Synthesize trivial spectrum metadata for MS2 trees
        assert arrays.spectrum_offsets is None, (
            "MS2 file should not contain spectrum_offsets"
        )

        spec_offsets, spec_frag_offsets, spec_frags_flat, spec_mslevels, spec_msn_precursors = (
            _make_ms2_spectrum_arrays(n_trees, arrays.node_offsets)
        )

        # Assert the one-spectrum-per-tree invariant
        per_tree_counts = spec_offsets[1:] - spec_offsets[:-1]
        assert (per_tree_counts == 1).all(), (
            "MS2 file invariant violated: expected exactly one spectrum per tree, "
            f"but got spectrum counts per tree: {per_tree_counts.tolist()}. "
            "If a future MS2 builder produces multiple spectra per tree, "
            "the MS2 file format must be extended to store spectrum_* fields explicitly."
        )
    else:
        # Full tree: spectrum arrays must be present
        assert arrays.spectrum_offsets is not None, (
            "Full tree file is missing spectrum_offsets. "
            "Cannot reconstruct FragmentationTree without spectrum metadata."
        )
        spec_offsets = arrays.spectrum_offsets
        spec_frag_offsets = arrays.spectrum_fragments_offsets
        spec_frags_flat = arrays.spectrum_fragments_flat
        spec_mslevels = arrays.spectrum_mslevels
        spec_msn_precursors = arrays.spectrum_msn_precursors

    # --- Reconstruct per graph ---
    trees: dict[tuple[str, str, str], FragmentationTree] = {}

    for i in range(n_trees):
        base_inchikey = str(arrays.base_inchikey[i])
        ion_mode = str(arrays.ion_mode[i])
        precursor_type = str(arrays.precursor_type[i])
        key = (base_inchikey, ion_mode, precursor_type)

        n_nodes_i = int(arrays.node_offsets[i + 1] - arrays.node_offsets[i])
        n_edges_i = int(arrays.edge_offsets[i + 1] - arrays.edge_offsets[i])

        node_start = int(arrays.node_offsets[i])
        node_end = int(arrays.node_offsets[i + 1])
        edge_start = int(arrays.edge_offsets[i])
        edge_end = int(arrays.edge_offsets[i + 1])

        # Slice per-graph arrays (copy because np.load returns read-only arrays)
        fragment_formulas = arrays.node_features[node_start:node_end].copy()
        fragment_errors_ppm = arrays.fragment_errors_ppm[node_start:node_end].copy()

        # Reconstruct dense edge_weights from COO
        edge_index_i = arrays.edge_index[:, edge_start:edge_end]
        edge_weights_i = arrays.edge_weights[edge_start:edge_end]

        edge_weights_dense = np.zeros((n_nodes_i, n_nodes_i), dtype=np.float64)
        if n_edges_i > 0:
            rows = edge_index_i[0, :]
            cols = edge_index_i[1, :]
            edge_weights_dense[rows, cols] = edge_weights_i

        # Precursor formula
        precursor_formula = arrays.precursor_formulas[i].copy()

        # Fragment formula strings
        if arrays.fragment_formulas_str is not None:
            fragment_formulas_str = [
                str(arrays.fragment_formulas_str[idx])
                for idx in range(node_start, node_end)
            ]
        else:
            fragment_formulas_str = ["" for _ in range(n_nodes_i)]

        # Spectrum metadata for this tree
        spec_start = int(spec_offsets[i])
        spec_end = int(spec_offsets[i + 1])
        n_spectra_i = spec_end - spec_start

        spectrum_fragments: list[np.ndarray] = []
        spectrum_mslevels_i: list[int] = []
        spectrum_msn_precursors_i: list[int] = []

        for s_idx in range(spec_start, spec_end):
            frag_start = int(spec_frag_offsets[s_idx])
            frag_end = int(spec_frag_offsets[s_idx + 1])
            frags = spec_frags_flat[frag_start:frag_end].copy()
            spectrum_fragments.append(frags)
            spectrum_mslevels_i.append(int(spec_mslevels[s_idx]))
            spectrum_msn_precursors_i.append(int(spec_msn_precursors[s_idx]))

        tree = FragmentationTree(
            base_inchikey=base_inchikey,
            ion_mode=ion_mode,
            precursor_formula=precursor_formula,
            fragment_formulas=fragment_formulas,
            fragment_formulas_str=fragment_formulas_str,
            edge_weights=edge_weights_dense,
            spectrum_fragments=spectrum_fragments,
            spectrum_mslevels=np.array(spectrum_mslevels_i, dtype=np.int32),
            spectrum_msn_precursors=spectrum_msn_precursors_i,
            fragment_errors_ppm=fragment_errors_ppm,
        )

        trees[key] = tree

    return trees


# ---------------------------------------------------------------------------
# Key alignment utilities
# ---------------------------------------------------------------------------


def build_key_index(keys: list[TreeKey]) -> dict[TreeKey, int]:
    """Map each TreeKey to its positional index in the list.

    Args:
        keys: List of TreeKey instances.

    Returns:
        Dict mapping each TreeKey to its index (0-based position).
    """
    return {key: idx for idx, key in enumerate(keys)}


def align_keys(
    ms2_keys: list[TreeKey],
    full_keys: list[TreeKey],
    fail_on_mismatch: bool = True,
) -> list[tuple[int, int]]:
    """Return (ms2_index, full_index) pairs for keys present in BOTH files.

    Keys present in only one file are dropped. If fail_on_mismatch is True
    (default), raises AssertionError with a clear message listing the
    mismatched keys. Pairing is robust to independently-built files
    because it uses key values, not positional equality.

    Args:
        ms2_keys: List of TreeKey instances from the MS2 file.
        full_keys: List of TreeKey instances from the full file.
        fail_on_mismatch: If True, raise on any key mismatch.

    Returns:
        List of (ms2_index, full_index) tuples for matching keys, sorted
        by canonical key order.

    Raises:
        AssertionError: If fail_on_mismatch is True and there are keys
            present in only one file.
    """
    ms2_index = build_key_index(ms2_keys)
    full_index = build_key_index(full_keys)

    ms2_set = set(ms2_index.keys())
    full_set = set(full_index.keys())

    only_ms2 = ms2_set - full_set
    only_full = full_set - ms2_set

    if fail_on_mismatch and (only_ms2 or only_full):
        messages: list[str] = []
        if only_ms2:
            keys_str = ", ".join(
                f"'{k.base_inchikey}/{k.ion_mode}/{k.precursor_type}'"
                for k in sorted(only_ms2, key=lambda x: (x.base_inchikey, x.ion_mode, x.precursor_type))
            )
            messages.append(
                f"Keys present in MS2 but missing in full "
                f"({len(only_ms2)}): {keys_str}"
            )
        if only_full:
            keys_str = ", ".join(
                f"'{k.base_inchikey}/{k.ion_mode}/{k.precursor_type}'"
                for k in sorted(only_full, key=lambda x: (x.base_inchikey, x.ion_mode, x.precursor_type))
            )
            messages.append(
                f"Keys present in full but missing in MS2 "
                f"({len(only_full)}): {keys_str}"
            )
        assert not (only_ms2 or only_full), "; ".join(messages)

    common = ms2_set & full_set
    pairs: list[tuple[int, int]] = [
        (ms2_index[key], full_index[key])
        for key in sorted(common, key=lambda k: (k.base_inchikey, k.ion_mode, k.precursor_type))
    ]

    return pairs
