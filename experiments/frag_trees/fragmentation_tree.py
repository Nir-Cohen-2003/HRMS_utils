"""
Fragmentation tree builder for MSn spectral libraries.

Groups spectra by compound (base_inchikey) and ionization mode,
builds a fragmentation graph where edges connect fragments to their
formula-parents, truncates based on MSn constraints, and weights edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
from numba import njit, prange

from hrms_utils.formula_annotation.utils import format_formula_string_to_array
from hrms_utils.hrms_core import NUM_ELEMENTS


# ---------------------------------------------------------------------------
# Numba helpers
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _is_superset(parent: np.ndarray, child: np.ndarray) -> bool:
    """Check if parent formula is a strict superset of child formula.

    Args:
        parent: 1-D array of element counts, shape (n_elements,).
        child: 1-D array of element counts, shape (n_elements,).

    Returns:
        True if parent >= child element-wise and parent != child.
    """
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
    """Build a boolean matrix where M[i, j] = True if formula[i] is superset of formula[j].

    Args:
        formulas: 2-D array, shape (n_fragments, n_elements).

    Returns:
        Boolean matrix, shape (n_fragments, n_fragments).
    """
    n = formulas.shape[0]
    matrix = np.zeros((n, n), dtype=np.bool_)
    for i in prange(n):
        for j in range(n):
            if i == j:
                continue
            matrix[i, j] = _is_superset(formulas[i], formulas[j])
    return matrix


@njit(cache=True, fastmath=True)
def _find_formula_index(formulas: np.ndarray, target: np.ndarray) -> int:
    """Find the index of a target formula in a formula array.

    Args:
        formulas: 2-D array, shape (n_fragments, n_elements).
        target: 1-D array, shape (n_elements,).

    Returns:
        Index of the matching formula, or -1 if not found.
    """
    n = formulas.shape[0]
    for i in range(n):
        match = True
        for j in range(formulas.shape[1]):
            if formulas[i, j] != target[j]:
                match = False
                break
        if match:
            return i
    return -1


@njit(cache=True, fastmath=True)
def _truncate_msn(
    superset_matrix: np.ndarray,
    fragment_indices: np.ndarray,
    msn_precursor_idx: int,
) -> np.ndarray:
    """Truncate parent edges for fragments in an MSn spectrum.

    For each fragment in the MSn spectrum (excluding the MSn precursor itself),
    keep only parents that are also children of the MSn precursor
    (i.e., the MSn precursor is a parent of the parent, or the parent IS
    the MSn precursor itself).

    Args:
        superset_matrix: Boolean matrix, shape (n_all, n_all).
        fragment_indices: Indices of fragments in this MSn spectrum.
        msn_precursor_idx: Index of the MSn precursor in the all-fragments array.

    Returns:
        Updated superset_matrix (modified in place).
    """
    n_all = superset_matrix.shape[0]
    for frag_idx in fragment_indices:
        # The MSn precursor itself is never truncated
        if frag_idx == msn_precursor_idx:
            continue
        for parent_idx in range(n_all):
            if not superset_matrix[parent_idx, frag_idx]:
                continue
            # The MSn precursor is always allowed as a parent
            if parent_idx == msn_precursor_idx:
                continue
            # Check if msn_precursor is a parent of parent_idx
            if not superset_matrix[msn_precursor_idx, parent_idx]:
                superset_matrix[parent_idx, frag_idx] = False
    return superset_matrix


@njit(cache=True, fastmath=True)
def _compute_edge_weights(superset_matrix: np.ndarray) -> np.ndarray:
    """Compute edge weights so incoming edges for each node sum to 1.

    Args:
        superset_matrix: Boolean matrix where M[i, j] means edge i -> j.

    Returns:
        Float matrix of same shape with weights.
    """
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
    """Drop fragments with zero parents (except the molecular precursor).

    Returns a boolean mask of kept fragments.
    """
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


@njit(cache=True, fastmath=True)
def _drop_orphans_iterative(superset_matrix: np.ndarray, precursor_idx: int) -> np.ndarray:
    """Iteratively drop orphans until no more exist.

    Dropping a fragment may cause its children to become orphans,
    so we repeat until convergence.
    """
    n = superset_matrix.shape[0]
    keep = np.ones(n, dtype=np.bool_)
    changed = True
    while changed:
        changed = False
        for j in range(n):
            if j == precursor_idx or not keep[j]:
                continue
            has_parent = False
            for i in range(n):
                if keep[i] and superset_matrix[i, j]:
                    has_parent = True
                    break
            if not has_parent:
                keep[j] = False
                changed = True
    return keep


def _minimize_dag(
    superset_matrix: np.ndarray,
    spectrum_peak_fragments: list[list[int]],
    spectrum_msn_precursor_indices: list[int],
    spectrum_mslevels: list[int],
    molecular_precursor_idx: int,
) -> np.ndarray:
    """Remove transitive edges when an intermediate has an MSn spectrum with the child.

    For each edge a -> c, if there exists b such that:
    - a -> b and b -> c both exist in the original graph
    - b is the precursor of an MSn spectrum (mslevel > 2, not molecular precursor)
    - c is a peak in that MSn spectrum

    Then remove a -> c, because c can be created via b and the direct edge
    is likely due to hidden intermediate fragmentation.

    If no intermediate b has an MSn spectrum with c, the edge a -> c is kept.
    """
    n = superset_matrix.shape[0]
    result = superset_matrix.copy()

    # Build a set of (precursor_idx, child_idx) pairs from MSn spectra
    observed_pairs = set()
    for peaks, prec_idx, mslevel in zip(
        spectrum_peak_fragments, spectrum_msn_precursor_indices, spectrum_mslevels
    ):
        if prec_idx < 0 or prec_idx == molecular_precursor_idx or mslevel <= 2:
            continue
        for child_idx in peaks:
            observed_pairs.add((prec_idx, child_idx))

    # For each edge a -> c, check if there's a path a -> b -> c
    # where (b, c) is observed in an MSn spectrum
    for a in range(n):
        for c in range(n):
            if not superset_matrix[a, c] or a == c:
                continue

            # Look for any b that creates a transitive path with observed MSn
            for b in range(n):
                if b == a or b == c:
                    continue
                if superset_matrix[a, b] and superset_matrix[b, c] and (b, c) in observed_pairs:
                    result[a, c] = False
                    break

    return result


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FragmentationTree:
    """A fragmentation tree for a single compound + ionization mode + precursor."""

    base_inchikey: str
    ion_mode: str
    precursor_formula: np.ndarray  # shape (NUM_ELEMENTS,)
    # All unique fragment formulas (including precursor as the largest)
    fragment_formulas: np.ndarray  # shape (n_fragments, NUM_ELEMENTS)
    fragment_formulas_str: list[str]
    # Edge weights matrix: weights[i, j] = weight of edge from fragment i to fragment j
    edge_weights: np.ndarray  # shape (n_fragments, n_fragments)
    # Mapping from original spectrum mslevel to fragment indices
    spectrum_fragments: list[np.ndarray]  # list of arrays of fragment indices per spectrum
    spectrum_mslevels: np.ndarray  # mslevel for each spectrum
    spectrum_msn_precursors: list[int]  # index of MSn precursor for each spectrum, -1 if none

    @property
    def n_fragments(self) -> int:
        return self.fragment_formulas.shape[0]

    @property
    def precursor_idx(self) -> int:
        """The molecular precursor is the largest fragment (superset of all others)."""
        # Find the fragment that is a superset of all others
        n = self.n_fragments
        for i in range(n):
            is_precursor = True
            for j in range(n):
                if i == j:
                    continue
                if not _is_superset(self.fragment_formulas[i], self.fragment_formulas[j]):
                    is_precursor = False
                    break
            if is_precursor:
                return i
        return 0  # fallback


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------


def build_fragmentation_trees(
    df: pl.DataFrame,
    mass_tolerance_ppm: float = 5.0,
) -> dict[tuple[str, str, str], FragmentationTree]:
    """Build fragmentation trees from a processed spectral library DataFrame.

    Groups by (base_inchikey, ion_mode, precursor_type), combines ALL
    spectra of the same molecule and adduct (MS2, MS3, MS4, different
    energies, etc.) into a single fragmentation graph.

    Fragments are identified by mass (within tolerance) rather than by
    exact formula, so isobaric fragments across spectra are merged.

    Args:
        df: Processed spectral library DataFrame with columns:
            - base_inchikey
            - ion_mode
            - precursor_type
            - molecular_formula_array (Array(Int32, NUM_ELEMENTS))
            - precursor_formula_array (Array(Int32, NUM_ELEMENTS))
            - precursor_mz (Float64)
            - cleaned_fragment_formulas (List(Array(Int32, NUM_ELEMENTS)))
            - cleaned_fragment_formulas_str (List(String), optional)
            - cleaned_normalized_mz (List(Float64))
            - mslevel (Int64)
        mass_tolerance_ppm: Mass tolerance in ppm for grouping fragments
            across spectra (default: 5.0).

    Returns:
        Dictionary mapping (base_inchikey, ion_mode, precursor_type) -> FragmentationTree
    """
    required = [
        "base_inchikey",
        "ion_mode",
        "precursor_type",
        "molecular_formula_array",
        "precursor_formula_array",
        "precursor_mz",
        "cleaned_fragment_formulas",
        "cleaned_normalized_mz",
        "mslevel",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    has_str = "cleaned_fragment_formulas_str" in df.columns

    # Group by compound + ionization mode + adduct (precursor_type)
    groups = df.group_by(["base_inchikey", "ion_mode", "precursor_type"])

    trees: dict[tuple[str, str, str], FragmentationTree] = {}

    for (base_inchikey, ion_mode, precursor_type), group_df in groups:
        # The molecular precursor is the precursor_formula_array of any MS2 spectrum
        # (all MS2 spectra in this group share the same molecular precursor)
        ms2_rows = group_df.filter(pl.col("mslevel") == 2)
        if ms2_rows.is_empty():
            # No MS2 data, skip this group
            continue

        molecular_precursor = np.array(
            ms2_rows["precursor_formula_array"][0], dtype=np.int32
        )
        molecular_precursor_str = _formula_array_to_string(molecular_precursor)
        molecular_precursor_mass = float(ms2_rows["precursor_mz"][0])

        # Global fragment registry: all fragments from all spectra in this group
        all_formulas: list[np.ndarray] = []
        all_masses: list[float] = []
        formula_to_str: dict[tuple, str] = {}

        def _register_formula(
            formula_arr: np.ndarray,
            mass: float | None,
            formula_str: str | None = None,
        ) -> int:
            """Register a fragment by mass (within tolerance) and return its index.

            If an existing fragment has a mass within ``mass_tolerance_ppm``
            of the supplied mass, the existing index is returned and the new
            fragment is treated as the same node.  The first formula/string
            encountered for a mass cluster is kept as the representative.
            """
            if mass is not None:
                tol = mass_tolerance_ppm * 1e-6
                for idx, existing_mass in enumerate(all_masses):
                    if existing_mass > 0 and abs(mass - existing_mass) / existing_mass <= tol:
                        return idx
            idx = len(all_formulas)
            all_formulas.append(formula_arr.copy())
            all_masses.append(mass if mass is not None else 0.0)
            formula_tuple = tuple(formula_arr)
            if formula_str is not None:
                formula_to_str[formula_tuple] = formula_str
            return idx

        # Register molecular precursor
        molecular_precursor_idx = _register_formula(
            molecular_precursor, molecular_precursor_mass, molecular_precursor_str
        )

        # Per-spectrum data for truncation
        # peak_fragments: actual fragment peaks from each spectrum (for truncation)
        # spectrum_all_fragments: peak_fragments + precursor (for tree building)
        spectrum_peak_fragments: list[list[int]] = []
        spectrum_all_fragments: list[list[int]] = []
        spectrum_mslevels: list[int] = []
        spectrum_msn_precursor_indices: list[int] = []

        for row in group_df.iter_rows(named=True):
            mslevel = row["mslevel"]
            frag_formulas = row["cleaned_fragment_formulas"]
            frag_strs = (
                row.get("cleaned_fragment_formulas_str", None)
                if has_str
                else None
            )
            frag_masses = row["cleaned_normalized_mz"]
            spec_precursor_mass = float(row["precursor_mz"])
            # The direct precursor of THIS spectrum (MS2 -> molecular, MS3+ -> fragment)
            spec_precursor_formula = np.array(
                row["precursor_formula_array"], dtype=np.int32
            )

            if frag_formulas is None:
                continue

            # Convert polars Series to plain lists
            if isinstance(frag_formulas, pl.Series):
                frag_formulas = frag_formulas.to_list()
            if frag_strs is not None and isinstance(frag_strs, pl.Series):
                frag_strs = frag_strs.to_list()
            if isinstance(frag_masses, pl.Series):
                frag_masses = frag_masses.to_list()

            peak_indices: list[int] = []

            # Register fragments (actual peaks) from this spectrum
            for frag_idx, frag in enumerate(frag_formulas):
                if frag is None:
                    continue
                frag_arr = np.array(frag, dtype=np.int32)
                frag_str = (
                    frag_strs[frag_idx]
                    if frag_strs is not None and frag_idx < len(frag_strs)
                    else None
                )
                frag_mass = (
                    frag_masses[frag_idx]
                    if frag_masses is not None and frag_idx < len(frag_masses)
                    else None
                )
                fidx = _register_formula(frag_arr, frag_mass, frag_str)
                peak_indices.append(fidx)

            # Register the spectrum's direct precursor as a fragment
            # (for MS2 this is the molecular precursor, already registered)
            spec_precursor_idx = _register_formula(
                spec_precursor_formula, spec_precursor_mass
            )

            # For the tree, the spectrum contains both peaks and precursor
            # (deduplicated: if precursor already in peaks, don't add twice)
            all_indices = peak_indices.copy()
            if spec_precursor_idx not in all_indices:
                all_indices.append(spec_precursor_idx)

            spectrum_peak_fragments.append(peak_indices)
            spectrum_all_fragments.append(all_indices)
            spectrum_mslevels.append(mslevel)
            spectrum_msn_precursor_indices.append(spec_precursor_idx)

        if not spectrum_all_fragments:
            continue

        tree = _build_tree_from_fragments(
            base_inchikey,
            ion_mode,
            molecular_precursor,
            all_formulas,
            formula_to_str,
            spectrum_peak_fragments,
            spectrum_all_fragments,
            spectrum_mslevels,
            spectrum_msn_precursor_indices,
        )
        if tree is not None:
            key = (base_inchikey, ion_mode, precursor_type)
            trees[key] = tree

    return trees


def _build_tree_from_fragments(
    base_inchikey: str,
    ion_mode: str,
    molecular_precursor: np.ndarray,
    all_formulas: list[np.ndarray],
    formula_to_str: dict[tuple, str],
    spectrum_peak_fragments: list[list[int]],
    spectrum_all_fragments: list[list[int]],
    spectrum_mslevels: list[int],
    spectrum_msn_precursor_indices: list[int],
) -> Optional[FragmentationTree]:
    """Build a single fragmentation tree from collected fragments."""

    n_fragments = len(all_formulas)
    if n_fragments == 0:
        return None

    all_formulas_arr = np.stack(all_formulas, axis=0)  # shape (n, NUM_ELEMENTS)

    # Build string representations
    all_formulas_str = []
    for formula in all_formulas:
        formula_tuple = tuple(formula)
        if formula_tuple in formula_to_str:
            all_formulas_str.append(formula_to_str[formula_tuple])
        else:
            all_formulas_str.append(_formula_array_to_string(formula))

    # 1. Build initial superset matrix
    superset_matrix = _build_superset_matrix(all_formulas_arr)

    # 2. Truncate based on MSn
    # Only apply truncation to actual peak fragments, NOT the spectrum precursor
    for peak_indices, mslevel, msn_precursor_idx in zip(
        spectrum_peak_fragments, spectrum_mslevels, spectrum_msn_precursor_indices
    ):
        if mslevel <= 2 or msn_precursor_idx < 0:
            continue

        peak_indices_arr = np.array(peak_indices, dtype=np.int32)
        superset_matrix = _truncate_msn(
            superset_matrix, peak_indices_arr, msn_precursor_idx
        )

    # 3. Find molecular precursor index
    molecular_precursor_idx = -1
    for i in range(n_fragments):
        is_precursor = True
        for j in range(n_fragments):
            if i == j:
                continue
            if not _is_superset(all_formulas_arr[i], all_formulas_arr[j]):
                is_precursor = False
                break
        if is_precursor:
            molecular_precursor_idx = i
            break

    if molecular_precursor_idx < 0:
        molecular_precursor_idx = 0  # fallback

    # 4. Minimize DAG: remove transitive edges when intermediate has MSn evidence
    superset_matrix = _minimize_dag(
        superset_matrix,
        spectrum_peak_fragments,
        spectrum_msn_precursor_indices,
        spectrum_mslevels,
        molecular_precursor_idx,
    )

    # 5. Iteratively drop orphans (fragments with 0 parents, except molecular precursor)
    keep_mask = _drop_orphans_iterative(superset_matrix, molecular_precursor_idx)

    # Remap indices
    kept_indices = np.where(keep_mask)[0]
    n_kept = len(kept_indices)
    if n_kept == 0:
        return None

    # Rebuild formulas and strings
    kept_formulas = all_formulas_arr[keep_mask]
    kept_formulas_str = [all_formulas_str[i] for i in kept_indices]

    # Rebuild superset matrix for kept fragments
    kept_superset = np.zeros((n_kept, n_kept), dtype=np.bool_)
    for new_i, old_i in enumerate(kept_indices):
        for new_j, old_j in enumerate(kept_indices):
            kept_superset[new_i, new_j] = superset_matrix[old_i, old_j]

    # 5. Compute edge weights
    edge_weights = _compute_edge_weights(kept_superset)

    # Remap spectrum fragment indices
    old_to_new = {old: new for new, old in enumerate(kept_indices)}
    remapped_spectrum_fragments = []
    remapped_mslevels = []
    remapped_msn_precursors = []
    for frag_indices, mslevel, msn_prec in zip(
        spectrum_all_fragments, spectrum_mslevels, spectrum_msn_precursor_indices
    ):
        remapped = [old_to_new[idx] for idx in frag_indices if idx in old_to_new]
        if remapped:
            remapped_spectrum_fragments.append(np.array(remapped, dtype=np.int32))
            remapped_mslevels.append(mslevel)
            remapped_msn_precursors.append(
                old_to_new.get(msn_prec, -1) if msn_prec >= 0 else -1
            )

    return FragmentationTree(
        base_inchikey=base_inchikey,
        ion_mode=ion_mode,
        precursor_formula=molecular_precursor,
        fragment_formulas=kept_formulas,
        fragment_formulas_str=kept_formulas_str,
        edge_weights=edge_weights,
        spectrum_fragments=remapped_spectrum_fragments,
        spectrum_mslevels=np.array(remapped_mslevels, dtype=np.int32),
        spectrum_msn_precursors=remapped_msn_precursors,
    )


def _formula_array_to_string(formula_array: np.ndarray) -> str:
    """Convert a formula array to a string representation."""
    from hrms_utils.formula_annotation.element_table import ELEMENT_SYMBOLS

    parts = []
    for symbol, count in zip(ELEMENT_SYMBOLS, formula_array):
        if count > 0:
            if count == 1:
                parts.append(symbol)
            else:
                parts.append(f"{symbol}{count}")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_tree(
    tree: FragmentationTree,
    output_path: Optional[str | Path] = None,
    figsize: tuple[int, int] = (14, 10),
) -> None:
    """Visualize a fragmentation tree using networkx and matplotlib.

    Args:
        tree: The FragmentationTree to visualize.
        output_path: If provided, save the figure to this path.
        figsize: Figure size in inches.
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    G = nx.DiGraph()

    # Add nodes with labels
    for i, formula_str in enumerate(tree.fragment_formulas_str):
        G.add_node(i, label=formula_str)

    # Add edges with weights
    n = tree.n_fragments
    for i in range(n):
        for j in range(n):
            w = tree.edge_weights[i, j]
            if w > 0:
                G.add_edge(i, j, weight=w)

    # Layout: hierarchical based on formula "size" (sum of elements)
    sizes = tree.fragment_formulas.sum(axis=1)
    # Group by size for y-position
    unique_sizes = np.sort(np.unique(sizes))[::-1]  # descending
    pos = {}
    for rank, size_val in enumerate(unique_sizes):
        nodes_at_rank = [i for i in range(n) if sizes[i] == size_val]
        n_nodes = len(nodes_at_rank)
        for col, node in enumerate(nodes_at_rank):
            x = (col - (n_nodes - 1) / 2.0) * 1.5
            y = -rank * 1.5
            pos[node] = (x, y)

    # Fallback for any missing positions
    for node in G.nodes():
        if node not in pos:
            pos[node] = (0, 0)

    fig, ax = plt.subplots(figsize=figsize)

    # Draw nodes
    node_colors = ["lightcoral" if i == tree.precursor_idx else "lightblue" for i in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1200, ax=ax)

    # Draw edges with width proportional to weight
    edges = G.edges()
    if edges:
        weights = [G[u][v]["weight"] * 5 for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=weights, alpha=0.6, arrows=True, arrowsize=15, ax=ax)

    # Draw labels
    labels = {i: G.nodes[i]["label"] for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)

    ax.set_title(f"Fragmentation Tree\n{tree.base_inchikey} | {tree.ion_mode}")
    ax.axis("off")
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI / utility
# ---------------------------------------------------------------------------

def load_and_build_tree(
    library_path: str | Path,
    base_inchikey: str,
    ion_mode: Optional[str] = None,
) -> Optional[FragmentationTree]:
    """Load a spectral library file and build the tree for a specific compound.

    Args:
        library_path: Path to the processed spectral library (parquet or msp/mgf).
        base_inchikey: The base InChIKey to select.
        ion_mode: Optional ion mode filter ("P" or "N").

    Returns:
        The FragmentationTree, or None if not found.
    """
    library_path = Path(library_path)

    if library_path.suffix.lower() in [".msp", ".mspec", ".mgf"]:
        from hrms_utils.formats.spectral_library import process_single_file

        df = process_single_file(library_path, includes_MSn=True)
    else:
        df = pl.read_parquet(library_path)

    # Filter
    mask = pl.col("base_inchikey") == base_inchikey
    if ion_mode is not None:
        mask = mask & (pl.col("ion_mode") == ion_mode)
    df = df.filter(mask)

    if df.is_empty():
        return None

    trees = build_fragmentation_trees(df)
    if not trees:
        return None

    # Return the first tree
    return next(iter(trees.values()))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build and visualize fragmentation trees")
    parser.add_argument("library", type=str, help="Path to spectral library file")
    parser.add_argument("inchikey", type=str, help="Base InChIKey")
    parser.add_argument("--ion-mode", type=str, default=None, help="Ion mode filter (P/N)")
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    args = parser.parse_args()

    tree = load_and_build_tree(args.library, args.inchikey, args.ion_mode)
    if tree is None:
        print(f"No tree found for {args.inchikey}")
    else:
        print(f"Built tree with {tree.n_fragments} fragments")
        visualize_tree(tree, output_path=args.output)
