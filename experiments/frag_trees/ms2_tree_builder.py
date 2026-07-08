"""
MS2 tree builder — builds MS2-only FragmentationTree objects.

Reuses the existing Phase 1 (MS2 merge + annotate) helpers from
fragmentation_tree.py, then constructs a FragmentationTree with
trivial MSn metadata (one spectrum covering all pooled MS2 fragments,
all mslevel=2, all msn_precursor=-1).

The one-spectrum-per-tree invariant is relied upon by load_trees_npz
when synthesizing trivial MS2 spectrum metadata for storage.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import polars as pl

from fragmentation_tree import (
    FragmentationTree,
    FragmentationTreeConfig,
    _annotate_mass_clusters,
    _build_superset_matrix,
    _collect_ms2_peaks,
    _compute_edge_weights,
    _formula_array_to_string,
    _is_superset,
    _match_precursor_to_fragments,
    _merge_peaks,
)
from hrms_utils.hrms_core import NUM_ELEMENTS


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_ms2_trees(
    spectral_library: pl.DataFrame,
    config: FragmentationTreeConfig | None = None,
) -> dict[tuple[str, str, str], FragmentationTree]:
    """Build MS2-only fragmentation trees from a spectral library DataFrame.

    Pools all MS2 peaks per (base_inchikey, ion_mode, precursor_type) group,
    annotates them against the molecular precursor formula, and connects
    fragments via subformula (superset/subset) relationships using the
    existing _build_superset_matrix and _compute_edge_weights helpers.

    Returns a dict keyed by (base_inchikey, ion_mode, precursor_type).
    The returned FragmentationTree objects have:

      - spectrum_mslevels  = all 2
      - spectrum_msn_precursors = all -1
      - spectrum_fragments = exactly ONE entry covering all pooled MS2
        fragment indices (this "one spectrum per tree" invariant is
        relied upon by load_trees_npz when synthesizing trivial MS2
        spectrum metadata — see tree_format_spec.md §4.4).

    so they are self-consistent FragmentationTree instances.

    Required columns:
        base_inchikey, ion_mode, precursor_type, precursor_formula_array,
        precursor_mz, cleaned_normalized_mz, cleaned_normalized_intensity, mslevel

    Args:
        spectral_library: Processed spectral library DataFrame.
        config: FragmentationTreeConfig instance. If None, uses defaults.

    Returns:
        Dictionary mapping (base_inchikey, ion_mode, precursor_type) -> FragmentationTree.
    """
    if config is None:
        config = FragmentationTreeConfig()

    required = [
        "base_inchikey",
        "ion_mode",
        "precursor_type",
        "precursor_formula_array",
        "precursor_mz",
        "cleaned_normalized_mz",
        "cleaned_normalized_intensity",
        "mslevel",
    ]
    missing = [c for c in required if c not in spectral_library.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Filter to MS2-only rows
    ms2_df = spectral_library.filter(pl.col("mslevel") == 2)
    if ms2_df.is_empty():
        return {}

    # Group by compound + ionization mode + adduct (precursor_type)
    groups = ms2_df.group_by(["base_inchikey", "ion_mode", "precursor_type"])

    trees: dict[tuple[str, str, str], FragmentationTree] = {}

    for (base_inchikey, ion_mode, precursor_type), group_df in groups:
        tree = _build_ms2_tree_for_group(
            group_df, base_inchikey, ion_mode, precursor_type, config,
        )
        if tree is not None:
            key = (base_inchikey, ion_mode, precursor_type)
            trees[key] = tree

    return trees


# ---------------------------------------------------------------------------
# Internal: build one tree per group
# ---------------------------------------------------------------------------


def _build_ms2_tree_for_group(
    group_df: pl.DataFrame,
    base_inchikey: str,
    ion_mode: str,
    precursor_type: str,
    config: FragmentationTreeConfig,
) -> Optional[FragmentationTree]:
    """Build an MS2-only FragmentationTree for a single compound group.

    The group_df must contain only mslevel==2 rows for one compound.
    """
    # -------------------------------------------------------------------
    # Step 1: Extract molecular precursor from first MS2 row
    # -------------------------------------------------------------------
    molecular_precursor = np.array(
        group_df["precursor_formula_array"][0], dtype=np.int32,
    )
    molecular_precursor_str = _formula_array_to_string(molecular_precursor)
    molecular_precursor_mass = float(group_df["precursor_mz"][0])

    # -------------------------------------------------------------------
    # Step 2: Collect and merge MS2 peaks (Phase 1 reuse)
    # -------------------------------------------------------------------
    masses_arr, intensities_arr, spec_indices, peak_indices = _collect_ms2_peaks(
        group_df, config,
    )
    if len(masses_arr) == 0:
        return None

    _, cluster_masses, _, _, _ = _merge_peaks(
        masses_arr, intensities_arr, spec_indices, peak_indices, config,
    )
    n_clusters = len(cluster_masses)
    if n_clusters == 0:
        return None

    # -------------------------------------------------------------------
    # Step 3: Annotate merged MS2 clusters against molecular precursor
    # -------------------------------------------------------------------
    # Use the molecular precursor as the upper bound for all annotations
    max_bounds = np.tile(molecular_precursor, (n_clusters, 1))
    formulas, formulas_str, errors_ppm = _annotate_mass_clusters(
        cluster_masses, max_bounds, config,
    )

    # -------------------------------------------------------------------
    # Step 4: Filter out zero-formula (unannotatable) fragments
    # -------------------------------------------------------------------
    valid_mask = np.any(formulas != 0, axis=1)
    n_valid = int(valid_mask.sum())
    if n_valid == 0:
        return None

    formulas_valid = formulas[valid_mask]
    formulas_str_valid = [s for s, v in zip(formulas_str, valid_mask) if v]
    errors_ppm_valid = errors_ppm[valid_mask]
    masses_valid = cluster_masses[valid_mask]

    # -------------------------------------------------------------------
    # Step 5: Ensure molecular precursor is present among fragments
    # -------------------------------------------------------------------
    # Try to match the precursor by mass against the valid cluster masses.
    # The mass array comes from _merge_peaks → _compute_cluster_representatives
    # which iterates clusters in ascending mass order, so it is sorted.
    prec_match = _match_precursor_to_fragments(
        molecular_precursor_mass, masses_valid, config.merge_tolerance_ppm,
    )

    if prec_match >= 0:
        # Precursor mass matches a valid cluster. Use that cluster as the
        # precursor fragment, overriding its formula/error to the exact
        # molecular precursor.
        final_formulas_list = [formulas_valid[i].copy() for i in range(n_valid)]
        final_formulas_str_list = list(formulas_str_valid)
        final_errors_list = [float(errors_ppm_valid[i]) for i in range(n_valid)]

        final_formulas_list[prec_match] = molecular_precursor.copy()
        final_formulas_str_list[prec_match] = molecular_precursor_str
        final_errors_list[prec_match] = 0.0
        precursor_idx = prec_match
    else:
        # Precursor mass not found among MS2 clusters. Prepend the molecular
        # precursor so it is the first fragment (index 0).
        precursor_idx = 0
        final_formulas_list = [molecular_precursor.copy()]
        final_formulas_str_list = [molecular_precursor_str]
        final_errors_list = [0.0]
        for i in range(n_valid):
            final_formulas_list.append(formulas_valid[i].copy())
            final_formulas_str_list.append(formulas_str_valid[i])
            final_errors_list.append(float(errors_ppm_valid[i]))

    n_final = len(final_formulas_list)
    final_formulas_arr = np.stack(final_formulas_list, axis=0)
    final_errors_arr = np.array(final_errors_list, dtype=np.float64)

    # -------------------------------------------------------------------
    # Step 6: Build superset matrix and edge weights
    # -------------------------------------------------------------------
    superset_matrix = _build_superset_matrix(final_formulas_arr)
    edge_weights = _compute_edge_weights(superset_matrix)

    # -------------------------------------------------------------------
    # Step 7: Construct FragmentationTree with trivial MSn metadata
    #
    # Exactly ONE spectrum entry covering ALL final fragments:
    #   - spectrum_fragments[0] = np.arange(n_final)
    #   - mslevel = 2
    #   - msn_precursor = -1
    # -------------------------------------------------------------------
    tree = FragmentationTree(
        base_inchikey=base_inchikey,
        ion_mode=ion_mode,
        precursor_formula=molecular_precursor,
        fragment_formulas=final_formulas_arr,
        fragment_formulas_str=final_formulas_str_list,
        edge_weights=edge_weights,
        spectrum_fragments=[np.arange(n_final, dtype=np.int32)],
        spectrum_mslevels=np.array([2], dtype=np.int32),
        spectrum_msn_precursors=[-1],
        fragment_errors_ppm=final_errors_arr,
    )

    return tree


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    parquet_path = sys.argv[1] if len(sys.argv) > 1 else "cladribine.parquet"

    print(f"Loading spectral library from: {parquet_path}")
    df = pl.read_parquet(parquet_path)
    print(f"  Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print(f"  MS2 rows: {df.filter(pl.col('mslevel') == 2).shape[0]}")

    config = FragmentationTreeConfig()
    trees = build_ms2_trees(df, config)

    print(f"\nBuilt {len(trees)} MS2 tree(s):")
    for key, tree in trees.items():
        base_inchikey, ion_mode, precursor_type = key
        n = tree.n_fragments
        n_edges = int(np.count_nonzero(tree.edge_weights))
        print(
            f"  {base_inchikey} | {ion_mode} | {precursor_type}: "
            f"{n} fragments, {n_edges} edges, "
            f"{len(tree.spectrum_fragments)} spectrum entry"
        )
        # Verify one-spectrum invariant
        assert len(tree.spectrum_fragments) == 1, (
            f"Expected 1 spectrum entry, got {len(tree.spectrum_fragments)}"
        )
        assert tree.spectrum_mslevels.tolist() == [2], (
            f"Expected mslevel=[2], got {tree.spectrum_mslevels.tolist()}"
        )
        assert tree.spectrum_msn_precursors == [-1], (
            f"Expected msn_precursors=[-1], got {tree.spectrum_msn_precursors}"
        )

    print("\nAll assertions passed. MS2 tree builder works correctly.")
