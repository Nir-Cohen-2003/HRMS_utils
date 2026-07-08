"""
Fragmentation tree builder for MSn spectral libraries.

Groups spectra by compound (base_inchikey) and ionization mode,
builds a fragmentation graph where edges connect fragments to their
formula-parents using a three-phase pipeline:

    Phase 1: MS2 merge + annotate
    Phase 2: Top-down MSn annotation with incremental precursor linking,
             global mass re-clustering, and tightest-bounds re-annotation.
    Phase 3: Edge construction using "lowest provable parent" rules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
from numba import njit, prange

from hrms_utils.formula_annotation.utils import format_formula_string_to_array
from hrms_utils.hrms_core import NUM_ELEMENTS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FragmentationTreeConfig:
    """Configuration for fragmentation tree construction.

    Attributes:
        merge_tolerance_ppm: Mass tolerance for merging peaks across spectra
            (same fragment observed in different spectra/energies). Default 5.0.
        annotation_tolerance_ppm: Mass tolerance passed to
            decompose_mass_with_bounds. Default 5.0.
        min_dbe: Minimum double-bond equivalent for formula decomposition.
            Default -0.5.
        max_dbe: Maximum DBE. Default 40.0.
        dbe_mode: DBE calculation mode ("half_integer" allows radicals).
            Default "half_integer".
        water_absorption: If True, add +2 H and +1 O to max_bounds
            (matching upstream clean_and_normalize_spectrum behavior).
            Default True.
    """
    merge_tolerance_ppm: float = 5.0
    annotation_tolerance_ppm: float = 5.0
    min_dbe: float = -0.5
    max_dbe: float = 40.0
    dbe_mode: str = "half_integer"
    water_absorption: bool = True


# ---------------------------------------------------------------------------
# Numba helpers — preserved from original
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


# ---------------------------------------------------------------------------
# New Numba helpers — Section 5 of the plan
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _cluster_masses_sorted(
    masses: np.ndarray,        # shape (n_peaks,), float64, MUST be sorted ascending
    tolerance_ppm: float,      # e.g. 5.0
) -> np.ndarray:               # shape (n_peaks,), int32 -- cluster ID per peak
    """Assign cluster IDs to sorted masses using greedy single-linkage clustering.

    A new cluster starts when |mass[i] - ref_mass| / ref_mass > tolerance_ppm * 1e-6,
    where ref_mass is the first mass of the current cluster.
    """
    n = masses.shape[0]
    cluster_ids = np.zeros(n, dtype=np.int32)
    if n == 0:
        return cluster_ids
    current_cluster = 0
    ref_mass = masses[0]
    cluster_ids[0] = current_cluster
    for i in range(1, n):
        if abs(masses[i] - ref_mass) / ref_mass > tolerance_ppm * 1e-6:
            current_cluster += 1
            ref_mass = masses[i]
        cluster_ids[i] = current_cluster
    return cluster_ids


@njit(cache=True, fastmath=True)
def _compute_cluster_representatives(
    masses: np.ndarray,         # shape (n_peaks,), float64, sorted
    intensities: np.ndarray,    # shape (n_peaks,), float64
    cluster_ids: np.ndarray,    # shape (n_peaks,), int32
    n_clusters: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute intensity-weighted mean mass and max intensity per cluster.

    Returns:
        cluster_masses: shape (n_clusters,), float64
        cluster_intensities: shape (n_clusters,), float64
    """
    cluster_masses = np.zeros(n_clusters, dtype=np.float64)
    cluster_intensities = np.zeros(n_clusters, dtype=np.float64)
    total_intensity = np.zeros(n_clusters, dtype=np.float64)
    count = np.zeros(n_clusters, dtype=np.int32)
    n = masses.shape[0]
    for i in range(n):
        cid = cluster_ids[i]
        cluster_masses[cid] += masses[i] * intensities[i]
        total_intensity[cid] += intensities[i]
        if intensities[i] > cluster_intensities[cid]:
            cluster_intensities[cid] = intensities[i]
        count[cid] += 1
    for cid in range(n_clusters):
        if total_intensity[cid] > 0:
            cluster_masses[cid] /= total_intensity[cid]
    return cluster_masses, cluster_intensities


@njit(cache=True, fastmath=True)
def _match_precursor_to_fragments(
    precursor_mz: float,            # m/z to match
    fragment_masses: np.ndarray,    # shape (n_fragments,), float64, sorted
    tolerance_ppm: float,
) -> int:                           # fragment index, or -1 if no match
    """Find the closest fragment mass to precursor_mz within tolerance.

    Uses binary search for O(log n) lookup. Returns -1 if no fragment
    is within tolerance_ppm of precursor_mz.
    """
    n = fragment_masses.shape[0]
    if n == 0:
        return -1

    # Binary search for insertion point
    lo = 0
    hi = n - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if fragment_masses[mid] < precursor_mz:
            lo = mid + 1
        elif fragment_masses[mid] > precursor_mz:
            hi = mid - 1
        else:
            # Exact match
            return mid

    # lo is the insertion point; check neighbors
    best_idx = -1
    best_diff = np.inf
    for candidate in (lo - 1, lo):
        if 0 <= candidate < n:
            diff = abs(fragment_masses[candidate] - precursor_mz)
            if diff / precursor_mz <= tolerance_ppm * 1e-6 and diff < best_diff:
                best_diff = diff
                best_idx = candidate
    return best_idx


@njit(cache=True, fastmath=True, parallel=True)
def _compute_tightest_bounds(
    cluster_ids_flat: np.ndarray,            # shape (k,), int32 -- cluster ID per (cluster, spectrum) pair
    spectrum_ids_flat: np.ndarray,           # shape (k,), int32 -- spectrum ID per pair
    spectrum_precursor_formulas: np.ndarray,  # shape (n_spectra, NUM_ELEMENTS), int32
    n_clusters: int,
) -> np.ndarray:                             # shape (n_clusters, NUM_ELEMENTS), int32
    """Compute element-wise minimum of all precursor formulas per cluster.

    For each cluster, collects all precursor formulas from spectra where
    the cluster's mass appears, and takes the element-wise min.
    This is the tightest provable upper bound for formula annotation.
    """
    n_elements = spectrum_precursor_formulas.shape[1]
    tightest = np.empty((n_clusters, n_elements), dtype=np.int32)
    # Initialize to large values
    for c in prange(n_clusters):
        for e in range(n_elements):
            tightest[c, e] = np.iinfo(np.int32).max

    k = cluster_ids_flat.shape[0]
    for i in range(k):
        cid = cluster_ids_flat[i]
        sid = spectrum_ids_flat[i]
        for e in range(n_elements):
            val = spectrum_precursor_formulas[sid, e]
            if val < tightest[cid, e]:
                tightest[cid, e] = val

    return tightest


@njit(cache=True, fastmath=True)
def _build_observation_matrix(
    peak_fragment_ids: np.ndarray,       # shape (total_peaks,), int32 -- flat peak fragment indices
    peak_offsets: np.ndarray,            # shape (n_spectra + 1,), int32 -- CSR offsets
    spectrum_precursor_indices: np.ndarray,  # shape (n_spectra,), int32
    n_fragments: int,
    n_spectra: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the spectral observation matrix.

    NOTE: The spectrum's own precursor fragment is EXCLUDED from in_spectrum.
    in_spectrum[p, s] is NOT set even though p is present in spectrum s (as the
    isolated ion). This is intentional for Rule 3 correctness: the precursor is
    the starting material, not a produced fragment. Including it would create
    false evidence that p was "produced" in spectrum s, which could prevent
    Rule 3 from correctly removing edges through intermediates. Only non-precursor
    peaks are recorded as "observed in" a spectrum.

    Returns:
        observation_matrix: shape (n_fragments, n_fragments), bool_
            P[A, B] = True if B is a peak in a spectrum whose precursor is A.
            (Precursor A itself is excluded -- P[A, A] is never set.)
        has_own_spectrum: shape (n_fragments,), bool_
            True if any spectrum has this fragment as precursor.
        in_spectrum: shape (n_fragments, n_spectra), bool_
            in_spectrum[B, s] = True if B is a peak in spectrum s.
            (Excludes the spectrum's own precursor.)
    """
    observation_matrix = np.zeros((n_fragments, n_fragments), dtype=np.bool_)
    has_own_spectrum = np.zeros(n_fragments, dtype=np.bool_)
    in_spectrum = np.zeros((n_fragments, n_spectra), dtype=np.bool_)

    for s in range(n_spectra):
        p = spectrum_precursor_indices[s]
        if p >= 0:
            has_own_spectrum[p] = True

        start = peak_offsets[s]
        end = peak_offsets[s + 1]
        for idx in range(start, end):
            b = peak_fragment_ids[idx]
            # Exclude the precursor itself
            if b == p:
                continue
            if p >= 0:
                observation_matrix[p, b] = True
            in_spectrum[b, s] = True

    return observation_matrix, has_own_spectrum, in_spectrum


@njit(cache=True, fastmath=True)
def _build_edges_lowest_provable_parent(
    superset_matrix: np.ndarray,         # shape (n, n), bool_ -- chemical validity
    observation_matrix: np.ndarray,      # shape (n, n), bool_ -- P[A, B]
    has_own_spectrum: np.ndarray,        # shape (n,), bool_
    in_spectrum: np.ndarray,             # shape (n, n_spectra), bool_
    n_spectra: int,
    molecular_precursor_idx: int,
) -> np.ndarray:                         # shape (n, n), bool_ -- final edges
    """Build edge matrix using 'lowest provable parent' rules.

    Rule 1 (transitive reduction with evidence):
      For edge A->C, if exists B where P[A,B] and P[B,C] and superset[B,C],
      remove A->C. (C is provably produced from B, a child of A.)

      NOTE: P[A, C] (A observed in C's spectrum) is NOT checked in Rule 1.
      The superset matrix is used as edge candidates -- if C is chemically valid
      as a child of A but is provably produced via B (an intermediate), there is
      no reason to keep the direct A->C edge regardless of whether A also appears
      in C's spectrum. The spectral evidence chain A->B->C is sufficient to prove
      C is produced from B, making the direct A->C edge redundant. We want the
      LOWEST provable parent, not all provable parents.

    Rule 3 (intermediate without spectrum):
      For edge B->C where B has no own spectrum (has_own_spectrum[B] is False):
      If any spectrum has B as peak but NOT C, remove B->C.
      (B existed without producing C.)

    Rule 2 (co-parenting, default):
      Edges not removed by Rules 1 or 3 are kept.
    """
    n = superset_matrix.shape[0]
    edge_matrix = superset_matrix.copy()

    # Rule 1: transitive reduction with spectral evidence
    for a in range(n):
        for c in range(n):
            if a == c or not edge_matrix[a, c]:
                continue
            for b in range(n):
                if b == a or b == c:
                    continue
                if (observation_matrix[a, b]
                        and observation_matrix[b, c]
                        and superset_matrix[b, c]):
                    edge_matrix[a, c] = False
                    break

    # Rule 3: intermediate without spectrum
    for b in range(n):
        if has_own_spectrum[b]:
            continue
        for c in range(n):
            if b == c or not edge_matrix[b, c]:
                continue
            # Check if B exists without C in any spectrum
            remove = False
            for s in range(n_spectra):
                if in_spectrum[b, s] and not in_spectrum[c, s]:
                    remove = True
                    break
            if remove:
                edge_matrix[b, c] = False

    # Rule 2 (co-parenting): edges not removed by Rules 1 or 3 are kept (default)
    return edge_matrix


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
    # Annotation error per fragment (backward-compatible, defaults to empty)
    fragment_errors_ppm: np.ndarray = field(default_factory=lambda: np.array([]))  # shape (n_fragments,)

    def __post_init__(self) -> None:
        """Validate that no fragment formula is all zeros."""
        n = self.fragment_formulas.shape[0]
        for i in range(n):
            assert np.any(self.fragment_formulas[i] != 0), (
                f"Fragment {i} has an empty (all-zero) formula. "
                f"Formula string: '{self.fragment_formulas_str[i]}'. "
                f"This is a bug in tree construction — zero-formula "
                f"clusters should have been filtered out."
            )

    @property
    def n_fragments(self) -> int:
        return self.fragment_formulas.shape[0]

    @property
    def precursor_idx(self) -> int:
        """The molecular precursor is the largest fragment (superset of all others)."""
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
# Python/Polars helper functions — Phase 2
# ---------------------------------------------------------------------------

def _collect_ms2_peaks(
    group_df: pl.DataFrame,
    config: FragmentationTreeConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect all MS2 peaks from the group DataFrame into flat parallel arrays.

    Returns:
        masses: shape (n_peaks,), float64 -- peak masses (sorted ascending)
        intensities: shape (n_peaks,), float64 -- peak intensities
        spectrum_indices: shape (n_peaks,), int32 -- which spectrum each peak came from
        peak_indices: shape (n_peaks,), int32 -- original peak position within spectrum
    """
    masses_list: list[float] = []
    intensities_list: list[float] = []
    spectrum_indices_list: list[int] = []
    peak_indices_list: list[int] = []

    for spec_idx, row in enumerate(group_df.iter_rows(named=True)):
        if row["mslevel"] != 2:
            continue
        mz = row["cleaned_normalized_mz"]
        intensity = row["cleaned_normalized_intensity"]
        if mz is None or intensity is None:
            continue
        if isinstance(mz, pl.Series):
            mz = mz.to_list()
        if isinstance(intensity, pl.Series):
            intensity = intensity.to_list()
        for pk_idx in range(len(mz)):
            masses_list.append(float(mz[pk_idx]))
            intensities_list.append(float(intensity[pk_idx]))
            spectrum_indices_list.append(spec_idx)
            peak_indices_list.append(pk_idx)

    masses_arr = np.array(masses_list, dtype=np.float64)
    intensities_arr = np.array(intensities_list, dtype=np.float64)

    # Sort by mass
    sort_order = np.argsort(masses_arr)
    return (
        masses_arr[sort_order],
        intensities_arr[sort_order],
        np.array(spectrum_indices_list, dtype=np.int32)[sort_order],
        np.array(peak_indices_list, dtype=np.int32)[sort_order],
    )


def _merge_peaks(
    masses: np.ndarray,
    intensities: np.ndarray,
    spectrum_indices: np.ndarray,
    peak_indices: np.ndarray,
    config: FragmentationTreeConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Merge peaks by mass tolerance into clusters.

    Sorts by mass, calls _cluster_masses_sorted and _compute_cluster_representatives,
    then builds cluster-spectrum membership as flat parallel arrays.

    Returns:
        cluster_ids: shape (n_peaks,), int32 -- cluster ID per peak
        cluster_masses: shape (n_clusters,), float64 -- intensity-weighted mean masses
        cluster_intensities: shape (n_clusters,), float64 -- max intensities
        cluster_ids_flat: shape (k,), int32 -- deduplicated (cluster_id, spectrum_idx) pairs
        spectrum_ids_flat: shape (k,), int32 -- deduplicated pairs
    """
    # Sort by mass
    sort_order = np.argsort(masses)
    masses_sorted = masses[sort_order]
    intensities_sorted = intensities[sort_order]
    spectrum_indices_sorted = spectrum_indices[sort_order]
    peak_indices_sorted = peak_indices[sort_order]

    # Cluster
    n = len(masses_sorted)
    cluster_ids = _cluster_masses_sorted(masses_sorted, config.merge_tolerance_ppm)
    n_clusters = int(np.max(cluster_ids)) + 1 if n > 0 else 0

    # Compute representatives
    cluster_masses, cluster_intensities = _compute_cluster_representatives(
        masses_sorted, intensities_sorted, cluster_ids, n_clusters,
    )

    # Build spectrum membership: deduplicate (cluster_id, spectrum_idx) pairs
    pairs_set: set[tuple[int, int]] = set()
    for i in range(n):
        pairs_set.add((int(cluster_ids[i]), int(spectrum_indices_sorted[i])))

    k = len(pairs_set)
    cluster_ids_flat = np.zeros(k, dtype=np.int32)
    spectrum_ids_flat = np.zeros(k, dtype=np.int32)
    for idx, (cid, sid) in enumerate(sorted(pairs_set)):
        cluster_ids_flat[idx] = cid
        spectrum_ids_flat[idx] = sid

    return cluster_ids, cluster_masses, cluster_intensities, cluster_ids_flat, spectrum_ids_flat


def _annotate_mass_clusters(
    cluster_masses: np.ndarray,
    max_bounds_per_cluster: np.ndarray,
    config: FragmentationTreeConfig,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Annotate mass clusters via decompose_mass_with_bounds.

    If config.water_absorption is True, adds +2 H and +1 O to max_bounds
    before calling the decomposition function.

    Formula selection: lowest errors_ppm, tie-break by lowest total atom count.
    If no candidates, formula is set to zeros and error to a large value.
    Zero-formula clusters are filtered out in _build_tree_for_group before
    tree assembly.

    Args:
        cluster_masses: shape (n_clusters,), float64
        max_bounds_per_cluster: shape (n_clusters, NUM_ELEMENTS), int32
        config: FragmentationTreeConfig

    Returns:
        formulas: shape (n_clusters, NUM_ELEMENTS), int32
        formulas_str: list of str, length n_clusters
        errors_ppm: shape (n_clusters,), float64
    """
    n = len(cluster_masses)
    if n == 0:
        return (
            np.zeros((0, NUM_ELEMENTS), dtype=np.int32),
            [],
            np.array([], dtype=np.float64),
        )

    # Apply water absorption to max_bounds
    max_bounds_adjusted = max_bounds_per_cluster.copy()
    if config.water_absorption:
        # H at index 0, O at index 3
        max_bounds_adjusted[:, 0] += 2
        max_bounds_adjusted[:, 3] += 1

    min_bounds = np.zeros((n, NUM_ELEMENTS), dtype=np.int32)

    df = pl.DataFrame({
        "mass_data": [
            {
                "mass": float(cluster_masses[i]),
                "min_bounds": min_bounds[i].tolist(),
                "max_bounds": max_bounds_adjusted[i].tolist(),
            }
            for i in range(n)
        ],
    }, schema={
        "mass_data": pl.Struct([
            pl.Field("mass", pl.Float64),
            pl.Field("min_bounds", pl.Array(pl.Int32, NUM_ELEMENTS)),
            pl.Field("max_bounds", pl.Array(pl.Int32, NUM_ELEMENTS)),
        ]),
    })

    result_df = df.with_columns(
        pl.col("mass_data").mass_decomposition.decompose_mass_with_bounds(
            tolerance_ppm=config.annotation_tolerance_ppm,
            min_dbe=config.min_dbe,
            max_dbe=config.max_dbe,
            dbe_mode=config.dbe_mode,
        ).alias("decomposed"),
    ).with_columns(
        pl.col("decomposed").struct.unnest(),
    )

    formulas_list = result_df["formulas"].to_list()
    formulas_str_list = result_df["formulas_str"].to_list()
    errors_list = result_df["errors_ppm"].to_list()

    final_formulas = np.zeros((n, NUM_ELEMENTS), dtype=np.int32)
    final_formulas_str: list[str] = []
    final_errors = np.full(n, np.inf, dtype=np.float64)

    for i in range(n):
        candidates_arr = formulas_list[i]
        candidates_str = formulas_str_list[i]
        errors = errors_list[i]

        if len(candidates_arr) == 0:
            # No candidates: set zeros and large error
            logger.debug(
                "No formula candidates for mass %.4f with max_bounds %s",
                cluster_masses[i],
                max_bounds_per_cluster[i].tolist(),
            )
            final_formulas_str.append("")
            continue

        # Lowest error_ppm, tie-break by lowest total atom count
        best_idx = 0
        best_error = errors[0]
        for j in range(1, len(errors)):
            if errors[j] < best_error:
                best_error = errors[j]
                best_idx = j
            elif errors[j] == best_error:
                # Tie-break by lowest total atom count
                if np.sum(candidates_arr[j]) < np.sum(candidates_arr[best_idx]):
                    best_idx = j

        final_formulas[i] = np.array(candidates_arr[best_idx], dtype=np.int32)
        final_formulas_str.append(str(candidates_str[best_idx]))
        final_errors[i] = best_error

    return final_formulas, final_formulas_str, final_errors


def _initial_msn_precursor_linking(
    group_df: pl.DataFrame,
    ms2_fragment_masses: np.ndarray,
    ms2_fragment_formulas: np.ndarray,
    molecular_precursor_formula: np.ndarray,
    config: FragmentationTreeConfig,
) -> np.ndarray:
    """Initial MS3 precursor linking against MS2 fragment masses.

    ONLY handles MS3 spectra (matching precursor_mz against MS2 masses).
    MS4+ spectra are left as zeros (to be filled incrementally during
    _annotate_msn_levels). MS2 spectra get the molecular precursor formula.

    Args:
        group_df: The group DataFrame.
        ms2_fragment_masses: shape (n_ms2_clusters,), float64, sorted cluster masses.
        ms2_fragment_formulas: shape (n_ms2_clusters, NUM_ELEMENTS), int32.
        molecular_precursor_formula: shape (NUM_ELEMENTS,), int32.
        config: FragmentationTreeConfig.

    Returns:
        spectrum_precursor_formulas: shape (n_spectra, NUM_ELEMENTS), int32.
    """
    n_spectra = group_df.height
    spectrum_precursor_formulas = np.zeros((n_spectra, NUM_ELEMENTS), dtype=np.int32)

    for s_idx, row in enumerate(group_df.iter_rows(named=True)):
        mslevel = row["mslevel"]
        if mslevel == 2:
            spectrum_precursor_formulas[s_idx] = molecular_precursor_formula
        elif mslevel == 3:
            precursor_mz = float(row["precursor_mz"])
            match_idx = _match_precursor_to_fragments(
                precursor_mz, ms2_fragment_masses, config.merge_tolerance_ppm,
            )
            if match_idx >= 0:
                spectrum_precursor_formulas[s_idx] = ms2_fragment_formulas[match_idx]
            # else: leave as zeros (unmatched MS3)
        # mslevel >= 4: leave as zeros (filled during _annotate_msn_levels)

    return spectrum_precursor_formulas


def _global_recluster_fragments(
    all_fragment_masses: np.ndarray,
    all_fragment_intensities: np.ndarray,
    cluster_ids_flat_input: np.ndarray,
    spectrum_ids_flat_input: np.ndarray,
    config: FragmentationTreeConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Globally re-cluster all fragment masses from all levels.

    This prevents duplicate cluster IDs for the same physical fragment that
    appeared at multiple levels and was clustered separately within each
    precursor group.

    Returns:
        unified_cluster_ids: shape (n_masses,), int32 -- unified cluster per input mass.
        unified_cluster_masses: shape (n_unified,), float64 -- representative masses.
        cluster_ids_flat_output: shape (k,), int32 -- deduplicated (unified_cluster, spectrum) pairs.
        spectrum_ids_flat_output: shape (k,), int32 -- parallel to cluster_ids_flat_output.
    """
    n = len(all_fragment_masses)
    if n == 0:
        return (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
        )

    # Sort by mass
    sort_order = np.argsort(all_fragment_masses)
    sorted_masses = all_fragment_masses[sort_order]
    sorted_intensities = all_fragment_intensities[sort_order]

    # Cluster
    unified_ids_sorted = _cluster_masses_sorted(sorted_masses, config.merge_tolerance_ppm)
    n_unified = int(np.max(unified_ids_sorted)) + 1

    # Compute representatives
    unified_masses, _ = _compute_cluster_representatives(
        sorted_masses, sorted_intensities, unified_ids_sorted, n_unified,
    )

    # Build mapping from original (sorted) position to unified cluster
    # Then remap the input cluster_ids_flat/spectrum_ids_flat pairs
    orig_to_unified = np.zeros(n, dtype=np.int32)
    for new_pos, orig_pos in enumerate(sort_order):
        orig_to_unified[orig_pos] = unified_ids_sorted[new_pos]

    # Remap spectrum pairs
    pairs_set: set[tuple[int, int]] = set()
    k_in = len(cluster_ids_flat_input)
    for i in range(k_in):
        cid = int(cluster_ids_flat_input[i])
        sid = int(spectrum_ids_flat_input[i])
        unified_cid = int(orig_to_unified[cid])
        pairs_set.add((unified_cid, sid))

    k_out = len(pairs_set)
    out_cluster_ids = np.zeros(k_out, dtype=np.int32)
    out_spectrum_ids = np.zeros(k_out, dtype=np.int32)
    for idx, (cid, sid) in enumerate(sorted(pairs_set)):
        out_cluster_ids[idx] = cid
        out_spectrum_ids[idx] = sid

    return orig_to_unified, unified_masses, out_cluster_ids, out_spectrum_ids


def _reannotate_with_tightest_bounds(
    unified_cluster_masses: np.ndarray,
    cluster_ids_flat: np.ndarray,
    spectrum_ids_flat: np.ndarray,
    spectrum_precursor_formulas: np.ndarray,
    config: FragmentationTreeConfig,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Re-annotate unified clusters with tightest element-wise bounds.

    Operates on UNIFIED clusters (must run AFTER global re-clustering).

    Returns:
        final_formulas: shape (n_unified, NUM_ELEMENTS), int32.
        final_formulas_str: list of str, length n_unified.
        final_errors_ppm: shape (n_unified,), float64.
    """
    n_clusters = len(unified_cluster_masses)
    if n_clusters == 0:
        return (
            np.zeros((0, NUM_ELEMENTS), dtype=np.int32),
            [],
            np.array([], dtype=np.float64),
        )

    tightest_bounds = _compute_tightest_bounds(
        cluster_ids_flat, spectrum_ids_flat, spectrum_precursor_formulas, n_clusters,
    )

    return _annotate_mass_clusters(unified_cluster_masses, tightest_bounds, config)


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_fragmentation_trees(
    df: pl.DataFrame,
    config: Optional[FragmentationTreeConfig] = None,
    mass_tolerance_ppm: Optional[float] = None,
) -> dict[tuple[str, str, str], FragmentationTree]:
    """Build fragmentation trees from a processed spectral library DataFrame.

    Groups by (base_inchikey, ion_mode, precursor_type), combines ALL
    spectra of the same molecule and adduct (MS2, MS3, MS4, different
    energies, etc.) into a single fragmentation graph.

    Three-phase pipeline:
      Phase 1: Merge-then-annotate MS2 peaks.
      Phase 2: Top-down MSn annotation with incremental precursor linking,
               global mass re-clustering, and tightest-bounds re-annotation.
      Phase 3: Edge construction using "lowest provable parent" rules.

    Backward-compatible: accepts either a FragmentationTreeConfig or the
    legacy mass_tolerance_ppm float.

    Required columns:
        base_inchikey, ion_mode, precursor_type, precursor_formula_array,
        precursor_mz, cleaned_normalized_mz, cleaned_normalized_intensity, mslevel

    Args:
        df: Processed spectral library DataFrame.
        config: FragmentationTreeConfig instance. If None and mass_tolerance_ppm
            is None, uses defaults.
        mass_tolerance_ppm: Legacy parameter. If provided, constructs a config
            with merge_tolerance_ppm=mass_tolerance_ppm and
            annotation_tolerance_ppm=mass_tolerance_ppm.

    Returns:
        Dictionary mapping (base_inchikey, ion_mode, precursor_type) -> FragmentationTree.
    """
    # Backward-compatible config construction
    if config is None:
        if mass_tolerance_ppm is not None:
            config = FragmentationTreeConfig(
                merge_tolerance_ppm=mass_tolerance_ppm,
                annotation_tolerance_ppm=mass_tolerance_ppm,
            )
        else:
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
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Group by compound + ionization mode + adduct (precursor_type)
    groups = df.group_by(["base_inchikey", "ion_mode", "precursor_type"])

    trees: dict[tuple[str, str, str], FragmentationTree] = {}

    for (base_inchikey, ion_mode, precursor_type), group_df in groups:
        tree = _build_tree_for_group(
            group_df, base_inchikey, ion_mode, precursor_type, config,
        )
        if tree is not None:
            key = (base_inchikey, ion_mode, precursor_type)
            trees[key] = tree

    return trees


def _build_tree_for_group(
    group_df: pl.DataFrame,
    base_inchikey: str,
    ion_mode: str,
    precursor_type: str,
    config: FragmentationTreeConfig,
) -> Optional[FragmentationTree]:
    """Build a FragmentationTree for a single compound group.

    Orchestrates Phase 1 → Phase 2 → Phase 3 with correct step ordering.
    """
    n_spectra = group_df.height

    # -----------------------------------------------------------------------
    # Step 1: Extract molecular precursor from MS2 rows
    # -----------------------------------------------------------------------
    ms2_rows_df = group_df.filter(pl.col("mslevel") == 2)
    if ms2_rows_df.is_empty():
        return None

    molecular_precursor = np.array(
        ms2_rows_df["precursor_formula_array"][0], dtype=np.int32,
    )
    molecular_precursor_str = _formula_array_to_string(molecular_precursor)
    molecular_precursor_mass = float(ms2_rows_df["precursor_mz"][0])

    # -----------------------------------------------------------------------
    # Step 2: Phase 1 — MS2 merge + annotate
    # -----------------------------------------------------------------------
    ms2_masses_arr, ms2_intensities_arr, ms2_spec_indices, ms2_peak_indices = (
        _collect_ms2_peaks(group_df, config)
    )
    if len(ms2_masses_arr) == 0:
        return None

    ms2_cluster_ids, ms2_cluster_masses, ms2_cluster_intensities, \
        ms2_cluster_ids_flat, ms2_spectrum_ids_flat = _merge_peaks(
            ms2_masses_arr, ms2_intensities_arr, ms2_spec_indices,
            ms2_peak_indices, config,
        )
    n_ms2 = len(ms2_cluster_masses)

    # Annotate MS2 clusters with molecular precursor as max_bounds
    ms2_max_bounds = np.tile(molecular_precursor, (n_ms2, 1))
    ms2_formulas, ms2_formulas_str, ms2_errors_ppm = _annotate_mass_clusters(
        ms2_cluster_masses, ms2_max_bounds, config,
    )

    # -----------------------------------------------------------------------
    # Step 3: Phase 2a — Initial MS3 linking (only MS3 against MS2 masses)
    # -----------------------------------------------------------------------
    spectrum_precursor_formulas = _initial_msn_precursor_linking(
        group_df, ms2_cluster_masses, ms2_formulas, molecular_precursor, config,
    )

    # -----------------------------------------------------------------------
    # Step 4: Phase 2b — Level-by-level MSn annotation with incremental rematch
    # -----------------------------------------------------------------------
    # Build per-level data for MS3+.
    # all_fragment_masses_from_msn: list of float, per-cluster masses from MS3+
    # all_fragment_intensities_from_msn: parallel list of intensities
    # msn_cluster_pos_pairs: list of (position_in_all_fragments, spectrum_idx)
    #   where position_in_all_fragments is relative to the combined array
    #   (MS2 masses at positions 0..n_ms2-1, then MS3+ appended after)
    all_fragment_masses: list[float] = list(ms2_cluster_masses)
    all_fragment_intensities: list[float] = list(ms2_cluster_intensities)
    all_spectrum_pairs: list[tuple[int, int]] = []

    # Add MS2 spectrum pairs (remap: MS2 cluster_id = position in all_fragment_masses)
    for i in range(len(ms2_cluster_ids_flat)):
        cid = int(ms2_cluster_ids_flat[i])
        sid = int(ms2_spectrum_ids_flat[i])
        all_spectrum_pairs.append((cid, sid))

    updated_precursor_formulas = spectrum_precursor_formulas.copy()
    max_mslevel = int(group_df["mslevel"].max())

    for current_level in range(3, max_mslevel + 1):
        # Collect spectra at this level with non-zero precursor formula
        level_spectra: list[int] = []
        for s_idx in range(n_spectra):
            row = group_df.row(s_idx, named=True)
            if row["mslevel"] == current_level and np.any(updated_precursor_formulas[s_idx] != 0):
                level_spectra.append(s_idx)

        if not level_spectra:
            continue

        # Group spectra by their precursor formula
        formula_groups: dict[tuple[int, ...], list[int]] = {}
        for s_idx in level_spectra:
            ft = tuple(int(x) for x in updated_precursor_formulas[s_idx])
            formula_groups.setdefault(ft, []).append(s_idx)

        # Track newly-annotated clusters for the REMATCH step
        level_masses_list: list[float] = []

        for prec_formula_tuple, spec_indices in formula_groups.items():
            prec_formula_arr = np.array(prec_formula_tuple, dtype=np.int32)

            # Collect peaks from all spectra in this group
            gm: list[float] = []
            gi: list[float] = []
            gs: list[int] = []
            gp: list[int] = []

            for s_idx in spec_indices:
                row = group_df.row(s_idx, named=True)
                mz = row["cleaned_normalized_mz"]
                intensity = row["cleaned_normalized_intensity"]
                if mz is None or intensity is None:
                    continue
                if isinstance(mz, pl.Series):
                    mz = mz.to_list()
                if isinstance(intensity, pl.Series):
                    intensity = intensity.to_list()
                for pk_idx in range(len(mz)):
                    gm.append(float(mz[pk_idx]))
                    gi.append(float(intensity[pk_idx]))
                    gs.append(s_idx)
                    gp.append(pk_idx)

            if not gm:
                continue

            gm_arr = np.array(gm, dtype=np.float64)
            gi_arr = np.array(gi, dtype=np.float64)
            gs_arr = np.array(gs, dtype=np.int32)
            gp_arr = np.array(gp, dtype=np.int32)

            cids, cmasses, cintens, cflat, sflat = _merge_peaks(
                gm_arr, gi_arr, gs_arr, gp_arr, config,
            )

            n_here = len(cmasses)
            if n_here == 0:
                continue

            # Record masses and spectrum membership
            start_pos = len(all_fragment_masses)
            for c in range(n_here):
                all_fragment_masses.append(float(cmasses[c]))
                all_fragment_intensities.append(float(cintens[c]))
                level_masses_list.append(float(cmasses[c]))

            for pair_idx in range(len(cflat)):
                cid = int(cflat[pair_idx])
                sid = int(sflat[pair_idx])
                all_spectrum_pairs.append((start_pos + cid, sid))

        # REMATCH: for spectra at current_level + 1, match precursor_mz
        # against the newly-annotated level-L masses, then annotate those
        # masses to get formulas for the next iteration
        if current_level < max_mslevel and level_masses_list:
            # Create unique sorted mass array for level-L
            level_unique_masses = np.array(sorted(set(level_masses_list)), dtype=np.float64)

            # Annotate level-L masses so we can set formulas as precursors
            # for L+1 spectra. Use each cluster's own max_bounds from its
            # precursor group. However, we don't have per-cluster mapping
            # here easily... Instead, we annotate each unique level mass with
            # a generous bound and use the result for matching.
            # Actually, for the REMATCH, we only need to know WHICH cluster
            # (by mass match) links to each L+1 spectrum. The cluster's formula
            # was already determined during the group annotation above.
            # But we didn't store the per-cluster formulas.

            # We need to store per-level-cluster formulas for the rematch.
            # Since we annotated within each group, we have formulas for those
            # clusters. Let's redo: annotate each level cluster with its
            # group's precursor bounds, and store the results.

            # Actually, we already annotated them above when we called
            # _annotate_mass_clusters... but we didn't store the result.
            # The returned ms2_formulas etc. are for MS2 only.
            # For the rematch, we need formulas for the level-L clusters.
            # Let's batch-annotate ALL level-L masses now with uniform bounds.
            # The bounds to use are the max of all precursor formulas for
            # this level (a conservative upper bound).
            # This is a temporary annotation; the final annotation happens
            # in Step 2d with tightest bounds.

            # Conservative: use molecular precursor as max_bounds for the
            # rematch annotation. This is loose but sufficient for matching.
            n_level = len(level_unique_masses)
            level_max_bounds = np.tile(molecular_precursor, (n_level, 1))
            level_formulas, level_formulas_str, _ = _annotate_mass_clusters(
                level_unique_masses, level_max_bounds, config,
            )

            for s_idx in range(n_spectra):
                row = group_df.row(s_idx, named=True)
                if row["mslevel"] != current_level + 1:
                    continue
                if np.any(updated_precursor_formulas[s_idx] != 0):
                    continue  # Already linked
                precursor_mz_val = float(row["precursor_mz"])
                match_idx = _match_precursor_to_fragments(
                    precursor_mz_val, level_unique_masses, config.merge_tolerance_ppm,
                )
                if match_idx >= 0:
                    updated_precursor_formulas[s_idx] = level_formulas[match_idx]
                # else: leave as zeros (unmatched at this level)

    # -----------------------------------------------------------------------
    # Step 5: Global re-clustering of ALL fragment masses
    #   (MUST precede tightest-bounds re-annotation)
    # -----------------------------------------------------------------------
    n_all = len(all_fragment_masses)
    if n_all == 0:
        return None

    all_masses_arr = np.array(all_fragment_masses, dtype=np.float64)
    all_intensities_arr = np.array(all_fragment_intensities, dtype=np.float64)

    all_spectrum_pairs_arr = np.array(all_spectrum_pairs, dtype=np.int32)
    all_cluster_in = all_spectrum_pairs_arr[:, 0]
    all_spectrum_in = all_spectrum_pairs_arr[:, 1]

    orig_to_unified, unified_cluster_masses, \
        unified_cluster_ids_flat, unified_spectrum_ids_flat = _global_recluster_fragments(
            all_masses_arr, all_intensities_arr,
            all_cluster_in, all_spectrum_in, config,
        )

    n_unified = len(unified_cluster_masses)

    # -----------------------------------------------------------------------
    # Step 6: Tightest-bounds re-annotation on unified clusters
    # -----------------------------------------------------------------------
    final_formulas, final_formulas_str, final_errors_ppm = _reannotate_with_tightest_bounds(
        unified_cluster_masses,
        unified_cluster_ids_flat, unified_spectrum_ids_flat,
        updated_precursor_formulas, config,
    )

    # Filter out clusters with zero formulas (un-annotatable masses).
    # These have no chemical meaning in a formula-driven fragmentation tree.
    valid_cluster_mask = np.any(final_formulas != 0, axis=1)
    n_valid = int(valid_cluster_mask.sum())
    n_removed = n_unified - n_valid

    if n_removed > 0:
        final_formulas = final_formulas[valid_cluster_mask]
        final_formulas_str = [s for s, v in zip(final_formulas_str, valid_cluster_mask) if v]
        final_errors_ppm = final_errors_ppm[valid_cluster_mask]
        unified_cluster_masses = unified_cluster_masses[valid_cluster_mask]
        n_unified = n_valid

    # -----------------------------------------------------------------------
    # Add molecular precursor as a fragment (if not already present)
    # -----------------------------------------------------------------------
    # Check if molecular precursor mass matches any unified cluster
    all_final_masses: list[np.ndarray] = []
    all_final_formulas: list[np.ndarray] = []
    all_final_formulas_str: list[str] = []
    all_final_errors: list[float] = []

    # Find if molecular precursor matches a unified cluster by mass
    precursor_cluster_idx = _match_precursor_to_fragments(
        molecular_precursor_mass, unified_cluster_masses, config.merge_tolerance_ppm,
    )

    if precursor_cluster_idx >= 0:
        # The precursor mass matches a cluster; check if its formula is
        # a superset of the cluster formula. If so, use the cluster as
        # the precursor fragment (it's the same physical ion).
        # If not, add the molecular precursor as a separate fragment.
        cluster_formula = final_formulas[precursor_cluster_idx]
        if _is_superset(molecular_precursor, cluster_formula) or np.array_equal(
            molecular_precursor, cluster_formula,
        ):
            # The molecular precursor IS this cluster
            precursor_idx_in_final = precursor_cluster_idx
            # Use the tighter annotation from the cluster
            all_final_formulas = [final_formulas[i] for i in range(n_unified)]
            all_final_formulas_str = list(final_formulas_str)
            all_final_errors = list(final_errors_ppm)
            all_final_masses = [float(unified_cluster_masses[i]) for i in range(n_unified)]
            # Ensure the precursor's formula is correct
            all_final_formulas[precursor_cluster_idx] = molecular_precursor
            all_final_formulas_str[precursor_cluster_idx] = molecular_precursor_str
            all_final_errors[precursor_cluster_idx] = 0.0
        else:
            # The mass-matched cluster is not chemically the precursor;
            # keep both: add molecular precursor as separate fragment
            precursor_idx_in_final = n_unified
            all_final_formulas = [final_formulas[i] for i in range(n_unified)]
            all_final_formulas_str = list(final_formulas_str)
            all_final_errors = list(final_errors_ppm)
            all_final_masses = [float(unified_cluster_masses[i]) for i in range(n_unified)]
            all_final_formulas.append(molecular_precursor)
            all_final_formulas_str.append(molecular_precursor_str)
            all_final_errors.append(0.0)
            all_final_masses.append(molecular_precursor_mass)
    else:
        # No mass match for molecular precursor; add it as separate fragment
        precursor_idx_in_final = n_unified
        all_final_formulas = [final_formulas[i] for i in range(n_unified)]
        all_final_formulas_str = list(final_formulas_str)
        all_final_errors = list(final_errors_ppm)
        all_final_masses = [float(unified_cluster_masses[i]) for i in range(n_unified)]
        all_final_formulas.append(molecular_precursor)
        all_final_formulas_str.append(molecular_precursor_str)
        all_final_errors.append(0.0)
        all_final_masses.append(molecular_precursor_mass)

    n_final = len(all_final_formulas)
    all_final_formulas_arr = np.stack(all_final_formulas, axis=0)
    all_final_masses_arr = np.array(all_final_masses, dtype=np.float64)
    all_final_errors_arr = np.array(all_final_errors, dtype=np.float64)

    # -----------------------------------------------------------------------
    # Step 7: Build per-spectrum peak-to-fragment mapping
    #   (used in Step 8 for observation matrix construction)
    # -----------------------------------------------------------------------
    # Build per-spectrum peak fragment indices and precursor indices for ALL spectra
    all_spectrum_peak_lists: list[list[int]] = []
    all_spectrum_precursor_indices: list[int] = []
    all_spectrum_mslevels: list[int] = []

    for s_idx in range(n_spectra):
        row = group_df.row(s_idx, named=True)
        mslevel = row["mslevel"]
        mz = row["cleaned_normalized_mz"]
        if mz is None:
            all_spectrum_peak_lists.append([])
            all_spectrum_precursor_indices.append(-1)
            all_spectrum_mslevels.append(int(mslevel))
            continue

        if isinstance(mz, pl.Series):
            mz_list = mz.to_list()
        else:
            mz_list = list(mz)

        peak_list: list[int] = []
        for mz_val in mz_list:
            fidx = _match_precursor_to_fragments(
                float(mz_val), all_final_masses_arr, config.merge_tolerance_ppm,
            )
            if fidx >= 0:
                peak_list.append(fidx)

        # Determine precursor index
        if mslevel == 2:
            prec_idx = precursor_idx_in_final
        else:
            prec_mz_val = float(row["precursor_mz"])
            prec_idx = _match_precursor_to_fragments(
                prec_mz_val, all_final_masses_arr, config.merge_tolerance_ppm,
            )

        all_spectrum_peak_lists.append(peak_list)
        all_spectrum_precursor_indices.append(prec_idx)
        all_spectrum_mslevels.append(int(mslevel))

    # -----------------------------------------------------------------------
    # Step 8: Phase 3 — Edge construction
    # -----------------------------------------------------------------------
    # 8a. Build superset matrix
    superset_matrix = _build_superset_matrix(all_final_formulas_arr)

    # 8b. Build CSR arrays for observation matrix
    total_peaks_all = sum(len(lst) for lst in all_spectrum_peak_lists)
    peak_frag_all = np.zeros(total_peaks_all, dtype=np.int32)
    peak_off_all = np.zeros(n_spectra + 1, dtype=np.int32)
    offset = 0
    for s_idx in range(n_spectra):
        peak_off_all[s_idx] = offset
        peak_list = all_spectrum_peak_lists[s_idx]
        for pk_idx, fidx in enumerate(peak_list):
            peak_frag_all[offset + pk_idx] = fidx
        offset += len(peak_list)
    peak_off_all[n_spectra] = offset

    prec_indices_arr = np.array(all_spectrum_precursor_indices, dtype=np.int32)

    # 8c. Build observation matrix
    obs_matrix, has_own_spec, in_spec = _build_observation_matrix(
        peak_frag_all, peak_off_all, prec_indices_arr,
        n_final, n_spectra,
    )

    # 8d. Build lowest provable parent edges
    edge_matrix = _build_edges_lowest_provable_parent(
        superset_matrix, obs_matrix, has_own_spec, in_spec,
        n_spectra, precursor_idx_in_final,
    )

    # 8e. Drop orphans iteratively
    keep_mask = _drop_orphans_iterative(edge_matrix, precursor_idx_in_final)
    kept_indices = np.where(keep_mask)[0]
    n_kept = len(kept_indices)
    if n_kept == 0:
        return None

    # -----------------------------------------------------------------------
    # Step 9: Remap indices after orphan removal
    # -----------------------------------------------------------------------
    kept_formulas = all_final_formulas_arr[keep_mask]
    kept_formulas_str = [all_final_formulas_str[i] for i in kept_indices]
    kept_errors = all_final_errors_arr[keep_mask]

    # Rebuild edge matrix for kept fragments
    kept_edges = np.zeros((n_kept, n_kept), dtype=np.bool_)
    for new_i, old_i in enumerate(kept_indices):
        for new_j, old_j in enumerate(kept_indices):
            kept_edges[new_i, new_j] = edge_matrix[old_i, old_j]

    # Compute edge weights
    edge_weights = _compute_edge_weights(kept_edges)

    # Remap spectrum fragment indices
    old_to_new = {old: new for new, old in enumerate(kept_indices)}
    remapped_spectrum_fragments: list[np.ndarray] = []
    remapped_mslevels: list[int] = []
    remapped_msn_precursors: list[int] = []

    for s_idx in range(n_spectra):
        frag_indices = all_spectrum_peak_lists[s_idx]
        prec_idx = all_spectrum_precursor_indices[s_idx]
        mslevel = all_spectrum_mslevels[s_idx]

        remapped = [old_to_new[idx] for idx in frag_indices if idx in old_to_new]
        # Also include the precursor if it's a kept fragment and not already in the list
        remapped_prec = old_to_new.get(prec_idx, -1) if prec_idx >= 0 else -1

        if remapped or (remapped_prec >= 0 and remapped_prec not in remapped):
            remapped_spectrum_fragments.append(np.array(remapped, dtype=np.int32))
            remapped_mslevels.append(mslevel)
            remapped_msn_precursors.append(remapped_prec)

    # -----------------------------------------------------------------------
    # Step 10: Assemble FragmentationTree
    # -----------------------------------------------------------------------
    # Find precursor index in kept fragments
    kept_precursor_idx = old_to_new.get(precursor_idx_in_final, 0)

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
        fragment_errors_ppm=kept_errors,
    )


def _formula_array_to_string(formula_array: np.ndarray) -> str:
    """Convert a formula array to a string representation."""
    from hrms_utils.formula_annotation.element_table import ELEMENT_SYMBOLS

    parts: list[str] = []
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
    from hrms_utils.formula_annotation.element_table import ELEMENT_MASSES

    element_masses_array = np.array(ELEMENT_MASSES, dtype=np.float64)

    G = nx.DiGraph()

    # Add nodes with labels
    for i, formula_str in enumerate(tree.fragment_formulas_str):
        mass = np.dot(tree.fragment_formulas[i], element_masses_array)
        G.add_node(i, label=f"{formula_str}\n{mass:.4f} Da")

    # Add edges with weights
    n = tree.n_fragments
    for i in range(n):
        for j in range(n):
            w = tree.edge_weights[i, j]
            if w > 0:
                G.add_edge(i, j, weight=w)

    # Layout: hierarchical based on formula "size" (sum of elements)
    sizes = tree.fragment_formulas.sum(axis=1)
    unique_sizes = np.sort(np.unique(sizes))[::-1]  # descending
    pos: dict[int, tuple[float, float]] = {}
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
