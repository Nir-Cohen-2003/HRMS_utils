"""
Standalone test for fragmentation tree builder core logic.
Does not require hrms_utils to be installed.

Tests match the rewritten fragmentation_tree.py Numba functions.
"""

import numpy as np

# We need to test the numba functions. Since the module imports hrms_utils,
# we'll copy the core numba functions here for testing.

from numba import njit, prange


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
# New Numba helpers — from fragmentation_tree.py
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


# =========================================================================
# Tests
# =========================================================================

# -------------------------------------------------------------------------
# KEEP: _is_superset and test_is_superset
# -------------------------------------------------------------------------

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


# -------------------------------------------------------------------------
# KEEP: _build_superset_matrix and test_build_superset_matrix
# -------------------------------------------------------------------------

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


# -------------------------------------------------------------------------
# KEEP: _compute_edge_weights and test_edge_weights
# -------------------------------------------------------------------------

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


# -------------------------------------------------------------------------
# KEEP: _drop_orphans_iterative and test
# -------------------------------------------------------------------------

def test_drop_orphans_iterative():
    """Test iterative orphan dropping (replaces non-iterative _drop_orphans)."""
    # 4 fragments: A (precursor, idx 0), B (parent A), C (parent B), D (no parent)
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
    print("test_drop_orphans_iterative PASSED")


def test_drop_orphans_iterative_cascade():
    """Test iterative orphan dropping cascades: dropping a node may orphan its children."""
    # A (precursor, idx 0) -> B -> C -> D
    matrix = np.array([
        [False, True, False, False],
        [False, False, True, False],
        [False, False, False, True],
        [False, False, False, False],
    ], dtype=np.bool_)

    # Remove A->B edge, so B has no parent -> B dropped -> C orphaned -> C dropped -> D orphaned
    matrix_broken = matrix.copy()
    matrix_broken[0, 1] = False

    keep = _drop_orphans_iterative(matrix_broken, 0)

    assert keep[0]   # precursor kept
    assert not keep[1]  # B orphaned
    assert not keep[2]  # C orphaned after B dropped
    assert not keep[3]  # D orphaned after C dropped
    print("test_drop_orphans_iterative_cascade PASSED")


# -------------------------------------------------------------------------
# ADD: _cluster_masses_sorted and test
# -------------------------------------------------------------------------

def test_cluster_masses_sorted():
    """Test greedy single-linkage clustering of sorted masses."""
    masses = np.array([100.0, 100.05, 100.1, 100.12, 200.0, 200.05, 200.06], dtype=np.float64)
    tol = 500.0  # 500 ppm = 0.05%, threshold = 5e-4

    cluster_ids = _cluster_masses_sorted(masses, tol)

    # Expected: {100.0, 100.05} cluster 0, {100.1, 100.12} cluster 1, {200.0, 200.05, 200.06} cluster 2
    expected = np.array([0, 0, 1, 1, 2, 2, 2], dtype=np.int32)
    np.testing.assert_array_equal(cluster_ids, expected)
    print("test_cluster_masses_sorted PASSED")


def test_cluster_masses_sorted_single():
    """Test clustering with a single mass."""
    masses = np.array([100.0], dtype=np.float64)
    cluster_ids = _cluster_masses_sorted(masses, 5.0)
    assert cluster_ids[0] == 0
    print("test_cluster_masses_sorted_single PASSED")


def test_cluster_masses_sorted_empty():
    """Test clustering with empty array."""
    masses = np.array([], dtype=np.float64)
    cluster_ids = _cluster_masses_sorted(masses, 5.0)
    assert len(cluster_ids) == 0
    print("test_cluster_masses_sorted_empty PASSED")


# -------------------------------------------------------------------------
# ADD: _compute_cluster_representatives and test
# -------------------------------------------------------------------------

def test_compute_cluster_representatives():
    """Test computing cluster representatives (weighted mean mass + max intensity)."""
    masses = np.array([100.0, 100.05, 100.1, 200.0, 200.05], dtype=np.float64)
    intensities = np.array([1.0, 2.0, 3.0, 5.0, 5.0], dtype=np.float64)
    cluster_ids = np.array([0, 0, 0, 1, 1], dtype=np.int32)
    n_clusters = 2

    cluster_masses, cluster_intensities = _compute_cluster_representatives(
        masses, intensities, cluster_ids, n_clusters,
    )

    # Cluster 0: total_intensity = 1+2+3=6
    #   weighted mass = (100*1 + 100.05*2 + 100.1*3)/6
    #   max intensity = 3.0
    expected_mass0 = (100.0*1.0 + 100.05*2.0 + 100.1*3.0) / 6.0
    assert abs(cluster_masses[0] - expected_mass0) < 1e-10
    assert cluster_intensities[0] == 3.0

    # Cluster 1: total_intensity = 5+5=10
    #   weighted mass = (200*5 + 200.05*5)/10 = 200.025
    #   max intensity = 5.0
    assert abs(cluster_masses[1] - 200.025) < 1e-10
    assert cluster_intensities[1] == 5.0

    print("test_compute_cluster_representatives PASSED")


def test_compute_cluster_representatives_empty():
    """Test with zero clusters."""
    masses = np.array([], dtype=np.float64)
    intensities = np.array([], dtype=np.float64)
    cluster_ids = np.array([], dtype=np.int32)
    cluster_masses, cluster_intensities = _compute_cluster_representatives(
        masses, intensities, cluster_ids, 0,
    )
    assert len(cluster_masses) == 0
    assert len(cluster_intensities) == 0
    print("test_compute_cluster_representatives_empty PASSED")


# -------------------------------------------------------------------------
# ADD: _match_precursor_to_fragments and test
# -------------------------------------------------------------------------

def test_match_precursor_to_fragments():
    """Test binary-search fragment matching."""
    fragment_masses = np.array([100.0, 150.0, 200.0, 250.0, 300.0], dtype=np.float64)
    tol = 10.0  # 10 * 1e-6 = 1e-5 relative

    # Exact match
    assert _match_precursor_to_fragments(200.0, fragment_masses, tol) == 2

    # Close match within tolerance: 150.001 vs 150.0, diff/150 = 6.67e-6 < 1e-5
    assert _match_precursor_to_fragments(150.001, fragment_masses, tol) == 1

    # Out of tolerance
    assert _match_precursor_to_fragments(150.1, fragment_masses, tol) == -1

    # Below smallest mass
    assert _match_precursor_to_fragments(50.0, fragment_masses, tol) == -1

    # Above largest mass
    assert _match_precursor_to_fragments(350.0, fragment_masses, tol) == -1

    print("test_match_precursor_to_fragments PASSED")


def test_match_precursor_to_fragments_empty():
    """Test matching against empty array."""
    fragment_masses = np.array([], dtype=np.float64)
    assert _match_precursor_to_fragments(100.0, fragment_masses, 5.0) == -1
    print("test_match_precursor_to_fragments_empty PASSED")


def test_match_precursor_to_fragments_prefers_closest():
    """Test that when two masses are within tolerance, the closer one is returned."""
    fragment_masses = np.array([150.0, 151.0], dtype=np.float64)
    tol = 10000.0  # 1% relative, both masses are within tolerance of 150.5

    # 150.5 is closer to 150.0 (diff 0.5) than to 151.0 (diff 0.5)
    # Both candidates would give diff=0.5, so the first one found (lo-1 or lo) wins.
    # Actually 150.5 - 150.0 = 0.5, 151.0 - 150.5 = 0.5. Both diffs equal.
    # Best_idx is set to the first candidate that matches. Candidate (lo-1) = 0 (150.0).
    # So the result should be 0.
    idx = _match_precursor_to_fragments(150.5, fragment_masses, tol)
    assert idx == 0, f"Expected 0 (closer to 150.0), got {idx}"
    print("test_match_precursor_to_fragments_prefers_closest PASSED")


def test_match_precursor_to_fragments_edge_insertion():
    """Test matching when the target is between two fragment masses."""
    fragment_masses = np.array([149.0, 150.0, 151.0], dtype=np.float64)
    tol = 5000.0  # 0.5% relative tolerance

    # Target 149.5: insertion point at index 1
    # Candidates: lo-1=0 (149.0), lo=1 (150.0)
    # 149.0 diff=0.5, 0.5/149.5=0.00334 < 0.005 -> match at idx 0 (checked first)
    idx = _match_precursor_to_fragments(149.5, fragment_masses, tol)
    assert idx == 0, f"Expected 0 (149.0 is closer to 149.5), got {idx}"

    # Target 150.5: insertion point at index 2
    # Candidates: lo-1=1 (150.0), lo=2 (151.0)
    # 150.0 diff=0.5, 151.0 diff=0.5, same diff; lo-1 checked first => idx=1
    idx = _match_precursor_to_fragments(150.5, fragment_masses, tol)
    assert idx == 1, f"Expected 1 (150.0 checked first), got {idx}"

    # Target 149.2: insertion point at index 1
    # Candidates: lo-1=0 (149.0), lo=1 (150.0)
    # 149.0 diff=0.2, 150.0 diff=0.8. 149.0 is clearly closer -> idx=0
    idx = _match_precursor_to_fragments(149.2, fragment_masses, tol)
    assert idx == 0, f"Expected 0 (149.0 is closest to 149.2), got {idx}"

    print("test_match_precursor_to_fragments_edge_insertion PASSED")


# -------------------------------------------------------------------------
# ADD: _compute_tightest_bounds and test
# -------------------------------------------------------------------------

def test_compute_tightest_bounds():
    """Test element-wise minimum of precursor formulas per cluster."""
    # 3 clusters, 4 elements (H, C, N, O)
    cluster_ids_flat = np.array([0, 0, 1, 1, 2], dtype=np.int32)
    spectrum_ids_flat = np.array([0, 1, 0, 2, 1], dtype=np.int32)
    # 3 spectra
    spectrum_precursor_formulas = np.array([
        [10, 5, 0, 1],   # spectrum 0: C5H10O
        [20, 10, 0, 2],  # spectrum 1: C10H20O2
        [8, 4, 0, 1],    # spectrum 2: C4H8O
    ], dtype=np.int32)
    n_clusters = 3

    tightest = _compute_tightest_bounds(
        cluster_ids_flat, spectrum_ids_flat,
        spectrum_precursor_formulas, n_clusters,
    )

    # Cluster 0: spectra 0 and 1 -> min([10,5,0,1], [20,10,0,2]) = [10,5,0,1]
    np.testing.assert_array_equal(tightest[0], [10, 5, 0, 1])

    # Cluster 1: spectra 0 and 2 -> min([10,5,0,1], [8,4,0,1]) = [8,4,0,1]
    np.testing.assert_array_equal(tightest[1], [8, 4, 0, 1])

    # Cluster 2: spectrum 1 -> [20,10,0,2]
    np.testing.assert_array_equal(tightest[2], [20, 10, 0, 2])

    print("test_compute_tightest_bounds PASSED")


def test_compute_tightest_bounds_empty():
    """Test with zero clusters."""
    cluster_ids_flat = np.array([], dtype=np.int32)
    spectrum_ids_flat = np.array([], dtype=np.int32)
    spectrum_precursor_formulas = np.array([[10, 5, 0, 1]], dtype=np.int32)
    tightest = _compute_tightest_bounds(
        cluster_ids_flat, spectrum_ids_flat,
        spectrum_precursor_formulas, 0,
    )
    assert tightest.shape == (0, 4)
    print("test_compute_tightest_bounds_empty PASSED")


def test_compute_tightest_bounds_no_overlap():
    """Test when each cluster belongs to exactly one spectrum."""
    cluster_ids_flat = np.array([0, 1, 2], dtype=np.int32)
    spectrum_ids_flat = np.array([0, 1, 2], dtype=np.int32)
    spectrum_precursor_formulas = np.array([
        [5, 3, 1, 0],
        [10, 6, 0, 1],
        [8, 4, 0, 2],
    ], dtype=np.int32)
    n_clusters = 3

    tightest = _compute_tightest_bounds(
        cluster_ids_flat, spectrum_ids_flat,
        spectrum_precursor_formulas, n_clusters,
    )

    np.testing.assert_array_equal(tightest[0], [5, 3, 1, 0])
    np.testing.assert_array_equal(tightest[1], [10, 6, 0, 1])
    np.testing.assert_array_equal(tightest[2], [8, 4, 0, 2])

    print("test_compute_tightest_bounds_no_overlap PASSED")


# -------------------------------------------------------------------------
# ADD: _build_observation_matrix and test
# -------------------------------------------------------------------------

def test_build_observation_matrix():
    """Test building observation matrix with precursor exclusion."""
    n_fragments = 3  # A=0, B=1, C=2
    n_spectra = 2

    # Spectrum 0: precursor A (0), peaks [A, B, C]
    # Spectrum 1: precursor B (1), peaks [B, C]
    peak_fragment_ids = np.array([0, 1, 2, 1, 2], dtype=np.int32)
    peak_offsets = np.array([0, 3, 5], dtype=np.int32)
    spectrum_precursor_indices = np.array([0, 1], dtype=np.int32)

    obs_matrix, has_own, in_spec = _build_observation_matrix(
        peak_fragment_ids, peak_offsets, spectrum_precursor_indices,
        n_fragments, n_spectra,
    )

    # --- observation_matrix (P[A, B]) ---
    # Spectrum 0 (precursor A=0): peaks [B=1, C=2] (A=0 excluded)
    assert obs_matrix[0, 1], "P[A, B] should be True (B is peak in A's spectrum)"
    assert obs_matrix[0, 2], "P[A, C] should be True"
    # Spectrum 1 (precursor B=1): peaks [C=2] (B=1 excluded)
    assert obs_matrix[1, 2], "P[B, C] should be True"
    # No other P should be set
    assert not obs_matrix[0, 0], "P[A, A] should be False (self excluded)"
    assert not obs_matrix[1, 1], "P[B, B] should be False (self excluded)"
    assert not obs_matrix[2, 0], "P[C, A] should be False"
    assert not obs_matrix[2, 1], "P[C, B] should be False"
    assert not obs_matrix[1, 0], "P[B, A] should be False"

    # --- has_own_spectrum ---
    assert has_own[0], "A has own spectrum (spectrum 0)"
    assert has_own[1], "B has own spectrum (spectrum 1)"
    assert not has_own[2], "C has no own spectrum"

    # --- in_spectrum ---
    # Spectrum 0: B and C are peaks, A is excluded
    assert in_spec[1, 0], "B is in spectrum 0"
    assert in_spec[2, 0], "C is in spectrum 0"
    assert not in_spec[0, 0], "A (precursor) is excluded from in_spectrum in its own spectrum"
    # Spectrum 1: C is peak, B is excluded
    assert in_spec[2, 1], "C is in spectrum 1"
    assert not in_spec[1, 1], "B (precursor) is excluded from in_spectrum in its own spectrum"
    assert not in_spec[0, 1], "A is not in spectrum 1"

    print("test_build_observation_matrix PASSED")


def test_build_observation_matrix_no_precursor():
    """Test building observation matrix when a spectrum has no matched precursor."""
    n_fragments = 2  # A=0, B=1
    n_spectra = 2

    # Spectrum 0: precursor A (0), peaks [B]
    # Spectrum 1: precursor -1 (unmatched), peaks [A, B]
    peak_fragment_ids = np.array([1, 0, 1], dtype=np.int32)
    peak_offsets = np.array([0, 1, 3], dtype=np.int32)
    spectrum_precursor_indices = np.array([0, -1], dtype=np.int32)

    obs_matrix, has_own, in_spec = _build_observation_matrix(
        peak_fragment_ids, peak_offsets, spectrum_precursor_indices,
        n_fragments, n_spectra,
    )

    # Spectrum 0 (precursor A=0): peak [B=1] -> P[A,B]=True
    assert obs_matrix[0, 1], "P[A, B] should be True"
    # Spectrum 1 (precursor -1): no P entries added (p < 0)
    assert not obs_matrix[0, 0]
    assert not obs_matrix[1, 0]
    assert not obs_matrix[1, 1]

    # has_own_spectrum: A has one (spectrum 0), B has none
    assert has_own[0], "A has own spectrum"
    assert not has_own[1], "B has no own spectrum"

    # in_spectrum: both A and B are recorded in spectrum 1 (no precursor exclusion)
    assert in_spec[1, 0], "B is in spectrum 0"
    assert in_spec[0, 1], "A is in spectrum 1"
    assert in_spec[1, 1], "B is in spectrum 1"
    assert not in_spec[0, 0], "A (precursor) is excluded from in_spectrum in spectrum 0"

    print("test_build_observation_matrix_no_precursor PASSED")


# -------------------------------------------------------------------------
# ADD: _build_edges_lowest_provable_parent — Rule 1 tests
# -------------------------------------------------------------------------

def test_rule1_transitive_reduction():
    """Rule 1: A->C removed when spectral evidence A->B->C exists."""
    # A(0) -> B(1) -> C(2) with superset edges
    # P[A,B] and P[B,C] both True -> A->C should be removed
    n = 3

    superset = np.array([
        [False, True,  True],   # A superset of B and C
        [False, False, True],   # B superset of C
        [False, False, False],
    ], dtype=np.bool_)

    obs = np.zeros((n, n), dtype=np.bool_)
    obs[0, 1] = True  # P[A, B]
    obs[1, 2] = True  # P[B, C]

    has_own = np.array([True, True, False], dtype=np.bool_)
    in_spec = np.zeros((n, 2), dtype=np.bool_)  # not needed for Rule 1

    edges = _build_edges_lowest_provable_parent(
        superset, obs, has_own, in_spec, 2, 0,
    )

    # A->C should be removed (transitive reduction through B)
    assert not edges[0, 2], "A->C should be removed by Rule 1"
    # A->B and B->C should remain
    assert edges[0, 1], "A->B should remain"
    assert edges[1, 2], "B->C should remain"

    print("test_rule1_transitive_reduction PASSED")


def test_rule1_no_pac_check():
    """Rule 1: P[A,C] being True does NOT prevent removal of A->C.

    The spectral evidence chain A->B->C is sufficient regardless of P[A,C].
    """
    n = 3

    superset = np.array([
        [False, True,  True],
        [False, False, True],
        [False, False, False],
    ], dtype=np.bool_)

    obs = np.zeros((n, n), dtype=np.bool_)
    obs[0, 1] = True  # P[A, B]
    obs[1, 2] = True  # P[B, C]
    obs[0, 2] = True  # P[A, C] also True — but should NOT prevent Rule 1

    has_own = np.array([True, True, False], dtype=np.bool_)
    in_spec = np.zeros((n, 2), dtype=np.bool_)

    edges = _build_edges_lowest_provable_parent(
        superset, obs, has_own, in_spec, 2, 0,
    )

    # A->C should STILL be removed despite P[A,C] being True
    assert not edges[0, 2], "A->C should be removed by Rule 1 even with P[A,C]=True"
    assert edges[0, 1], "A->B should remain"
    assert edges[1, 2], "B->C should remain"

    print("test_rule1_no_pac_check PASSED")


# -------------------------------------------------------------------------
# ADD: _build_edges_lowest_provable_parent — Rule 3 tests
# -------------------------------------------------------------------------

def test_rule3_remove_when_b_existed_without_c():
    """Rule 3: B->C removed when B has no own spectrum and B appears without C."""
    n = 3  # A=0, B=1, C=2
    n_spectra = 2

    superset = np.array([
        [False, True,  True],   # A superset of B and C
        [False, False, True],   # B superset of C
        [False, False, False],
    ], dtype=np.bool_)

    obs = np.zeros((n, n), dtype=np.bool_)  # no spectral evidence for Rule 1
    has_own = np.array([True, False, False], dtype=np.bool_)  # B has NO own spectrum
    in_spec = np.zeros((n, n_spectra), dtype=np.bool_)
    in_spec[1, 0] = True  # B is in spectrum 0
    # C is NOT in spectrum 0 — B existed without C

    edges = _build_edges_lowest_provable_parent(
        superset, obs, has_own, in_spec, n_spectra, 0,
    )

    # B->C should be removed by Rule 3 (B existed without C)
    assert not edges[1, 2], "B->C should be removed by Rule 3"
    # A->B and A->C should remain (not affected by Rule 3)
    assert edges[0, 1], "A->B should remain"
    assert edges[0, 2], "A->C should remain (A has own spectrum, Rule 3 doesn't apply)"

    print("test_rule3_remove_when_b_existed_without_c PASSED")


def test_rule3_keep_when_b_always_with_c():
    """Rule 3: B->C kept when B has no own spectrum and B always appears with C."""
    n = 3  # A=0, B=1, C=2
    n_spectra = 2

    superset = np.array([
        [False, True,  True],
        [False, False, True],
        [False, False, False],
    ], dtype=np.bool_)

    obs = np.zeros((n, n), dtype=np.bool_)
    has_own = np.array([True, False, False], dtype=np.bool_)  # B has NO own spectrum
    in_spec = np.zeros((n, n_spectra), dtype=np.bool_)
    in_spec[1, 0] = True  # B is in spectrum 0
    in_spec[2, 0] = True  # C is ALSO in spectrum 0 — B always produced C
    in_spec[1, 1] = True  # B is in spectrum 1
    in_spec[2, 1] = True  # C is also in spectrum 1

    edges = _build_edges_lowest_provable_parent(
        superset, obs, has_own, in_spec, n_spectra, 0,
    )

    # B->C should be kept (B never appears without C)
    assert edges[1, 2], "B->C should be kept (B always appears with C)"
    assert edges[0, 1], "A->B should remain"
    assert edges[0, 2], "A->C should remain"

    print("test_rule3_keep_when_b_always_with_c PASSED")


# -------------------------------------------------------------------------
# ADD: _build_edges_lowest_provable_parent — Rule 2 (co-parenting) test
# -------------------------------------------------------------------------

def test_rule2_coparenting():
    """Rule 2: edges not removed by Rules 1 or 3 are kept (co-parenting)."""
    n = 3  # A=0, B=1, C=2
    n_spectra = 1

    # A is superset of B and C (two direct fragments, no chain)
    superset = np.array([
        [False, True,  True],   # A -> B, A -> C
        [False, False, False],  # B -> nothing
        [False, False, False],  # C -> nothing
    ], dtype=np.bool_)

    obs = np.zeros((n, n), dtype=np.bool_)    # no spectral evidence
    has_own = np.array([True, True, True], dtype=np.bool_)  # all have own spectra
    in_spec = np.zeros((n, n_spectra), dtype=np.bool_)

    edges = _build_edges_lowest_provable_parent(
        superset, obs, has_own, in_spec, n_spectra, 0,
    )

    # Both edges should be kept (co-parenting is the default)
    assert edges[0, 1], "A->B should be kept (Rule 2)"
    assert edges[0, 2], "A->C should be kept (Rule 2)"

    print("test_rule2_coparenting PASSED")


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

if __name__ == "__main__":
    test_is_superset()
    test_build_superset_matrix()
    test_edge_weights()
    test_drop_orphans_iterative()
    test_drop_orphans_iterative_cascade()
    test_cluster_masses_sorted()
    test_cluster_masses_sorted_single()
    test_cluster_masses_sorted_empty()
    test_compute_cluster_representatives()
    test_compute_cluster_representatives_empty()
    test_match_precursor_to_fragments()
    test_match_precursor_to_fragments_empty()
    test_match_precursor_to_fragments_prefers_closest()
    test_match_precursor_to_fragments_edge_insertion()
    test_compute_tightest_bounds()
    test_compute_tightest_bounds_empty()
    test_compute_tightest_bounds_no_overlap()
    test_build_observation_matrix()
    test_build_observation_matrix_no_precursor()
    test_rule1_transitive_reduction()
    test_rule1_no_pac_check()
    test_rule3_remove_when_b_existed_without_c()
    test_rule3_keep_when_b_always_with_c()
    test_rule2_coparenting()
    print("\nAll tests passed!")
