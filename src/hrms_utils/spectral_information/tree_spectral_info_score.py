'''
goal of this file:
implement an algorithm to give spectral info score to a mass spectrum, the following way:
given a spectrum (so a precursor formula and list of fragment formulas) build a tree the following way- any pair of fragmnets (precursor included) is connected if one can be obtained from the other by removing a subformula (so C2H4O for example). make the edges directed from the smaller formula to the larger one. then, normalize all the formulas by the precursor formula- so the precursor "length" is 1, and is equal in all directions where the precursor is not 0. avoid division by 0 of course, and remember that all fragments are subsets of the precursor. then, for each edge its weight is the distance (makeu this be chosen between l1, l2 and cosine distance) between the two formulas. for each node, compute the minimal weight edge going out of it (if any), keep thsi weight, the edges are not needed now. (so there isn't really a need to construct edges, just compute the weights directly while taking the inclusion criteria into account). Call this weight P.
Then for each node, its contribution to the score is -P * ln (P) * M, where M is the length of the fragmnet formula (after normlization). the score is then the sum of contirbutuions.
'''
import numpy as np
from numpy.typing import NDArray
from numba import njit, prange
import polars as pl

# ##############################################################################
# Numba-jitted helper functions
# ##############################################################################

@njit(fastmath=True)
def _l1_distance(a: NDArray[np.floating], b: NDArray[np.floating]) -> np.floating:
    """Computes L1 (Manhattan) distance between two vectors."""
    return np.sum(np.abs(a - b))

@njit(fastmath=True)
def _l2_distance(a: NDArray[np.floating], b: NDArray[np.floating]) -> np.floating:
    """Computes L2 (Euclidean) distance between two vectors."""
    return np.sqrt(np.sum((a - b)**2))

@njit(fastmath=True)
def _cosine_distance(a: NDArray[np.floating], b: NDArray[np.floating]) -> np.floating:
    """Computes cosine distance between two vectors."""
    dot_product = np.sum(a * b)
    norm_a = np.sqrt(np.sum(a**2))
    norm_b = np.sqrt(np.sum(b**2))
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    return 1.0 - (dot_product / (norm_a * norm_b))

@njit(cache=True, fastmath=True)
def _is_superformula(super_formula: NDArray[np.floating], sub_formula: NDArray[np.floating]) -> bool:
    """Checks if super_formula contains sub_formula."""
    tolerance = 1e-12  # Why: guard against floating point noise introduced by normalization.
    return bool(
        np.all(super_formula >= sub_formula - tolerance)
        and np.any(super_formula > sub_formula + tolerance)
    )

# ##############################################################################
# Core batch processing function
# ##############################################################################

@njit(parallel=True, fastmath=True)
def _tree_score_core_batch(
    concat_norm_formulas: np.ndarray,
    offsets: np.ndarray,
    formula_counts: np.ndarray,
    dims: np.ndarray,
    distance_metric: int
) -> np.ndarray:
    """
    Core batch implementation for tree-based spectral information score.
    Why: A single numba-jitted parallel loop avoids Python-level overhead and
    computes scores for all spectra in parallel.
    """
    num_spectra = len(formula_counts)
    scores = np.zeros(num_spectra, dtype=np.float64)

    for i in prange(num_spectra):
        k = formula_counts[i]
        n = dims[i]

        if k <= 1 or n == 0:
            continue

        spectrum_offset = offsets[i]
        total_score = 0.0

        norm_formulas = concat_norm_formulas[spectrum_offset : spectrum_offset + k * n].reshape(k, n)

        for j in range(k):
            node_A_norm = norm_formulas[j]
            min_dist = np.inf

            for l in range(k):
                if j == l:
                    continue

                node_B_norm = norm_formulas[l]

                if _is_superformula(node_B_norm, node_A_norm):
                    if distance_metric == 0:
                        dist = _l1_distance(node_A_norm, node_B_norm)
                    elif distance_metric == 1:
                        dist = _l2_distance(node_A_norm, node_B_norm)
                    else:
                        dist = _cosine_distance(node_A_norm, node_B_norm)

                    if dist < min_dist:
                        min_dist = dist

            if np.isfinite(min_dist):
                if distance_metric == 0:
                    distance_cap = 2.0  # Why: L1 distance between length-1 vectors is bounded by 2.
                elif distance_metric == 1:
                    distance_cap = np.sqrt(2.0)  # Why: L2 distance between length-1 vectors is bounded by sqrt(2).
                else:
                    distance_cap = 2.0  # Why: Cosine distance ranges [0, 2] for normalized vectors.

                scaled_dist = min_dist / distance_cap
                if scaled_dist <= 1e-12:
                    continue  # Why: avoid log(0) while keeping zero-distance edges non-contributory.
                if scaled_dist >= 1.0:
                    scaled_dist = 1.0 - 1e-12  # Why: keep entropy term finite and positive.

                M = np.sum(node_A_norm)
                if M > 0.0:
                    total_score += -scaled_dist * np.log(scaled_dist) * M

        scores[i] = total_score

    return scores

# ##############################################################################
# Polars wrapper function
# ##############################################################################

def tree_spectral_info_score_polars(
    precursors: pl.Series,
    fragments: pl.Series,
    *,
    distance_metric: str = "l2"
) -> pl.Series:
    """
    Calculates a tree-based spectral information score for each spectrum in a Polars DataFrame.

    The algorithm models relationships between fragments (and precursor) as a tree,
    where edges connect sub-formulas to super-formulas. The score is derived from
    the entropy of the minimum distances between connected nodes in a normalized space.

    Args:
        precursors: A Polars Series of precursor formulas (List[Float64]).
        fragments: A Polars Series of fragment formulas (List[List[Float64]]).
        distance_metric: The distance metric for comparing normalized formulas.
                         One of 'l1', 'l2', or 'cosine'. Defaults to 'l2'.

    Returns:
        A Polars Series of Float64 scores, one for each input spectrum.
    """
    if len(precursors) != len(fragments):
        raise AssertionError("precursors and fragments must have the same length")

    dist_map = {"l1": 0, "l2": 1, "cosine": 2}
    if distance_metric not in dist_map:
        raise ValueError(f"Unknown distance_metric: '{distance_metric}'. Must be one of {list(dist_map.keys())}")
    dist_metric_int = dist_map[distance_metric]

    prec_array = precursors.to_numpy()
    frag_array = fragments.to_numpy()
    n_rows = len(prec_array)

    # Data preparation for batch processing
    concat_norm_list = []
    offsets = np.empty(n_rows, dtype=np.int64)
    formula_counts = np.empty(n_rows, dtype=np.int64)
    dims = np.empty(n_rows, dtype=np.int64)

    current_offset = 0
    for i in range(n_rows):
        p_list = prec_array[i]
        f_list_of_lists = frag_array[i]

        if p_list is None or len(p_list) == 0:
            offsets[i] = current_offset
            formula_counts[i] = 0
            dims[i] = 0
            continue

        p_vec = np.asarray(p_list, dtype=np.float64)
        active_mask = p_vec > 0
        n_active = int(np.sum(active_mask))

        if n_active == 0:
            offsets[i] = current_offset
            formula_counts[i] = 0
            dims[i] = 0
            continue

        total_precursor_mass = float(np.sum(p_vec[active_mask]))
        assert total_precursor_mass > 0.0, "normalized precursor length must be positive before scoring"

        all_formulas_list = [p_vec]
        if f_list_of_lists is not None:
            for f in f_list_of_lists:
                all_formulas_list.append(np.asarray(f, dtype=np.float64))

        k = len(all_formulas_list)
        formula_counts[i] = k
        dims[i] = n_active
        offsets[i] = current_offset

        if k <= 1:
            continue

        spec_norm_formulas = np.empty((k, n_active), dtype=np.float64)

        for j, formula_vec in enumerate(all_formulas_list):
            if len(formula_vec) != len(p_vec):
                raise ValueError(
                    f"Row {i}: Fragment formula length mismatch. Precursor has {len(p_vec)} elements, fragment has {len(formula_vec)}."
                )

            f_active = formula_vec[active_mask]
            spec_norm_formulas[j, :] = f_active / total_precursor_mass  # Why: ensures precursor length sums to 1 and fragments stay bounded.

        concat_norm_list.extend(spec_norm_formulas.ravel())
        current_offset += spec_norm_formulas.size

    if len(concat_norm_list) == 0:
        return pl.Series(values=np.zeros(n_rows), dtype=pl.Float64)

    concat_norm_formulas = np.asarray(concat_norm_list, dtype=np.float64)

    scores = _tree_score_core_batch(
        concat_norm_formulas,
        offsets,
        formula_counts,
        dims,
        dist_metric_int
    )
    np.nan_to_num(scores, copy=False, nan=0.0, posinf=0.0, neginf=0.0)  # Why: downstream consumers expect finite scores.

    return pl.Series(values=scores, dtype=pl.Float64)
