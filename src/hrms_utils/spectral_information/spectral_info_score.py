import numpy as np
from numpy.typing import NDArray
from numba import njit, prange
import polars as pl

def _rows_unique(arr: NDArray[np.floating]) -> bool:
    # Assumes 2D array of numeric rows.
    # Why: quick deterministic duplicate check used by callers to fail-fast when duplicates exist.
    seen = set()
    for row in arr:
        t = tuple(row.tolist())
        if t in seen:
            return False
        seen.add(t)
    return True

def _make_fragments_unique(fragments: NDArray[np.floating], active_mask: NDArray[np.bool_]) -> NDArray[np.floating]:
    """
    Remove duplicate fragment rows while preserving the first-seen order.

    Note: `fragments` is expected to already be projected to the active dimensions
    (i.e. shape (k, n_active)). `active_mask` is accepted for API compatibility
    but intentionally ignored to avoid boolean-index size mismatches.
    """
    unique_fragments = np.asarray(fragments, dtype=float, order="C").copy()

    # Fast-path empty input
    if unique_fragments.size == 0:
        return unique_fragments

    # Fail fast on unexpected shape to make debugging upstream data easier.
    assert unique_fragments.ndim == 2, (
        f"fragments must be a 2D array (k, n_active); got ndim={unique_fragments.ndim}. "
        "Fix: ensure fragments are projected to active dimensions before calling this helper."
    )

    # If already unique, return as-is.
    if _rows_unique(unique_fragments):
        return unique_fragments

    # Use numpy.unique to get unique rows. np.unique(..., axis=0) may reorder rows;
    # we preserve the original first-seen order by using return_index and sorting indices.
    uniq_rows, first_indices = np.unique(unique_fragments, axis=0, return_index=True)
    order = np.argsort(first_indices)
    unique_fragments = uniq_rows[order]

    return unique_fragments


# New: vectorized batch scoring using numba parallel loop.
@njit(parallel=True, cache=True, fastmath=True)
def _score_core_batch(
    concat_F: np.ndarray,    # 1D array of all normalized fragment coordinates concatenated
    offsets: np.ndarray,     # start index in concat_F for each spectrum (int64)
    ks: np.ndarray,          # number of fragments per spectrum (int64)
    dims: np.ndarray,        # active-dimension count per spectrum (int64)
    num_points: int,
    bandwidth: float,
    alpha: float,
    rng_seed: int
) -> np.ndarray:
    """
    Compute scores for a batch of spectra. Each spectrum i has:
      - k = ks[i] fragments
      - n = dims[i] active dimensions
      - data located in concat_F[offsets[i] : offsets[i] + k * n] laid out row-major (k rows of length n)

    Why: a single numba-jitted parallel loop avoids Python-level per-row overhead and lets
    the Monte Carlo sampling be performed in native code across the batch.
    """
    N = ks.shape[0]
    results = np.empty(N, dtype=np.float64)

    inv2h2 = 1.0 / (2.0 * bandwidth * bandwidth)

    for i in prange(N):
        k = int(ks[i])
        n = int(dims[i])
        if k == 0 or n == 0:
            results[i] = 0.0
            continue

        base = int(offsets[i])
        total = 0.0
        x = np.empty(n, dtype=np.float64)

        # deterministic per-spectrum RNG: offset seed by index for reproducibility
        np.random.seed(rng_seed + i)

        for m in range(num_points):
            # sample x ~ Uniform([0,1]^n)
            for d in range(n):
                x[d] = np.random.random()

            # coverage c(x) = sum_j exp(-||x - f_j||^2 / (2h^2))
            c = 0.0
            for j in range(k):
                off_j = base + j * n
                s = 0.0
                for d in range(n):
                    t = x[d] - concat_F[off_j + d]
                    s += t * t
                c += np.exp(-s * inv2h2)

            total += np.log1p(alpha * c)

        results[i] = (total / num_points) * 1e2

    return results

def spectral_info_polars(
        precursors: pl.Series,
        fragments: pl.Series,
        *,
        bandwidth: float = 0.12,
        alpha: float = 1.0,
        num_points: int = 2048,
        rng_seed: int = 0,
        require_unique_fragments: bool = True
) -> pl.Series:
    """

    This refactored wrapper performs per-row reshaping/normalization in Python,
    concatenates all normalized fragment coordinates into a single flat array,
    and then calls a single numba-jitted vectorized routine that computes scores
    for the entire batch in parallel.

    Expectations (no heavy assertions here — preparation is done proactively):
      - precursors: Series of List(Float64)
      - fragments: Series of List(List(Float64))
    Returns:
      - Series of Float64 scores

    Algorithm:
    Kernelized local coverage score (monotone and submodular-like).

    Shapes:
      - precursor: np.ndarray(shape=(n,))
      - fragments: np.ndarray(shape=(k, n))

    Behavior:
      - Only dimensions with precursor > 0 are used for normalization and scoring.
      - Dimensions with precursor == 0 are ignored.
      - If no dimensions have precursor > 0, returns 0.0.

    Coverage field over active dims A:
      - c(x) = sum_j exp(-||x - f_j||^2 / (2 h^2)), with h=bandwidth.

    Score (approximated integral):
      - Score = mean_x log(1 + alpha * c(x)) over x ~ Uniform([0,1]^|A|).
    """
    # Minimal input checks only (skip detailed assertions per request).
    if len(precursors) != len(fragments):
        raise AssertionError("precursors and fragments must have same length")

    prec_array = precursors.to_numpy()
    frag_array = fragments.to_numpy()
    n_rows = len(prec_array)

    concat_list: list[np.float64] = []
    offsets = np.empty(n_rows, dtype=np.int64)
    ks = np.empty(n_rows, dtype=np.int64)
    dims = np.empty(n_rows, dtype=np.int64)

    cur_offset = 0
    for i in range(n_rows):
        p = prec_array[i]
        f = frag_array[i]

        # Empty or null fragments -> zero score (handled by batch scorer)
        if f is None or len(f) == 0:
            offsets[i] = cur_offset
            ks[i] = 0
            dims[i] = 0
            continue

        p_arr = np.asarray(p, dtype=float)
        active_mask = p_arr > 0.0
        n_active = int(active_mask.sum())
        if n_active == 0:
            offsets[i] = cur_offset
            ks[i] = 0
            dims[i] = 0
            continue

        # Project fragments to active dims and ensure numpy array
        frag_arr = np.asarray([np.asarray(fi, dtype=float)[active_mask] for fi in f], dtype=float)
        if frag_arr.size == 0 or frag_arr.ndim != 2:
            offsets[i] = cur_offset
            ks[i] = 0
            dims[i] = 0
            continue

        # Optionally remove duplicates (preserve first-seen order)
        if require_unique_fragments:
            if not _rows_unique(np.ascontiguousarray(frag_arr)):
                raise AssertionError(
                    "Duplicate fragments detected after ignoring zero-precursor dimensions. "
                    "Fix: ensure each fragment is unique in the active dimensions (remove duplicates or perturb values) or use `require_unique_fragments=False` to have this function remove duplicates for you"
                )
        else:
            frag_arr = _make_fragments_unique(frag_arr, active_mask)

        k_i = frag_arr.shape[0]
        # Normalize to [0,1]^n_active
        scale = (1.0 / p_arr[active_mask]).astype(np.float64)
        F_i = (frag_arr.astype(np.float64, copy=False) * scale[None, :]).ravel()  # row-major flatten (k_i * n_active)

        offsets[i] = cur_offset
        ks[i] = k_i
        dims[i] = n_active

        # extend concatenated list
        concat_list.extend(F_i.tolist())
        cur_offset += F_i.size

    # Build contiguous concat_F; if empty, return zeros
    if len(concat_list) == 0:
        return pl.Series(values=[0.0] * n_rows, dtype=pl.Float64)

    concat_F = np.asarray(concat_list, dtype=np.float64)

    # Call vectorized numba scorer
    scores = _score_core_batch(concat_F, offsets, ks, dims, int(num_points), float(bandwidth), float(alpha), int(rng_seed))

    # Return polars Series
    return pl.Series(values=scores.tolist(), dtype=pl.Float64)

def _build_scores_from_python_lists(
    prec_array: np.ndarray,
    frags_list: list,
    *,
    bandwidth: float,
    alpha: float,
    num_points: int,
    rng_seed: int,
    require_unique_fragments: bool,
) -> np.ndarray:
    """
    Internal helper: given a numpy object-array of precursor lists and a list of fragment
    row-lists per spectrum, produce scores via the numba batch kernel.
    """
    n_rows = len(prec_array)
    concat_list: list[np.float64] = []
    offsets = np.empty(n_rows, dtype=np.int64)
    ks = np.empty(n_rows, dtype=np.int64)
    dims = np.empty(n_rows, dtype=np.int64)

    cur_offset = 0
    for i in range(n_rows):
        p_raw = prec_array[i]
        f = frags_list[i]

        if f is None or len(f) == 0:
            offsets[i] = cur_offset
            ks[i] = 0
            dims[i] = 0
            continue

        p_arr = np.asarray(p_raw, dtype=float)
        active_mask = p_arr > 0.0
        n_active = int(active_mask.sum())
        if n_active == 0:
            offsets[i] = cur_offset
            ks[i] = 0
            dims[i] = 0
            continue

        # Project fragments to active dims
        frag_arr = np.asarray([np.asarray(fi, dtype=float)[active_mask] for fi in f], dtype=float)
        if frag_arr.size == 0 or frag_arr.ndim != 2:
            offsets[i] = cur_offset
            ks[i] = 0
            dims[i] = 0
            continue

        if require_unique_fragments:
            if not _rows_unique(np.ascontiguousarray(frag_arr)):
                raise AssertionError(
                    "Duplicate fragments detected after ignoring zero-precursor dimensions. "
                    "Fix or set require_unique_fragments=False to drop duplicates."
                )
        else:
            frag_arr = _make_fragments_unique(frag_arr, active_mask)

        k_i = frag_arr.shape[0]
        scale = (1.0 / p_arr[active_mask]).astype(np.float64)
        F_i = (frag_arr.astype(np.float64, copy=False) * scale[None, :]).ravel()

        offsets[i] = cur_offset
        ks[i] = k_i
        dims[i] = n_active

        concat_list.extend(F_i.tolist())
        cur_offset += F_i.size

    if len(concat_list) == 0:
        return np.zeros(n_rows, dtype=np.float64)

    concat_F = np.asarray(concat_list, dtype=np.float64)
    scores = _score_core_batch(
        concat_F, offsets, ks, dims,
        int(num_points), float(bandwidth), float(alpha), int(rng_seed)
    )
    return scores


def spectral_info_search(
    query_precursor: list[float] | NDArray[np.floating],
    query_fragments: list[list[float]] | None,
    library_precursor: list[float] | NDArray[np.floating],
    library_fragments: list[list[float]] | None,
    *,
    bandwidth: float = 0.12,
    alpha: float = 1.0,
    num_points: int = 2048,
    rng_seed: int = 0,
    require_unique_fragments: bool = True,
) -> tuple[float, float, float, float]:
    """
    Compute 4 values for a single query/library pair:
      - query: info score on all query fragments
      - library: info score on all library fragments
      - query_extra: info score on fragments in query but not in library (set difference)
      - library_extra: info score on fragments in library but not in query (set difference)
    """
    # Normalize inputs to arrays/lists
    qp = np.asarray(query_precursor, dtype=float)
    lp = np.asarray(library_precursor, dtype=float)
    qf = query_fragments or []
    lf = library_fragments or []

    # Fast paths for empties
    qf_arr = np.asarray([np.asarray(r, dtype=float) for r in qf], dtype=float) if len(qf) > 0 else np.empty((0, qp.size), dtype=float)
    lf_arr = np.asarray([np.asarray(r, dtype=float) for r in lf], dtype=float) if len(lf) > 0 else np.empty((0, lp.size), dtype=float)

    # Intersection of active dimensions for set difference
    inter_len = min(qp.size, lp.size)
    if inter_len == 0:
        q_extra_rows = qf
        l_extra_rows = lf
    else:
        q_mask = (qp[:inter_len] > 0.0)
        l_mask = (lp[:inter_len] > 0.0)
        inter_mask = q_mask & l_mask
        if inter_mask.sum() == 0 or qf_arr.shape[0] == 0 or lf_arr.shape[0] == 0:
            # No shared active dims or one side empty => all rows are extra on their side
            q_extra_rows = qf
            l_extra_rows = lf
        else:
            q_proj = qf_arr[:, :inter_len][:, inter_mask]
            l_proj = lf_arr[:, :inter_len][:, inter_mask]

            l_set = {tuple(row.tolist()) for row in l_proj}
            q_set = {tuple(row.tolist()) for row in q_proj}

            q_extra_rows = [qf[i] for i, row in enumerate(q_proj) if tuple(row.tolist()) not in l_set]
            l_extra_rows = [lf[i] for i, row in enumerate(l_proj) if tuple(row.tolist()) not in q_set]

    # Build batched scoring for the 3 lists (query_all, library_all, extras on each)
    prec_array = np.array([qp], dtype=object)
    lib_prec_array = np.array([lp], dtype=object)

    q_scores = _build_scores_from_python_lists(
        prec_array, [qf],
        bandwidth=bandwidth, alpha=alpha, num_points=num_points, rng_seed=rng_seed,
        require_unique_fragments=require_unique_fragments,
    )
    l_scores = _build_scores_from_python_lists(
        lib_prec_array, [lf],
        bandwidth=bandwidth, alpha=alpha, num_points=num_points, rng_seed=rng_seed,
        require_unique_fragments=require_unique_fragments,
    )
    qx_scores = _build_scores_from_python_lists(
        prec_array, [q_extra_rows],
        bandwidth=bandwidth, alpha=alpha, num_points=num_points, rng_seed=rng_seed,
        require_unique_fragments=require_unique_fragments,
    )
    lx_scores = _build_scores_from_python_lists(
        lib_prec_array, [l_extra_rows],
        bandwidth=bandwidth, alpha=alpha, num_points=num_points, rng_seed=rng_seed,
        require_unique_fragments=require_unique_fragments,
    )

    return float(q_scores[0]), float(l_scores[0]), float(qx_scores[0]), float(lx_scores[0])


def spectral_info_search_polars(
    query_precursors: pl.Series,
    query_fragments: pl.Series,
    library_precursors: pl.Series,
    library_fragments: pl.Series,
    *,
    bandwidth: float = 0.12,
    alpha: float = 1.0,
    num_points: int = 2048,
    rng_seed: int = 0,
    require_unique_fragments: bool = True,
) -> pl.Series:
    """
    Vectorized wrapper over Polars:
      - Takes series of query/library precursors and fragments
      - Computes 4 scores per row: query, library, query_extra, library_extra
      - Returns a Struct Series with those 4 fields.
    """
    if len(query_precursors) != len(query_fragments):
        raise AssertionError("query_precursors and query_fragments must have same length")
    if len(library_precursors) != len(library_fragments):
        raise AssertionError("library_precursors and library_fragments must have same length")
    if len(query_precursors) != len(library_precursors):
        raise AssertionError("query and library must have the same number of rows")

    n_rows = len(query_precursors)

    qp_array = query_precursors.to_numpy()
    qf_array = query_fragments.to_numpy()
    lp_array = library_precursors.to_numpy()
    lf_array = library_fragments.to_numpy()

    # Prepare 4 fragment lists: q_all, l_all, q_extra, l_extra
    q_all = [None] * n_rows
    l_all = [None] * n_rows
    q_extra = [None] * n_rows
    l_extra = [None] * n_rows

    for i in range(n_rows):
        qp = np.asarray(qp_array[i], dtype=float)
        lp = np.asarray(lp_array[i], dtype=float)

        qf = qf_array[i]
        lf = lf_array[i]

        q_all[i] = [] if (qf is None) else [np.asarray(r, dtype=float).tolist() for r in qf]
        l_all[i] = [] if (lf is None) else [np.asarray(r, dtype=float).tolist() for r in lf]

        # Build arrays for difference
        qf_arr = np.asarray([np.asarray(r, dtype=float) for r in qf], dtype=float) if (qf is not None and len(qf) > 0) else np.empty((0, qp.size), dtype=float)
        lf_arr = np.asarray([np.asarray(r, dtype=float) for r in lf], dtype=float) if (lf is not None and len(lf) > 0) else np.empty((0, lp.size), dtype=float)

        # Determine intersection of active dimensions
        inter_len = min(qp.size, lp.size)
        if inter_len == 0:
            q_extra[i] = q_all[i]
            l_extra[i] = l_all[i]
            continue

        q_mask = (qp[:inter_len] > 0.0)
        l_mask = (lp[:inter_len] > 0.0)
        inter_mask = q_mask & l_mask

        if inter_mask.sum() == 0 or qf_arr.shape[0] == 0 or lf_arr.shape[0] == 0:
            q_extra[i] = q_all[i]
            l_extra[i] = l_all[i]
            continue

        q_proj = qf_arr[:, :inter_len][:, inter_mask]
        l_proj = lf_arr[:, :inter_len][:, inter_mask]

        l_set = {tuple(row.tolist()) for row in l_proj}
        q_set = {tuple(row.tolist()) for row in q_proj}

        q_extra_rows = [q_all[i][idx] for idx, row in enumerate(q_proj) if tuple(row.tolist()) not in l_set]
        l_extra_rows = [l_all[i][idx] for idx, row in enumerate(l_proj) if tuple(row.tolist()) not in q_set]

        q_extra[i] = q_extra_rows
        l_extra[i] = l_extra_rows

    # Build scores via batch kernel
    qp_np = np.array([np.asarray(x, dtype=float) for x in qp_array], dtype=object)
    lp_np = np.array([np.asarray(x, dtype=float) for x in lp_array], dtype=object)

    q_scores = _build_scores_from_python_lists(
        qp_np, q_all,
        bandwidth=bandwidth, alpha=alpha, num_points=num_points, rng_seed=rng_seed,
        require_unique_fragments=require_unique_fragments,
    )
    l_scores = _build_scores_from_python_lists(
        lp_np, l_all,
        bandwidth=bandwidth, alpha=alpha, num_points=num_points, rng_seed=rng_seed,
        require_unique_fragments=require_unique_fragments,
    )
    qx_scores = _build_scores_from_python_lists(
        qp_np, q_extra,
        bandwidth=bandwidth, alpha=alpha, num_points=num_points, rng_seed=rng_seed,
        require_unique_fragments=require_unique_fragments,
    )
    lx_scores = _build_scores_from_python_lists(
        lp_np, l_extra,
        bandwidth=bandwidth, alpha=alpha, num_points=num_points, rng_seed=rng_seed,
        require_unique_fragments=require_unique_fragments,
    )

    s_query = pl.Series(values=q_scores.tolist(), dtype=pl.Float64, name="query")
    s_library = pl.Series(values=l_scores.tolist(), dtype=pl.Float64, name="library")
    s_query_extra = pl.Series(values=qx_scores.tolist(), dtype=pl.Float64, name="query_extra")
    s_library_extra = pl.Series(values=lx_scores.tolist(), dtype=pl.Float64, name="library_extra")

    return pl.struct([s_query, s_library, s_query_extra, s_library_extra])