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

@njit(cache=True, fastmath=True)
def _score_core(F: np.ndarray, num_points: int, bandwidth: float, alpha: float, rng_seed: int) -> float:
    # F: (k, n) normalized fragments in [0,1]^n
    k = F.shape[0]
    n = F.shape[1]
    if k == 0:
        return 0.0

    # Seed RNG compatible with Numba
    np.random.seed(rng_seed)

    inv2h2 = 1.0 / (2.0 * bandwidth * bandwidth)
    total = 0.0
    x = np.empty(n, dtype=np.float64)

    for m in range(num_points):
        # sample x ~ Uniform([0,1]^n)
        for d in range(n):
            x[d] = np.random.random()

        # coverage c(x) = sum_j exp(-||x - f_j||^2 / (2h^2))
        c = 0.0
        for j in range(k):
            s = 0.0
            Fj = F[j]
            for d in range(n):
                t = x[d] - Fj[d]
                s += t * t
            c += np.exp(-s * inv2h2)

        total += np.log1p(alpha * c)

    return (total / num_points) * 1e2

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

def score_fragments_local_coverage(
    precursor: NDArray[np.floating],
    fragments: NDArray[np.floating],
    *,
    bandwidth: float = 0.12,   # kernel width (in normalized [0,1] space)
    alpha: float = 1.0,        # scales log(1 + alpha * c)
    num_points: int = 2048,    # Monte Carlo samples for the integral
    rng_seed: int = 0,         # fixed sampling for exact monotonic comparisons
    require_unique_fragments: bool = True
) -> float:
    """
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
    precursor = np.asarray(precursor, dtype=float)

    # Why: fail-fast and give actionable remediation to callers when shapes/dtypes are wrong.
    assert precursor.ndim == 1, (
        f"precursor must be a 1D array; got {precursor.ndim}D array. "
        "Fix: supply a 1D sequence of positive precursor values (e.g. np.array([...]))."
    )

    # Convert fragments to numpy and validate shape consistently (fail-fast).
    fragments = np.asarray(fragments, dtype=float)
    assert fragments.ndim == 2, (
        f"fragments must be a 2D array with shape (k, n); got ndim={fragments.ndim}. "
        "Fix: provide fragments as an array-like of fragment vectors, e.g. list-of-lists."
    )
    assert fragments.shape[1] == precursor.shape[0], (
        f"fragments second dimension must match precursor length (expected {precursor.shape[0]}, got {fragments.shape[1]}). "
        "Fix: ensure each fragment has the same number of dimensions as the precursor."
    )

    # Why: precursor values < 0 are invalid for normalization; fail loudly with remediation.
    assert np.all(precursor >= 0), (
        "precursor entries must be non-negative. "
        "Fix: remove or correct negative precursor values before calling this function."
    )

    # Select only dimensions with strictly positive precursor (active dims).
    active_mask = precursor > 0.0
    if not np.any(active_mask):
        # No active dimensions -> no meaningful normalization/coverage
        return 0.0

    precursor_active = precursor[active_mask]
    fragments_active = np.asarray(fragments)[:, active_mask]

    k = fragments_active.shape[0]
    if k == 0:
        return 0.0

    # Duplicate check after projecting to active dimensions only.
    # Why: duplicates would make the score degenerate for set-like operations; fail-fast with guidance.
    frag_check_arr = np.ascontiguousarray(fragments_active)

    if require_unique_fragments:
        assert _rows_unique(frag_check_arr), (
            "Duplicate fragments detected after ignoring zero-precursor dimensions. "
            "Fix: ensure each fragment is unique in the active dimensions (remove duplicates or perturb values)."
        )
    else:
        fragments_active = _make_fragments_unique(fragments_active, active_mask)

    # Normalize to [0,1]^|A| using only active dims (ensure C-contiguous float64 for Numba).
    # Why: ignore dims with zero precursor to avoid division by zero and keep score invariant to them.
    scale = (1.0 / precursor_active).astype(np.float64)
    F = fragments_active.astype(np.float64, copy=False) * scale[None, :]
    F = np.ascontiguousarray(F)

    # If, after ignoring zero dims, there are 0 active dims, return 0.0.
    if F.shape[1] == 0:
        return 0.0

    return _score_core(F, int(num_points), float(bandwidth), float(alpha), int(rng_seed))

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
    Polars wrapper for score_fragments_local_coverage.

    This refactored wrapper performs per-row reshaping/normalization in Python,
    concatenates all normalized fragment coordinates into a single flat array,
    and then calls a single numba-jitted vectorized routine that computes scores
    for the entire batch in parallel.

    Expectations (no heavy assertions here — preparation is done proactively):
      - precursors: Series of List(Float64)
      - fragments: Series of List(List(Float64))
    Returns:
      - Series of Float64 scores
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