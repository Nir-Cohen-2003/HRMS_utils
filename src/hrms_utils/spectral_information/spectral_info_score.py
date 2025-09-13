import numpy as np
from numpy.typing import NDArray
from numba import njit
import polars as pl

def _rows_unique(arr: NDArray[np.int64]) -> bool:
    # Assumes 2D
    seen = set()
    for row in arr:
        t = tuple(row.tolist())
        if t in seen:
            return False
        seen.add(t)
    return True

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

def score_fragments_local_coverage(
    precursor: NDArray[np.floating],
    fragments: NDArray[np.floating],
    *,
    bandwidth: float = 0.12,   # kernel width (in normalized [0,1] space)
    alpha: float = 1.0,        # scales log(1 + alpha * c)
    num_points: int = 2048,    # Monte Carlo samples for the integral
    rng_seed: int = 0,         # fixed sampling for exact monotonic comparisons
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
    assert _rows_unique(frag_check_arr), (
        "Duplicate fragments detected after ignoring zero-precursor dimensions. "
        "Fix: ensure each fragment is unique in the active dimensions (remove duplicates or perturb values)."
    )

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
) -> pl.Series:
    """
    Polars wrapper for score_fragments_local_coverage.

    Expects:
      - precursors: Series of List(Float64)
      - fragments: Series of List(List(Float64))
    Returns:
      - Series of Float64 scores
    """
    # Why: fail-fast on invalid inputs with actionable messages so callers can fix upstream data.
    assert isinstance(precursors, pl.Series), "precursors must be a polars.Series; got %r" % type(precursors)
    assert isinstance(fragments, pl.Series), "fragments must be a polars.Series; got %r" % type(fragments)

    assert len(precursors) == len(fragments), "precursors and fragments must have the same length"

    # Convert to numpy object arrays for per-row validation and iteration.
    prec_array = precursors.to_numpy()
    frag_array = fragments.to_numpy()

    # Find first non-null entries to validate expected structure (fail fast and explicitly).
    first_idx = None
    for i in range(len(prec_array)):
        if prec_array[i] is not None:
            first_idx = i
            break

    assert first_idx is not None, "precursors series contains only null/None values; at least one precursor is required to infer shape"

    first_prec = np.asarray(prec_array[first_idx], dtype=float)
    assert first_prec.ndim == 1, f"precursor entries must be 1D sequences; first non-null entry has ndim={first_prec.ndim}"
    assert np.all(first_prec >= 0), "precursor values must be non-negative; negative values detected in the first non-null precursor"

    # Validate the corresponding fragments structure for the first non-null row (shape check).
    first_frag = frag_array[first_idx]
    assert first_frag is not None and len(first_frag) >= 0, "fragments entries must be list-like (can be empty list for no fragments)"

    if len(first_frag) > 0:
        # Ensure fragments are list of 1D numeric sequences and match precursor length.
        first_frag_item = np.asarray(first_frag[0], dtype=float)
        assert first_frag_item.ndim == 1, "each fragment must be a 1D sequence of numeric values"
        assert first_frag_item.shape[0] == first_prec.shape[0], (
            "fragment dimensionality must match precursor length; "
            f"expected {first_prec.shape[0]}, got {first_frag_item.shape[0]}"
        )

    scores: list[float] = []
    # Per-row validation inside the loop so we can return a meaningful error for a specific index.
    for idx in range(len(prec_array)):
        p = prec_array[idx]
        f = frag_array[idx]

        if f is None or len(f) == 0:
            scores.append(0.0)
            continue

        # Validate shapes for this row explicitly and fail with actionable message.
        try:
            p_arr = np.asarray(p, dtype=float)
        except Exception as e:
            raise AssertionError(f"precursor at index {idx} is not convertible to numeric 1D array: {e}")
        assert p_arr.ndim == 1, f"precursor at index {idx} must be 1D, got ndim={p_arr.ndim}"
        assert np.all(p_arr >= 0), f"precursor at index {idx} contains negative values; precursors must be >= 0"

        # fragments row must be list-like of 1D numeric sequences of same length as precursor
        assert isinstance(f, (list, tuple, np.ndarray)), f"fragments at index {idx} must be list-like of fragments"
        # Convert fragments into a 2D numpy array if possible for score function
        try:
            frag_arr = np.asarray([np.asarray(fi, dtype=float) for fi in f], dtype=float)
        except Exception as e:
            raise AssertionError(f"fragments at index {idx} must be list-like of numeric 1D sequences: {e}")

        assert frag_arr.ndim == 2, f"fragments at index {idx} must be 2D (k, n); got ndim={frag_arr.ndim}"
        assert frag_arr.shape[1] == p_arr.shape[0], (
            f"fragment dimensionality at index {idx} ({frag_arr.shape[1]}) "
            f"does not match precursor length ({p_arr.shape[0]})"
        )

        score = score_fragments_local_coverage(
            p_arr,
            frag_arr,
            bandwidth=bandwidth,
            alpha=alpha,
            num_points=num_points,
            rng_seed=rng_seed,
        )
        scores.append(float(score))

    # Why Float64: maintain numerical precision across downstream analytic steps.
    return pl.Series(values=scores, dtype=pl.Float64)