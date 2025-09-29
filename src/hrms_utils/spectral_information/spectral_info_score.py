import numpy as np
from numpy.typing import NDArray
from numba import njit, prange
import polars as pl


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
) -> pl.Series:
    """
    This refactored wrapper prepares data and calls a numba-jitted routine to compute scores in parallel.

    Expectations:
      - precursors: Series of List(Float64)
      - fragments: Series of List(List(Float64))
    Returns:
      - Series of Float64 scores

    Algorithm:
    Kernelized local coverage score.

    Behavior:
      - Only dimensions with precursor > 0 are used for normalization and scoring.
      - If require_unique_fragments is False, fragment rows are deduplicated *after* projection to active dimensions.
      - If require_unique_fragments is True, an assertion is raised if duplicates are found after projection.

    Coverage field over active dims A:
      - c(x) = sum_j exp(-||x - f_j||^2 / (2 h^2)), with h=bandwidth.

    Score (approximated integral):
      - Score = mean_x log(1 + alpha * c(x)) over x ~ Uniform([0,1]^|A|).
    """
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
        p_list = prec_array[i]
        f_list_of_lists = frag_array[i]

        if f_list_of_lists is None or len(f_list_of_lists) == 0 or p_list is None or len(p_list) == 0:
            offsets[i] = cur_offset
            ks[i] = 0
            dims[i] = 0
            continue

        p_arr = np.asarray(p_list, dtype=float)
        active_mask = p_arr > 0.0
        n_active = int(active_mask.sum())
        if n_active == 0:
            offsets[i] = cur_offset
            ks[i] = 0
            dims[i] = 0
            continue

        # Project fragments to active dimensions. This must be done in Python before batching.
        frag_arr_list = [np.asarray(fi, dtype=float)[active_mask] for fi in f_list_of_lists]
        frag_arr_list = [f for f in frag_arr_list if f.size > 0]

        if not frag_arr_list:
            offsets[i] = cur_offset
            ks[i] = 0
            dims[i] = 0
            continue
            
        frag_arr = np.array(frag_arr_list)

        if frag_arr.ndim != 2:
            offsets[i] = cur_offset
            ks[i] = 0
            dims[i] = 0
            continue

        if frag_arr.shape[0] > 1:
            unique_rows, first_indices = np.unique(frag_arr, axis=0, return_index=True)
            if unique_rows.shape[0] < frag_arr.shape[0]:
                order = np.argsort(first_indices)
                frag_arr = unique_rows[order]

        k_i = frag_arr.shape[0]
        scale = (1.0 / p_arr[active_mask]).astype(np.float64)
        F_i = (frag_arr.astype(np.float64, copy=False) * scale[None, :]).ravel()

        offsets[i] = cur_offset
        ks[i] = k_i
        dims[i] = n_active

        concat_list.extend(F_i.tolist())
        cur_offset += F_i.size

    if len(concat_list) == 0:
        return pl.Series(values=[0.0] * n_rows, dtype=pl.Float64)

    concat_F = np.asarray(concat_list, dtype=np.float64)

    scores = _score_core_batch(
        concat_F, 
        offsets, 
        ks, 
        dims, 
        int(num_points), 
        float(bandwidth), 
        float(alpha), 
        int(rng_seed)
    )

    return pl.Series(values=scores.tolist(), dtype=pl.Float64)
