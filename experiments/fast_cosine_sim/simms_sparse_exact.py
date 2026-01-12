#!/usr/bin/env python
"""
SimMS-based sparse exact cosine kernel and batching helper.

This module uses a lightweight CUDA kernel inspired by SimMS' cosine kernel to
compute exact cosine similarities only for candidate (row, col) pairs produced
by the approximate stage. It returns a COO sparse matrix with the filled
scores, enabling a two-stage pipeline: approximate (fast, sparse) -> exact
(focused on candidates).

Key choices:
- Uses numba.cuda directly; assumes spectra are individually sorted by m/z.
- Omits precursor shift and mass-shift logic for simplicity per request.
- Fails fast when required inputs are missing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numba
import numpy as np
import scipy.sparse as sp
from numba import cuda


@dataclass
class SparseExactConfig:
    """Configuration for sparse exact cosine on GPU.

    Attributes
    ----------
    tolerance_ppm : float
        MS2 tolerance in ppm for matching peaks.
    mz_power : float
        Exponent for m/z scaling in cosine computation.
    intensity_power : float
        Exponent for intensity scaling in cosine computation.
    match_limit : int
        Maximum matches per pair (bounds local buffers).
    n_max_peaks : int | None
        Max peaks per spectrum; if None uses per-side max length.
    pair_batch_size : int
        Number of candidate pairs per GPU launch to bound memory.
    """

    tolerance_ppm: float = 10.0
    mz_power: float = 0.0
    intensity_power: float = 0.5
    match_limit: int = 1024
    n_max_peaks: int | None = None
    pair_batch_size: int = 65_536  # ~64k pairs per launch

    def __post_init__(self) -> None:
        assert self.tolerance_ppm > 0, "tolerance_ppm must be positive"
        assert self.match_limit > 0, "match_limit must be positive"
        assert self.pair_batch_size > 0, "pair_batch_size must be positive"


def _pad_spectra_to_dense(
    mz_list: Sequence[np.ndarray],
    int_list: Sequence[np.ndarray],
    n_max_peaks: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Pad spectra to dense arrays.

    Returns
    -------
    mz_dense : np.ndarray, shape (n_spectra, max_peaks)
    int_dense : np.ndarray, shape (n_spectra, max_peaks)
    lengths : np.ndarray, shape (n_spectra,)
    max_peaks : int
    """
    assert len(mz_list) == len(int_list), "mz_list and int_list length mismatch"
    lengths = np.array([len(m) for m in mz_list], dtype=np.int32)
    max_peaks = int(np.max(lengths)) if lengths.size > 0 else 0
    if n_max_peaks is not None:
        max_peaks = min(max_peaks, int(n_max_peaks))
    mz_dense = np.zeros((len(mz_list), max_peaks), dtype=np.float32)
    int_dense = np.zeros((len(int_list), max_peaks), dtype=np.float32)
    for i, (mz, inten) in enumerate(zip(mz_list, int_list)):
        l_i = min(len(mz), max_peaks)
        mz_dense[i, :l_i] = mz[:l_i]
        int_dense[i, :l_i] = inten[:l_i]
        lengths[i] = l_i
    return mz_dense, int_dense, lengths, max_peaks


def _compute_norms(
    mz_dense: np.ndarray,
    int_dense: np.ndarray,
    lengths: np.ndarray,
    mz_power: float,
    intensity_power: float,
) -> np.ndarray:
    """Compute cosine norms for each spectrum."""
    weights = (mz_dense**mz_power) * (int_dense**intensity_power)
    sq = (weights**2).sum(axis=1)
    norms = np.sqrt(sq).astype(np.float32)
    assert np.all(norms > 0), "All spectra must have non-zero norm"
    return norms


def _build_sparse_exact_cosine_kernel(
    match_limit_const: int,
    max_peaks_const: int,
):
    assert match_limit_const > 0, "match_limit_const must be positive"
    assert max_peaks_const > 0, "max_peaks_const must be positive"

    @cuda.jit
    def _kernel(
        mz_left,
        int_left,
        len_left,
        mz_right,
        int_right,
        len_right,
        norms_left,
        norms_right,
        pair_left_idx,
        pair_right_idx,
        tolerance_ppm,
        mz_power,
        int_power,
        out_scores,
        out_matches,
    ):
        """Compute exact cosine for candidate pairs (1D grid).

        Matches up to `match_limit_const` peaks per pair; overflow is signaled
        by returning a negative match count.
        """
        idx = cuda.grid(1)
        n_pairs = pair_left_idx.size
        if idx >= n_pairs:
            return

        li = pair_left_idx[idx]
        ri = pair_right_idx[idx]

        l_len = len_left[li]
        r_len = len_right[ri]
        if l_len == 0 or r_len == 0:
            out_scores[idx] = 0.0
            out_matches[idx] = 0
            return

        match_r = cuda.local.array(match_limit_const, numba.int32)
        match_q = cuda.local.array(match_limit_const, numba.int32)
        match_vals = cuda.local.array(match_limit_const, numba.float32)
        used_r = cuda.local.array(max_peaks_const, numba.boolean)
        used_q = cuda.local.array(max_peaks_const, numba.boolean)

        for k in range(max_peaks_const):
            used_r[k] = False
            used_q[k] = False

        count = numba.int32(0)
        overflow = numba.int32(0)

        i = numba.int32(0)
        j = numba.int32(0)
        while i < l_len and j < r_len:
            mz_l = mz_left[li, i]
            mz_r = mz_right[ri, j]

            base = mz_l
            if base < 200.0:
                base = 200.0
            tol_da = base * tolerance_ppm * 1e-6

            diff = mz_l - mz_r
            if diff > tol_da:
                j += 1
                continue
            if diff < -tol_da:
                i += 1
                continue

            int_l = int_left[li, i]
            int_r = int_right[ri, j]
            amp_l = math.pow(mz_l, mz_power) * math.pow(int_l, int_power)
            amp_r = math.pow(mz_r, mz_power) * math.pow(int_r, int_power)

            if count < match_limit_const:
                match_r[count] = i
                match_q[count] = j
                match_vals[count] = amp_l * amp_r
            else:
                overflow = 1
                break
            count += 1
            i += 1
            j += 1

        score = numba.float32(0.0)
        used_matches = numba.int32(0)
        for k in range(count):
            r_idx = match_r[k]
            q_idx = match_q[k]
            if (not used_r[r_idx]) and (not used_q[q_idx]):
                used_r[r_idx] = True
                used_q[q_idx] = True
                score += match_vals[k]
                used_matches += 1

        norm = norms_left[li] * norms_right[ri]
        if norm > 0:
            out_scores[idx] = score / norm
        else:
            out_scores[idx] = 0.0
        out_matches[idx] = -used_matches if overflow else used_matches

    return _kernel


def sparse_exact_cosine_from_pairs_gpu(
    left_mz: Sequence[np.ndarray],
    left_int: Sequence[np.ndarray],
    right_mz: Sequence[np.ndarray],
    right_int: Sequence[np.ndarray],
    pair_left_idx: np.ndarray,
    pair_right_idx: np.ndarray,
    config: SparseExactConfig | None = None,
) -> sp.coo_matrix:
    """Compute sparse exact cosine on GPU for provided candidate pairs.

    Parameters
    ----------
    left_mz, left_int, right_mz, right_int : sequences of np.ndarray
        Sorted spectra (per spectrum) for left/right sides.
    pair_left_idx, pair_right_idx : np.ndarray[int]
        Indices into left/right spectra (same length).
    config : SparseExactConfig
        Kernel and batching settings.

    Returns
    -------
    scipy.sparse.coo_matrix
        COO matrix with shape (len(left_mz), len(right_mz)) containing scores.
    """
    assert pair_left_idx.shape == pair_right_idx.shape, "pair index shape mismatch"
    if config is None:
        config = SparseExactConfig()

    n_left = len(left_mz)
    n_right = len(right_mz)
    assert n_left > 0 and n_right > 0, "left/right spectra must be non-empty"
    if pair_left_idx.size == 0:
        return sp.coo_matrix((n_left, n_right))
    min_left = int(pair_left_idx.min())
    max_left = int(pair_left_idx.max())
    min_right = int(pair_right_idx.min())
    max_right = int(pair_right_idx.max())
    assert min_left >= 0 and min_right >= 0, "pair indices must be non-negative"
    assert max_left < n_left, "pair_left_idx has out-of-bounds entries"
    assert max_right < n_right, "pair_right_idx has out-of-bounds entries"

    mz_l, int_l, len_l, max_peaks_l = _pad_spectra_to_dense(
        left_mz, left_int, config.n_max_peaks
    )
    mz_r, int_r, len_r, max_peaks_r = _pad_spectra_to_dense(
        right_mz, right_int, config.n_max_peaks
    )
    assert max_peaks_l == max_peaks_r, (
        "Left and right must share padded peak dimension for the kernel"
    )

    norms_l = _compute_norms(
        mz_l, int_l, len_l, config.mz_power, config.intensity_power
    )
    norms_r = _compute_norms(
        mz_r, int_r, len_r, config.mz_power, config.intensity_power
    )

    d_mz_l = cuda.to_device(mz_l)
    d_int_l = cuda.to_device(int_l)
    d_len_l = cuda.to_device(len_l)
    d_mz_r = cuda.to_device(mz_r)
    d_int_r = cuda.to_device(int_r)
    d_len_r = cuda.to_device(len_r)
    d_norms_l = cuda.to_device(norms_l)
    d_norms_r = cuda.to_device(norms_r)

    kernel = _build_sparse_exact_cosine_kernel(
        match_limit_const=config.match_limit,
        max_peaks_const=max_peaks_l,
    )
    scores: list[np.ndarray] = []
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []

    threads_per_block = 256
    n_pairs_total = pair_left_idx.size
    batch_size = config.pair_batch_size

    for start in range(0, n_pairs_total, batch_size):
        end = min(start + batch_size, n_pairs_total)
        batch_left = pair_left_idx[start:end].astype(np.int32, copy=False)
        batch_right = pair_right_idx[start:end].astype(np.int32, copy=False)

        d_left_idx = cuda.to_device(batch_left)
        d_right_idx = cuda.to_device(batch_right)
        d_scores = cuda.device_array(shape=batch_left.shape, dtype=np.float32)
        d_matches = cuda.device_array(shape=batch_left.shape, dtype=np.int32)

        blocks = (batch_left.size + threads_per_block - 1) // threads_per_block
        kernel[blocks, threads_per_block](
            d_mz_l,
            d_int_l,
            d_len_l,
            d_mz_r,
            d_int_r,
            d_len_r,
            d_norms_l,
            d_norms_r,
            d_left_idx,
            d_right_idx,
            np.float32(config.tolerance_ppm),
            np.float32(config.mz_power),
            np.float32(config.intensity_power),
            d_scores,
            d_matches,
        )

        scores.append(d_scores.copy_to_host())
        rows.append(batch_left.astype(np.int64, copy=False))
        cols.append(batch_right.astype(np.int64, copy=False))

    data = np.concatenate(scores) if scores else np.array([], dtype=np.float32)
    row = np.concatenate(rows) if rows else np.array([], dtype=np.int64)
    col = np.concatenate(cols) if cols else np.array([], dtype=np.int64)

    return sp.coo_matrix((data, (row, col)), shape=(len(left_mz), len(right_mz)))
