import math

import numpy as np
from numba import cuda, float32, int32, uint64

# CONSTANTS
WARP_SIZE = 32
MAX_MATCHES_SMALL = 128  # Buffer size for matches in small kernel
MAX_MATCHES_MEDIUM = 512  # Buffer size for matches in medium kernel
MAX_MATCHES_LARGE = 1024  # Buffer size for matches in large kernel
LARGE_THREAD_BLOCK = 256  # Threads for large kernel


@cuda.jit(device=True, inline=True)
def swap_matches(scores, r_idxs, q_idxs, i, j):
    """Swap match i and match j in the arrays."""
    temp_s = scores[i]
    temp_r = r_idxs[i]
    temp_q = q_idxs[i]

    scores[i] = scores[j]
    r_idxs[i] = r_idxs[j]
    q_idxs[i] = q_idxs[j]

    scores[j] = temp_s
    r_idxs[j] = temp_r
    q_idxs[j] = temp_q


@cuda.jit
def greedy_cosine_fast_small(
    mz_a,
    int_a,
    len_a,
    mz_b,
    int_b,
    len_b,
    out_scores,
    n_pairs,
    stride,
    tolerance,
    shift,
    mz_power,
    int_power,
    pair_a_indices,
    pair_b_indices,
):
    """
    Optimized for small spectra (< 64 peaks) using Warp-per-Pair strategy.
    Supports arbitrary pairings via pair_a_indices/pair_b_indices.
    Grid: (N_PAIRS + (Threads/32) - 1) / (Threads/32), 1
    Block: 256 threads (8 pairs per block)
    Shared Mem: 8 warps * 128 matches * (4+4+4 bytes) = 12KB
    """
    shared_scores = cuda.shared.array(1024, float32)
    shared_r = cuda.shared.array(1024, int32)
    shared_q = cuda.shared.array(1024, int32)

    warp_counters = cuda.shared.array(32, int32)

    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    bdim = cuda.blockDim.x

    warp_id_in_block = tid // 32
    lane_id = tid % 32

    global_warp_id = (bid * (bdim // 32)) + warp_id_in_block

    if global_warp_id >= n_pairs:
        return

    a_idx = pair_a_indices[global_warp_id]
    b_idx = pair_b_indices[global_warp_id]

    base_idx = warp_id_in_block * MAX_MATCHES_SMALL
    if lane_id == 0:
        warp_counters[warp_id_in_block] = 0

    for i in range(lane_id, MAX_MATCHES_SMALL, 32):
        shared_scores[base_idx + i] = -1.0
        shared_r[base_idx + i] = -1
        shared_q[base_idx + i] = -1

    cuda.syncwarp()

    la = len_a[a_idx]
    lb = len_b[b_idx]

    # 1. FIND MATCHES (Parallel)
    for i in range(lane_id, la, 32):
        val_mz_a = mz_a[a_idx, i]
        val_int_a = int_a[a_idx, i]

        if mz_power == 0 and int_power == 0.5:
            amp_a = math.sqrt(val_int_a)
        else:
            amp_a = (val_mz_a**mz_power) * (val_int_a**int_power)

        base_mz = val_mz_a
        if base_mz < 200.0:
            base_mz = 200.0
        tol_da = base_mz * tolerance * 1e-6

        for j in range(lb):
            val_mz_b = mz_b[b_idx, j]

            if shift == 0:
                mz_q_shifted = val_mz_b
            else:
                mz_q_shifted = val_mz_b + shift

            if mz_q_shifted > val_mz_a + tol_da:
                break
            if mz_q_shifted < val_mz_a - tol_da:
                continue

            val_int_b = int_b[b_idx, j]

            if mz_power == 0 and int_power == 0.5:
                amp_b = math.sqrt(val_int_b)
            else:
                amp_b = (val_mz_b**mz_power) * (val_int_b**int_power)

            score = amp_a * amp_b

            idx = cuda.atomic.add(warp_counters, warp_id_in_block, 1)

            if idx < MAX_MATCHES_SMALL:
                shared_scores[base_idx + idx] = score
                shared_r[base_idx + idx] = i
                shared_q[base_idx + idx] = j
                
    cuda.syncwarp()
    
    count = warp_counters[warp_id_in_block]
    if count > MAX_MATCHES_SMALL:
        count = MAX_MATCHES_SMALL

    # 2. BITONIC SORT
    k = 2
    while k <= MAX_MATCHES_SMALL:
        j = k // 2
        while j > 0:
            for i in range(lane_id, MAX_MATCHES_SMALL, 32):
                ixj = i ^ j
                if ixj > i:
                    s_i = shared_scores[base_idx + i]
                    s_j = shared_scores[base_idx + ixj]

                    should_swap = False
                    if (i & k) == 0:
                        if s_i < s_j:
                            should_swap = True
                    else:
                        if s_i > s_j:
                            should_swap = True

                    if should_swap:
                        swap_matches(
                            shared_scores,
                            shared_r,
                            shared_q,
                            base_idx + i,
                            base_idx + ixj,
                        )
            cuda.syncwarp()
            j //= 2
        k *= 2

    # 3. GREEDY SELECTION
    if lane_id == 0:
        used_a = uint64(0)
        used_b = uint64(0)
        final_score = float32(0.0)

        for i in range(count):
            s = shared_scores[base_idx + i]
            if s <= 0:
                break

            r = shared_r[base_idx + i]
            q = shared_q[base_idx + i]

            mask_a = uint64(1) << uint64(r)
            mask_b = uint64(1) << uint64(q)

            is_free = ((used_a & mask_a) == 0) and ((used_b & mask_b) == 0)

            if is_free:
                final_score += s
                used_a |= mask_a
                used_b |= mask_b

        out_scores[global_warp_id] = final_score

    # Norm calculation (Post-Greedy)
    my_norm_a = float32(0.0)
    my_norm_b = float32(0.0)

    for i in range(lane_id, la, 32):
        val_mz = mz_a[a_idx, i]
        val_int = int_a[a_idx, i]
        if mz_power == 0 and int_power == 0.5:
            val = math.sqrt(val_int)
        else:
            val = (val_mz**mz_power) * (val_int**int_power)
        my_norm_a += val * val

    for i in range(lane_id, lb, 32):
        val_mz = mz_b[b_idx, i]
        val_int = int_b[b_idx, i]
        if mz_power == 0 and int_power == 0.5:
            val = math.sqrt(val_int)
        else:
            val = (val_mz**mz_power) * (val_int**int_power)
        my_norm_b += val * val

    if lane_id == 0:
        shared_scores[base_idx] = 0.0
        shared_scores[base_idx + 1] = 0.0
    cuda.syncwarp()

    cuda.atomic.add(shared_scores, base_idx, my_norm_a)
    cuda.atomic.add(shared_scores, base_idx + 1, my_norm_b)
    cuda.syncwarp()

    if lane_id == 0:
        norm_a = math.sqrt(shared_scores[base_idx])
        norm_b = math.sqrt(shared_scores[base_idx + 1])
        if norm_a > 0 and norm_b > 0:
            out_scores[global_warp_id] /= norm_a * norm_b


@cuda.jit
def greedy_cosine_fast_medium(
    mz_a,
    int_a,
    len_a,
    mz_b,
    int_b,
    len_b,
    out_scores,
    n_pairs,
    stride,
    tolerance,
    shift,
    mz_power,
    int_power,
    pair_a_indices,
    pair_b_indices,
):
    """
    Optimized for medium spectra (64 < N <= 256) using Block-per-Pair strategy.
    Supports arbitrary pairings via pair_a_indices/pair_b_indices.
    Grid: N_PAIRS, 1
    Block: 128 threads
    """
    pair_idx = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    if pair_idx >= n_pairs:
        return

    a_idx = pair_a_indices[pair_idx]
    b_idx = pair_b_indices[pair_idx]

    s_scores = cuda.shared.array(MAX_MATCHES_MEDIUM, float32)
    s_r = cuda.shared.array(MAX_MATCHES_MEDIUM, int32)
    s_q = cuda.shared.array(MAX_MATCHES_MEDIUM, int32)
    s_counter = cuda.shared.array(1, int32)

    if tid == 0:
        s_counter[0] = 0

    for i in range(tid, MAX_MATCHES_MEDIUM, cuda.blockDim.x):
        s_scores[i] = -1.0
        s_r[i] = -1
        s_q[i] = -1

    cuda.syncthreads()

    la = len_a[a_idx]
    lb = len_b[b_idx]

    # 1. Match Finding
    for i in range(tid, la, cuda.blockDim.x):
        val_mz_a = mz_a[a_idx, i]
        val_int_a = int_a[a_idx, i]

        if mz_power == 0 and int_power == 0.5:
            amp_a = math.sqrt(val_int_a)
        else:
            amp_a = (val_mz_a**mz_power) * (val_int_a**int_power)

        base_mz = val_mz_a
        if base_mz < 200.0:
            base_mz = 200.0
        tol_da = base_mz * tolerance * 1e-6

        for j in range(lb):
            val_mz_b = mz_b[b_idx, j]

            if shift == 0:
                mz_q_shifted = val_mz_b
            else:
                mz_q_shifted = val_mz_b + shift

            if mz_q_shifted > val_mz_a + tol_da:
                break
            if mz_q_shifted < val_mz_a - tol_da:
                continue

            val_int_b = int_b[b_idx, j]

            if mz_power == 0 and int_power == 0.5:
                amp_b = math.sqrt(val_int_b)
            else:
                amp_b = (val_mz_b**mz_power) * (val_int_b**int_power)

            score = amp_a * amp_b

            idx = cuda.atomic.add(s_counter, 0, 1)
            if idx < MAX_MATCHES_MEDIUM:
                s_scores[idx] = score
                s_r[idx] = i
                s_q[idx] = j

    cuda.syncthreads()

    count = min(s_counter[0], MAX_MATCHES_MEDIUM)

    # 2. Bitonic Sort
    k = 2
    while k <= MAX_MATCHES_MEDIUM:
        j = k // 2
        while j > 0:
            for i in range(tid, MAX_MATCHES_MEDIUM, cuda.blockDim.x):
                ixj = i ^ j
                if ixj > i:
                    s_i = s_scores[i]
                    s_j = s_scores[ixj]

                    should_swap = False
                    if (i & k) == 0:
                        if s_i < s_j:
                            should_swap = True
                    else:
                        if s_i > s_j:
                            should_swap = True

                    if should_swap:
                        swap_matches(s_scores, s_r, s_q, i, ixj)
            cuda.syncthreads()
            j //= 2
        k *= 2

    # 3. Greedy Selection
    if tid == 0:
        used_a_0 = uint64(0)
        used_a_1 = uint64(0)
        used_a_2 = uint64(0)
        used_a_3 = uint64(0)
        used_b_0 = uint64(0)
        used_b_1 = uint64(0)
        used_b_2 = uint64(0)
        used_b_3 = uint64(0)

        final_score = float32(0.0)

        for i in range(count):
            s = s_scores[i]
            if s <= 0:
                break

            r = s_r[i]
            q = s_q[i]

            r_bucket = r // 64
            r_bit = r % 64
            q_bucket = q // 64
            q_bit = q % 64

            is_used_a = False
            if r_bucket == 0:
                is_used_a = (used_a_0 >> r_bit) & 1
            elif r_bucket == 1:
                is_used_a = (used_a_1 >> r_bit) & 1
            elif r_bucket == 2:
                is_used_a = (used_a_2 >> r_bit) & 1
            elif r_bucket == 3:
                is_used_a = (used_a_3 >> r_bit) & 1

            is_used_b = False
            if q_bucket == 0:
                is_used_b = (used_b_0 >> q_bit) & 1
            elif q_bucket == 1:
                is_used_b = (used_b_1 >> q_bit) & 1
            elif q_bucket == 2:
                is_used_b = (used_b_2 >> q_bit) & 1
            elif q_bucket == 3:
                is_used_b = (used_b_3 >> q_bit) & 1

            if not is_used_a and not is_used_b:
                final_score += s

                mask_r = uint64(1) << uint64(r_bit)
                if r_bucket == 0:
                    used_a_0 |= mask_r
                elif r_bucket == 1:
                    used_a_1 |= mask_r
                elif r_bucket == 2:
                    used_a_2 |= mask_r
                elif r_bucket == 3:
                    used_a_3 |= mask_r

                mask_q = uint64(1) << uint64(q_bit)
                if q_bucket == 0:
                    used_b_0 |= mask_q
                elif q_bucket == 1:
                    used_b_1 |= mask_q
                elif q_bucket == 2:
                    used_b_2 |= mask_q
                elif q_bucket == 3:
                    used_b_3 |= mask_q

        out_scores[pair_idx] = final_score

    # Norm calc (Parallel Reduce)
    s_norm_a = cuda.shared.array(128, float32)
    s_norm_b = cuda.shared.array(128, float32)

    my_a = float32(0.0)
    for i in range(tid, la, cuda.blockDim.x):
        if mz_power == 0 and int_power == 0.5:
            val = math.sqrt(int_a[a_idx, i])
        else:
            val = (mz_a[a_idx, i] ** mz_power) * (int_a[a_idx, i] ** int_power)
        my_a += val * val
    s_norm_a[tid] = my_a

    my_b = float32(0.0)
    for i in range(tid, lb, cuda.blockDim.x):
        if mz_power == 0 and int_power == 0.5:
            val = math.sqrt(int_b[b_idx, i])
        else:
            val = (mz_b[b_idx, i] ** mz_power) * (int_b[b_idx, i] ** int_power)
        my_b += val * val
    s_norm_b[tid] = my_b

    cuda.syncthreads()

    s = cuda.blockDim.x // 2
    while s > 0:
        if tid < s:
            s_norm_a[tid] += s_norm_a[tid + s]
            s_norm_b[tid] += s_norm_b[tid + s]
        cuda.syncthreads()
        s //= 2

    if tid == 0:
        norm_a = math.sqrt(s_norm_a[0])
        norm_b = math.sqrt(s_norm_b[0])
        if norm_a > 0 and norm_b > 0:
            out_scores[pair_idx] /= norm_a * norm_b


@cuda.jit
def greedy_cosine_fast_large(
    mz_a,
    int_a,
    len_a,
    mz_b,
    int_b,
    len_b,
    out_scores,
    n_pairs,
    stride,
    tolerance,
    shift,
    mz_power,
    int_power,
    pair_a_indices,
    pair_b_indices,
):
    """
    Optimized for large spectra (256 < N <= 1024) using Block-per-Pair strategy with shared memory flags.
    Supports arbitrary pairings via pair_a_indices/pair_b_indices.
    Grid: N_PAIRS, 1
    Block: 256 threads
    """
    pair_idx = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    if pair_idx >= n_pairs:
        return

    a_idx = pair_a_indices[pair_idx]
    b_idx = pair_b_indices[pair_idx]

    s_scores = cuda.shared.array(MAX_MATCHES_LARGE, float32)
    s_r = cuda.shared.array(MAX_MATCHES_LARGE, int32)
    s_q = cuda.shared.array(MAX_MATCHES_LARGE, int32)
    s_counter = cuda.shared.array(1, int32)

    if tid == 0:
        s_counter[0] = 0

    for i in range(tid, MAX_MATCHES_LARGE, cuda.blockDim.x):
        s_scores[i] = -1.0
        s_r[i] = -1
        s_q[i] = -1

    cuda.syncthreads()

    la = len_a[a_idx]
    lb = len_b[b_idx]

    # 1. Match Finding
    for i in range(tid, la, cuda.blockDim.x):
        val_mz_a = mz_a[a_idx, i]
        val_int_a = int_a[a_idx, i]

        if mz_power == 0 and int_power == 0.5:
            amp_a = math.sqrt(val_int_a)
        else:
            amp_a = (val_mz_a**mz_power) * (val_int_a**int_power)

        base_mz = val_mz_a
        if base_mz < 200.0:
            base_mz = 200.0
        tol_da = base_mz * tolerance * 1e-6

        for j in range(lb):
            val_mz_b = mz_b[b_idx, j]

            if shift == 0:
                mz_q_shifted = val_mz_b
            else:
                mz_q_shifted = val_mz_b + shift

            if mz_q_shifted > val_mz_a + tol_da:
                break
            if mz_q_shifted < val_mz_a - tol_da:
                continue

            val_int_b = int_b[b_idx, j]

            if mz_power == 0 and int_power == 0.5:
                amp_b = math.sqrt(val_int_b)
            else:
                amp_b = (val_mz_b**mz_power) * (val_int_b**int_power)

            score = amp_a * amp_b

            idx = cuda.atomic.add(s_counter, 0, 1)
            if idx < MAX_MATCHES_LARGE:
                s_scores[idx] = score
                s_r[idx] = i
                s_q[idx] = j

    cuda.syncthreads()

    count = min(s_counter[0], MAX_MATCHES_LARGE)

    # 2. Bitonic Sort
    k = 2
    while k <= MAX_MATCHES_LARGE:
        j = k // 2
        while j > 0:
            for i in range(tid, MAX_MATCHES_LARGE, cuda.blockDim.x):
                ixj = i ^ j
                if ixj > i:
                    s_i = s_scores[i]
                    s_j = s_scores[ixj]

                    should_swap = False
                    if (i & k) == 0:
                        if s_i < s_j:
                            should_swap = True
                    else:
                        if s_i > s_j:
                            should_swap = True

                    if should_swap:
                        swap_matches(s_scores, s_r, s_q, i, ixj)
            cuda.syncthreads()
            j //= 2
        k *= 2

    # 3. Greedy Selection
    s_used_a = cuda.shared.array(1024, int32)
    s_used_b = cuda.shared.array(1024, int32)

    for i in range(tid, 1024, cuda.blockDim.x):
        s_used_a[i] = 0
        s_used_b[i] = 0

    cuda.syncthreads()

    if tid == 0:
        final_score = float32(0.0)

        for i in range(count):
            s = s_scores[i]
            if s <= 0:
                break

            r = s_r[i]
            q = s_q[i]

            if s_used_a[r] == 0 and s_used_b[q] == 0:
                final_score += s
                s_used_a[r] = 1
                s_used_b[q] = 1

        out_scores[pair_idx] = final_score

    # Norm calc (Parallel Reduce)
    s_norm_a = cuda.shared.array(LARGE_THREAD_BLOCK, float32)
    s_norm_b = cuda.shared.array(LARGE_THREAD_BLOCK, float32)

    my_a = float32(0.0)
    for i in range(tid, la, cuda.blockDim.x):
        if mz_power == 0 and int_power == 0.5:
            val = math.sqrt(int_a[a_idx, i])
        else:
            val = (mz_a[a_idx, i] ** mz_power) * (int_a[a_idx, i] ** int_power)
        my_a += val * val
    s_norm_a[tid] = my_a

    my_b = float32(0.0)
    for i in range(tid, lb, cuda.blockDim.x):
        if mz_power == 0 and int_power == 0.5:
            val = math.sqrt(int_b[b_idx, i])
        else:
            val = (mz_b[b_idx, i] ** mz_power) * (int_b[b_idx, i] ** int_power)
        my_b += val * val
    s_norm_b[tid] = my_b

    cuda.syncthreads()

    s = cuda.blockDim.x // 2
    while s > 0:
        if tid < s:
            s_norm_a[tid] += s_norm_a[tid + s]
            s_norm_b[tid] += s_norm_b[tid + s]
        cuda.syncthreads()
        s //= 2

    if tid == 0:
        norm_a = math.sqrt(s_norm_a[0])
        norm_b = math.sqrt(s_norm_b[0])
        if norm_a > 0 and norm_b > 0:
            out_scores[pair_idx] /= norm_a * norm_b


def run_greedy_cosine_fast(
    mz_a,
    int_a,
    len_a,
    mz_b,
    int_b,
    len_b,
    tolerance=0.1,
    shift=0.0,
    mz_power=0.0,
    int_power=1.0,
    pair_a_indices=None,
    pair_b_indices=None,
):
    """
    Dispatcher for fast greedy cosine kernels.
    Supports arbitrary pair lists via pair_a_indices/pair_b_indices.
    """
    stride = mz_a.shape[1]

    if pair_a_indices is None and pair_b_indices is None:
        n_pairs = mz_a.shape[0]
        pair_a_indices_host = np.arange(n_pairs, dtype=np.int32)
        pair_b_indices_host = np.arange(n_pairs, dtype=np.int32)
    else:
        assert pair_a_indices is not None and pair_b_indices is not None, (
            "Both pair_a_indices and pair_b_indices must be provided together."
        )
        pair_a_indices_host = np.asarray(pair_a_indices, dtype=np.int32)
        pair_b_indices_host = np.asarray(pair_b_indices, dtype=np.int32)
        n_pairs = pair_a_indices_host.shape[0]
        assert pair_b_indices_host.shape[0] == n_pairs, (
            "pair_a_indices and pair_b_indices must be the same length."
        )

    assert n_pairs > 0, "At least one pair must be provided."
    assert pair_a_indices_host.max() < mz_a.shape[0], (
        f"pair_a_indices reference spectrum {pair_a_indices_host.max()}, "
        f"but mz_a has {mz_a.shape[0]} spectra."
    )
    assert pair_b_indices_host.max() < mz_b.shape[0], (
        f"pair_b_indices reference spectrum {pair_b_indices_host.max()}, "
        f"but mz_b has {mz_b.shape[0]} spectra."
    )

    pair_a_indices_device = cuda.to_device(pair_a_indices_host)
    pair_b_indices_device = cuda.to_device(pair_b_indices_host)

    out_scores = cuda.device_array(n_pairs, dtype=np.float32)

    if stride <= 64:
        threads = 256
        warps_per_block = threads // WARP_SIZE
        blocks = (n_pairs + warps_per_block - 1) // warps_per_block
        greedy_cosine_fast_small[blocks, threads](
            mz_a,
            int_a,
            len_a,
            mz_b,
            int_b,
            len_b,
            out_scores,
            n_pairs,
            stride,
            tolerance,
            shift,
            mz_power,
            int_power,
            pair_a_indices_device,
            pair_b_indices_device,
        )
    elif stride <= 256:
        threads = 128
        blocks = n_pairs
        greedy_cosine_fast_medium[blocks, threads](
            mz_a,
            int_a,
            len_a,
            mz_b,
            int_b,
            len_b,
            out_scores,
            n_pairs,
            stride,
            tolerance,
            shift,
            mz_power,
            int_power,
            pair_a_indices_device,
            pair_b_indices_device,
        )
    elif stride <= 1024:
        threads = 256
        blocks = n_pairs
        greedy_cosine_fast_large[blocks, threads](
            mz_a,
            int_a,
            len_a,
            mz_b,
            int_b,
            len_b,
            out_scores,
            n_pairs,
            stride,
            tolerance,
            shift,
            mz_power,
            int_power,
            pair_a_indices_device,
            pair_b_indices_device,
        )
    else:
        raise NotImplementedError(
            "Spectra larger than 1024 peaks not yet supported in fast kernel"
        )

    return out_scores
