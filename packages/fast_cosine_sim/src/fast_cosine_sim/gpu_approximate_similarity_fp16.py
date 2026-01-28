"""
GPU-accelerated approximate similarity using FP16 (Half Precision).

This module provides a custom CUDA implementation for the approximate similarity pipeline,
specifically designed to operate entirely in FP16 to maximize performance and reduce memory usage.

It handles:
1. Normalization (L2) in FP16
2. Adaptive Expansion (MS2 Tolerance) in FP16
3. Similarity Computation (Sparse x Sparse -> Dense) in FP16
4. Thresholding and Extraction
"""

import cupy as cp
import numpy as np

# C++ Source for Kernels
KERNELS_SRC = r'''
#include <cuda_fp16.h>

extern "C" {

// -----------------------------------------------------------------------------
// 1. Normalization Kernel (In-place)
// -----------------------------------------------------------------------------
// Calculates L2 norm of each row and divides data.
// Handles zero-division safely.

__global__ void normalize_csr_rows_fp16(
    int n_rows,
    const int* __restrict__ indptr,
    half* __restrict__ data
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    int start = indptr[row];
    int end = indptr[row + 1];

    if (start >= end) return;

    // Calculate sum of squares
    float sum_sq = 0.0f;
    for (int i = start; i < end; ++i) {
        float val = __half2float(data[i]);
        sum_sq += val * val;
    }

    float norm = sqrtf(sum_sq);
    half h_norm = (norm > 1e-8f) ? __float2half(norm) : __float2half(1.0f);

    // Normalize
    for (int i = start; i < end; ++i) {
        // data[i] /= norm
        data[i] = __hdiv(data[i], h_norm);
    }
}

// -----------------------------------------------------------------------------
// 2. Expansion Kernels (Adaptive)
// -----------------------------------------------------------------------------
// Pass 1: Count output size
__global__ void expand_count_fp16(
    int n_rows,
    const int* __restrict__ indptr,
    const int* __restrict__ indices,
    float bin_size,
    float tolerance_ppm,
    float mass_cutoff,
    int* __restrict__ row_counts // Output: number of items per row
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    int start = indptr[row];
    int end = indptr[row + 1];
    
    int count = 0;
    for (int i = start; i < end; ++i) {
        int col_idx = indices[i];
        float mz = (float)col_idx * bin_size;
        float eff_mz = fmaxf(mz, mass_cutoff);
        float tol_da = eff_mz * tolerance_ppm * 1.0e-6f;
        int window = (int)ceilf(tol_da / bin_size);
        count += (2 * window + 1);
    }
    row_counts[row] = count;
}

// Pass 2: Fill expanded CSR (with duplicates)
// Note: We don't sum duplicates here. The dot product kernel handles them.
__global__ void expand_fill_fp16(
    int n_rows,
    const int* __restrict__ src_indptr,
    const int* __restrict__ src_indices,
    const half* __restrict__ src_data,
    const int* __restrict__ dst_indptr,
    int* __restrict__ dst_indices,
    half* __restrict__ dst_data,
    float bin_size,
    float tolerance_ppm,
    float mass_cutoff,
    int n_bins
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    int src_start = src_indptr[row];
    int src_end = src_indptr[row + 1];
    int dst_offset = dst_indptr[row];

    for (int i = src_start; i < src_end; ++i) {
        int col = src_indices[i];
        half val = src_data[i];
        
        float mz = (float)col * bin_size;
        float eff_mz = fmaxf(mz, mass_cutoff);
        float tol_da = eff_mz * tolerance_ppm * 1.0e-6f;
        int window = (int)ceilf(tol_da / bin_size);
        
        int start_bin = col - window;
        int end_bin = col + window; // inclusive

        for (int b = start_bin; b <= end_bin; ++b) {
            if (b >= 0 && b < n_bins) {
                dst_indices[dst_offset] = b;
                dst_data[dst_offset] = val;
                dst_offset++;
            }
        }
    }
}


// -----------------------------------------------------------------------------
// 3. Similarity Kernel (Sparse x Sparse -> Dense)
// -----------------------------------------------------------------------------
// Computes D[i, j] = dot(L[i], R[j])
// L is CSR (possibly expanded with duplicates, sorted-ish).
// R is CSR (standard, sorted indices).
// Output is Dense (M x N).
//
// Optimized for "Merge Path" style dot product on sparse vectors.
// Threads mapped to (i, j) would be inefficient (random access).
// Better: Thread per L-row (i). Loop over R-rows (j).
// Or: Thread block per L-row? 
// Given typical sparsity (~100-1000 peaks), a single thread can handle one dot product fast.
//
// Grid: (M, N_blocks_for_R) ?? 
// Batch sizes are ~1000x1000. 1M threads is fine.
// Grid: (dim3( (N+31)/32, M )) -> Thread(x, y) computes L[y] . R[x]
// This assumes indices in L[y] and R[x] are SORTED.
// Expanded L has duplicates but is locally sorted (chunks). Merging is complex.
//
// Brute force scan for L[y] against R[x]?
// Or binary search R[x] for each element of L[y]?
// Since R[x] is usually much sparser than L[y] (if L is expanded), or similar?
// If L is expanded, it's dense-ish locally.
//
// Algorithm:
// For each thread (representing pair i, j):
//   Initialize sum = 0
//   Iterate k_L in L[i]:
//     Binary search for col(k_L) in R[j].
//     If found, sum += ...
//
// Since R[j] is sparse (~500-2000 items), binary search is fast (log 1000 ~ 10 ops).
// L[i] is expanded (~10000 items?). 10k * 10 = 100k ops per pair.
// 1000x1000 pairs = 100 G ops. Doable on GPU in ms.
//
// Precondition: R indices MUST be sorted (Standard CSR).
// L indices need NOT be sorted/merged (Expanded CSR).

__device__ int binary_search(const int* indices, int size, int target) {
    int l = 0;
    int r = size - 1;
    while (l <= r) {
        int m = l + (r - l) / 2;
        int val = indices[m];
        if (val == target) return m;
        if (val < target) l = m + 1;
        else r = m - 1;
    }
    return -1;
}

__global__ void similarity_dense_fp16(
    int n_left,
    int n_right,
    const int* __restrict__ L_indptr,
    const int* __restrict__ L_indices,
    const half* __restrict__ L_data,
    const int* __restrict__ R_indptr,
    const int* __restrict__ R_indices,
    const half* __restrict__ R_data,
    half* __restrict__ result, // (n_left x n_right)
    float threshold
) {
    // Thread mapping:
    // x: right index (j)
    // y: left index (i)
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= n_left || j >= n_right) return;

    // Pointers to row i of L
    int l_start = L_indptr[i];
    int l_end = L_indptr[i + 1];

    // Pointers to row j of R
    int r_start = R_indptr[j];
    int r_end = R_indptr[j + 1];
    int r_size = r_end - r_start;
    const int* r_idx_ptr = &R_indices[r_start];

    float score = 0.0f;

    // For each nonzero in L[i]
    for (int k = l_start; k < l_end; ++k) {
        int col = L_indices[k];
        
        // Find col in R[j]
        // Since L is often expanded (sequential clusters), we could optimize the search hint.
        // But binary search is robust.
        int match_idx = binary_search(r_idx_ptr, r_size, col);
        
        if (match_idx != -1) {
            float val_l = __half2float(L_data[k]);
            float val_r = __half2float(R_data[r_start + match_idx]);
            score += val_l * val_r;
        }
    }

    // Write if > threshold
    if (score >= threshold) {
        result[i * n_right + j] = __float2half(score);
    } else {
        result[i * n_right + j] = __float2half(0.0f);
    }
}

}
'''

class GPUApproximateSimilarityFP16:
    def __init__(self):
        # Compile kernels
        self.module = cp.RawModule(code=KERNELS_SRC, options=('-std=c++11',))
        self.norm_kernel = self.module.get_function('normalize_csr_rows_fp16')
        self.expand_count = self.module.get_function('expand_count_fp16')
        self.expand_fill = self.module.get_function('expand_fill_fp16')
        self.sim_kernel = self.module.get_function('similarity_dense_fp16')

    def normalize(self, csr_mat):
        """In-place L2 normalization of CSR matrix (fp16)."""
        n_rows = csr_mat.shape[0]
        block = 128
        grid = (n_rows + block - 1) // block
        self.norm_kernel((grid,), (block,), (
            n_rows,
            csr_mat.indptr,
            csr_mat.data
        ))

    def expand(self, csr_mat, bin_size, tol_ppm, mass_cutoff, nbins):
        """Expand CSR matrix with adaptive window."""
        n_rows = csr_mat.shape[0]
        
        # 1. Count output size
        row_counts = cp.zeros(n_rows, dtype=cp.int32)
        block = 128
        grid = (n_rows + block - 1) // block
        
        self.expand_count((grid,), (block,), (
            n_rows,
            csr_mat.indptr,
            csr_mat.indices,
            cp.float32(bin_size),
            cp.float32(tol_ppm),
            cp.float32(mass_cutoff),
            row_counts
        ))
        
        # 2. Allocate output
        # Calculate new indptr using scan
        new_indptr = cp.zeros(n_rows + 1, dtype=cp.int32)
        cp.cumsum(row_counts, out=new_indptr[1:])
        total_nnz = int(new_indptr[-1])
        
        new_indices = cp.empty(total_nnz, dtype=cp.int32)
        new_data = cp.empty(total_nnz, dtype=cp.float16)
        
        # 3. Fill
        self.expand_fill((grid,), (block,), (
            n_rows,
            csr_mat.indptr,
            csr_mat.indices,
            csr_mat.data,
            new_indptr,
            new_indices,
            new_data,
            cp.float32(bin_size),
            cp.float32(tol_ppm),
            cp.float32(mass_cutoff),
            cp.int32(nbins)
        ))
        
        # Create new CSR (note: may contain duplicates, which is fine for our sim kernel)
        return cp.sparse.csr_matrix(
            (new_data, new_indices, new_indptr),
            shape=csr_mat.shape,
            dtype=cp.float16
        )

    def compute_similarity(self, L_fp16, R_fp16, threshold):
        """Compute L @ R.T -> Dense -> COO."""
        M = L_fp16.shape[0]
        N = R_fp16.shape[0]
        
        if M == 0 or N == 0:
            return cp.array([], dtype=cp.int32), cp.array([], dtype=cp.int32), cp.array([], dtype=cp.float16)
            
        result_dense = cp.zeros((M, N), dtype=cp.float16)
        
        # Grid dimensions
        block_x = 32
        block_y = 16
        grid_x = (N + block_x - 1) // block_x
        grid_y = (M + block_y - 1) // block_y
        
        self.sim_kernel((grid_x, grid_y), (block_x, block_y), (
            M, N,
            L_fp16.indptr, L_fp16.indices, L_fp16.data,
            R_fp16.indptr, R_fp16.indices, R_fp16.data,
            result_dense,
            cp.float32(threshold)
        ))
        
        # Extract > 0 (threshold logic applied in kernel, setting < thresh to 0)
        # Using cupy nonzero is fast for dense
        rows, cols = result_dense.nonzero()
        data = result_dense[rows, cols]
        
        return rows, cols, data

def run_pipeline_fp16(left_data, left_indices, left_indptr, 
                      right_data, right_indices, right_indptr,
                      config):
    """
    Main entry point for FP16 pipeline.
    
    Args:
        left_data, left_indices, left_indptr: CSR components (host or device)
        right_data, right_indices, right_indptr: CSR components (host or device)
        config: GPUApproximateConfig object
        
    Returns:
        (rows, cols, data) as numpy arrays (CPU)
    """
    
    pipeline = GPUApproximateSimilarityFP16()
    
    # 1. Create FP16 CSRs on GPU
    # Assuming inputs are already on GPU or need transfer. 
    # To be safe, we cast to fp16.
    L_csr = cp.sparse.csr_matrix(
        (cp.asarray(left_data).astype(cp.float16), 
         cp.asarray(left_indices).astype(cp.int32), 
         cp.asarray(left_indptr).astype(cp.int32))
    )
    
    R_csr = cp.sparse.csr_matrix(
        (cp.asarray(right_data).astype(cp.float16), 
         cp.asarray(right_indices).astype(cp.int32), 
         cp.asarray(right_indptr).astype(cp.int32))
    )
    
    # 2. Normalize (In-place)
    pipeline.normalize(L_csr)
    pipeline.normalize(R_csr)
    
    # 3. Expand Left (Adaptive)
    if config.ms2_tolerance_ppm > 0:
        L_csr = pipeline.expand(
            L_csr, config.bin_size, config.ms2_tolerance_ppm, 
            config.mass_tolerance_cutoff_mz, config.nbins
        )
        
    # 4. Compute Similarity
    # Note: R is used as-is (rows of R are columns of R.T). 
    # Logic is L @ R.T, computed as dot(L[i], R[j]).
    l_rows, r_cols, sims = pipeline.compute_similarity(
        L_csr, R_csr, config.approx_threshold
    )
    
    # 5. Return to CPU
    return cp.asnumpy(l_rows), cp.asnumpy(r_cols), cp.asnumpy(sims)
    
if __name__ == "__main__":
    