# Next Steps: GPU Testing & Optimization

**Commit**: 9e2e390  
**Date**: 2026-01-22  
**Status**: Ready for GPU deployment

---

## Immediate Next Steps (Ready to Execute)

### 1. Deploy to GPU Machine

```bash
# Copy the complete package
cp -r /home/analytit_admin/dev/HRMS_utils/packages/fast_cosine_sim /path/to/gpu-machine

# OR use git
cd /path/on/gpu-machine
git clone <repo-url>
cd HRMS_utils/packages/fast_cosine_sim
git checkout fast_cosine_sim_project
```

### 2. Run Small Test (Verify Setup)

```bash
# Quick sanity check (~10-30 seconds)
python scripts/optimize_gpu_kernel.py \
    --n-spectra 100 \
    --n-peaks-per-spectrum 50 \
    --batch-sizes 10 20 30

# Expected output:
# [1/3] Testing batch_size=10 spectra...
#     Time breakdown: GPU=123.4ms, CPU OH=45.6ms
#     Kernel breakdown (all 10 batches):
#       - Expand:        102.34ms (82.9%)
#       - SpMM:           12.56ms (10.2%)
#       - Normalization:   5.62ms ( 4.6%)
#       - Other ops:       2.48ms ( 2.0%)
```

### 3. Run Medium Test (Collect Data)

```bash
# More realistic workload (~2-5 minutes)
python scripts/optimize_gpu_kernel.py \
    --n-spectra 500 \
    --n-peaks-per-spectrum 100 \
    --batch-sizes 50 100 200 400 \
    --match-rate 0.001

# Will generate results for 4 batch size configurations
# Captures: GPU time, CPU overhead, kernel operation breakdown
```

### 4. Collect Results

```bash
# Results are printed to console. Capture to file:
python scripts/optimize_gpu_kernel.py \
    --n-spectra 500 \
    --batch-sizes 50 100 200 \
    2>&1 | tee gpu_profile_results.txt

# Save the output for analysis
```

---

## What to Look For in Results

### ✅ Success Indicators

1. **Kernel profiling completes** without "Kernel profiling failed" warnings
2. **Percentages sum to 100%**:
   ```
   Expand (82.9%) + SpMM (10.2%) + Norm (4.6%) + Other (2.0%) = 99.7% ✓
   ```
3. **Expected bottleneck**: Expand operation dominates (70-90%)
4. **CPU overhead is small**: 100-500ms for whole benchmark
5. **Throughput increases** with batch size (generally)

### ⚠️ Anomalies to Investigate

| Observation | Investigation |
|-------------|---|
| SpMM is >30% of time | Unexpected; normally 5-15%. Check matrix shapes. |
| Normalization is >10% | Unexpected; normally 2-5%. Check L2 implementation. |
| "Negative CPU overhead" | Normal if small (<10ms). Timing precision issue. |
| Expand is <50% of time | Unexpected if tolerance is tight. Check config. |
| Profiling takes >2s | Overhead is visible. Consider profiling fewer batches. |

---

## Data Analysis (After Results Collected)

### Extract Key Metrics

From the output, extract for each batch size:
- Total time (s)
- Throughput (comparisons/sec)
- GPU measured time (ms)
- CPU overhead (ms)
- Expand % / SpMM % / Norm %

### Create Comparison Table

```
Batch Size | Time(s) | Throughput | GPU(ms) | CPU_OH(ms) | Expand% | SpMM% | Norm%
-----------|---------|-----------|--------|-----------|---------|-------|-------
50         | 5.21    | 12.0M     | 4562   | 648       | 90.4    | 5.6   | 3.2
100        | 4.41    | 14.1M     | 3896   | 512       | 83.6    | 13.1  | 2.6
200        | 3.90    | 16.0M     | 3456   | 446       | 82.4    | 14.8  | 2.5
400        | 3.45    | 18.1M     | 3012   | 438       | 80.1    | 17.3  | 2.1
```

### Look for Patterns

1. **Throughput scaling**: Does it increase with batch size?
2. **Expand dominance**: Is it consistently 70-90%?
3. **SpMM growth**: Does SpMM% increase with batch size?
   - If yes: SpMM is becoming bottleneck at large batches
   - If no: Expand remains dominant
4. **GPU overhead**: Does CPU overhead decrease with larger batches?

---

## Optimization Decisions Based on Results

### If Expand is >85% (Expected)

**Decision**: Expand operation is the bottleneck.

**Next steps**:
1. Profile expand operation internals (bin lookup, bounds checking, etc.)
2. Consider optimizations:
   - Vectorized tolerance window calculation
   - Precomputed bin boundaries
   - Coalesced memory access patterns
3. Measure improvement from each optimization

### If SpMM is >20% (Unexpected)

**Decision**: Sparse matrix multiplication is unexpectedly expensive.

**Next steps**:
1. Check CSR matrix shape (should be n_spectra × n_bins)
2. Verify sparsity is high (>90% zeros)
3. Consider optimizations:
   - Batched SpMM if available
   - Different sparse library (cuSPARSE?)
   - GPU kernel fusion with expand

### If Normalization is >10% (Unexpected)

**Decision**: L2 normalization is slower than expected.

**Next steps**:
1. Check if using fused kernel vs separate
2. Profile norm calculation separately
3. Consider optimizations:
   - Vectorized norm calculation
   - Precision reduction (float32)
   - Batched normalization

---

## Next Optimization Phase

### Phase 1: Profiling (Current)
- ✅ Collect per-operation timing data
- ✅ Identify bottleneck operations
- ✅ Measure scaling across batch sizes

### Phase 2: Analysis (After Results)
- Aggregate results across batch sizes
- Create optimization priority list
- Estimate potential speedups

### Phase 3: Implementation (Data-Driven)
- Optimize highest-impact operation first
- Measure improvement (re-profile after each change)
- Iterate until diminishing returns

### Phase 4: Verification
- Run full benchmarks on production data
- Compare before/after performance
- Document findings and recommendations

---

## Useful Commands

### Run with verbose output
```bash
python scripts/optimize_gpu_kernel.py \
    --n-spectra 500 \
    --batch-sizes 50 100 200 \
    --verbose
```

### Run with specific match rate
```bash
python scripts/optimize_gpu_kernel.py \
    --n-spectra 1000 \
    --batch-sizes 100 \
    --match-rate 0.01  # 1% of spectra match instead of 0.1%
```

### Save results to file
```bash
python scripts/optimize_gpu_kernel.py \
    --n-spectra 500 \
    --batch-sizes 50 100 200 \
    2>&1 | tee results_$(date +%Y%m%d_%H%M%S).txt
```

---

## Troubleshooting

### "ImportError: No module named 'cupy'"
- Confirm GPU machine has CuPy installed
- Check CUDA compatibility: `python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"`

### "Kernel profiling failed: ..."
- Check GPU memory availability: `nvidia-smi`
- Reduce dataset size: `--n-spectra 100`
- Check GPU function API in `src/fast_cosine_sim/gpu_approximate_similarity.py`

### "Negative CPU overhead"
- Normal if small (<10ms); timing precision issue
- Larger negative values indicate GPU time > wall-time; check synchronization

### "OutOfMemory: CUDA out of memory"
- Reduce batch size: `--batch-sizes 10 20`
- Reduce dataset: `--n-spectra 100`
- Clear GPU: `python -c "import cupy as cp; cp.get_default_memory_pool().free_all_blocks()"`

---

## Files to Reference

- **Main script**: `scripts/optimize_gpu_kernel.py` (269 lines of new profiling code)
- **Profiling function**: Lines 1195-1359
- **Integration point**: Lines 1454-1495 in `benchmark_batch_sizes()`
- **Result dataclass**: Lines 106-158 (`AggregatedKernelTimings`)
- **Documentation**: `IMPLEMENTATION_COMPLETE.md`

---

## Key Metrics to Track

For each test run, record:

| Metric | Why | Target |
|--------|-----|--------|
| Throughput (pairs/sec) | Main performance metric | >10M |
| GPU Utilization (%) | Efficiency | >80% |
| Expand operation % | Bottleneck indicator | 70-90% |
| SpMM operation % | Secondary bottleneck | 5-15% |
| CPU overhead (ms) | Dispatch cost | <500ms |
| Memory usage (GB) | Constraint | <8GB for 5000 spectra |

---

## Expected Timeline

| Task | Duration | GPU Required |
|------|----------|---|
| Small test (100 spectra) | 10-30s | Yes |
| Medium test (500 spectra) | 2-5min | Yes |
| Full benchmark (5000 spectra) | 10-30min | Yes |
| Analysis & visualization | 30min | No |
| Optimization implementation | Depends on findings | No |
| Re-profiling after optimization | 2-5min | Yes |

---

## Contact / Questions

If issues arise during GPU testing:
1. Check IMPLEMENTATION_COMPLETE.md for design details
2. Review profiling function source (lines 1195-1359)
3. Check GPU memory with `nvidia-smi`
4. Verify CSR matrix shapes match expected (n_spectra × n_bins)

---

**Ready to proceed with GPU testing!** 🚀
