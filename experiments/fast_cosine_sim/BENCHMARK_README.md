# Benchmark Script: Old vs New GPU Similarity

## Critical Fix Applied: Threshold Matching

### The Problem
The old implementation (`experiments/fast_cosine_sim/`) has **automatic threshold reduction** logic:

```python
# In SimilarityConfig.__post_init__:
if self.approx_threshold < 0.0:
    self.approx_threshold = max(0.0, float(self.threshold) - 0.15)
```

This means:
- If you set `threshold=0.8` and leave `approx_threshold=-1.0` (default)
- The old implementation automatically sets `approx_threshold = 0.8 - 0.15 = 0.65`

### The Solution
To ensure **both implementations use the exact same threshold**, we:

1. **Explicitly set `approx_threshold=0.65`** in the benchmark configuration
2. **Disable auto-reduction** by passing a non-negative value
3. **Add assertion** to verify the threshold is set correctly

```python
SIMILARITY_PARAMS = {
    "approx_threshold": 0.65,  # Explicitly set (not auto-reduced)
    "exact_threshold": 0.8,     # For reference only
    ...
}

# Old implementation config
approx_cfg = SimilarityConfig(
    threshold=0.8,              # Exact stage threshold (not used in this test)
    approx_threshold=0.65,      # Explicitly set to match new impl
)

# Verify it's set correctly
assert approx_cfg.approx_threshold == 0.65
```

### Verification Test
Run this to verify threshold behavior:
```bash
cd /home/analytit_admin/dev/HRMS_utils
python experiments/fast_cosine_sim/test_threshold_config.py
```

## Running the Benchmark

### Quick Test (Recommended)
```bash
cd /home/analytit_admin/dev/HRMS_utils
python experiments/fast_cosine_sim/benchmark_old_vs_new.py
```
- Tests: fraghub_100k only
- Runs: 3 per implementation
- Time: ~5-10 minutes
- Reports: Minimum time (most reliable)

### Full Test (All Datasets)
```bash
python experiments/fast_cosine_sim/benchmark_old_vs_new.py --full-test --num-runs 5
```
- Tests: fraghub_100k, fraghub_300k, fraghub_all
- Runs: 5 per implementation per dataset
- Time: ~30-60 minutes

### Fast Iteration
```bash
python experiments/fast_cosine_sim/benchmark_old_vs_new.py --num-runs 1
```
- Single run (less reliable but faster)

## What Gets Compared

✅ **Approximate stage ONLY** (fair comparison)
- Old: Extracts `approx_compute_seconds` from log
- New: Measures entire approximate stage time

✅ **Identical parameters**:
- `upper_mass_bound=1000.0`
- `bin_size=0.0001`
- `ms2_tolerance_ppm=10.0`
- `intensity_power=0.5`
- `approx_threshold=0.65` ← **Both use this exact value**
- `target_gpu_mem_ratio=0.3`

## Expected Output

```
================================================================================
GPU-ACCELERATED COSINE SIMILARITY BENCHMARK
================================================================================
Comparing: OLD (experiments) vs NEW (packages)
Datasets: fraghub_100k
Runs per config: 3
Parameters: {...}

IMPORTANT: Both implementations use approx_threshold=0.65
           (Old impl auto-reduction disabled by explicit approx_threshold setting)
================================================================================

Dataset: fraghub_100k
  Size: 100,000 spectra
  Pairs found: 1,234,567

  OLD implementation (min):   45.123s  (avg: 45.678s)
  NEW implementation (min):   32.456s  (avg: 33.012s)
  Speedup: 1.39x FASTER
```

## Files Created

1. `benchmark_old_vs_new.py` - Main benchmark script
2. `test_threshold_config.py` - Threshold verification test
3. `benchmark_results.csv` - Detailed results (auto-generated when run)
