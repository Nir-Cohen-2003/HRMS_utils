# Guide for Engineering Porting: MS-DIAL Peak Detection Algorithm

This document provides engineers with a clear step-by-step outline for porting the MS-DIAL chromatogram peak picking algorithm from C# (see `PeakPick.cs` and helpers) into another language (such as Rust), so that the same detected peaks are filtered with equivalent results.

Each section explains the function's responsibility, the core logic, and *how* to reproduce equivalent logic—even if the implementation details differ. Pay special attention to matching thresholds, smoothing, and filtering, since these determine results.

***

## 1. **Input Data Preparation**
- **Purpose:** Accept intensity vs. retention time profile (a chromatogram). Input is usually a pair of arrays: times (X), intensities (Y).
- **How to Port:**
  - Make sure to keep both as floats/doubles—integer rounding may ruin precision.

***

## 2. **Smoothing/Noise Reduction**
- **Purpose:** Reduce high-frequency noise to clarify the true peak structure.
- **C# Logic:** Applies a linearly weighted moving average (see `Smoothing.LinearWeightedMovingAverage`). The window width and kind of smoothing (e.g., Savitzky-Golay) may be chosen by parameters.
- **How to Port:**
  - Re-implement moving average smoothing with weights matching the original (typically linear; e.g. central points get higher weights).
  - Given window size N, smoothed Y[i] = weighted sum over window around i.
  - Make smoothing function modular so the same weights, window, and method can be chosen.

***

## 3. **Differential Filtering (Edge & Peak Finding)**
- **Purpose:** Find peak starts, tops, and ends using slopes (first and second derivatives).
- **C# Logic:** Calculates first and second order numerical derivatives using coefficients (`generateDifferencialCoefficients`).
  - Uses sliding window finite difference, e.g.
    $$ f'(x_i) \approx \sum\limits_{k=-m}^m c_k y_{i+k} $$
  - Also finds sign changes, zero-crossing for peak tops.
- **How to Port:**
  - Compute numerical derivatives using the same window-size and coefficients (or derive the equivalent for your target language).
  - Implement helper to detect sign changes (slope up to down) and regions where second derivative indicates peak top (low point).

***

## 4. **Adaptive Noise Estimation**
- **Purpose:** Separates true peaks from noise using dynamic noise baseline calculation.
- **C# Logic:** For amplitude and slope, computes median in lowest 5% of data, then takes maximum in sliding window (see `calculateSlopeNoises`).
  - Thresholds for peak detection are calculated as a fold (e.g., 3x, 5x) above this baseline.
- **How to Port:**
  - Implement function to:
    1. Extract lowest N% of amplitude and derivative values.
    2. Calculate median.
    3. For every window, find maximum within that region.
    4. Use these locally-adaptive values as the background threshold for peak detection.
  - Use robust statistics to avoid outliers.

***

## 5. **Peak Boundary and Top Detection**
- **Purpose:** Clearly define where a peak starts, ends, and reaches its apex.
- **C# Logic:**
  - Searches outward from local maxima to find edges where Y (intensity) and first derivative (slope) fall below background.
  - Uses derivative sign and intensity comparisons (`searchRealLeftEdge`, `searchRightEdgeCandidate`, `searchRealRightEdge`).
- **How to Port:**
  - For each local maximum:
    - Step backward to find where amplitude and slope fall below threshold: left boundary.
    - Step forward for right boundary.
    - Top is where first derivative changes sign (up to down) and second derivative is near its minimum.
  - Store indices marking each detected peak region.

***

## 6. **Peak Qualification & Filtering**
- **Purpose:** Remove false or low-value peaks; assign quantitative/quality metrics.
- **C# Logic:**
  - Metrics: S/N (signal to noise), peak area, symmetry, sharpness, width, purity.
  - Filters out peaks if below minimum amplitude, area, width, S/N, or fail fold over noise.
- **How to Port:**
  - For each bracketed peak, calculate:
    - S/N: Ratio of peak max vs. local baseline or noise.
    - Width: Time (or points) between left/right edges.
    - Symmetry, sharpness: Check definitions in code.
  - Discard peaks that do not meet the same numeric thresholds.
  - All filters must match numeric/comparison rules to get "identical" accepted/rejected peaks.

***

## 7. **Output Formatting**
- **Purpose:** Allow further analysis/alignment in pipeline.
- **C# Logic:** Output list of peaks with index/range, apex, area, height, and quality metrics.
- **How to Port:**
  - Return vector/list/array or write to file (e.g., CSV/TSV/JSON) with all above properties per peak.

***

## **Porting Tips for Rust (or similar languages)**
- Use strict types: `f64` or `f32` for intensities.
- Prefer iterators and slices for performance, avoid unnecessary copies.
- Encapsulate smoothing and derivative logic in pure functions.
- Use unit tests: For porting, check your Rust outputs match a range of test chromatograms and verify against the C# implementation.
- If available, cross-check with exported results from the original MS-DIAL for identical filtering on the same data: this is your "ground truth".

***

## **Summary Table: C# Components to Port**
| Feature         | Main C# File         | Rust Module/Purpose          |
|-----------------|---------------------|-----------------------------|
| Smoothing       | Smoothing.cs        | moving_avg.rs (weighted)    |
| Derivatives     | PeakPick.cs         | diff_filters.rs             |
| Noise baseline  | PeakPick.cs         | noise_estimate.rs           |
| Peak picking    | PeakPick.cs         | peak_pick.rs                |
| QC & filtering  | PeakPick.cs         | qc_filter.rs                |
| Output struct   | PeakPickResult.cs   | peak.rs (struct)            |

Be sure to reproduce *the numeric logic for thresholds and filtering* as precisely as possible—insofar as floating-point roundings differ, results may be very slightly different, but peaks should be filtered identically.
