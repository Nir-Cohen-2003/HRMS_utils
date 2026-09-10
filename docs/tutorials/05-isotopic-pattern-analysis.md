# Isotopic-pattern analysis

`deduce_isotopic_pattern()` compares the MS1 isotope lists with expected isotope
mass shifts and returns an `Array(Int32, 24)`: twelve lower bounds followed by twelve
upper bounds. Formula elements always use the order:

```text
H, C, N, O, F, Na, P, S, Cl, K, Br, I
```

The implementation uses isotope evidence for C, S, Cl, and Br. Bounds can be supplied
to bounded mass decomposition, which reduces candidate formulas before fragment
annotation. This is constraint generation, not compound identification.

```python
import polars as pl
from hrms_utils import hrms_core  # registers the Polars namespaces

features = pl.DataFrame({
    "precursor_mz": [180.0634],
    "ms1_mz": [[180.0634, 181.0668]],
    "ms1_intensity": [[1_000_000.0, 10_000.0]],
})
with_bounds = features.with_columns(
    pl.col("precursor_mz").mass_decomposition.deduce_isotopic_pattern(
        ms1_mzs=pl.col("ms1_mz"),
        ms1_intensities=pl.col("ms1_intensity"),
        ms1_mass_tolerance_ppm=3.0,
        isotopic_mass_tolerance_ppm=2.0,
        minimum_intensity=0.0,
    ).alias("isotope_bounds")
)
```

For instrument-specific tolerance calibration, use the
[calibration guide](../how-to/calibrate-isotopic-parameters.md). Calibration is not
loaded automatically; the resulting runtime model must be passed explicitly to
`deduce_isotopic_pattern()` or `annotate_chromatogram_with_formulas()`.
