# Data structures

HRMS Utils uses Polars DataFrames, including nested `List` and fixed-size `Array`
columns. These schemas describe the current MS-DIAL parser and annotation output.

## Parsed MS-DIAL chromatogram

`get_chromatogram()` consumes a tab-delimited table with the required MS-DIAL headers
and returns:

| Column | Polars type |
|---|---|
| `Peak ID` | `Int64` |
| `RT (min)`, `Precursor_mz_MSDIAL`, `Height` | `Float64` |
| `Precursor_type_MSDIAL` | `String` |
| `Isotope` | `Int32` |
| `msms_m/z`, `msms_intensity` | `List(Float64)` |
| `isobars` | `List(Int64)` |
| `msms_m/z_cleaned`, `msms_intensity_cleaned` | `List(Float64)` |
| `energy_is_too_low`, `energy_is_too_high` | `Boolean` |
| `ms1_isotopes_m/z`, `ms1_isotopes_intensity` | `List(Float64)` |

MS/MS intensities are normalized to the maximum intensity while parsing. MS1
intensities remain absolute. The two cleaned MS/MS columns are parser output; formula
annotation currently consumes the raw `msms_m/z` and `msms_intensity` columns.

## Formula and isotope representation

All formula arrays use this fixed element order and width 12:

```text
H, C, N, O, F, Na, P, S, Cl, K, Br, I
```

`deduce_isotopic_pattern()` returns `Array(Int32, 24)` with twelve minimum counts
followed by twelve maximum counts. The Rust plugin uses isotope evidence to constrain
C, S, Cl, and Br.

## Annotation output

`annotate_chromatogram_with_formulas()` retains the parsed columns and adds:

| Column | Polars type |
|---|---|
| `min_bounds`, `max_bounds`, `precursor_formula` | `Array(Int32, 12)` |
| `precursor_formula_str` | `String` |
| `precursor_errors_ppm`, `explained_intensity` | `Float64` |
| `cleaned_msms_mz`, `cleaned_msms_intensity` | `List(Float64)` |
| `cleaned_spectrum_formulas` | `List(Array(Int32, 12))` |
| `cleaned_spectrum_formulas_str` | `List(String)` |
| `cleaned_fragment_errors_ppm` | `List(Float64)` |

The pipeline deduces bounds, performs bounded decomposition on the precursor m/z,
explodes candidate formulas, and annotates fragments. It decomposes
`Precursor_mz_MSDIAL` directly without adduct/charge neutralization; these candidates
are not automatically neutral molecular formulas or compound identifications.

## Recommended filtering and storage

Use Polars expressions to select monoisotopic features with usable spectra:

```python
import polars as pl

usable = annotated.filter(
    pl.col("Isotope") == 0,
    pl.col("ms1_isotopes_m/z").is_not_null(),
    pl.col("ms1_isotopes_m/z").list.len() > 0,
    pl.col("msms_m/z").is_not_null(),
    pl.col("msms_m/z").list.len() > 0,
)
```

Use Parquet to preserve nested columns:

```python
annotated.write_parquet("annotated.parquet")
```

For CSV/Excel, select scalar columns (for example `Peak ID`, `Height`,
`precursor_formula_str`, and `precursor_errors_ppm`) rather than serializing nested
lists and arrays directly.

## Library spectra

Processed spectral-library tables commonly include `name`, `precursor_mz`,
`cleaned_normalized_mz`, `cleaned_normalized_intensity`, quality scores, and `inchikey`.
Their exact schema depends on the library-processing path; see the
[formats API](api/formats.md).
