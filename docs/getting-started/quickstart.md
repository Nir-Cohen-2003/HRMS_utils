# Quickstart

This guide introduces the core concepts and workflows of HRMS Utils in 5 minutes.

## Core Concepts

HRMS Utils is built around three main ideas:

1. **Polars DataFrames** - All data is stored in Polars DataFrames for fast, memory-efficient processing
2. **Expression Plugins** - Mass spec operations are Polars expression methods (e.g., `.mass_decomposition.decompose_mass()`)
3. **Type Safety** - Full type hints and `.pyi` stubs for excellent IDE support

## Basic Workflow

### 1. Import the Library

```python
import polars as pl
from hrms_utils import hrms_core, formats
```

### 2. Load Data

HRMS Utils supports multiple formats:

```python
# MSDIAL chromatogram
from hrms_utils.formats.msdial import get_chromatogram
chromatogram = get_chromatogram("sample.txt")

# MSP/MSPEC library
from hrms_utils.formats.nist_mspec import read_MSPEC_file
library = read_MSPEC_file("library.msp")

# mzML files
mzml_data = hrms_core.read_mzml(["file1.mzML", "file2.mzML"])
```

### 3. Explore Data

All functions return Polars DataFrames:

```python
# View structure
print(chromatogram.schema)

# Inspect data
print(chromatogram.select([
    "Peak ID", "RT (min)", "Precursor_mz_MSDIAL", "Height"
]).head())

# Filter and sort
high_intensity = chromatogram.filter(
    pl.col("Height") > 1e6
).sort("Height", descending=True)
```

## Common Operations

### Mass Decomposition

Decompose a mass into candidate elemental formulas:

```python
# Create DataFrame with masses
df = pl.DataFrame({"mass": [156.0423, 180.0634, 342.1162]})

# Decompose
result = df.with_columns(
    formulas=pl.col("mass").mass_decomposition.decompose_mass(
        tolerance_ppm=5.0,
        min_dbe=0.0,
        max_dbe=40.0
    )
)

# Extract formula strings
result = result.with_columns(
    formula_str=pl.col("formulas").struct.field("formulas_str")
)
print(result)
```

### Spectral Similarity

Compare two spectra:

```python
# Create struct with both spectra
comparison = pl.DataFrame({
    "spectra": [{
        "mz1": [100.0, 200.0, 300.0],
        "intensities1": [0.5, 0.8, 1.0],
        "mz2": [100.0, 200.0, 300.0],
        "intensities2": [0.5, 0.8, 1.0],
        "precursor_mz1": 400.0,
        "precursor_mz2": 400.0,
    }]
})

# Compute similarity
similarity = comparison.with_columns(
    score=pl.col("spectra").spectral_similarity.dotprod_similarity(
        ms2_tolerance_in_ppm=10.0
    )
)
print(similarity["score"])
```

### Isotopic Pattern Analysis

Deduce elemental bounds from MS1 isotopes:

```python
# Assuming you have MS1 isotope data
df = pl.DataFrame({
    "precursor_mz": [342.1162],
    "ms1_mz": [[342.1162, 343.1195, 344.1229]],
    "ms1_intensity": [[100000.0, 15000.0, 2000.0]]
})

# Deduce bounds
result = df.with_columns(
    bounds=pl.col("precursor_mz").mass_decomposition.deduce_isotopic_pattern(
        ms1_mzs=pl.col("ms1_mz"),
        ms1_intensities=pl.col("ms1_intensity"),
        ms1_mass_tolerance_ppm=5.0,
        isotopic_mass_tolerance_ppm=3.0
    )
)
```

## Data Structures

HRMS Utils uses nested Polars types for spectra:

```python
# Spectrum stored as Lists
spectrum_df = pl.DataFrame({
    "mz": [[100.0, 200.0, 300.0]],
    "intensity": [[0.5, 0.8, 1.0]]
})

# Formula stored as Array(Int32, 15)
# Represents element counts: [H, C, N, O, P, S, F, Cl, Br, I, Si, B, Se, As, Na]
formula_df = pl.DataFrame({
    "formula_array": [[2, 6, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]  # C6H2O
})
```

## Polars Expression Chaining

HRMS Utils integrates seamlessly with Polars expressions:

```python
result = (
    chromatogram
    .filter(pl.col("Height") > 1e6)
    .with_columns(
        # Annotate with formulas
        formulas=pl.col("Precursor_mz_MSDIAL").mass_decomposition.decompose_mass(
            tolerance_ppm=3.0
        )
    )
    .with_columns(
        # Extract formula strings
        formula_str=pl.col("formulas").struct.field("formulas_str")
    )
    .explode("formula_str")  # One row per candidate formula
    .sort("Height", descending=True)
)
```

## Performance Tips

1. **Use LazyFrames** for large datasets:
   ```python
   lazy_result = (
       pl.scan_parquet("large_library.parquet")
       .filter(pl.col("quality_score") > 0.8)
       .collect(engine="streaming")
   )
   ```

2. **Batch operations** instead of row-by-row:
   ```python
   # Good: vectorized
   df.with_columns(formulas=pl.col("mass").mass_decomposition.decompose_mass(...))
   
   # Avoid: row-by-row iteration
   for row in df.iter_rows():
       ...
   ```

3. **Filter early** to reduce data size:
   ```python
   df.filter(pl.col("Height") > threshold).with_columns(...)
   ```

## Next Steps

Ready to dive deeper? Check out the tutorials:

- [**MSDIAL Chromatogram Annotation**](../tutorials/01-msdial-chromatogram-annotation.md) - Complete workflow for processing MSDIAL data
- [**MSP Library Processing**](../tutorials/02-msp-library-processing.md) - Clean and prepare spectral libraries
- [**Spectral Similarity Search**](../tutorials/03-spectral-similarity-search.md) - Match spectra against libraries

Or explore the [API Reference](../reference/index.md) for detailed function documentation.
