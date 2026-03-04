# Data Structures

Documentation for the Polars DataFrame schemas used in HRMS Utils.

## Overview

HRMS Utils uses Polars DataFrames with nested types (Lists, Arrays, Structs) to efficiently represent mass spectrometry data.

## Common Types

### Spectrum Representation

Spectra are stored as separate List columns for m/z and intensity:

```python
{
    "mz": List[Float64],            # m/z values
    "intensity": List[Float64]      # Intensities (0-1 normalized)
}
```

### Formula Representation

Chemical formulas are stored as fixed-size arrays of element counts:

```python
{
    "formula": Array[Int32, 15]     # Element counts [H, C, N, O, P, S, F, Cl, Br, I, Si, B, Se, As, Na]
}
```

Element order: H, C, N, O, P, S, F, Cl, Br, I, Si, B, Se, As, Na

## Schema Definitions

### MSDIAL Chromatogram

Schema returned by `get_chromatogram()`:

| Column | Type | Description |
|--------|------|-------------|
| `Peak ID` | Int64 | Unique peak identifier |
| `RT (min)` | Float64 | Retention time in minutes |
| `Precursor_mz_MSDIAL` | Float64 | Precursor m/z |
| `Height` | Float64 | Peak height |
| `Precursor_type_MSDIAL` | String | Adduct type |
| `msms_m/z` | List[Float64] | MS/MS fragment m/z |
| `msms_intensity` | List[Float64] | MS/MS intensities (0-1) |
| `ms1_isotopes_m/z` | List[Float64] | MS1 isotope m/z values |
| `ms1_isotopes_intensity` | List[Float64] | MS1 isotope intensities |

### MSP Library

Schema returned by `process_single_file()`:

| Column | Type | Description |
|--------|------|-------------|
| `name` | String | Compound name |
| `precursor_mz` | Float64 | Precursor m/z |
| `molecular_formula` | String | Formula as text |
| `molecular_formula_array` | Array[Int32, 15] | Formula as element counts |
| `cleaned_normalized_mz` | List[Float64] | Cleaned fragment m/z |
| `cleaned_normalized_intensity` | List[Float64] | Cleaned intensities (0-1) |
| `spectral_information_score` | Float64 | Quality score |
| `inchikey` | String | InChIKey identifier |

See [API Reference](api/formats.md) for complete schemas.

## Working with Nested Types

### Accessing List Elements

```python
# Get first element
df.with_columns(first_mz=pl.col("mz").list.first())

# Get specific index
df.with_columns(second_intensity=pl.col("intensity").list.get(1))

# List length
df.with_columns(num_peaks=pl.col("mz").list.len())
```

### Exploding Lists

```python
# One row per fragment
df.explode(["mz", "intensity"])
```

### Array Operations

```python
# Access array element by index
df.with_columns(carbon_count=pl.col("formula_array").arr.get(1))

# Sum array
df.with_columns(total_atoms=pl.col("formula_array").arr.sum())
```

## See Also

- [Polars Data Types Documentation](https://docs.pola.rs/user-guide/expressions/data-types/)
- [Tutorials](../tutorials/01-msdial-chromatogram-annotation.md)
