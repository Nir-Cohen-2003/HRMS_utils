# Rust mzML Reader Schema Extensions

This document explains how to add new fields to the Rust-based mzML reader (`src/hrms_core/io_mzml.rs`).

## Overview

The reader iterates over spectra using `mzdata` and collects data into vectors, which are then converted into Polars Series and finally a DataFrame.

## How to Add a New Field

To add a new field (column) to the output DataFrame, follow these steps in `src/hrms_core/io_mzml.rs`:

### 1. Add a Vector for Storage

At the beginning of `read_single_mzml`, define a mutable vector to hold the data for the new field. Use `Option<T>` if the field might be missing.

```rust
// Inside read_single_mzml function
let mut my_new_field: Vec<Option<f64>> = Vec::new();
```

### 2. Extract Data in the Loop

Inside the `for spectrum in reader` loop, extract the data from the `spectrum` object or its components (`precursor`, `params`, etc.).

```rust
// Inside the loop
let val = if let Some(param) = spectrum.params().iter().find(|p| p.name == "my param") {
    param.value.to_f64().ok()
} else {
    None
};
my_new_field.push(val);
```

**Tips for Extraction:**
- **CV Params:** Use `spectrum.params()` or `precursor.activation.params()` to search for CV parameters by accession or name.
- **User Params:** Similar to CV params, search by name.
- **Struct Fields:** Check `mzdata` documentation for available fields on `Spectrum`, `Precursor`, `Activation`, etc.

### 3. Create a Series

After the loop, convert the vector into a Polars `Series`.

```rust
let s_my_new_field = Series::new("my_new_field".into(), my_new_field);
```

### 4. Add to DataFrame

Add the new Series to the `DataFrame::new` call at the end of the function.

```rust
let df = DataFrame::new(vec![
    // ... existing columns ...
    s_my_new_field.into(),
])?;
```

## Example: Adding "Base Peak Intensity"

1.  **Vector:** `let mut base_peak_intensities = Vec::new();`
2.  **Extraction:**
    ```rust
    // mzdata Spectrum often has this in params or computed
    let bp_int = spectrum.peaks().base_peak().map(|p| p.intensity);
    base_peak_intensities.push(bp_int);
    ```
3.  **Series:** `let s_bp = Series::new("base_peak_intensity".into(), base_peak_intensities);`
4.  **DataFrame:** Add `s_bp.into()` to the list.
