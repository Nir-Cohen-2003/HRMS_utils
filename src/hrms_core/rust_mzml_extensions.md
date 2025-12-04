# Extending the Rust mzML Reader

This document explains how to add new fields to the Rust-based mzML reader located in `src/hrms_core/io_mzml.rs`.

## Overview

The reader uses the `mzdata` crate to parse mzML files and `polars` to construct DataFrames. It iterates over spectra, collects values into vectors, and then builds a DataFrame.

## Adding a New Field

To add a new field (e.g., "ion_mobility"), follow these steps in `src/hrms_core/io_mzml.rs`:

1.  **Declare a Vector for the Column**:
    Inside `read_single_mzml`, add a new vector to hold the data for the new column.
    ```rust
    let mut ion_mobilities = Vec::new();
    ```

2.  **Extract Data in the Loop**:
    Inside the `for spectrum in reader` loop, extract the value from the `spectrum` object.
    You may need to access `spectrum.params()` to find specific CV parameters if they are not exposed as methods on the `spectrum` object.

    ```rust
    // Example: Extracting a CV param
    let mut im = None;
    for param in spectrum.params() {
        if param.accession == Some(1002476) { // Accession for ion mobility drift time
             im = param.value.to_f64().ok();
             break;
        }
    }
    ion_mobilities.push(im);
    ```

3.  **Create a Series**:
    After the loop, convert the vector into a Polars `Series`.
    ```rust
    let s_im = Series::new("ion_mobility".into(), ion_mobilities);
    ```

4.  **Add to DataFrame**:
    Add the new Series to the `DataFrame::new` call.
    ```rust
    let df = DataFrame::new(vec![
        // ... existing columns ...
        s_im.into(),
    ])?;
    ```

## Common Accessions

Refer to `scripts/schema_extensions.md` or the PSI-MS ontology for accession numbers.

## Handling Optional Fields

Always use `Option<T>` in your vectors (e.g., `Vec<Option<f64>>`) to handle cases where the field is missing in some spectra. Polars handles `None` as null values.
