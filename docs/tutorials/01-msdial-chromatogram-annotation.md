# Tutorial 1: MSDIAL Chromatogram Annotation

Learn how to read, process, and annotate chromatogram data from MSDIAL with candidate elemental formulas.

## What You'll Learn

- Reading MSDIAL `.txt` export files
- Understanding the chromatogram data structure
- Subtracting blank samples
- Annotating peaks with candidate formulas using isotopic patterns
- Filtering and exporting annotated results

## Prerequisites

- HRMS Utils installed (see [Installation](../getting-started/installation.md))
- MSDIAL chromatogram exported as tab-delimited text (use "trim content for Excel" option)

## Overview

This tutorial walks through a complete workflow:

1. Load sample chromatogram
2. (Optional) Load and subtract blank
3. Annotate with candidate formulas using isotopic pattern analysis
4. Explore and filter results

## Step 1: Load Chromatogram Data

MSDIAL exports chromatograms as tab-delimited text files containing peak information, retention times, MS/MS spectra, and MS1 isotope patterns.

```python
from pathlib import Path
import polars as pl
from hrms_utils.formats.msdial import get_chromatogram

# Load sample chromatogram
sample_path = Path("tests/data/MSDIAL_output.txt")
chromatogram = get_chromatogram(sample_path)

# View the data structure
print(chromatogram.schema)
```

### Understanding the Output Schema

`get_chromatogram()` returns a Polars DataFrame with these columns:

- `Peak ID` (Int64): Unique identifier for each peak
- `RT (min)` (Float64): Retention time in minutes
- `Precursor_mz_MSDIAL` (Float64): Precursor m/z measured by MSDIAL
- `Height` (Float64): Peak height (intensity)
- `Precursor_type_MSDIAL` (String): Adduct type (e.g., "[M+H]+", "[M+Na]+")
- `Isotope` (Int32): Isotope number (0 for monoisotopic)
- `msms_m/z` (List[Float64]): MS/MS fragment m/z values
- `msms_intensity` (List[Float64]): MS/MS fragment intensities (normalized 0-1)
- `isobars` (List[Int64]): Peak IDs of potential isobaric contaminants
- `msms_m/z_cleaned` (List[Float64]): MS/MS after isobar subtraction
- `msms_intensity_cleaned` (List[Float64]): Cleaned intensities
- `energy_is_too_low` (Boolean): Collision energy too low (molecular ion dominates)
- `energy_is_too_high` (Boolean): Collision energy too high (no molecular ion)
- `ms1_isotopes_m/z` (List[Float64]): MS1 isotope peak m/z values
- `ms1_isotopes_intensity` (List[Float64]): MS1 isotope intensities

### Explore the Data

```python
# View first few peaks
print(chromatogram.select([
    "Peak ID", "RT (min)", "Precursor_mz_MSDIAL", 
    "Height", "Precursor_type_MSDIAL"
]).head(10))

# Check how many peaks have MS/MS data
with_msms = chromatogram.filter(pl.col("msms_m/z").is_not_null())
print(f"Peaks with MS/MS: {len(with_msms)}/{len(chromatogram)}")

# Filter high-intensity peaks
high_intensity = chromatogram.filter(
    pl.col("Height") > 1e6
).sort("Height", descending=True)
print(f"\nTop 5 peaks by intensity:")
print(high_intensity.select([
    "Peak ID", "RT (min)", "Precursor_mz_MSDIAL", "Height"
]).head())
```

## Step 2: Blank Subtraction (Optional)

If you have a blank sample, you can subtract it to remove background peaks:

```python
from hrms_utils.formats.msdial import subtract_blank_frame, blank_config

# Load blank chromatogram
blank_path = Path("path/to/blank.txt")
blank = get_chromatogram(blank_path)

# Configure subtraction parameters
config = blank_config(
    ms1_mass_tolerance=3.0,  # ppm tolerance for precursor matching
    dRT_min=0.1,             # retention time window (minutes)
    ratio=5,                 # intensity ratio threshold
    use_ms2=False            # whether to use MS/MS similarity
)

# Subtract blank
chromatogram_cleaned = subtract_blank_frame(
    sample_df=chromatogram,
    blank_df=blank,
    config=config
)

print(f"Peaks before: {len(chromatogram)}")
print(f"Peaks after blank subtraction: {len(chromatogram_cleaned)}")
```

### Blank Subtraction Parameters

- **`ms1_mass_tolerance`**: Precursor m/z tolerance in ppm. Peaks matching within this tolerance are considered the same compound.
- **`dRT_min`**: Retention time tolerance in minutes. Tighter window = more specific matching.
- **`ratio`**: Minimum sample/blank intensity ratio. If sample peak is less than `ratio` times the blank peak intensity, it's removed.
- **`use_ms2`**: If `True`, requires MS/MS similarity match in addition to RT/mass matching. More specific but requires both samples to have MS/MS.

## Step 3: Annotate with Candidate Formulas

Now annotate peaks with candidate elemental formulas using isotopic pattern analysis and mass decomposition:

```python
from hrms_utils.formats.msdial import annotate_chromatogram_with_formulas

# Annotate chromatogram
annotated = annotate_chromatogram_with_formulas(
    chromatogram,
    max_bounds=None,  # Auto-deduce from isotopic pattern
    precursor_mass_accuracy_ppm=3.0,
    fragment_mass_accuracy_ppm=5.0,
    normalized_fragment_mass_accuracy_ppm=4.0,
    isotopic_mass_accuracy_ppm=2.0,
    isotopic_minimum_intensity=5e4,
)

print(annotated.schema)
```

### What Happens During Annotation

1. **Isotopic Pattern Deduction**: Analyzes MS1 isotope peaks to infer elemental composition constraints (min/max bounds for each element)
2. **Mass Decomposition**: Generates candidate formulas matching the precursor mass within tolerance, constrained by isotopic bounds
3. **Spectrum Cleaning**: For each candidate formula, cleans and normalizes MS/MS fragments by matching to expected fragment masses
4. **Explosion**: Each peak expands into multiple rows, one per candidate formula

### New Columns Added

- `min_bounds` (Array[Int32, 15]): Minimum element counts from isotopic pattern
- `max_bounds` (Array[Int32, 15]): Maximum element counts from isotopic pattern
- `precursor_formula` (Array[Int32, 15]): Candidate formula as element count array
- `precursor_formula_str` (String): Human-readable formula (e.g., "C15H10O5")
- `precursor_errors_ppm` (Float64): Mass error for this candidate formula (ppm)
- `cleaned_msms_mz` (List[Float64]): Fragment m/z values matched to candidate formula
- `cleaned_msms_intensity` (List[Float64]): Corresponding intensities
- `cleaned_spectrum_formulas` (List[Array[Int32, 15]]): Assigned formulas for each fragment
- `cleaned_spectrum_formulas_str` (List[String]): Fragment formulas as strings
- `cleaned_fragment_errors_ppm` (List[Float64]): Mass errors for fragments (ppm)

## Step 4: Explore Annotated Results

```python
# View annotated peaks
print(annotated.select([
    "Peak ID", "RT (min)", "Precursor_mz_MSDIAL",
    "precursor_formula_str", "precursor_errors_ppm"
]).head(20))

# Count candidates per peak
candidates_per_peak = (
    annotated
    .group_by("Peak ID")
    .agg(pl.col("precursor_formula_str").count().alias("num_candidates"))
    .sort("num_candidates", descending=True)
)
print("\nCandidates per peak:")
print(candidates_per_peak.head(10))

# Filter by mass error
high_confidence = annotated.filter(
    pl.col("precursor_errors_ppm").abs() < 2.0
)
print(f"\nCandidates with error < 2 ppm: {len(high_confidence)}")
```

### Filter by Fragment Coverage

Prefer formulas that explain more fragments:

```python
# Count explained fragments
annotated_with_coverage = annotated.with_columns(
    num_explained_fragments=pl.col("cleaned_msms_mz").list.len()
)

# Filter peaks with good fragment coverage
good_coverage = annotated_with_coverage.filter(
    pl.col("num_explained_fragments") >= 3
)

print(good_coverage.select([
    "Peak ID", "precursor_formula_str", 
    "num_explained_fragments", "precursor_errors_ppm"
]).head())
```

### Select Best Candidate per Peak

Choose the formula with lowest mass error and most explained fragments:

```python
# Rank candidates within each peak
ranked = (
    annotated_with_coverage
    .with_columns(
        score=(
            pl.col("num_explained_fragments") * 10  # Reward fragment coverage
            - pl.col("precursor_errors_ppm").abs()  # Penalize mass error
        )
    )
    .sort(["Peak ID", "score"], descending=[False, True])
    .group_by("Peak ID")
    .first()  # Take best candidate per peak
)

print(ranked.select([
    "Peak ID", "RT (min)", "precursor_formula_str",
    "num_explained_fragments", "precursor_errors_ppm", "score"
]))
```

## Step 5: Export Results

Save annotated results for downstream analysis:

```python
# Export to Parquet (recommended for Polars)
annotated.write_parquet("annotated_chromatogram.parquet")

# Or export to CSV
annotated.select([
    "Peak ID", "RT (min)", "Precursor_mz_MSDIAL", "Height",
    "precursor_formula_str", "precursor_errors_ppm",
    "num_explained_fragments"
]).write_csv("annotated_chromatogram.csv")

# Export to Excel
annotated.write_excel("annotated_chromatogram.xlsx")
```

## Complete Example

Here's the full workflow in one script:

```python
from pathlib import Path
import polars as pl
from hrms_utils.formats.msdial import (
    get_chromatogram,
    subtract_blank_frame,
    blank_config,
    annotate_chromatogram_with_formulas
)

# Configure Polars display
pl.Config.set_tbl_rows(20)
pl.Config.set_tbl_cols(15)

# 1. Load sample
sample_path = Path("tests/data/MSDIAL_output.txt")
chromatogram = get_chromatogram(sample_path)
print(f"Loaded {len(chromatogram)} peaks")

# 2. Optional: subtract blank
# blank = get_chromatogram("blank.txt")
# chromatogram = subtract_blank_frame(chromatogram, blank, blank_config())

# 3. Annotate with formulas
annotated = annotate_chromatogram_with_formulas(
    chromatogram,
    precursor_mass_accuracy_ppm=3.0,
    fragment_mass_accuracy_ppm=5.0
)

# 4. Filter and rank
best_candidates = (
    annotated
    .with_columns(
        num_fragments=pl.col("cleaned_msms_mz").list.len(),
        score=(
            pl.col("cleaned_msms_mz").list.len() * 10
            - pl.col("precursor_errors_ppm").abs()
        )
    )
    .filter(
        pl.col("precursor_errors_ppm").abs() < 3.0,
        pl.col("num_fragments") >= 2
    )
    .sort(["Peak ID", "score"], descending=[False, True])
    .group_by("Peak ID").first()
)

# 5. Export
best_candidates.write_parquet("annotated_results.parquet")
print(f"\nExported {len(best_candidates)} annotated peaks")
```

## Tips and Best Practices

1. **Isotopic patterns are key**: The isotopic pattern deduction greatly reduces the formula search space. Peaks without good isotope patterns may not be annotated.

2. **Adjust tolerances based on instrument**: High-resolution Orbitrap data can use tighter tolerances (1-3 ppm), while TOF data may need 5-10 ppm.

3. **Fragment coverage matters**: Formulas that explain more fragments are generally more reliable.

4. **Check energy levels**: Use `energy_is_too_low` and `energy_is_too_high` to identify poorly fragmented peaks.

5. **Isobar cleaning**: The `msms_m/z_cleaned` column contains spectra after subtracting isobaric contaminants. Use this for better annotation.

## Next Steps

- [Tutorial 2: MSP Library Processing](02-msp-library-processing.md) - Process reference libraries
- [Tutorial 3: Spectral Similarity Search](03-spectral-similarity-search.md) - Match annotated peaks against libraries
- [API Reference: MSDIAL](../reference/api/formats.md#msdial) - Detailed function documentation
