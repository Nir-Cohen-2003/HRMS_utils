# Tutorial 2: MSP Library Processing

Learn how to load, clean, and prepare spectral libraries in MSP/MSPEC format for similarity searches and annotation.

## What You'll Learn

- Reading MSP/MSPEC library files
- Understanding library data structure
- Automatic spectrum cleaning and normalization
- Filtering by quality metrics
- Combining multiple libraries
- Preparing libraries for similarity searches

## Prerequisites

- HRMS Utils installed (see [Installation](../getting-started/installation.md))
- MSP or MSPEC library files (NIST format)

## Overview

This tutorial covers:

1. Loading single MSP/MSPEC files
2. Understanding the library schema
3. Automatic cleaning and normalization
4. Quality filtering
5. Combining multiple libraries
6. Exporting clean libraries

## Step 1: Load an MSP Library

MSP (Mass Spectral Library) files contain reference spectra with metadata like chemical formulas, collision energies, and instrument information.

```python
from pathlib import Path
import polars as pl
from hrms_utils.formats.nist_mspec import read_MSPEC_file

# Load library
library_path = Path("tests/data/msp_sample.msp")
library = read_MSPEC_file(
    library_path,
    raw_fragment_tolerance_ppm=10.0,
    normalized_fragment_tolerance_ppm=5.0,
    molecular_ion_tolerance_ppm=5.0
)

print(f"Loaded {len(library)} spectra")
print(library.schema)
```

### Loading Parameters

- **`raw_fragment_tolerance_ppm`**: Tolerance for initial fragment matching during cleaning (typically 5-10 ppm)
- **`normalized_fragment_tolerance_ppm`**: Tolerance for normalized fragment mass error after cleaning (typically 2-5 ppm)
- **`molecular_ion_tolerance_ppm`**: Tolerance for detecting molecular ion peak (typically 3-5 ppm)

## Step 2: Understanding the Library Schema

The loaded library is a Polars DataFrame with extensive metadata and cleaned spectra:

### Metadata Columns

```python
metadata_cols = [
    "name",                  # Compound name
    "nist_id",              # NIST library ID
    "cas",                  # CAS registry number
    "inchikey",             # Full InChIKey
    "base_inchikey",        # InChIKey without stereochemistry
    "smiles",               # SMILES string
    "inchi",                # InChI string
    "molecular_formula",    # Formula as string (e.g., "C15H10O5")
    "precursor_mz",         # Precursor m/z
    "exact_mass",           # Calculated exact mass
]

print(library.select(metadata_cols).head())
```

### Instrument Information

```python
instrument_cols = [
    "instrument_type",      # e.g., "LC-ESI-QTOF"
    "instrument",           # Specific instrument model
    "ionization",          # e.g., "ESI"
    "ion_mode",            # "POSITIVE" or "NEGATIVE"
    "is_orbitrap",         # Boolean: Orbitrap instrument
    "is_TOF",              # Boolean: TOF instrument
    "is_ESI",              # Boolean: ESI ionization
]

print(library.select(instrument_cols).value_counts().head(10))
```

### Collision Energy

```python
energy_cols = [
    "collision_energy_NCE",              # Normalized collision energy (%)
    "collision_energy_ev",               # Collision energy in eV
    "multiple_collision_energies",       # Boolean: merged from multiple energies
    "collision_energy_mean",             # Mean energy if multiple
    "collision_energy_list",             # List of energies if multiple
]

print(library.select(energy_cols).head())
```

### Spectral Data

```python
spectral_cols = [
    # Raw spectrum (as stored in MSP)
    "raw_spectrum_mz",                   # List[Float64]: original m/z
    "raw_spectrum_intensity",            # List[Float64]: original intensities
    
    # Cleaned and normalized spectrum
    "cleaned_normalized_mz",             # List[Float64]: cleaned m/z
    "cleaned_normalized_intensity",      # List[Float64]: cleaned intensities (0-1)
    "cleaned_formulas",                  # List[Array[Int32, 15]]: fragment formulas
    "cleaned_formulas_str",              # List[String]: fragment formulas as text
    "cleaned_errors_ppm",                # List[Float64]: mass errors for fragments
]

# View a spectrum
spectrum = library.select(spectral_cols).row(0, named=True)
print(f"Raw peaks: {len(spectrum['raw_spectrum_mz'])}")
print(f"Cleaned peaks: {len(spectrum['cleaned_normalized_mz'])}")
```

### Quality Metrics

```python
quality_cols = [
    "spectral_information_score",        # Float64: information content score
    "molecular_ion_present",             # Boolean: molecular ion detected
    "molecular_ion_intensity",           # Float64: molecular ion intensity
    "base_peak_intensity",               # Float64: highest peak intensity
    "num_peaks_cleaned",                 # Int64: number of cleaned peaks
]

print(library.select(quality_cols).describe())
```

## Step 3: Automatic Cleaning Process

When you call `read_MSPEC_file()`, it automatically:

1. **Extracts metadata** from MSP entries
2. **Validates formulas** against precursor mass
3. **Cleans spectra**:
   - Removes noise (< 0.1% intensity)
   - Matches fragments to expected masses from formula
   - Normalizes intensities to 0-1 range
4. **Computes quality scores**:
   - Spectral information score
   - Molecular ion presence
   - Fragment coverage

### Example: Comparing Raw vs Cleaned

```python
# Select one spectrum
idx = 0
example = library.row(idx, named=True)

print(f"Compound: {example['name']}")
print(f"Formula: {example['molecular_formula']}")
print(f"Precursor m/z: {example['precursor_mz']:.4f}")
print(f"\nRaw peaks: {len(example['raw_spectrum_mz'])}")
print(f"Cleaned peaks: {len(example['cleaned_normalized_mz'])}")
print(f"Spectral info score: {example['spectral_information_score']:.2f}")
print(f"Molecular ion present: {example['molecular_ion_present']}")
```

## Step 4: Filter by Quality

Remove low-quality spectra before using the library:

```python
# Filter by multiple quality criteria
high_quality = library.filter(
    # Require good information content
    pl.col("spectral_information_score") > 5.0,
    
    # Require minimum number of fragments
    pl.col("num_peaks_cleaned") >= 3,
    
    # Require valid formula
    pl.col("clean_precursor").is_not_null(),
    
    # Require molecular ion (optional, depends on use case)
    # pl.col("molecular_ion_present") == True,
)

print(f"Original library: {len(library)} spectra")
print(f"High quality: {len(high_quality)} spectra")
print(f"Filtered out: {len(library) - len(high_quality)} spectra")
```

### Quality Metrics Explained

**Spectral Information Score**: Measures how "informative" a spectrum is based on fragment diversity and distribution. Higher scores indicate more distinctive spectra.

- < 3.0: Low information (few or redundant fragments)
- 3.0-7.0: Moderate information
- > 7.0: High information (diverse, distinctive fragments)

**Num Peaks Cleaned**: Number of fragments after cleaning. More fragments generally means better coverage.

**Molecular Ion Present**: Whether the precursor ion is detected in the spectrum. Some fragmentation methods (e.g., high energy CID) may not preserve it.

### Filter by Instrument Type

Create subsets for specific instrument types:

```python
# Orbitrap spectra only
orbitrap_lib = library.filter(pl.col("is_orbitrap") == True)
print(f"Orbitrap spectra: {len(orbitrap_lib)}")

# ESI positive mode only
esi_pos = library.filter(
    pl.col("is_ESI") == True,
    pl.col("ion_mode") == "POSITIVE"
)
print(f"ESI positive mode: {len(esi_pos)}")

# TOF instruments with high quality
tof_hq = library.filter(
    pl.col("is_TOF") == True,
    pl.col("spectral_information_score") > 6.0
)
print(f"High-quality TOF spectra: {len(tof_hq)}")
```

## Step 5: Combine Multiple Libraries

Merge libraries from different sources:

```python
from hrms_utils.formats.nist_mspec import create_nist_dataframe

# List of (file_path, database_name) tuples
library_files = [
    (Path("library1.msp"), "NIST"),
    (Path("library2.msp"), "MassBank"),
    (Path("library3.msp"), "GNPS"),
]

# Combine into one DataFrame
combined = create_nist_dataframe(library_files)

print(f"Combined library: {len(combined)} spectra")

# Check source distribution
print("\nSpectra per database:")
print(combined.group_by("DB_Name").agg(
    pl.count().alias("num_spectra")
))
```

### Deduplicate Combined Libraries

Remove duplicate entries based on InChIKey:

```python
# Deduplicate by base InChIKey (ignoring stereochemistry)
deduplicated = combined.unique(
    subset=["base_inchikey"],
    keep="first"
)

print(f"Before deduplication: {len(combined)}")
print(f"After deduplication: {len(deduplicated)}")
```

### Prefer High-Quality Duplicates

Keep the best spectrum for each compound:

```python
# For each InChIKey, keep the spectrum with highest information score
best_per_compound = (
    combined
    .sort("spectral_information_score", descending=True)
    .unique(subset=["base_inchikey"], keep="first")
)

print(f"Best spectra per compound: {len(best_per_compound)}")
```

## Step 6: Export Clean Library

Save the cleaned library for later use:

```python
# Export to Parquet (recommended - preserves nested types)
high_quality.write_parquet("clean_library.parquet")

# Or export selected columns to CSV
high_quality.select([
    "name", "molecular_formula", "precursor_mz",
    "spectral_information_score", "num_peaks_cleaned",
    "DB_Name"
]).write_csv("clean_library_metadata.csv")

# For use in similarity searches, save just the essential columns
similarity_library = high_quality.select([
    "name",
    "inchikey",
    "precursor_mz",
    "molecular_formula_array",
    "cleaned_normalized_mz",
    "cleaned_normalized_intensity",
    "spectral_information_score"
])
similarity_library.write_parquet("similarity_search_library.parquet")
```

## Step 7: Analyze Library Statistics

Explore library composition:

```python
# Mass distribution
mass_stats = library.select([
    pl.col("precursor_mz").min().alias("min_mass"),
    pl.col("precursor_mz").max().alias("max_mass"),
    pl.col("precursor_mz").mean().alias("mean_mass"),
    pl.col("precursor_mz").median().alias("median_mass"),
])
print("Mass distribution:")
print(mass_stats)

# Fragment count distribution
fragment_dist = (
    library
    .with_columns(
        num_fragments=pl.col("cleaned_normalized_mz").list.len()
    )
    .group_by("num_fragments")
    .agg(pl.count().alias("num_spectra"))
    .sort("num_fragments")
)
print("\nFragment count distribution:")
print(fragment_dist.head(20))

# Information score distribution
info_score_dist = library.select([
    pl.col("spectral_information_score").min().alias("min_score"),
    pl.col("spectral_information_score").max().alias("max_score"),
    pl.col("spectral_information_score").mean().alias("mean_score"),
    pl.col("spectral_information_score").quantile(0.25).alias("q25"),
    pl.col("spectral_information_score").quantile(0.5).alias("q50"),
    pl.col("spectral_information_score").quantile(0.75).alias("q75"),
])
print("\nInformation score distribution:")
print(info_score_dist)
```

## Complete Example

Here's a full workflow for processing an MSP library:

```python
from pathlib import Path
import polars as pl
from hrms_utils.formats.nist_mspec import read_MSPEC_file, create_nist_dataframe

# Configure display
pl.Config.set_tbl_rows(20)

# 1. Load library
library = read_MSPEC_file(
    "tests/data/msp_sample.msp",
    raw_fragment_tolerance_ppm=10.0,
    normalized_fragment_tolerance_ppm=5.0
)
print(f"Loaded {len(library)} spectra")

# 2. Filter by quality
clean_library = library.filter(
    pl.col("spectral_information_score") > 5.0,
    pl.col("num_peaks_cleaned") >= 3,
    pl.col("clean_precursor").is_not_null()
)
print(f"After quality filter: {len(clean_library)} spectra")

# 3. Analyze
print("\nInstrument distribution:")
print(clean_library.group_by("instrument_type").agg(
    pl.count().alias("count")
).sort("count", descending=True).head(10))

print("\nMass range:")
print(f"Min: {clean_library['precursor_mz'].min():.2f}")
print(f"Max: {clean_library['precursor_mz'].max():.2f}")

# 4. Export for similarity searches
clean_library.select([
    "name", "inchikey", "precursor_mz",
    "molecular_formula_array",
    "cleaned_normalized_mz",
    "cleaned_normalized_intensity",
    "spectral_information_score"
]).write_parquet("processed_library.parquet")

print("\nExported clean library to processed_library.parquet")
```

## Tips and Best Practices

1. **Use Parquet format**: Preserves nested types (Lists, Arrays) efficiently. CSV cannot handle nested data well.

2. **Filter early**: Apply quality filters before combining libraries to reduce memory usage.

3. **Check mass range**: Ensure your library covers the mass range of your samples.

4. **Information score threshold**: Start with 5.0, adjust based on your needs. Lower thresholds include more spectra but lower quality.

5. **Deduplicate carefully**: Use `base_inchikey` to merge stereoisomers, or full `inchikey` to keep them separate.

6. **Instrument matching**: For best results, match library instrument type to your sample instrument type.

## Next Steps

- [Tutorial 3: Spectral Similarity Search](03-spectral-similarity-search.md) - Use this library to match unknown spectra
- [API Reference: MSP/MSPEC](../reference/api/formats.md#msp-mspec-nist) - Detailed function documentation
- [How-To: Batch Processing](../how-to/batch-processing.md) - Process large libraries efficiently
