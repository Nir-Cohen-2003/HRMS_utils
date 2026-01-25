# Tutorial 3: Spectral Similarity Search

Learn how to match query spectra against reference libraries using precursor mass filtering and spectral similarity metrics.

## What You'll Learn

- Matching precursors by mass tolerance
- Computing spectral similarity scores
- Understanding different similarity metrics
- Filtering and ranking matches
- Batch processing large searches
- Optimizing search performance

## Prerequisites

- HRMS Utils installed (see [Installation](../getting-started/installation.md))
- Completed [Tutorial 1](01-msdial-chromatogram-annotation.md) (MSDIAL processing)
- Completed [Tutorial 2](02-msp-library-processing.md) (library processing)

## Overview

Spectral similarity search involves:

1. Loading query spectra (e.g., from MSDIAL)
2. Loading reference library (e.g., from MSP)
3. Matching precursors by mass tolerance
4. Computing spectral similarity for matched pairs
5. Filtering and ranking results

## Step 1: Prepare Query and Library Data

```python
from pathlib import Path
import polars as pl
from hrms_utils.formats.msdial import get_chromatogram, annotate_chromatogram_with_formulas
from hrms_utils.formats.nist_mspec import read_MSPEC_file

# Load query chromatogram
query = get_chromatogram("tests/data/MSDIAL_output.txt")

# Optionally annotate with formulas
query_annotated = annotate_chromatogram_with_formulas(
    query,
    precursor_mass_accuracy_ppm=3.0
)

# Load reference library
library = read_MSPEC_file(
    "tests/data/msp_sample.msp",
    raw_fragment_tolerance_ppm=10.0,
    normalized_fragment_tolerance_ppm=5.0
)

# Filter library by quality
library_clean = library.filter(
    pl.col("spectral_information_score") > 5.0,
    pl.col("num_peaks_cleaned") >= 3
)

print(f"Query spectra: {len(query)}")
print(f"Library spectra: {len(library_clean)}")
```

## Step 2: Precursor Mass Matching

Match query and library spectra based on precursor m/z within tolerance:

```python
# Set mass tolerance (in ppm)
precursor_tolerance_ppm = 10.0

# Join on precursor mass tolerance
matches = query.lazy().join_where(
    library_clean.lazy(),
    # Precursor mass within tolerance (ppm)
    (pl.col("Precursor_mz_MSDIAL") / pl.col("precursor_mz") - 1.0).abs() 
        <= (precursor_tolerance_ppm * 1e-6),
    suffix="_lib"
).collect()

print(f"Precursor matches: {len(matches)}")
print(f"Avg matches per query: {len(matches) / len(query):.1f}")
```

### Understanding the Join

`join_where()` creates a row for every query-library pair where precursors match within tolerance:

- **Query columns**: Original names (e.g., `Precursor_mz_MSDIAL`, `msms_m/z`)
- **Library columns**: Suffixed with `_lib` (e.g., `precursor_mz_lib`, `cleaned_normalized_mz`)

### Narrower Mass Windows

For high-resolution instruments, use tighter tolerances:

```python
# Orbitrap: 3 ppm
orbitrap_matches = query.lazy().join_where(
    library_clean.filter(pl.col("is_orbitrap") == True).lazy(),
    (pl.col("Precursor_mz_MSDIAL") / pl.col("precursor_mz") - 1.0).abs() <= 3e-6,
    suffix="_lib"
).collect()

# TOF: 10 ppm
tof_matches = query.lazy().join_where(
    library_clean.filter(pl.col("is_TOF") == True).lazy(),
    (pl.col("Precursor_mz_MSDIAL") / pl.col("precursor_mz") - 1.0).abs() <= 10e-6,
    suffix="_lib"
).collect()
```

## Step 3: Compute Spectral Similarity

HRMS Utils provides multiple similarity metrics. All work on Polars struct columns containing both spectra.

### Dot Product Similarity (NIST-style)

The classic NIST dot product with sqrt intensity weighting:

```python
# Create struct with both spectra
similarity_results = matches.with_columns(
    pl.struct(
        # Query spectrum
        pl.col("msms_m/z").alias("mz1"),
        pl.col("msms_intensity").alias("intensities1"),
        pl.col("Precursor_mz_MSDIAL").alias("precursor_mz1"),
        # Library spectrum
        pl.col("cleaned_normalized_mz").alias("mz2"),
        pl.col("cleaned_normalized_intensity").alias("intensities2"),
        pl.col("precursor_mz").alias("precursor_mz2"),
    ).spectral_similarity.dotprod_similarity(
        ms2_tolerance_in_ppm=10.0,
        clean_spectra_first=True,
        noise_threshold=0.001,
        ignore_precursor=True
    ).alias("dotprod_score")
)

print(similarity_results.select([
    "Peak ID", "name", "dotprod_score",
    "Precursor_mz_MSDIAL", "precursor_mz"
]).sort("dotprod_score", descending=True).head(10))
```

### Entropy Similarity

Spectral entropy similarity is more robust to noise:

```python
similarity_results = similarity_results.with_columns(
    pl.struct(
        pl.col("msms_m/z").alias("mz1"),
        pl.col("msms_intensity").alias("intensities1"),
        pl.col("Precursor_mz_MSDIAL").alias("precursor_mz1"),
        pl.col("cleaned_normalized_mz").alias("mz2"),
        pl.col("cleaned_normalized_intensity").alias("intensities2"),
        pl.col("precursor_mz").alias("precursor_mz2"),
    ).spectral_similarity.entropy_similarity(
        ms2_tolerance_in_ppm=10.0,
        clean_spectra_first=True,
        noise_threshold=0.001,
        ignore_precursor=True
    ).alias("entropy_score")
)

print(similarity_results.select([
    "Peak ID", "name", "dotprod_score", "entropy_score"
]).sort("entropy_score", descending=True).head(10))
```

### Mass-Weighted Cosine Similarity

Emphasizes higher m/z fragments:

```python
similarity_results = similarity_results.with_columns(
    pl.struct(
        pl.col("msms_m/z").alias("mz1"),
        pl.col("msms_intensity").alias("intensities1"),
        pl.col("Precursor_mz_MSDIAL").alias("precursor_mz1"),
        pl.col("cleaned_normalized_mz").alias("mz2"),
        pl.col("cleaned_normalized_intensity").alias("intensities2"),
        pl.col("precursor_mz").alias("precursor_mz2"),
    ).spectral_similarity.mass_weighted_cosine_similarity(
        ms2_tolerance_in_ppm=10.0,
        clean_spectra_first=True,
        ignore_precursor=True
    ).alias("cosine_score")
)
```

### Similarity Metric Comparison

Each metric has different characteristics:

| Metric | Best For | Range |
|--------|----------|-------|
| Dot Product | General purpose, NIST-style searches | 0-1 |
| Entropy Similarity | Noisy spectra, robust to intensity variations | 0-1 |
| Mass-Weighted Cosine | Emphasizing high-mass fragments | 0-1 |

## Step 4: Filter and Rank Results

### Filter by Similarity Threshold

```python
# Keep only high-confidence matches
high_confidence = similarity_results.filter(
    pl.col("dotprod_score") >= 0.7
)

print(f"Total matches: {len(similarity_results)}")
print(f"High confidence (>0.7): {len(high_confidence)}")
```

### Select Top Match per Query

Get the best library match for each query peak:

```python
top_matches = (
    similarity_results
    .filter(pl.col("dotprod_score") >= 0.6)  # Minimum threshold
    .sort(["Peak ID", "dotprod_score"], descending=[False, True])
    .group_by("Peak ID")
    .first()  # Take best match per query
)

print(f"Queries with matches: {len(top_matches)}")
print(top_matches.select([
    "Peak ID", "RT (min)", "name",
    "dotprod_score", "Precursor_mz_MSDIAL", "precursor_mz"
]).head(10))
```

### Rank by Combined Criteria

Combine multiple factors for scoring:

```python
ranked = similarity_results.with_columns(
    # Combined score: similarity + mass error + info score
    combined_score=(
        pl.col("dotprod_score") * 10.0  # Spectral similarity (0-10)
        - (pl.col("Precursor_mz_MSDIAL") / pl.col("precursor_mz") - 1.0).abs() * 1e6 / 10  # Mass error penalty
        + pl.col("spectral_information_score") / 10  # Library quality bonus
    )
).sort(["Peak ID", "combined_score"], descending=[False, True])

top_ranked = ranked.group_by("Peak ID").first()

print(top_ranked.select([
    "Peak ID", "name", "dotprod_score", "combined_score"
]).head(10))
```

## Step 5: Handle Edge Cases

### Null MS/MS Data

Some peaks may lack MS/MS spectra:

```python
# Filter out peaks without MS/MS
valid_matches = similarity_results.filter(
    pl.col("msms_m/z").is_not_null(),
    pl.col("dotprod_score").is_not_null()
)

print(f"Valid similarity calculations: {len(valid_matches)}/{len(similarity_results)}")
```

### Multiple Matches per Library Entry

One library compound may match multiple query peaks:

```python
# Group by library compound
library_hits = (
    high_confidence
    .group_by("name")
    .agg(
        pl.col("Peak ID").n_unique().alias("num_query_matches"),
        pl.col("dotprod_score").max().alias("best_score"),
        pl.col("dotprod_score").mean().alias("avg_score")
    )
    .sort("num_query_matches", descending=True)
)

print("Library compounds matching multiple queries:")
print(library_hits.head(10))
```

## Step 6: Export Results

Save search results for further analysis:

```python
# Export top matches
top_matches.select([
    "Peak ID",
    "RT (min)",
    "Precursor_mz_MSDIAL",
    "Height",
    "name",
    "molecular_formula",
    "inchikey",
    "precursor_mz",
    "dotprod_score",
    "entropy_score",
    "spectral_information_score"
]).write_parquet("similarity_search_results.parquet")

# Export summary CSV
top_matches.select([
    "Peak ID", "RT (min)", "name",
    "dotprod_score", "Precursor_mz_MSDIAL", "precursor_mz"
]).write_csv("similarity_matches_summary.csv")

print("Results exported")
```

## Complete Example

Full workflow from loading data to exporting results:

```python
from pathlib import Path
import polars as pl
from hrms_utils.formats.msdial import get_chromatogram
from hrms_utils.formats.nist_mspec import read_MSPEC_file

# Configure display
pl.Config.set_tbl_rows(20)

# 1. Load data
print("Loading data...")
query = get_chromatogram("tests/data/MSDIAL_output.txt")
library = read_MSPEC_file("tests/data/msp_sample.msp")

# Filter library
library_clean = library.filter(
    pl.col("spectral_information_score") > 5.0,
    pl.col("num_peaks_cleaned") >= 3
)

print(f"Query: {len(query)} peaks")
print(f"Library: {len(library_clean)} spectra")

# 2. Precursor matching
print("\nMatching precursors...")
precursor_tolerance_ppm = 10.0
matches = query.lazy().join_where(
    library_clean.lazy(),
    (pl.col("Precursor_mz_MSDIAL") / pl.col("precursor_mz") - 1.0).abs() 
        <= (precursor_tolerance_ppm * 1e-6),
    suffix="_lib"
).collect()

print(f"Precursor matches: {len(matches)}")

# 3. Compute similarity
print("\nComputing spectral similarity...")
similarity_results = matches.with_columns(
    pl.struct(
        pl.col("msms_m/z").alias("mz1"),
        pl.col("msms_intensity").alias("intensities1"),
        pl.col("Precursor_mz_MSDIAL").alias("precursor_mz1"),
        pl.col("cleaned_normalized_mz").alias("mz2"),
        pl.col("cleaned_normalized_intensity").alias("intensities2"),
        pl.col("precursor_mz").alias("precursor_mz2"),
    ).spectral_similarity.dotprod_similarity(
        ms2_tolerance_in_ppm=10.0,
        clean_spectra_first=True,
        ignore_precursor=True
    ).alias("similarity_score")
)

# 4. Filter and rank
print("\nFiltering results...")
top_matches = (
    similarity_results
    .filter(
        pl.col("similarity_score").is_not_null(),
        pl.col("similarity_score") >= 0.7
    )
    .sort(["Peak ID", "similarity_score"], descending=[False, True])
    .group_by("Peak ID").first()
)

print(f"High-confidence matches: {len(top_matches)}")

# 5. Display results
print("\nTop matches:")
print(top_matches.select([
    "Peak ID", "RT (min)", "name",
    "similarity_score", "Precursor_mz_MSDIAL", "precursor_mz"
]).head(10))

# 6. Export
top_matches.write_parquet("search_results.parquet")
print("\nResults saved to search_results.parquet")
```

## Performance Optimization

### Use LazyFrames for Large Searches

```python
# Process in streaming mode for large datasets
similarity_lazy = (
    query.lazy()
    .join_where(
        library_clean.lazy(),
        (pl.col("Precursor_mz_MSDIAL") / pl.col("precursor_mz") - 1.0).abs() <= 10e-6,
        suffix="_lib"
    )
    .with_columns(
        pl.struct(...).spectral_similarity.dotprod_similarity(...).alias("score")
    )
    .filter(pl.col("score") >= 0.7)
    .collect(streaming=True)
)
```

### GPU-Accelerated Searches

For very large libraries (>100k spectra), use the `fast_cosine_sim` package:

```python
from fast_cosine_sim import (
    compute_gpu_batched_approximate_similarity_pairs,
    ApproximateGpuBatchedSimilarityConfig
)

# See experiments/spectral_information/similarity_vs_info.py for complete example
# This uses GPU acceleration for massive speedup on large datasets
```

### Batch Processing

Process multiple chromatograms against the same library:

```python
# Load multiple query files
query_files = ["sample1.txt", "sample2.txt", "sample3.txt"]
all_results = []

for query_file in query_files:
    query = get_chromatogram(query_file)
    
    # Run search
    matches = query.lazy().join_where(...).collect()
    similarity = matches.with_columns(...)
    top = similarity.filter(...).group_by("Peak ID").first()
    
    # Add source file info
    top = top.with_columns(pl.lit(query_file).alias("source_file"))
    all_results.append(top)

# Combine all results
combined = pl.concat(all_results)
combined.write_parquet("batch_search_results.parquet")
```

## Tips and Best Practices

1. **Precursor tolerance**: Use 3-5 ppm for Orbitrap, 5-10 ppm for TOF, 10-20 ppm for lower-resolution instruments

2. **MS/MS tolerance**: Similar to precursor tolerance, but can be slightly larger

3. **Clean spectra first**: Set `clean_spectra_first=True` to remove noise before comparison

4. **Ignore precursor**: Set `ignore_precursor=True` to exclude precursor ion from similarity calculation

5. **Filter library by quality**: Remove low-quality library spectra before searching to reduce false positives

6. **Combine multiple metrics**: Use both dot product and entropy similarity for validation

7. **Mass error check**: Even with good spectral match, large mass errors indicate incorrect matches

8. **Use LazyFrames**: For large searches, LazyFrames with streaming collection save memory

## Next Steps

- [How-To: GPU Acceleration](../how-to/gpu-acceleration.md) - Scale to millions of comparisons
- [How-To: Batch Processing](../how-to/batch-processing.md) - Process multiple samples efficiently
- [API Reference: Spectral Similarity](../reference/api/hrms_core.md#spectral-similarity) - Detailed metric documentation
