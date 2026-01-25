# HRMS Utils

**High-resolution mass spectrometry utilities for Python**

HRMS Utils is a Python library for processing, annotating, and analyzing high-resolution mass spectrometry (HRMS) data. Built on Polars DataFrames with Rust-accelerated plugins, it provides fast and memory-efficient workflows for metabolomics and small molecule analysis.

## Key Features

- **Mass Decomposition**: Generate candidate elemental formulas from exact masses with customizable constraints
- **Spectral Similarity**: Compare MS/MS spectra using entropy similarity, cosine similarity, and dot product metrics
- **Isotopic Pattern Analysis**: Deduce elemental composition constraints from MS1 isotope distributions
- **Format Support**: Read and process MSDIAL chromatograms, MSP/MSPEC libraries, MGF files, and mzML data
- **Polars Integration**: Native Polars DataFrame operations with expression plugins for seamless data manipulation
- **Rust Performance**: Critical algorithms implemented in Rust for speed and memory efficiency

## Quick Example

```python
import polars as pl
from hrms_utils.formats.msdial import get_chromatogram, annotate_chromatogram_with_formulas

# Read MSDIAL chromatogram
chromatogram = get_chromatogram("sample.txt")

# Annotate with candidate formulas
annotated = annotate_chromatogram_with_formulas(
    chromatogram,
    precursor_mass_accuracy_ppm=3.0,
    fragment_mass_accuracy_ppm=5.0
)

# Explore results
print(annotated.select([
    "Peak ID", "RT (min)", "precursor_formula_str", 
    "cleaned_msms_mz", "spectral_information_score"
]))
```

## Typical Workflows

### 1. MSDIAL Chromatogram Annotation
Read chromatogram data from MSDIAL, subtract blanks, and annotate peaks with candidate formulas using isotopic patterns and MS/MS fragmentation.

[→ Tutorial: MSDIAL Chromatogram Annotation](tutorials/01-msdial-chromatogram-annotation.md)

### 2. MSP Library Processing
Load and clean spectral libraries in MSP/MSPEC format, filter by quality metrics, and prepare for similarity searches.

[→ Tutorial: MSP Library Processing](tutorials/02-msp-library-processing.md)

### 3. Spectral Similarity Search
Match query spectra against reference libraries using precursor mass filtering and spectral similarity scoring.

[→ Tutorial: Spectral Similarity Search](tutorials/03-spectral-similarity-search.md)

## Why HRMS Utils?

**Polars-Native**: All data operations use Polars DataFrames for consistent, fast data manipulation. No need to convert between formats.

**Type-Safe**: Full type hints and `.pyi` stubs for excellent IDE support and type checking.

**Domain-Specific**: Designed specifically for HRMS workflows with functions that understand mass spectrometry concepts (ppm tolerances, isotope patterns, fragmentation).

**Extensible**: Built on Polars expression plugins, making it easy to add custom algorithms as Rust extensions.

## Getting Started

New to HRMS Utils? Start here:

1. [**Installation**](getting-started/installation.md) - Install via conda, pip, or build from source
2. [**Quickstart**](getting-started/quickstart.md) - 5-minute introduction to core concepts
3. [**Tutorials**](tutorials/01-msdial-chromatogram-annotation.md) - Step-by-step guides for common workflows

## Project Status

HRMS Utils is under active development. The API is stabilizing but may have breaking changes in minor versions until 1.0 release.

## License

Licensed under the Apache License 2.0. See [LICENSE](https://github.com/Nir-Cohen-2003/HRMS_utils/blob/main/LICENSE) for details.

## Contributing

Contributions are welcome! See the [Development Guide](contributing/development.md) for setup instructions.
