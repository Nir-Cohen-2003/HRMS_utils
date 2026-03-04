# API Reference

Complete API documentation for HRMS Utils, automatically generated from docstrings.

## Modules

HRMS Utils is organized into several modules:

### [HRMS Core](api/hrms_core.md)
Core functionality implemented as Polars expression plugins:

- **Mass Decomposition**: Generate candidate formulas from exact masses
- **Spectral Similarity**: Compare MS/MS spectra using multiple metrics
- **Spectral Information**: Calculate spectral entropy and information scores
- **Isotopic Patterns**: Deduce elemental constraints from MS1 isotopes
- **File I/O**: Read mzML files

### [Formats](api/formats.md)
Parsers and readers for mass spectrometry data formats:

- **MSDIAL**: Read and process MSDIAL chromatograms
- **MSP/MSPEC**: Read NIST spectral libraries
- **MGF**: Read Mascot Generic Format files
- **mzML**: Read mzML files (via core module)

### [Formula Annotation](api/formula_annotation.md)
Tools for working with chemical formulas:

- **Element Tables**: Periodic table data, masses, and adduct definitions
- **Utilities**: Convert formulas between string and array representations

### [RDKit Utilities](api/rdkit.md)
RDKit integration for molecular structure operations:

- **Molecular Fingerprints**: Generate molecular fingerprints
- **Fragmentation**: In-silico fragmentation predictions
- **Structure Utilities**: SMILES/InChI conversions

## Data Structures

See [Data Structures](data-structures.md) for details on the Polars DataFrame schemas used throughout HRMS Utils.

## Quick Links

### Common Operations

- [Decompose mass to formulas](api/hrms_core.md#hrms_utils.hrms_core.MassDecomposerUtils.decompose_mass)
- [Compute spectral similarity](api/hrms_core.md#hrms_utils.hrms_core.SpectralUtils.dotprod_similarity)
- [Read MSDIAL chromatogram](api/formats.md#hrms_utils.formats.msdial.get_chromatogram)
- [Read MSP library](api/formats.md#hrms_utils.formats.spectral_library.process_single_file)

### Advanced Features

- [Deduce isotopic pattern constraints](api/hrms_core.md#hrms_utils.hrms_core.MassDecomposerUtils.deduce_isotopic_pattern)
- [Clean and normalize spectrum](api/hrms_core.md#hrms_utils.hrms_core.MassDecomposerUtils.clean_and_normalize_spectrum)
- [Spectral information score](api/hrms_core.md#hrms_utils.hrms_core.SpectralInfoNamespace.spectral_info_score)

## Usage Patterns

### Polars Expression Plugins

HRMS Core functions are exposed as Polars expression methods via namespaces:

```python
import polars as pl
from hrms_utils import hrms_core

# Mass decomposition namespace
df.with_columns(
    formulas=pl.col("mass").mass_decomposition.decompose_mass(...)
)

# Spectral similarity namespace
df.with_columns(
    similarity=pl.col("spectra").spectral_similarity.dotprod_similarity(...)
)

# Spectral information namespace
df.with_columns(
    info_score=pl.col("spectrum").spectral_info.spectral_info_score(...)
)
```

### Regular Functions

Format parsers and utilities are standard Python functions:

```python
from hrms_utils.formats import msdial, spectral_library

# Returns Polars DataFrame
chromatogram = msdial.get_chromatogram("file.txt")

# Returns Polars DataFrame
library = spectral_library.process_single_file("library.msp")
```

## Type Hints

All functions have complete type hints. Use your IDE's autocomplete and type checking features:

```python
import polars as pl
from hrms_utils.formats.msdial import get_chromatogram

# IDE will show:
# def get_chromatogram(path: str | Path) -> pl.DataFrame: ...
chromatogram = get_chromatogram("data.txt")
```

## See Also

- [Tutorials](../tutorials/01-msdial-chromatogram-annotation.md) - Step-by-step guides
- [How-To Guides](../how-to/custom-tolerances.md) - Solutions to specific problems
- [Explanation](../explanation/architecture.md) - Deep dives into concepts
