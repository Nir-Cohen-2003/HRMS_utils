# HRMS Core API

Core functionality for high-resolution mass spectrometry data processing, implemented as Polars expression plugins.

## Overview

The `hrms_core` module provides Polars expression plugins for:

- Mass decomposition (exact mass → candidate formulas)
- Spectral similarity comparison
- Spectral information scoring
- Isotopic pattern analysis
- mzML file reading

All plugins are accessed through Polars expression namespaces:

```python
import polars as pl
from hrms_utils import hrms_core

# Mass decomposition namespace
df.with_columns(
    pl.col("mass").mass_decomposition.decompose_mass(...)
)

# Spectral similarity namespace
df.with_columns(
    pl.col("spectra").spectral_similarity.dotprod_similarity(...)
)
```

## Mass Decomposition

::: hrms_utils.hrms_core.MassDecomposerUtils
    options:
      show_root_heading: true
      members:
        - decompose_mass
        - decompose_mass_with_bounds
        - clean_and_normalize_spectrum
        - deduce_isotopic_pattern

## Spectral Similarity

::: hrms_utils.hrms_core.SpectralUtils
    options:
      show_root_heading: true
      members:
        - entropy_similarity
        - general_cosine_similarity
        - mass_weighted_cosine_similarity
        - dotprod_similarity
        - explained_intensity

## Spectral Information

::: hrms_utils.hrms_core.SpectralInfoNamespace
    options:
      show_root_heading: true
      members:
        - spectral_info_score

## File I/O

::: hrms_utils.hrms_core.read_mzml

::: hrms_utils.hrms_core.read_thermo

## Module Constants

::: hrms_utils.hrms_core.__version__
    options:
      show_root_heading: false

::: hrms_utils.hrms_core.NUM_ELEMENTS
    options:
      show_root_heading: false

## See Also

- [Tutorial: Mass Decomposition](../tutorials/04-mass-decomposition.md)
- [Tutorial: Spectral Similarity](../tutorials/03-spectral-similarity-search.md)
- [Explanation: Mass Decomposition Algorithm](../explanation/mass-decomposition-algorithm.md)
