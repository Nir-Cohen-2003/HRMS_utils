# Mass Decomposer C++ Python API Documentation

This document describes the Python API for the C++-accelerated mass decomposition functions provided by `mass_decomposer_cpp`.

## Function List

### 1. `decompose_mass`
Decompose a single target mass into all possible molecular formulas within the specified constraints.

**Signature:**
```python
def decompose_mass(
    target_mass: float,
    element_bounds: dict[str, tuple[int, int]],
    strategy: str = "recursive",
    tolerance_ppm: float = 5.0,
    min_dbe: float = 0.0,
    max_dbe: float = 40.0,
    max_hetero_ratio: float = 100.0,
    max_results: int = 100000
) -> list[dict[str, int]]
```

**Arguments:**
- `target_mass`: The mass to decompose.
- `element_bounds`: Dictionary mapping element symbols to (min_count, max_count) bounds, e.g. `{'C': (0, 50), 'H': (0, 100)}`.
- `strategy`: Decomposition algorithm, either `"recursive"` or `"money_changing"`.
- `tolerance_ppm`: Allowed mass error in parts per million.
- `min_dbe`: Minimum double bond equivalent (DBE) allowed.
- `max_dbe`: Maximum DBE allowed.
- `max_hetero_ratio`: Maximum allowed ratio of heteroatoms to carbon.
- `max_results`: Maximum number of formulas to return.

**Returns:**
- List of formulas as dictionaries mapping element symbols to counts.

---

### 2. `decompose_mass_parallel`
Decompose multiple target masses in parallel, each with the same element bounds and constraints.

**Signature:**
```python
def decompose_mass_parallel(
    target_masses: list[float],
    element_bounds: dict[str, tuple[int, int]],
    strategy: str = "recursive",
    tolerance_ppm: float = 5.0,
    min_dbe: float = 0.0,
    max_dbe: float = 40.0,
    max_hetero_ratio: float = 100.0,
    max_results: int = 100000
) -> list[list[dict[str, int]]]
```

**Arguments:**
- `target_masses`: List of masses to decompose.
- `element_bounds`: See above.
- `strategy`: See above.
- `tolerance_ppm`: See above.
- `min_dbe`: See above.
- `max_dbe`: See above.
- `max_hetero_ratio`: See above.
- `max_results`: See above.

**Returns:**
- List of results for each mass, each being a list of formulas.

---

### 3. `decompose_spectrum`
Decompose a spectrum: find all possible precursor formulas for a given precursor mass, and for each, decompose the fragment masses as subsets of the precursor.

**Signature:**
```python
def decompose_spectrum(
    precursor_mass: float,
    fragment_masses: list[float],
    element_bounds: dict[str, tuple[int, int]],
    strategy: str = "recursive",
    tolerance_ppm: float = 5.0,
    min_dbe: float = 0.0,
    max_dbe: float = 40.0,
    max_hetero_ratio: float = 100.0,
    max_results: int = 100000
) -> list[dict]
```

**Arguments:**
- `precursor_mass`: The precursor ion mass.
- `fragment_masses`: List of fragment masses to decompose.
- `element_bounds`: See above.
- `strategy`: See above.
- `tolerance_ppm`: See above.
- `min_dbe`: See above.
- `max_dbe`: See above.
- `max_hetero_ratio`: See above.
- `max_results`: See above.

**Returns:**
- List of precursor/fragment decomposition results. Each dict contains:
    - `'precursor'`: dict of precursor formula,
    - `'fragments'`: list of lists of fragment formulas,
    - `'precursor_mass'`: float,
    - `'precursor_error_ppm'`: float,
    - `'fragment_masses'`: list of lists of floats,
    - `'fragment_errors_ppm'`: list of lists of floats.

---

### 4. `decompose_spectra_parallel`
Decompose multiple spectra in parallel. Each spectrum is defined by a precursor mass and a list of fragment masses.

**Signature:**
```python
def decompose_spectra_parallel(
    spectra_data: list[tuple[float, list[float]]],
    element_bounds: dict[str, tuple[int, int]],
    strategy: str = "recursive",
    tolerance_ppm: float = 5.0,
    min_dbe: float = 0.0,
    max_dbe: float = 40.0,
    max_hetero_ratio: float = 100.0,
    max_results: int = 100000
) -> list[list[dict]]
```

**Arguments:**
- `spectra_data`: List of spectra, each as `(precursor_mass, fragment_masses)`.
- `element_bounds`: See above.
- `strategy`: See above.
- `tolerance_ppm`: See above.
- `min_dbe`: See above.
- `max_dbe`: See above.
- `max_hetero_ratio`: See above.
- `max_results`: See above.

**Returns:**
- List of results for each spectrum (see `decompose_spectrum` for result structure).

---

### 5. `decompose_spectrum_known_precursor`
Decompose fragment masses for a spectrum where the precursor formula is already known.

**Signature:**
```python
def decompose_spectrum_known_precursor(
    precursor_formula: dict[str, int],
    fragment_masses: list[float],
    element_bounds: dict[str, tuple[int, int]],
    strategy: str = "recursive",
    tolerance_ppm: float = 5.0,
    min_dbe: float = 0.0,
    max_dbe: float = 40.0,
    max_hetero_ratio: float = 100.0,
    max_results: int = 100000
) -> list[list[dict[str, int]]]
```

**Arguments:**
- `precursor_formula`: The known precursor formula as a dictionary.
- `fragment_masses`: List of fragment masses to decompose.
- `element_bounds`: See above.
- `strategy`: See above.
- `tolerance_ppm`: See above.
- `min_dbe`: See above.
- `max_dbe`: See above.
- `max_hetero_ratio`: See above.
- `max_results`: See above.

**Returns:**
- List of fragment results, one per fragment mass, each a list of formulas.

---

### 6. `decompose_spectra_known_precursor_parallel`
Decompose multiple spectra in parallel, where each spectrum has a known precursor formula and a list of fragment masses.

**Signature:**
```python
def decompose_spectra_known_precursor_parallel(
    spectra_data: list[tuple[dict[str, int], list[float]]],
    element_bounds: dict[str, tuple[int, int]],
    strategy: str = "recursive",
    tolerance_ppm: float = 5.0,
    min_dbe: float = 0.0,
    max_dbe: float = 40.0,
    max_hetero_ratio: float = 100.0,
    max_results: int = 100000
) -> list[list[list[dict[str, int]]]]
```

**Arguments:**
- `spectra_data`: List of spectra, each as `(precursor_formula, fragment_masses)`.
- `element_bounds`: See above.
- `strategy`: See above.
- `tolerance_ppm`: See above.
- `min_dbe`: See above.
- `max_dbe`: See above.
- `max_hetero_ratio`: See above.
- `max_results`: See above.

**Returns:**
- List of results for each spectrum; for each spectrum, a list of fragment results (as in `decompose_spectrum_known_precursor`).
