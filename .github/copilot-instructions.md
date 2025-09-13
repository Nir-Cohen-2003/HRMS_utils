# Copilot / Assistant Guidelines

Concise, actionable rules for editing this repository. Keep changes explicit, fail fast on missing resources, and prefer clarity over cleverness.

---

## Project context (why this matters)
- This is a project containing utils for working with high-resolution mass spectrometry (HRMS) data, including formula annotation, conversion between formats, spectra and data cleaning.
- the codebase works with polars dataframes as the major entity, with spectra stored as nested datatypes inside of polars dataframes.
---

## Core rules (highest priority)
- Use Polars for all dataframe work. Example: `import polars as pl` and use `pl.DataFrame`, `pl.read_csv`, etc.
- Use dataclasses for configs. Include explicit type hints (typing + np.typing.NDArray) and document array/tensor shapes.
- If you introduce a breaking change in one module, update every dependent module to match the new contract. Remove legacy fallbacks unless backward-compatibility is explicitly requested.
- Fail fast when a required resource, file, or configuration is missing (raise AssertionError or custom exception).
- Comments should explain *why* a decision was made or why the code is required, not restate *what* the code does.
- Use clear, descriptive names (no abbreviations). Prefer longer names that convey purpose.
- Avoid nested functions. Define helpers as private methods on a class or as standalone functions in a module.
- Use type hints everywhere. Avoid `Any` unless absolutely necessary, and document why if you do.
- When needing to accelerate function execution, use numba jit compilation and vectorization unless there's a compelling reason to use multiprocessing or threading, in which case document the reason and use processpoolexecutor/threadpoolexecutor with clear comments.
- when needing even more performance, use c++ with cython as compatibility layer, and edit the top level setup.py to include the new compiled extension. When working with code that is already in cython/c++, prefer keeping ocre logic in c++ and only use cython as a thin compatibility layer.
- when using cython, put all cdef declaration at the top of the function, and use cdef types for all variables that can be typed.
- when using assertions, include a message that explains what went wrong and how.
---

## Practical patterns and examples

Polars
```python
# Why: Polars improves performance and memory consistency across the codebase.
import polars as pl
df: pl.DataFrame = pl.read_csv("data.csv")
```

HRMS and domain libraries
- Use hrms_utils for parsing/standardizing spectra. Prefer its utilities over ad-hoc parsing.
- Document any domain-specific assumptions (e.g., mass tolerance, ionization mode) in code comments.

Array shape comments
- Always document shapes where ambiguity exists.
  - Good: `# features: np.ndarray(shape=(n_spectra, n_fragments, 2))`

Type hints
- Use typing and numpy typing.
- Use explicit return types for functions.
- Avoid Any unless unavoidable; document why.

Naming
- Use long, descriptive names. E.g.:
  - good: `compute_mass_spectrum_similarity_score`
  - bad: `compSim`

Breaking-change propagation
- If you change a function signature or return type, update:
  - callers in the same package
  - tests
  - any serialization/deserialization code
- Remove code that handled the old contract unless the change request explicitly asked for backward compatibility.

assertion examples
bad: 
if precursor.ndim != 1:
  raise ValueError("precursor must be a 1D array")
good:
assert precursor.ndim == 1, f"precursor must be a 1D array, got {precursor.ndim}D array instead"
---

## Testing and CI
- Add/update unit tests for any changed contract.
- Use polars in tests that exercise dataframe logic.
- Tests should assert shapes and types where relevant (e.g., tensor shapes produced by dataset/collate logic).

---

## Commenting style
- Explain the rationale, trade-offs, and non-obvious constraints.
  - Good: `# Why: use deterministic seed so training runs are reproducible across CI`
  - Bad: `# sets random seed`

---

## Pull request checklist
- All changed modules that depend on modified logic are updated.
- Unit tests updated/added for new contracts.
- No silent fallbacks left behind.
- Dataframe code uses Polars.
- Configs are dataclasses with type hints and documented shapes.
- Domain-specific settings (HRMS tolerances, ionization mode) are documented and validated.

---

## When in doubt
- Prefer clarity and explicit contracts that make reasoning about data and errors straightforward.
- Ask the reviewer if backward compatibility is required before adding fallbacks.
- If a required dependency or config is missing, fail fast and explain the missing requirement in the exception message.