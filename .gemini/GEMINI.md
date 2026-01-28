# Copilot / Assistant Guidelines

Concise, actionable rules for editing this repository. Keep changes explicit, fail fast on missing resources, and prefer clarity over cleverness.

---

## Project Context
- **Domain:** Utilities for high-resolution mass spectrometry (HRMS) data.
- **Architecture:** The codebase relies heavily on **Polars dataframes**. Spectra are often stored as nested datatypes within these dataframes.
- **Environment:** This project uses **Pixi** for dependency management. Always use `pixi run` or the environment defined in `pixi.toml` to execute files.

---

## Core Rules (Highest Priority)

### 1. Performance & Dataframes
- **Polars Only:** Use `polars` for all dataframe work. Never use pandas.
- **Vectorization:** In performance-critical code (iterating over spectra, masses, or large datasets), **avoid Python loops**.
- **Acceleration:** Use accelerated libraries for heavy operations: Polars expressions, Numpy, Cupy, or custom extensions.

### 2. Architecture & execution
- **Breaking Changes:** If you introduce a breaking change, update **every** impacted part of the codebase immediately. Do not leave "fallbacks" or legacy support code.
- **Pixi:** Do not  run python files directly. run them via `pixi`, e.g., `pixi run -e experiments python python_file.py`, substitute the appropriate environment.

### 3. Simplicity & Scope
- **Simple Code:** Avoid `try/catch` blocks unless strictly necessary. Failures should be loud, especially in experimental code.
- **Testing & Docs:** Add or update unit tests and documentation **only when explicitly instructed**.

---

## Coding Style & Standards

### Type Hints & Shapes
- Use explicit type hints everywhere (standard `typing` + `np.typing`).
- **Configs:** Always use `dataclasses` for configuration objects.
- **Array Shapes:** Explicitly document the shape of tensors/arrays in comments where ambiguity exists.
  - *Example:* `# features: np.ndarray(shape=(n_spectra, n_fragments, 2))`

### Structure & Naming
- **Naming:** Use long, descriptive names. No abbreviations (e.g., use `compute_similarity` instead of `compSim`).
- **Functions:** Avoid nested functions. Define helpers as private methods or standalone functions.
- **Comments:** Explain *why* a decision was made, not *what* the code does.

---

## Error Handling

### Fail Fast
- **No Defaults:** Do not silently fallback to default values if a required resource or config is missing. Raise an error immediately.
- **Assertions:** When using assertions, include a clear message explaining what went wrong and what the expected state was.
  - *Bad:* `assert len(x) == 5`
  - *Good:* `assert len(x) == 5, f"Expected 5 items, got {len(x)}"`
