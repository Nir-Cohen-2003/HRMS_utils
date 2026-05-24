# HRMS Utils Documentation

Documentation for HRMS Utils, built with MkDocs and Material theme.

## Building and Viewing Docs

### Prerequisites

Install documentation dependencies:

```bash
pixi install -e docs
```

### Local Development

Serve documentation locally with live reload:

```bash
pixi run -e docs docs-serve
```

Then open http://127.0.0.1:8000 in your browser. Changes to markdown files will automatically reload.

### Build Static Site

Build the documentation as static HTML:

```bash
pixi run -e docs docs-build
```

Output will be in the `site/` directory.

### Deploy to GitHub Pages

Deploy documentation to GitHub Pages:

```bash
pixi run -e docs docs-deploy
```

This will:
1. Build the docs
2. Push to the `gh-pages` branch
3. Make docs available at https://nir-cohen-2003.github.io/HRMS_utils/

## Documentation Structure

```
docs/
├── index.md                          # Landing page
├── getting-started/
│   ├── installation.md               # Installation guide
│   └── quickstart.md                 # 5-minute quickstart
├── tutorials/
│   ├── 01-msdial-chromatogram-annotation.md    # MSDIAL workflow
│   ├── 02-msp-library-processing.md            # MSP library processing
│   └── 03-spectral-similarity-search.md        # Similarity search
├── how-to/
│   ├── custom-tolerances.md          # Setting mass/RT tolerances
│   ├── blank-subtraction.md          # Blank removal
│   ├── batch-processing.md           # Large-scale processing
│   ├── export-results.md             # Saving results
│   └── gpu-acceleration.md           # GPU-accelerated searches
├── reference/
│   ├── index.md                      # API overview
│   ├── api/
│   │   ├── hrms_core.md             # Auto-generated from docstrings
│   │   ├── formats.md               # Auto-generated from docstrings
│   │   ├── formula_annotation.md    # Auto-generated from docstrings

│   └── data-structures.md           # DataFrame schemas
├── explanation/
│   ├── architecture.md              # System overview
│   ├── mass-decomposition-algorithm.md
│   ├── spectral-entropy.md
│   ├── isotopic-patterns.md
│   └── polars-plugins.md
└── contributing/
    ├── development.md               # Developer setup
    └── writing-plugins.md           # Extending with plugins
```

## Writing Documentation

### Adding New Pages

1. Create markdown file in appropriate directory
2. Add to `mkdocs.yml` navigation
3. Use relative links to other pages

### API Documentation

API reference pages use mkdocstrings to auto-extract from docstrings:

```markdown
::: hrms_utils.hrms_core.read_mzml
    options:
      show_root_heading: false
```

The `:::` syntax tells mkdocstrings to extract documentation from the specified module/function.

### Code Examples

Use fenced code blocks with language specification:

````markdown
```python
import polars as pl
from hrms_utils import hrms_core

# Your code here
```
````

### Admonitions

Use admonitions for notes, warnings, tips:

```markdown
!!! note
    This is a note

!!! warning
    This is a warning

!!! tip
    This is a tip
```

### Math Equations

Use LaTeX for equations (MathJax rendering):

```markdown
Inline math: \(E = mc^2\)

Display math:
\[
\text{ppm error} = \frac{|m_{\text{obs}} - m_{\text{calc}}|}{m_{\text{calc}}} \times 10^6
\]
```

## Docstring Style

HRMS Utils uses Google-style docstrings. Example:

```python
def decompose_mass(
    self,
    tolerance_ppm: float = 5.0,
    min_dbe: float = 0.0,
    max_dbe: float = 40.0,
) -> pl.Expr:
    """
    Decompose a mass into possible chemical formulas.
    
    Args:
        tolerance_ppm: The mass tolerance in ppm.
        min_dbe: The minimum degree of unsaturation.
        max_dbe: The maximum degree of unsaturation.
    
    Returns:
        A Polars expression with the decomposition results.
    """
```

## Configuration

Documentation is configured in `mkdocs.yml`. Key sections:

- `theme`: Material theme settings
- `plugins`: mkdocstrings configuration
- `markdown_extensions`: Enabled markdown features
- `nav`: Navigation structure

## Deployment

### GitHub Actions (Recommended)

Add `.github/workflows/docs.yml`:

```yaml
name: Documentation

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: 3.12
      - run: pip install mkdocs mkdocs-material 'mkdocstrings[python]' mkdocstrings-python
      - run: mkdocs gh-deploy --force
```

### Manual Deployment

```bash
pixi run -e docs docs-deploy
```

## Troubleshooting

### mkdocstrings can't find modules

Ensure `src` directory is in the Python path. In `mkdocs.yml`:

```yaml
plugins:
  - mkdocstrings:
      handlers:
        python:
          paths: [src]
```

### Live reload not working

Check that you're running from the project root and port 8000 is available:

```bash
# From project root
pixi run -e docs docs-serve

# Use different port if needed
pixi run -e docs mkdocs serve --dev-addr localhost:8001
```

## Contributing to Docs

1. Create/edit markdown files in `docs/`
2. Test locally with `pixi run -e docs docs-serve`
3. Commit changes
4. Documentation will auto-deploy on push to main (if GitHub Actions configured)

For questions, see [Contributing Guide](contributing/development.md).
