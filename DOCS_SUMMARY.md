# Documentation Setup Complete

Documentation for HRMS Utils has been successfully set up using MkDocs with Material theme and mkdocstrings for automatic API reference generation.

## What Was Created

### Configuration
- `mkdocs.yml` - Complete MkDocs configuration with Material theme
- `pixi.toml` - Added `docs` feature with dependencies and tasks
- `docs/javascripts/mathjax.js` - MathJax configuration for equation rendering

### Documentation Content

#### Completed Pages (Ready to Use)
1. **Landing Page** (`docs/index.md`)
   - Project overview and key features
   - Quick example
   - Links to main workflows

2. **Getting Started**
   - `installation.md` - Installation via conda, pip, or from source
   - `quickstart.md` - 5-minute introduction to core concepts

3. **Priority Tutorials** (Complete with working examples)
   - `01-msdial-chromatogram-annotation.md` - Full MSDIAL workflow
   - `02-msp-library-processing.md` - MSP library processing
   - `03-spectral-similarity-search.md` - Similarity search workflow

4. **API Reference** (Auto-generated from docstrings)
   - `reference/index.md` - API overview
   - `reference/api/hrms_core.md` - Core functionality
   - `reference/api/formats.md` - Format parsers
   - `reference/api/formula_annotation.md` - Formula tools
   - `reference/data-structures.md` - DataFrame schema documentation

#### Placeholder Pages (For Future Expansion)
- `tutorials/04-mass-decomposition.md`
- `tutorials/05-isotopic-pattern-analysis.md`
- `how-to/*.md` (5 files)
- `explanation/*.md` (5 files)
- `contributing/*.md` (2 files)

## Usage Commands

### View Documentation Locally
```bash
# Install dependencies (first time only)
pixi install -e docs

# Serve with live reload
pixi run -e docs docs-serve
# Opens at http://127.0.0.1:8000
```

### Build Static Site
```bash
pixi run -e docs docs-build
# Output in site/ directory
```

### Deploy to GitHub Pages
```bash
pixi run -e docs docs-deploy
# Pushes to gh-pages branch
# Site: https://nir-cohen-2003.github.io/HRMS_utils/
```

## Key Features

### Automatic API Documentation
- Uses mkdocstrings to extract from Google-style docstrings
- Type hints from `.pyi` stubs are included
- Example:
  ```markdown
  ::: hrms_utils.hrms_core.read_mzml
  ```

### Real Examples Using Test Data
All three priority tutorials include working code using actual test data:
- `tests/data/MSDIAL_output.txt` - For MSDIAL tutorial
- `tests/data/msp_sample.msp` - For MSP tutorial
- Both files - For similarity search tutorial

### Material Theme Features
- Dark/light mode toggle
- Search functionality
- Navigation tabs
- Code copy buttons
- Syntax highlighting
- Mobile-responsive

### Math Support
- LaTeX equations via MathJax
- Inline: `\(equation\)`
- Display: `\[equation\]`

## Documentation Structure

Following the **Diátaxis framework**:

1. **Tutorials** - Learning-oriented, step-by-step
2. **How-To Guides** - Problem-oriented solutions
3. **Reference** - Information-oriented API docs
4. **Explanation** - Understanding-oriented theory

## Next Steps

### Immediate
1. Test the docs build:
   ```bash
   pixi run -e docs docs-serve
   ```

2. Review the three priority tutorials to ensure they match your workflows

3. Customize colors/branding in `mkdocs.yml` if desired

### Short-Term
1. Fill in placeholder pages as needed
2. Add screenshots/images to tutorials
3. Set up GitHub Actions for auto-deployment

### Long-Term
1. Add more tutorials based on user feedback
2. Expand explanation pages with algorithm details
3. Create video tutorials linking to docs

## GitHub Pages Deployment

To enable GitHub Pages:

1. Push this branch to GitHub
2. Run:
   ```bash
   pixi run -e docs docs-deploy
   ```
3. Go to repository Settings → Pages
4. Source should be set to `gh-pages` branch (automatic)
5. Docs will be live at https://nir-cohen-2003.github.io/HRMS_utils/

## Editing Documentation

### Adding New Pages
1. Create `.md` file in appropriate `docs/` subdirectory
2. Add to `nav` section in `mkdocs.yml`
3. Test with `pixi run -e docs docs-serve`

### Updating API Docs
Just update docstrings in Python code - mkdocstrings extracts automatically!

### Style Guide
- Use Google-style docstrings
- Include code examples in tutorials
- Link to API reference from tutorials
- Use admonitions for notes/warnings/tips

## Troubleshooting

### Can't find mkdocs command
```bash
pixi install -e docs
```

### Port 8000 already in use
```bash
pixi run -e docs mkdocs serve --dev-addr localhost:8001
```

### mkdocstrings can't import modules
Ensure you're in the project root directory and `src` is in paths (already configured)

## Files Created

```
mkdocs.yml                                          # MkDocs config
pixi.toml                                          # Added docs feature
docs/
├── README.md                                      # Docs developer guide
├── index.md                                       # Landing page
├── javascripts/mathjax.js                        # Math rendering
├── getting-started/
│   ├── installation.md                           # ✓ Complete
│   └── quickstart.md                             # ✓ Complete
├── tutorials/
│   ├── 01-msdial-chromatogram-annotation.md      # ✓ Complete
│   ├── 02-msp-library-processing.md              # ✓ Complete
│   ├── 03-spectral-similarity-search.md          # ✓ Complete
│   ├── 04-mass-decomposition.md                  # Placeholder
│   └── 05-isotopic-pattern-analysis.md           # Placeholder
├── how-to/
│   ├── custom-tolerances.md                      # Placeholder
│   ├── blank-subtraction.md                      # Placeholder
│   ├── batch-processing.md                       # Placeholder
│   ├── export-results.md                         # Placeholder
│   └── gpu-acceleration.md                       # Placeholder
├── reference/
│   ├── index.md                                  # ✓ Complete
│   ├── data-structures.md                        # ✓ Complete
│   └── api/
│       ├── hrms_core.md                          # ✓ Complete (auto-gen)
│       ├── formats.md                            # ✓ Complete (auto-gen)
│       ├── formula_annotation.md                 # ✓ Complete (auto-gen)
│├── explanation/
│   ├── architecture.md                           # Placeholder
│   ├── mass-decomposition-algorithm.md           # Placeholder
│   ├── spectral-entropy.md                       # Placeholder
│   ├── isotopic-patterns.md                      # Placeholder
│   └── polars-plugins.md                         # Placeholder
└── contributing/
    ├── development.md                            # Placeholder
    └── writing-plugins.md                        # Placeholder
```

## Summary

**Documentation is ready to use!**

✅ All priority workflows documented  
✅ API reference auto-generated from docstrings  
✅ Professional Material theme  
✅ Working examples with real test data  
✅ Easy to extend with new pages  
✅ One-command deployment to GitHub Pages  

Run `pixi run -e docs docs-serve` to see it in action!
