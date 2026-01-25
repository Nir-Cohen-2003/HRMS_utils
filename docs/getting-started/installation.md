# Installation

HRMS Utils can be installed via conda (recommended), pip, or built from source.

## Requirements

- Python 3.12 or later
- Operating System: Linux, macOS, or Windows

## Conda Installation (Recommended)

The easiest way to install HRMS Utils is via conda:

```bash
conda install -c conda-forge hrms_utils
```

Or using mamba for faster dependency resolution:

```bash
mamba install -c conda-forge hrms_utils
```

This installs HRMS Utils along with all required dependencies including Polars, RDKit, and NumPy.

## Pip Installation

HRMS Utils can also be installed via pip:

```bash
pip install hrms_utils
```

Note: RDKit installation via pip may require additional system dependencies. Using conda is recommended for easier RDKit setup.

## Building from Source

For development or to use the latest unreleased features, you can build from source.

### Prerequisites

1. **Rust toolchain** (for building Rust extensions):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **Pixi** (recommended for managing environments):
   ```bash
   curl -fsSL https://pixi.sh/install.sh | bash
   ```

### Clone and Build

```bash
# Clone the repository
git clone https://github.com/Nir-Cohen-2003/HRMS_utils.git
cd HRMS_utils

# Using pixi (recommended)
pixi install
pixi run python -c "import hrms_utils; print(hrms_utils.hrms_core.__version__)"

# Or using maturin directly
pip install maturin
maturin develop --release
```

### Running Tests

After building from source, verify the installation:

```bash
# Using pixi
pixi run pytest

# Or using pytest directly
pytest tests/
```

## Verifying Installation

Check that HRMS Utils is correctly installed:

```python
import hrms_utils
from hrms_utils import hrms_core, formats

print(f"HRMS Utils version: {hrms_core.__version__}")
print(f"Number of elements supported: {hrms_core.NUM_ELEMENTS}")
```

You should see output showing the version number and the number of elements (15) supported for formula decomposition.

## Optional Dependencies

### Documentation Tools

To build documentation locally:

```bash
pixi add --feature docs mkdocs mkdocs-material 'mkdocstrings[python]' mkdocstrings-python
pixi run -e docs mkdocs serve
```

### GPU-Accelerated Similarity Search

For large-scale spectral similarity searches, install the fast_cosine_sim package:

```bash
# This requires CUDA-capable GPU and CUDA toolkit
pip install fast-cosine-sim
```

## Troubleshooting

### Import Errors

If you encounter import errors, ensure all dependencies are installed:

```bash
pip install polars rdkit numpy numba scipy aiohttp requests
```

### Rust Build Failures

If building from source fails with Rust errors:

1. Ensure you have the latest Rust toolchain:
   ```bash
   rustup update
   ```

2. Check that you have the required system libraries (Linux):
   ```bash
   sudo apt-get install build-essential libssl-dev pkg-config
   ```

3. On macOS, ensure Xcode Command Line Tools are installed:
   ```bash
   xcode-select --install
   ```

### RDKit Installation Issues

RDKit can be tricky to install via pip. If you encounter issues:

1. Use conda/mamba instead (recommended)
2. Or install RDKit separately via conda, then install hrms_utils via pip:
   ```bash
   conda create -n hrms python=3.12 rdkit -c conda-forge
   conda activate hrms
   pip install hrms_utils
   ```

## Next Steps

Once installed, head to the [Quickstart Guide](quickstart.md) to learn the basics.
