#!/bin/bash
set -ex

# Debug: List files in the current directory to check for Cargo.toml
ls -la

# First, build the rust extension in-place so that it is available in the package
# We use develop to build and install in the current environment.
maturin develop --manifest-path "$PWD/Cargo.toml" --release --strip --skip-install

# Then, install the package with pip, which will run setup.py for the C++ extensions
# and package everything together.
$PYTHON -m pip install . --no-deps --ignore-installed -vv
