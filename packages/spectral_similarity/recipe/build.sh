#!/bin/bash

rm -rf dist build .eggs *.egg-info

cargo-bundle-licenses --format yaml --output $SRC_DIR/THIRDPARTY.yml

$PYTHON -m build -w -n -x

$PYTHON -m pip install --no-deps dist/*.whl