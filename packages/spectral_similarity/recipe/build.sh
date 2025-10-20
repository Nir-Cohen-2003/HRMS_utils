#!/bin/bash

cargo-bundle-licenses --format yaml --output $SRC_DIR/THIRDPARTY.yml

maturin build --release -o dist --strip

pip install --prefix=$PREFIX dist/*.whl --no-deps
