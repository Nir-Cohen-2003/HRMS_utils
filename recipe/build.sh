#!/bin/bash
export DIST_EXTRA_CONFIG=$PWD/../setup.cfg
$PYTHON -m pip install . -v --no-deps --no-build-isolation
