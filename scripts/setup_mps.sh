#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
python3 -m venv .venv-mps
.venv-mps/bin/python -m pip install -r requirements-mps.txt
.venv-mps/bin/python scripts/check_mps.py
