#!/bin/sh
set -eu
python -m pytest -q
python scripts/train.py --epochs 2
python scripts/run_zero_training_diagnostic.py

