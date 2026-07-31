#!/bin/sh
set -eu
python scripts/run_synthetic_validation.py --seeds 5
echo "Real-data benchmark intentionally not started: configure a licensed synchronized EEG-audio dataset first."

