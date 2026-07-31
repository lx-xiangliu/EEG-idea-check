#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/real_dataset_manifest.json" >&2
  echo "No synthetic fallback is permitted for the real benchmark." >&2
  exit 2
fi

./.venv/bin/python scripts/prepare_data.py --manifest "$1"
echo "Manifest passed leakage validation. Dataset-specific audio/EEG loader and licensed local files are still required." >&2
exit 3
