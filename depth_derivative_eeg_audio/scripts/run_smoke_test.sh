#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
./.venv/bin/pytest -q
./.venv/bin/python scripts/smoke_test.py
