#!/usr/bin/env python3
"""Download/cache the configured MOABB data without running models."""

from __future__ import annotations

from _common import common_parser, configured
from src.config import resolve_project_path
from src.data import load_dataset


def main() -> None:
    parser = common_parser("Download BNCI2014_001 into the project cache")
    args = parser.parse_args()
    config, _ = configured(args, mode="real")
    config["data"]["fallback_to_synthetic"] = False
    cache_dir = resolve_project_path(config["project"]["cache_dir"]) / "data"
    dataset = load_dataset(config, int(config["project"]["seed"]), cache_dir)
    print(
        f"Cached {dataset.dataset_name}: {dataset.X.shape[0]} epochs, "
        f"{len(set(dataset.subjects.tolist()))} subjects at {cache_dir}"
    )


if __name__ == "__main__":
    main()
