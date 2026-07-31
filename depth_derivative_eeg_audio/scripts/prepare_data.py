#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import load_config
from src.data import load_manifest, make_synthetic_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate real manifests or explicitly generate synthetic metadata")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--manifest", help="Path to a real-data JSON manifest")
    group.add_argument("--synthetic", action="store_true", help="Explicitly use controlled synthetic data")
    parser.add_argument("--config", default="configs/experiment/synthetic.yaml")
    args = parser.parse_args()
    if args.manifest:
        manifest = load_manifest(args.manifest)
        print(json.dumps({"status": "valid", "dataset": manifest.dataset_name, "records": len(manifest.records)}, indent=2))
        return
    cfg = load_config(args.config)
    bundle = make_synthetic_bundle(cfg.data, cfg.train.seed)
    print(
        json.dumps(
            {
                "status": "generated_in_memory",
                "mode": bundle.mode,
                "train": len(bundle.train),
                "val": len(bundle.val),
                "test": len(bundle.test),
                "note": "Synthetic data are not a fallback for a missing real manifest.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
