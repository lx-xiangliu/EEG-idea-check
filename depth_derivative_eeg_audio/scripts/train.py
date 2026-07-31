#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from src.config import load_config
from src.data import make_synthetic_bundle
from src.training import train_synthetic
from src.utils import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one explicitly configured synthetic run")
    parser.add_argument("--config", required=True)
    parser.add_argument("--synthetic", action="store_true", help="Required acknowledgement for synthetic mode")
    args = parser.parse_args()
    if not args.synthetic:
        raise SystemExit("Real training requires a dataset-specific loader; pass --synthetic for the controlled task")
    configure_logging()
    cfg = load_config(args.config)
    result = train_synthetic(cfg, make_synthetic_bundle(cfg.data, cfg.train.seed))
    print(json.dumps(asdict(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
