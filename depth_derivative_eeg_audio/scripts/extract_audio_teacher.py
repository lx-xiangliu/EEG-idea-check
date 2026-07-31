#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.models import AudioTeacher


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract controlled synthetic audio-teacher hidden states")
    parser.add_argument("--input", required=True, help="Torch tensor file with shape [batch,time,12]")
    parser.add_argument("--output", required=True)
    parser.add_argument("--hierarchy", default="hierarchical", choices=["hierarchical", "flat", "nonmonotonic", "parallel", "teacher_shuffled"])
    args = parser.parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input tensor does not exist: {input_path}")
    value = torch.load(input_path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor) or value.ndim != 3 or value.shape[-1] != 12:
        raise ValueError("Input must be a tensor with shape [batch,time,12]")
    teacher = AudioTeacher(input_dim=12, hierarchy=args.hierarchy)
    with torch.no_grad():
        output = teacher(value, return_hidden_states=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"hidden_states": output["hidden_states"], "hierarchy": args.hierarchy}, output_path)


if __name__ == "__main__":
    main()
