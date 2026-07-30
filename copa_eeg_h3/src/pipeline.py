"""End-to-end H3 experiment with bounded intermediate storage."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import csv
import hashlib
import json
import platform
import resource
import time

import numpy as np
import scipy
import sklearn
import torch
import yaml

from .config import resolve_path
from .data import load_dataset
from .extraction import extract_all
from .model import build_adapter
from .operators import build_operators
from .plotting import family_paths, head_heatmap, heatmap, qkv_plot
from .probes import (
    duplicate_identity_control, grouped_probe, paired_feature_distances,
    run_main_probes, shuffled_label_control,
)
from .report import evidence_table, write_report


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


def _json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class H3Pipeline:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.seed = int(config["project"]["seed"])
        self.output = resolve_path(config["project"]["output_dir"])

    def run(self) -> Path:
        started = time.perf_counter()
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.set_num_threads(int(self.config["inference"].get("torch_threads", 1)))
        self.output.mkdir(parents=True, exist_ok=True)
        probes_dir, attention_dir, figures_dir = (
            self.output / "probes", self.output / "attention", self.output / "figures"
        )
        for directory in (probes_dir, attention_dir, figures_dir):
            directory.mkdir(parents=True, exist_ok=True)
        dataset = load_dataset(self.config, self.seed)
        operators = build_operators(self.config["operators"])
        adapter, resolution = build_adapter(
            self.config, dataset.X.shape[1], dataset.X.shape[2], self.seed
        )
        store, attention_rows, distance_rows, operator_metadata = extract_all(
            dataset, operators, adapter, self.config, self.seed
        )
        operator_rows, task_rows, family_rows, operator_ovr_rows = run_main_probes(
            store, self.config, self.seed
        )
        evidence = evidence_table(family_rows)

        # Main final-pooled block supports controls without re-running all layers.
        pooled = next(
            (item for item in store.finalized() if item[0].representation == "pooled_representation" and item[0].head is None),
            None,
        )
        controls = []
        if pooled is not None:
            _, X, indices = pooled
            rows = [store.metadata[int(index)] for index in indices]
            subjects = np.asarray([row["subject_id"] for row in rows])
            operator_y = np.asarray([row["operator_label"] for row in rows])
            shuffled = shuffled_label_control(X, operator_y, subjects, self.config["probe"], self.seed + 501)
            controls.append({"control": "shuffled_operator_labels", **shuffled})
            rng = np.random.default_rng(self.seed + 502)
            shuffled_subject_groups = subjects.copy()
            rng.shuffle(shuffled_subject_groups)
            subject_control = grouped_probe(
                X, operator_y, shuffled_subject_groups, ["LogisticRegression"],
                int(self.config["probe"]["bootstrap_iterations"]),
                float(self.config["probe"]["confidence_level"]), self.seed + 503,
            )
            controls.append({"control": "shuffled_subject_labels_as_split_groups", **(subject_control[0] if subject_control else {})})
            identity_mask = operator_y == "identity"
            duplicate = duplicate_identity_control(
                X[identity_mask], subjects[identity_mask], len(operators),
                self.config["probe"], self.seed + 504,
            )
            controls.append({"control": "identity_only_duplicate_views", **duplicate})
            source_hash = np.asarray([row["source_epoch_id"] % 2 for row in rows])
            source_probe = grouped_probe(
                X, source_hash, subjects, ["LogisticRegression"],
                int(self.config["probe"]["bootstrap_iterations"]),
                float(self.config["probe"]["confidence_level"]), self.seed + 505,
            )
            controls.append({"control": "source_id_hash_probe", **(source_probe[0] if source_probe else {})})
            same_source, same_operator = paired_feature_distances(X, rows)
            controls.extend([
                {"control": "same_source_epoch_different_operator", "mean_cosine_distance": same_source},
                {"control": "same_operator_different_source_epoch", "mean_cosine_distance": same_operator},
            ])
        controls.extend([
            {
                "control": "random_initialized_model",
                "status": "same_as_primary_architecture_fallback" if not adapter.pretrained else "separate baseline required",
                "model_kind": adapter.model_kind,
            },
            {
                "control": "dataset_id_probe",
                "status": "not_estimable_single_dataset",
                "n_dataset_ids": 1,
            },
        ])

        _csv(probes_dir / "operator_probe_by_layer.csv", operator_rows)
        _csv(probes_dir / "task_probe_by_layer.csv", task_rows)
        _csv(probes_dir / "operator_family_probe.csv", family_rows)
        _csv(probes_dir / "operator_one_vs_rest.csv", operator_ovr_rows)
        _csv(probes_dir / "controls.csv", controls)
        _csv(probes_dir / "evidence_table.csv", evidence)
        _csv(attention_dir / "attention_distance.csv", distance_rows)
        _csv(attention_dir / "attention_statistics.csv", attention_rows)
        with (self.output / "operator_metadata.jsonl").open("w", encoding="utf-8") as handle:
            for row in operator_metadata:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        if bool(self.config["inference"].get("save_compressed_summaries", True)):
            summary_dir = self.output / "compressed_summaries"
            summary_dir.mkdir(exist_ok=True)
            for block, X, indices in store.finalized():
                head = "all" if block.head is None else str(block.head)
                np.savez_compressed(
                    summary_dir / f"{block.representation}__layer{block.layer}__head{head}.npz",
                    features=X, metadata_indices=indices,
                )
            _json(summary_dir / "metadata.json", store.metadata)

        dataset_summary = {
            "mode": dataset.mode, "dataset_name": dataset.dataset_name, "loader": dataset.loader,
            "n_epochs": len(dataset.X), "n_subjects": int(np.unique(dataset.subjects).size),
            "subjects": [int(value) for value in np.unique(dataset.subjects)],
            "shape": list(dataset.X.shape), "sfreq": dataset.sfreq,
            "operator_count_including_identity": len(operators),
        }
        model_summary = {
            **resolution,
            "model_kind": adapter.model_kind, "pretrained": adapter.pretrained,
            "provenance": adapter.provenance,
            "parameter_count": sum(parameter.numel() for parameter in adapter.model.parameters()),
        }
        _json(self.output / "dataset_summary.json", dataset_summary)
        _json(self.output / "model_summary.json", model_summary)
        heatmap(operator_rows, figures_dir / "operator_leakage_heatmap.png", "Operator decoding by layer and representation")
        heatmap(task_rows, figures_dir / "task_information_heatmap.png", "Task decoding by layer and representation")
        qkv_plot(operator_rows, figures_dir / "qkv_comparison.png")
        family_paths(evidence, figures_dir / "operator_family_paths.png")
        head_heatmap(operator_rows, figures_dir / "operator_head_heatmap.png")
        write_report(
            self.output, dataset_summary, model_summary, operator_rows,
            operator_ovr_rows, family_rows, evidence, controls,
        )

        runtime = {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024),
            "platform": platform.platform(), "python": platform.python_version(),
            "torch": torch.__version__, "numpy": np.__version__,
            "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
            "device": "cpu", "inference_mode": True,
        }
        _json(self.output / "runtime.json", runtime)
        (self.output / "resolved_config.yaml").write_text(
            yaml.safe_dump(self.config, sort_keys=False), encoding="utf-8"
        )
        manifest = []
        for path in sorted(item for item in self.output.rglob("*") if item.is_file()):
            if path.name == "manifest.json":
                continue
            manifest.append({
                "path": str(path.relative_to(self.output)),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
        _json(self.output / "manifest.json", manifest)
        print("Generated files:")
        for path in sorted(item for item in self.output.rglob("*") if item.is_file()):
            print(path.relative_to(self.output))
        return self.output
