"""End-to-end H2 experiment pipeline."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import csv
import gzip
import hashlib
import json
import platform
import resource
import time

import numpy as np
import scipy
import sklearn
import yaml

from .config import resolve_path
from .controls import CONTROL_NAMES, build_controls
from .data import EEGDataset, load_dataset
from .linalg import (
    explained_by_subspace,
    principal_angle_summary,
    random_subspace,
    spectrum_metrics,
    top_right_subspace,
)
from .operators import EEGOperator, apply_operator, build_operators
from .plotting import plot_cumulative_spectra, plot_operator_matrices, plot_transfer_heatmap
from .report import write_report
from .representations import (
    PatchRepresentation,
    assert_no_fit_leakage,
    build_representations,
)
from .statistics import holm_adjust, paired_wilcoxon, subject_bootstrap_difference


H1_CONTEXT = (
    "H1 的 9-subject BNCI2014_001 运行显示观测算子可跨受试者识别"
    "（最佳 balanced accuracy 0.7952），且跨算子任务泛化相对对角线平均下降 0.0724。"
    "H2 因此进一步检验该稳定效应是否集中在可迁移低秩子空间；H1 的结果本身不预设 H2 成立。"
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _flatten(tensor: np.ndarray, epoch_mask: np.ndarray | None = None) -> np.ndarray:
    selected = tensor if epoch_mask is None else tensor[epoch_mask]
    return selected.reshape(-1, selected.shape[-1])


def _seed(*values: int) -> int:
    sequence = np.random.SeedSequence([int(value) for value in values])
    return int(sequence.generate_state(1)[0])


class RepresentationMetadataWriter:
    def __init__(self, path: Path):
        self.handle = gzip.open(path, "wt", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(
            self.handle,
            fieldnames=[
                "operator", "subject", "sample_id", "token_id", "embedding_type",
                "embedding_dimension", "fit_subjects",
            ],
        )
        self.writer.writeheader()

    def write(
        self,
        operator: str,
        dataset: EEGDataset,
        n_tokens: int,
        embedding: str,
        dimension: int,
        fit_subjects: Iterable[int],
    ) -> None:
        fit_string = "|".join(str(int(value)) for value in fit_subjects)
        for subject, sample_id in zip(dataset.subjects, dataset.sample_ids):
            for token_id in range(n_tokens):
                self.writer.writerow({
                    "operator": operator,
                    "subject": int(subject),
                    "sample_id": int(sample_id),
                    "token_id": token_id,
                    "embedding_type": embedding,
                    "embedding_dimension": dimension,
                    "fit_subjects": fit_string,
                })

    def close(self) -> None:
        self.handle.close()


class H2Pipeline:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.seed = int(config["project"].get("seed", 42))
        self.output_dir = resolve_path(config["project"]["output_dir"])
        self.cache_dir = resolve_path(config["project"].get("cache_dir", "cache"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dataset: EEGDataset | None = None
        self.operators: list[EEGOperator] = []
        self.cache_signature = ""
        self.operator_cache_dir = Path()

    def _prepare(self) -> None:
        self.dataset = load_dataset(self.config, self.seed, self.cache_dir / "data")
        self.operators = build_operators(self.config["operators"])
        payload = {
            "dataset": self.dataset.dataset_name,
            "loader": self.dataset.loader,
            "shape": self.dataset.X.shape,
            "subjects": self.dataset.subjects.tolist(),
            "sample_ids": self.dataset.sample_ids.tolist(),
            "seed": self.seed,
            "operators": [operator.name for operator in self.operators],
        }
        self.cache_signature = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        self.operator_cache_dir = self.cache_dir / "operator_views" / self.cache_signature
        self.operator_cache_dir.mkdir(parents=True, exist_ok=True)

    def _operator_view(self, operator: EEGOperator) -> tuple[np.ndarray, list[dict]]:
        assert self.dataset is not None
        npz_path = self.operator_cache_dir / f"{operator.name}.npz"
        metadata_path = self.operator_cache_dir / f"{operator.name}.json"
        if npz_path.exists() and metadata_path.exists():
            with np.load(npz_path) as archive:
                data = np.asarray(archive["X"], dtype=np.float32)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            return data, metadata
        data, metadata = apply_operator(
            self.dataset.X, operator, self.dataset.sfreq, self.dataset.channel_names, self.seed
        )
        np.savez_compressed(npz_path, X=data)
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
        return data, metadata

    def _noise_view(self) -> np.ndarray:
        assert self.dataset is not None
        ratio = float(self.config["analysis"].get("gaussian_noise_std_ratio", 0.1))
        rng = np.random.default_rng(self.seed + 404)
        scales = np.std(self.dataset.X, axis=(1, 2), keepdims=True) * ratio
        return np.asarray(
            self.dataset.X + rng.normal(size=self.dataset.X.shape) * scales,
            dtype=np.float32,
        )

    def _dataset_summary(self) -> dict[str, Any]:
        assert self.dataset is not None
        return {
            "mode": self.dataset.mode,
            "dataset_name": self.dataset.dataset_name,
            "loader": self.dataset.loader,
            "shape": list(self.dataset.X.shape),
            "n_epochs": int(len(self.dataset.X)),
            "n_subjects": int(np.unique(self.dataset.subjects).size),
            "subjects": [int(value) for value in np.unique(self.dataset.subjects)],
            "n_channels": int(self.dataset.X.shape[1]),
            "n_times": int(self.dataset.X.shape[2]),
            "sfreq": float(self.dataset.sfreq),
            "class_counts": {
                str(int(label)): int(np.sum(self.dataset.y == label))
                for label in np.unique(self.dataset.y)
            },
            "channel_names": list(self.dataset.channel_names),
            "cache_signature": self.cache_signature,
        }

    def _append_spectral_rows(
        self,
        embedding: str,
        operator: str,
        control: str,
        difference: np.ndarray,
        explained_rows: list[dict],
        threshold_rows: list[dict],
        effective_rows: list[dict],
    ) -> dict[int, dict[int, float]]:
        """Add overall/subject results and return rank -> subject -> rho."""
        assert self.dataset is not None
        analysis = self.config["analysis"]
        ranks = [int(value) for value in analysis["ranks"]]
        thresholds = [float(value) for value in analysis["thresholds"]]
        subject_values: dict[int, dict[int, float]] = {rank: {} for rank in ranks}
        scopes: list[tuple[str, int | str, np.ndarray]] = [
            ("overall", "all", np.ones(len(difference), dtype=bool))
        ]
        scopes.extend(
            ("subject", int(subject), self.dataset.subjects == subject)
            for subject in np.unique(self.dataset.subjects)
        )
        for scope, subject, mask in scopes:
            metrics = spectrum_metrics(_flatten(difference, mask), ranks, thresholds)
            for rank, value in metrics["rho"].items():
                explained_rows.append({
                    "embedding": embedding,
                    "operator": operator,
                    "control": control,
                    "scope": scope,
                    "subject": subject,
                    "rank": rank,
                    "explained_variance": value,
                })
                if scope == "subject":
                    subject_values[rank][int(subject)] = value
            for threshold, rank in metrics["threshold_ranks"].items():
                threshold_rows.append({
                    "embedding": embedding,
                    "operator": operator,
                    "control": control,
                    "scope": scope,
                    "subject": subject,
                    "threshold": threshold,
                    "minimum_rank": rank,
                })
            effective_rows.append({
                "embedding": embedding,
                "operator": operator,
                "control": control,
                "scope": scope,
                "subject": subject,
                "effective_rank": metrics["effective_rank"],
                "stable_rank": metrics["stable_rank"],
                "matrix_rows": int(np.sum(mask) * difference.shape[1]),
                "embedding_dimension": int(difference.shape[2]),
            })
        return subject_values

    def _global_analysis(
        self,
        metadata_writer: RepresentationMetadataWriter,
    ) -> tuple[list[dict], list[dict], list[dict], list[dict], dict[str, dict[str, np.ndarray]]]:
        assert self.dataset is not None
        explained_rows: list[dict] = []
        threshold_rows: list[dict] = []
        effective_rows: list[dict] = []
        comparison_rows: list[dict] = []
        global_differences: dict[str, dict[str, np.ndarray]] = {}
        subjects = np.unique(self.dataset.subjects)
        fit_subjects = subjects[:-1]
        fit_mask = np.isin(self.dataset.subjects, fit_subjects)
        noise_X = self._noise_view()
        representations = build_representations(
            self.config["representations"], self.dataset.sfreq, self.seed
        )
        representation_cache = self.cache_dir / "representations" / self.cache_signature
        representation_cache.mkdir(parents=True, exist_ok=True)

        for representation_index, representation in enumerate(representations):
            print(f"[global] embedding={representation.name}", flush=True)
            representation.fit(
                self.dataset.X[fit_mask], self.dataset.sample_ids[fit_mask]
            )
            identity_embedding = representation.transform(self.dataset.X)
            noise_embedding = representation.transform(noise_X)
            metadata_writer.write(
                "identity", self.dataset, identity_embedding.shape[1],
                representation.name, identity_embedding.shape[2], fit_subjects,
            )
            global_differences[representation.name] = {}
            for operator_index, operator in enumerate(self.operators):
                operated_X, operator_metadata = self._operator_view(operator)
                operated_embedding = representation.transform(operated_X)
                metadata_writer.write(
                    operator.name, self.dataset, operated_embedding.shape[1],
                    representation.name, operated_embedding.shape[2], fit_subjects,
                )
                controls = build_controls(
                    identity_embedding,
                    operated_embedding,
                    noise_embedding,
                    self.dataset.y,
                    _seed(self.seed, representation_index, operator_index),
                )
                global_differences[representation.name][operator.name] = controls["paired"]
                if bool(self.config["analysis"].get("cache_representations", True)):
                    np.savez_compressed(
                        representation_cache / f"{representation.name}__{operator.name}.npz",
                        paired_difference=controls["paired"],
                    )

                metrics_by_control: dict[str, dict[int, dict[int, float]]] = {}
                for control_name in CONTROL_NAMES:
                    metrics_by_control[control_name] = self._append_spectral_rows(
                        representation.name,
                        operator.name,
                        control_name,
                        controls[control_name],
                        explained_rows,
                        threshold_rows,
                        effective_rows,
                    )
                for rank in [int(value) for value in self.config["analysis"]["ranks"]]:
                    paired_by_subject = metrics_by_control["paired"][rank]
                    for control_position, control_name in enumerate(CONTROL_NAMES[1:]):
                        control_by_subject = metrics_by_control[control_name][rank]
                        mean, low, high = subject_bootstrap_difference(
                            paired_by_subject,
                            control_by_subject,
                            int(self.config["analysis"]["bootstrap_iterations"]),
                            float(self.config["analysis"]["confidence_level"]),
                            _seed(self.seed, representation_index, operator_index, rank, control_position),
                        )
                        p_value, effect_size, n_subjects = paired_wilcoxon(
                            paired_by_subject,
                            control_by_subject,
                            int(self.config["analysis"]["minimum_subjects_for_test"]),
                        )
                        comparison_rows.append({
                            "embedding": representation.name,
                            "operator": operator.name,
                            "rank": rank,
                            "control": control_name,
                            "mean_difference": mean,
                            "ci_low": low,
                            "ci_high": high,
                            "wilcoxon_p": p_value,
                            "effect_size_rank_biserial": effect_size,
                            "n_subjects": n_subjects,
                            "bootstrap_unit": "subject",
                        })
                print(f"  operator={operator.name} complete", flush=True)
        # Each operator/embedding/rank asks one four-control question family.
        # Correcting all ranks, operators, and embeddings as one family would
        # mix scientifically distinct hypotheses and be unnecessarily opaque.
        families: dict[tuple[str, str, int], list[int]] = defaultdict(list)
        for index, row in enumerate(comparison_rows):
            families[(row["embedding"], row["operator"], int(row["rank"]))].append(index)
        for indices in families.values():
            adjusted = holm_adjust(
                [float(comparison_rows[index]["wilcoxon_p"]) for index in indices]
            )
            for index, value in zip(indices, adjusted):
                comparison_rows[index]["holm_p"] = value
        return (
            explained_rows,
            threshold_rows,
            effective_rows,
            comparison_rows,
            global_differences,
        )

    def _fold_differences(
        self,
        embedding: str,
        global_differences: dict[str, dict[str, np.ndarray]],
        train_mask: np.ndarray,
        test_mask: np.ndarray,
    ) -> dict[str, np.ndarray]:
        assert self.dataset is not None
        if embedding != "pca":
            return global_differences[embedding]
        template = build_representations(
            {**self.config["representations"], "types": ["pca"]},
            self.dataset.sfreq,
            self.seed,
        )[0]
        template.fit(self.dataset.X[train_mask], self.dataset.sample_ids[train_mask])
        assert_no_fit_leakage(template, self.dataset.sample_ids[test_mask])
        components = np.asarray(template.model.components_, dtype=np.float32)
        raw_differences = global_differences["raw_patch"]
        return {
            operator: difference @ components.T
            for operator, difference in raw_differences.items()
        }

    def _locality_analysis(
        self,
        embedding: str,
        operator: str,
        difference: np.ndarray,
        train_mask: np.ndarray,
        test_mask: np.ndarray,
        global_basis: np.ndarray,
        fold_subject: int,
        principal_rows: list[dict],
        locality_rows: list[dict],
        seed: int,
    ) -> None:
        assert self.dataset is not None
        rank = int(self.config["analysis"]["transfer_rank"])
        # Label-local subspaces and their mutual angles.
        label_bases: dict[int, np.ndarray] = {}
        for label in np.unique(self.dataset.y):
            local_train = train_mask & (self.dataset.y == label)
            local_test = test_mask & (self.dataset.y == label)
            if not np.any(local_test):
                continue
            basis = top_right_subspace(_flatten(difference, local_train), rank, seed)
            label_bases[int(label)] = basis
            local_score = explained_by_subspace(_flatten(difference, local_test), basis)
            global_score = explained_by_subspace(_flatten(difference, local_test), global_basis)
            angle, maximum, minimum = principal_angle_summary(global_basis, basis)
            locality_rows.append({
                "embedding": embedding,
                "operator": operator,
                "fold_subject": fold_subject,
                "group_type": "label",
                "group": int(label),
                "metric": "transfer_gain",
                "local_explained": local_score,
                "global_explained": global_score,
                "local_minus_global": local_score - global_score,
                "mean_angle_deg": angle,
            })
        label_keys = sorted(label_bases)
        for i, first in enumerate(label_keys):
            for second in label_keys[i + 1:]:
                mean, maximum, minimum = principal_angle_summary(label_bases[first], label_bases[second])
                principal_rows.append({
                    "embedding": embedding,
                    "comparison_type": "label_group",
                    "fold_subject": fold_subject,
                    "source_operator": operator,
                    "target_operator": operator,
                    "group_a": first,
                    "group_b": second,
                    "rank": rank,
                    "mean_angle_deg": mean,
                    "max_angle_deg": maximum,
                    "min_angle_deg": minimum,
                })

        # Power-quartile local subspaces; thresholds come from train subjects only.
        powers = np.mean(self.dataset.X.astype(np.float64) ** 2, axis=(1, 2))
        bounds = np.quantile(powers[train_mask], [0.25, 0.5, 0.75])
        bins = np.digitize(powers, bounds)
        quartile_bases: dict[int, np.ndarray] = {}
        for quartile in range(4):
            local_train = train_mask & (bins == quartile)
            local_test = test_mask & (bins == quartile)
            if not np.any(local_train) or not np.any(local_test):
                continue
            basis = top_right_subspace(_flatten(difference, local_train), rank, seed + quartile + 1)
            quartile_bases[quartile] = basis
            local_score = explained_by_subspace(_flatten(difference, local_test), basis)
            global_score = explained_by_subspace(_flatten(difference, local_test), global_basis)
            angle, maximum, minimum = principal_angle_summary(global_basis, basis)
            locality_rows.append({
                "embedding": embedding,
                "operator": operator,
                "fold_subject": fold_subject,
                "group_type": "power_quartile",
                "group": quartile + 1,
                "metric": "transfer_gain",
                "local_explained": local_score,
                "global_explained": global_score,
                "local_minus_global": local_score - global_score,
                "mean_angle_deg": angle,
            })
        quartile_keys = sorted(quartile_bases)
        for i, first in enumerate(quartile_keys):
            for second in quartile_keys[i + 1:]:
                mean, maximum, minimum = principal_angle_summary(
                    quartile_bases[first], quartile_bases[second]
                )
                principal_rows.append({
                    "embedding": embedding,
                    "comparison_type": "power_quartile",
                    "fold_subject": fold_subject,
                    "source_operator": operator,
                    "target_operator": operator,
                    "group_a": first + 1,
                    "group_b": second + 1,
                    "rank": rank,
                    "mean_angle_deg": mean,
                    "max_angle_deg": maximum,
                    "min_angle_deg": minimum,
                })

    def _append_subject_group_angles(
        self,
        embedding: str,
        differences: dict[str, np.ndarray],
        principal_rows: list[dict],
        seed: int,
    ) -> None:
        """Estimate each subject subspace once per embedding/operator.

        Subject-local subspaces do not depend on a LOSO train/test split for
        the requested descriptive angle comparison. Computing them once avoids
        repeating the same SVD in every fold.
        """
        assert self.dataset is not None
        rank = int(self.config["analysis"]["transfer_rank"])
        subjects = [int(value) for value in np.unique(self.dataset.subjects)]
        for operator_index, (operator, difference) in enumerate(differences.items()):
            bases = {
                subject: top_right_subspace(
                    _flatten(difference, self.dataset.subjects == subject),
                    rank,
                    _seed(seed, operator_index, subject),
                )
                for subject in subjects
            }
            for first_index, first in enumerate(subjects):
                for second in subjects[first_index + 1:]:
                    mean, maximum, minimum = principal_angle_summary(
                        bases[first], bases[second]
                    )
                    principal_rows.append({
                        "embedding": embedding,
                        "comparison_type": "subject_group",
                        "fold_subject": "all",
                        "source_operator": operator,
                        "target_operator": operator,
                        "group_a": first,
                        "group_b": second,
                        "rank": rank,
                        "mean_angle_deg": mean,
                        "max_angle_deg": maximum,
                        "min_angle_deg": minimum,
                    })

    def _transfer_analysis(
        self,
        global_differences: dict[str, dict[str, np.ndarray]],
    ) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
        assert self.dataset is not None
        transfer_rows: list[dict] = []
        specificity_rows: list[dict] = []
        principal_rows: list[dict] = []
        locality_rows: list[dict] = []
        ranks = [int(value) for value in self.config["analysis"]["ranks"]]
        fixed_rank = int(self.config["analysis"]["transfer_rank"])
        repeats = int(self.config["analysis"]["random_subspace_repeats"])
        embeddings = list(global_differences)
        operator_names = [operator.name for operator in self.operators]
        rng = np.random.default_rng(self.seed + 909)

        for embedding_index, embedding in enumerate(embeddings):
            print(f"[transfer] embedding={embedding}", flush=True)
            self._append_subject_group_angles(
                embedding,
                global_differences[embedding],
                principal_rows,
                _seed(self.seed, embedding_index, 707),
            )
            for fold_index, held_out in enumerate(np.unique(self.dataset.subjects)):
                print(f"  held_out_subject={int(held_out)}", flush=True)
                test_mask = self.dataset.subjects == held_out
                train_mask = ~test_mask
                differences = self._fold_differences(
                    embedding, global_differences, train_mask, test_mask
                )
                maximum_rank = max(max(ranks), fixed_rank)
                train_bases = {
                    operator: top_right_subspace(
                        _flatten(differences[operator], train_mask),
                        maximum_rank,
                        _seed(self.seed, embedding_index, fold_index, position),
                    )
                    for position, operator in enumerate(operator_names)
                }
                oracle_bases = {
                    operator: top_right_subspace(
                        _flatten(differences[operator], test_mask),
                        maximum_rank,
                        _seed(self.seed, embedding_index, fold_index, position, 606),
                    )
                    for position, operator in enumerate(operator_names)
                }
                fixed_bases = {
                    operator: basis[:, :min(fixed_rank, basis.shape[1])]
                    for operator, basis in train_bases.items()
                }
                dimension = next(iter(differences.values())).shape[-1]
                random_bases = [
                    random_subspace(dimension, maximum_rank, rng)
                    for _ in range(repeats)
                ]
                for operator_index, operator in enumerate(operator_names):
                    test_matrix = _flatten(differences[operator], test_mask)
                    other_operator = operator_names[(operator_index + 1) % len(operator_names)]
                    for rank in ranks:
                        train_basis = train_bases[operator][
                            :, :min(rank, train_bases[operator].shape[1])
                        ]
                        train_score = explained_by_subspace(test_matrix, train_basis)
                        random_scores = [
                            explained_by_subspace(
                                test_matrix,
                                basis[:, :min(rank, basis.shape[1])],
                            )
                            for basis in random_bases
                        ]
                        random_score = float(np.mean(random_scores))
                        other_basis = train_bases[other_operator][
                            :, :min(rank, train_bases[other_operator].shape[1])
                        ]
                        other_score = explained_by_subspace(test_matrix, other_basis)
                        oracle_basis = oracle_bases[operator][
                            :, :min(rank, oracle_bases[operator].shape[1])
                        ]
                        oracle_score = explained_by_subspace(test_matrix, oracle_basis)
                        for baseline, score, source in (
                            ("train_subspace", train_score, operator),
                            ("random_subspace", random_score, "random"),
                            ("other_operator", other_score, other_operator),
                            ("oracle", oracle_score, "test_oracle"),
                        ):
                            transfer_rows.append({
                                "embedding": embedding,
                                "operator": operator,
                                "held_out_subject": int(held_out),
                                "rank": rank,
                                "baseline": baseline,
                                "source_subspace": source,
                                "explained_test": score,
                                "residual_ratio": 1 - score,
                                "difference_vs_random": (
                                    score - random_score if baseline == "train_subspace" else ""
                                ),
                            })
                    self._locality_analysis(
                        embedding,
                        operator,
                        differences[operator],
                        train_mask,
                        test_mask,
                        fixed_bases[operator],
                        int(held_out),
                        principal_rows,
                        locality_rows,
                        _seed(self.seed, embedding_index, fold_index, operator_index, 808),
                    )

                for source_operator in operator_names:
                    for target_operator in operator_names:
                        score = explained_by_subspace(
                            _flatten(differences[target_operator], test_mask),
                            fixed_bases[source_operator],
                        )
                        specificity_rows.append({
                            "embedding": embedding,
                            "held_out_subject": int(held_out),
                            "rank": fixed_rank,
                            "source_operator": source_operator,
                            "target_operator": target_operator,
                            "explained_test": score,
                        })
                for source_index, source_operator in enumerate(operator_names):
                    for target_index, target_operator in enumerate(operator_names):
                        mean, maximum, minimum = principal_angle_summary(
                            fixed_bases[source_operator], fixed_bases[target_operator]
                        )
                        principal_rows.append({
                            "embedding": embedding,
                            "comparison_type": "operator",
                            "fold_subject": int(held_out),
                            "source_operator": source_operator,
                            "target_operator": target_operator,
                            "group_a": "",
                            "group_b": "",
                            "rank": fixed_rank,
                            "mean_angle_deg": mean,
                            "max_angle_deg": maximum,
                            "min_angle_deg": minimum,
                        })
        return transfer_rows, specificity_rows, principal_rows, locality_rows

    def run(self) -> Path:
        start = time.perf_counter()
        started_at = datetime.now(timezone.utc)
        self._prepare()
        assert self.dataset is not None
        summary = self._dataset_summary()
        (self.output_dir / "dataset_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (self.output_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(self.config, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )

        # Materialize operator metadata once and preserve the exact stochastic masks.
        with (self.output_dir / "operator_metadata.jsonl").open("w", encoding="utf-8") as handle:
            for operator in self.operators:
                _, metadata = self._operator_view(operator)
                for sample_id, subject, item in zip(
                    self.dataset.sample_ids, self.dataset.subjects, metadata
                ):
                    handle.write(json.dumps({
                        "sample_id": int(sample_id),
                        "subject": int(subject),
                        **item,
                    }, ensure_ascii=False) + "\n")

        metadata_writer = RepresentationMetadataWriter(
            self.output_dir / "representation_metadata.csv.gz"
        )
        try:
            (
                explained_rows,
                threshold_rows,
                effective_rows,
                comparison_rows,
                global_differences,
            ) = self._global_analysis(metadata_writer)
        finally:
            metadata_writer.close()

        transfer_rows, specificity_rows, principal_rows, locality_rows = (
            self._transfer_analysis(global_differences)
        )
        _write_csv(self.output_dir / "explained_variance.csv", explained_rows)
        _write_csv(self.output_dir / "rank_thresholds.csv", threshold_rows)
        _write_csv(self.output_dir / "effective_rank.csv", effective_rows)
        _write_csv(self.output_dir / "control_comparisons.csv", comparison_rows)
        _write_csv(self.output_dir / "cross_subject_transfer.csv", transfer_rows)
        _write_csv(self.output_dir / "operator_specificity.csv", specificity_rows)
        _write_csv(self.output_dir / "principal_angles.csv", principal_rows)
        _write_csv(self.output_dir / "locality_analysis.csv", locality_rows)

        plot_cumulative_spectra(explained_rows, self.output_dir)
        plot_transfer_heatmap(transfer_rows, self.output_dir)
        plot_operator_matrices(specificity_rows, principal_rows, self.output_dir)
        write_report(
            self.output_dir,
            summary,
            explained_rows,
            threshold_rows,
            transfer_rows,
            specificity_rows,
            comparison_rows,
            locality_rows,
            H1_CONTEXT,
        )
        runtime = {
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": time.perf_counter() - start,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
                1024**2 if platform.system() == "Darwin" else 1024
            ),
            "cpu_only": True,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        }
        (self.output_dir / "runtime.json").write_text(
            json.dumps(runtime, indent=2), encoding="utf-8"
        )
        manifest = {
            "files": sorted(
                str(path.relative_to(self.output_dir))
                for path in self.output_dir.rglob("*")
                if path.is_file()
            ),
            "row_counts": {
                "explained_variance": len(explained_rows),
                "rank_thresholds": len(threshold_rows),
                "effective_rank": len(effective_rows),
                "control_comparisons": len(comparison_rows),
                "cross_subject_transfer": len(transfer_rows),
                "operator_specificity": len(specificity_rows),
                "principal_angles": len(principal_rows),
                "locality_analysis": len(locality_rows),
            },
        }
        (self.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[complete] outputs={self.output_dir}", flush=True)
        return self.output_dir
