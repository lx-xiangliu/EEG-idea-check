"""Streaming inference and bounded summaries of Transformer internals."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

import numpy as np
import torch

from .data import EEGDataset
from .model import EEGTransformerAdapter
from .operators import EEGOperator, apply_operator


def _token_summary(x: torch.Tensor) -> np.ndarray:
    """mean/std/max/first/covariance summary without retaining token tensors."""
    x = x.float()
    centered = x - x.mean(dim=1, keepdim=True)
    cov_diag = centered.square().mean(dim=1)
    covariance = centered.transpose(1, 2) @ centered / max(1, x.shape[1] - 1)
    mask = ~torch.eye(x.shape[-1], dtype=torch.bool, device=x.device)
    off = covariance[:, mask]
    parts = [
        x.mean(1), x.std(1, unbiased=False), x.amax(1), x[:, 0], cov_diag,
        off.mean(1, keepdim=True), off.std(1, unbiased=False, keepdim=True),
    ]
    return torch.cat(parts, dim=1).cpu().numpy().astype(np.float32)


def _qkv_summary(x: torch.Tensor) -> tuple[np.ndarray, list[np.ndarray]]:
    # [batch, head, token, d] -> global token summary over concatenated heads,
    # plus one bounded feature vector per head.
    global_tokens = x.transpose(1, 2).reshape(x.shape[0], x.shape[2], -1)
    global_summary = _token_summary(global_tokens)
    per_head = [_token_summary(x[:, head]) for head in range(x.shape[1])]
    return global_summary, per_head


def _attention_summary(x: torch.Tensor, probabilities: bool) -> tuple[np.ndarray, list[np.ndarray], dict[str, np.ndarray]]:
    x = x.float()
    batch, heads, tokens, _ = x.shape
    diagonal = x.diagonal(dim1=-2, dim2=-1)
    eye = torch.eye(tokens, dtype=torch.bool, device=x.device)[None, None]
    off = x.masked_select(~eye).reshape(batch, heads, tokens * (tokens - 1))
    top_count = max(1, round(tokens * 0.1))
    topk = x.topk(top_count, dim=-1).values
    if probabilities:
        entropy = -(x.clamp_min(1e-12) * x.clamp_min(1e-12).log()).sum(-1).mean(-1)
        diagonal_mass = diagonal.sum(-1) / tokens
        off_diagonal_mass = 1.0 - diagonal_mass
        top_concentration = topk.sum(-1).mean(-1)
    else:
        entropy = torch.zeros((batch, heads), device=x.device)
        diagonal_mass = diagonal.mean(-1)
        off_diagonal_mass = off.mean(-1)
        top_concentration = topk.mean((-1, -2))
    features = torch.stack(
        [
            x.mean((-1, -2)), x.std((-1, -2), unbiased=False),
            x.amax((-1, -2)), diagonal_mass, off_diagonal_mass,
            top_concentration, entropy,
        ],
        dim=-1,
    )
    stats = {
        "mean": features[..., 0].cpu().numpy(),
        "std": features[..., 1].cpu().numpy(),
        "max": features[..., 2].cpu().numpy(),
        "diagonal_mass": diagonal_mass.cpu().numpy(),
        "off_diagonal_mass": off_diagonal_mass.cpu().numpy(),
        "top_k_concentration": top_concentration.cpu().numpy(),
        "entropy": entropy.cpu().numpy(),
    }
    return (
        features.reshape(batch, -1).cpu().numpy().astype(np.float32),
        [features[:, head].cpu().numpy().astype(np.float32) for head in range(heads)],
        stats,
    )


def _js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    p = p.flatten(2).clamp_min(1e-12)
    q = q.flatten(2).clamp_min(1e-12)
    p = p / p.sum(-1, keepdim=True)
    q = q / q.sum(-1, keepdim=True)
    midpoint = 0.5 * (p + q)
    return 0.5 * (
        (p * (p.log() - midpoint.log())).sum(-1)
        + (q * (q.log() - midpoint.log())).sum(-1)
    )


@dataclass
class FeatureBlock:
    representation: str
    layer: int
    head: int | None
    chunks: list[np.ndarray] = field(default_factory=list)
    metadata_indices: list[np.ndarray] = field(default_factory=list)


class FeatureStore:
    def __init__(self):
        self.blocks: dict[tuple[str, int, int | None], FeatureBlock] = {}
        self.metadata: list[dict[str, Any]] = []

    def add_metadata(self, rows: list[dict[str, Any]]) -> np.ndarray:
        start = len(self.metadata)
        self.metadata.extend(rows)
        return np.arange(start, start + len(rows), dtype=int)

    def add(self, representation: str, layer: int, head: int | None, values: np.ndarray, indices: np.ndarray) -> None:
        key = (representation, layer, head)
        block = self.blocks.setdefault(key, FeatureBlock(*key))
        block.chunks.append(np.asarray(values, dtype=np.float32))
        block.metadata_indices.append(indices.copy())

    def finalized(self):
        for block in self.blocks.values():
            yield block, np.concatenate(block.chunks), np.concatenate(block.metadata_indices)


def extract_all(
    dataset: EEGDataset,
    operators: list[EEGOperator],
    adapter: EEGTransformerAdapter,
    config: dict[str, Any],
    seed: int,
) -> tuple[FeatureStore, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run small CPU batches and retain only fixed-size summaries."""
    batch_size = int(config["inference"]["batch_size"])
    requested_layers = config["inference"].get("layers", "all")
    requested_heads = config["inference"].get("heads", "all")
    store, attention_rows, distance_rows, operator_metadata = FeatureStore(), [], [], []
    for operator in operators:
        operated, metadata = apply_operator(dataset.X, operator, dataset.sfreq, dataset.channel_names, seed)
        for sample_id, subject, task, item in zip(dataset.sample_ids, dataset.subjects, dataset.y, metadata):
            operator_metadata.append({
                "source_epoch_id": int(sample_id), "subject_id": int(subject),
                "task_label": int(task), "operator_label": operator.name,
                "operator_family": operator.family, "dataset_id": dataset.dataset_name,
                "operator_metadata": item,
            })
        for start in range(0, len(operated), batch_size):
            stop = min(len(operated), start + batch_size)
            batch_rows = [
                {
                    "source_epoch_id": int(dataset.sample_ids[index]),
                    "subject_id": int(dataset.subjects[index]),
                    "task_label": int(dataset.y[index]),
                    "operator_label": operator.name,
                    "operator_family": operator.family,
                    "dataset_id": dataset.dataset_name,
                    "operator_metadata": json.dumps(metadata[index], sort_keys=True),
                }
                for index in range(start, stop)
            ]
            meta_idx = store.add_metadata(batch_rows)
            tensor = torch.from_numpy(operated[start:stop])
            current = adapter.forward_with_internals(tensor, batch_rows)
            store.add("input_embedding", -1, None, _token_summary(current["input_embedding"]), meta_idx)
            store.add("final", len(current["layers"]), None, _token_summary(current["final_tokens"]), meta_idx)
            store.add(
                "pooled_representation", len(current["layers"]), None,
                current["pooled_representation"].cpu().numpy(), meta_idx,
            )
            identity = None
            if operator.name != "identity":
                identity = adapter.forward_with_internals(torch.from_numpy(dataset.X[start:stop]))
            for layer_index, layer in enumerate(current["layers"]):
                if requested_layers != "all" and layer_index not in [int(v) for v in requested_layers]:
                    continue
                for name in ("q", "k", "v"):
                    global_features, per_head = _qkv_summary(layer[name])
                    store.add(name, layer_index, None, global_features, meta_idx)
                    for head, values in enumerate(per_head):
                        if requested_heads == "all" or head in [int(v) for v in requested_heads]:
                            store.add(name, layer_index, head, values, meta_idx)
                for name in (
                    "attention_output", "residual_before_attention",
                    "residual_after_attention", "mlp_output",
                ):
                    label = "residual" if name == "residual_after_attention" else name
                    store.add(label, layer_index, None, _token_summary(layer[name]), meta_idx)
                for name, probabilities in (("attention_logits", False), ("attention_probs", True)):
                    global_features, per_head, stats = _attention_summary(layer[name], probabilities)
                    store.add(name, layer_index, None, global_features, meta_idx)
                    for head, values in enumerate(per_head):
                        if requested_heads == "all" or head in [int(v) for v in requested_heads]:
                            store.add(name, layer_index, head, values, meta_idx)
                    if probabilities:
                        for local_index, row in enumerate(batch_rows):
                            for head in range(layer[name].shape[1]):
                                attention_rows.append({
                                    **{key: row[key] for key in (
                                        "source_epoch_id", "subject_id", "task_label",
                                        "operator_label", "operator_family",
                                    )},
                                    "layer": layer_index, "head": head,
                                    **{key: float(value[local_index, head]) for key, value in stats.items()},
                                })
                if identity is not None:
                    cur_probs = layer["attention_probs"]
                    base_probs = identity["layers"][layer_index]["attention_probs"]
                    frobenius = (cur_probs - base_probs).square().sum((-1, -2)).sqrt()
                    js = _js_divergence(cur_probs, base_probs)
                    for local_index, row in enumerate(batch_rows):
                        for head in range(cur_probs.shape[1]):
                            distance_rows.append({
                                **{key: row[key] for key in (
                                    "source_epoch_id", "subject_id", "operator_label", "operator_family",
                                )},
                                "layer": layer_index, "head": head,
                                "frobenius_distance": float(frobenius[local_index, head]),
                                "jensen_shannon_divergence": float(js[local_index, head]),
                                "pairing": "same_source_identity_vs_operator",
                            })
    return store, attention_rows, distance_rows, operator_metadata

