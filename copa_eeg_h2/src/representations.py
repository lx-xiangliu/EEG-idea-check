"""Leakage-aware temporal-patch representations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.decomposition import PCA


def patchify(X: np.ndarray, patch_samples: int) -> np.ndarray:
    """Convert [epoch, channel, time] into [epoch, token, channel*patch]."""
    if X.ndim != 3 or patch_samples <= 0:
        raise ValueError("patchify expects [epoch, channel, time] and positive patch_samples")
    n_epochs, n_channels, n_times = X.shape
    n_tokens = n_times // patch_samples
    if n_tokens < 1:
        raise ValueError("patch size exceeds epoch duration")
    cropped = X[..., : n_tokens * patch_samples]
    patches = cropped.reshape(n_epochs, n_channels, n_tokens, patch_samples)
    return patches.transpose(0, 2, 1, 3).reshape(n_epochs, n_tokens, n_channels * patch_samples)


@dataclass
class PatchRepresentation:
    name: str
    patch_samples: int
    embedding_dim: int
    seed: int
    pca_batch_size: int = 2048
    activation: str = "tanh"

    def __post_init__(self) -> None:
        self.model: Any = None
        self.fit_sample_ids: set[int] = set()

    def fit(self, identity_X: np.ndarray, sample_ids: np.ndarray) -> "PatchRepresentation":
        tokens = patchify(identity_X, self.patch_samples)
        flat = tokens.reshape(-1, tokens.shape[-1])
        self.fit_sample_ids = {int(value) for value in sample_ids}
        output_dim = min(self.embedding_dim, flat.shape[0], flat.shape[1])
        if self.name == "raw_patch":
            self.model = None
        elif self.name == "pca":
            # Randomized PCA directly targets the requested low-dimensional
            # embedding. On the full 9-subject run this avoids dozens of
            # expensive exact SVDs over IncrementalPCA batches while retaining
            # the strict train-identity-only fit boundary.
            model = PCA(
                n_components=output_dim,
                svd_solver="randomized",
                random_state=self.seed,
                iterated_power=3,
            )
            model.fit(flat)
            self.model = model
        elif self.name in {"random_projection", "frozen_patch"}:
            rng = np.random.default_rng(self.seed + (7919 if self.name == "frozen_patch" else 0))
            weight = rng.normal(0, 1 / np.sqrt(flat.shape[1]), (flat.shape[1], output_dim))
            bias = rng.normal(0, 0.01, output_dim) if self.name == "frozen_patch" else np.zeros(output_dim)
            self.model = (weight.astype(np.float32), bias.astype(np.float32))
        else:
            raise ValueError(f"Unknown representation: {self.name}")
        return self

    @property
    def fitted_dimension(self) -> int | None:
        if self.name == "raw_patch":
            return None
        if self.name == "pca":
            return int(self.model.n_components_)
        return int(self.model[0].shape[1])

    def transform(self, X: np.ndarray) -> np.ndarray:
        patches = patchify(X, self.patch_samples)
        shape = patches.shape
        flat = patches.reshape(-1, shape[-1])
        if self.name == "raw_patch":
            embedded = flat
        elif self.name == "pca":
            embedded = self.model.transform(flat)
        else:
            weight, bias = self.model
            embedded = flat @ weight + bias
            if self.name == "frozen_patch":
                if self.activation == "relu":
                    embedded = np.maximum(embedded, 0)
                else:
                    embedded = np.tanh(embedded)
        output = np.asarray(embedded, dtype=np.float32).reshape(shape[0], shape[1], -1)
        if not np.isfinite(output).all():
            raise ValueError(f"{self.name} produced NaN or Inf")
        return output


def build_representations(
    config: dict[str, Any],
    sfreq: float,
    seed: int,
) -> list[PatchRepresentation]:
    patch_samples = max(1, int(round(float(config.get("patch_seconds", 0.1)) * sfreq)))
    return [
        PatchRepresentation(
            name=str(name),
            patch_samples=patch_samples,
            embedding_dim=int(config.get("embedding_dim", 64)),
            seed=seed + position * 101,
            pca_batch_size=int(config.get("pca_batch_size", 2048)),
            activation=str(config.get("frozen_activation", "tanh")),
        )
        for position, name in enumerate(config.get(
            "types", ["raw_patch", "pca", "random_projection", "frozen_patch"]
        ))
    ]


def assert_no_fit_leakage(representation: PatchRepresentation, test_sample_ids: np.ndarray) -> None:
    overlap = representation.fit_sample_ids.intersection(int(value) for value in test_sample_ids)
    if overlap:
        raise AssertionError(f"Representation leakage: {len(overlap)} test samples used in fit")
