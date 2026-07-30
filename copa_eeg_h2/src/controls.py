"""Matched difference-matrix controls."""

from __future__ import annotations

import numpy as np


CONTROL_NAMES = (
    "paired",
    "unpaired",
    "gaussian_noise",
    "random_orthogonal",
    "label_preserving_permutation",
)


def derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    if n < 2:
        raise ValueError("A derangement requires at least two samples")
    order = np.arange(n)
    for _ in range(100):
        rng.shuffle(order)
        if np.all(order != np.arange(n)):
            return order.copy()
    return np.roll(np.arange(n), 1)


def label_preserving_derangement(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    permutation = np.arange(len(labels))
    for label in np.unique(labels):
        index = np.flatnonzero(labels == label)
        if len(index) < 2:
            raise ValueError("Each label needs at least two epochs for permutation control")
        permutation[index] = index[derangement(len(index), rng)]
    if not np.all(labels[permutation] == labels) or np.any(permutation == np.arange(len(labels))):
        raise AssertionError("Invalid label-preserving derangement")
    return permutation


def random_orthogonal(dimension: int, rng: np.random.Generator) -> np.ndarray:
    matrix = rng.normal(size=(dimension, dimension))
    q, r = np.linalg.qr(matrix)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    return q * signs


def build_controls(
    identity: np.ndarray,
    operated: np.ndarray,
    noise_view: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> dict[str, np.ndarray]:
    """Return matched [epoch, token, dimension] difference tensors."""
    if identity.shape != operated.shape or identity.shape != noise_view.shape:
        raise ValueError("All control inputs must have identical shape")
    rng = np.random.default_rng(seed)
    unpaired_order = derangement(len(identity), rng)
    label_order = label_preserving_derangement(labels, rng)
    q = random_orthogonal(identity.shape[-1], rng)
    controls = {
        "paired": operated - identity,
        "unpaired": operated[unpaired_order] - identity,
        "gaussian_noise": noise_view - identity,
        "random_orthogonal": identity @ q - identity,
        "label_preserving_permutation": operated[label_order] - identity,
    }
    shapes = {value.shape for value in controls.values()}
    if shapes != {identity.shape}:
        raise AssertionError("Controls are not sample/dimension matched")
    return {key: np.asarray(value, dtype=np.float32) for key, value in controls.items()}
