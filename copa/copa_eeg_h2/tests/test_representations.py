import numpy as np
import pytest

from src.data import make_synthetic_eeg
from src.representations import PatchRepresentation, assert_no_fit_leakage, patchify


def test_patch_shapes_and_all_embeddings():
    dataset = make_synthetic_eeg(n_subjects=3, epochs_per_subject=4, seed=5)
    patches = patchify(dataset.X, 25)
    assert patches.shape == (12, 20, 8 * 25)
    for name in ("raw_patch", "pca", "random_projection", "frozen_patch"):
        model = PatchRepresentation(name, patch_samples=25, embedding_dim=16, seed=2)
        model.fit(dataset.X[:8], dataset.sample_ids[:8])
        result = model.transform(dataset.X)
        assert result.shape[:2] == (12, 20)
        assert result.shape[2] == (200 if name == "raw_patch" else 16)
        assert np.isfinite(result).all()


def test_pca_fit_ids_detect_leakage():
    dataset = make_synthetic_eeg(n_subjects=3, epochs_per_subject=4, seed=6)
    model = PatchRepresentation("pca", patch_samples=25, embedding_dim=8, seed=4)
    model.fit(dataset.X[:8], dataset.sample_ids[:8])
    assert_no_fit_leakage(model, dataset.sample_ids[8:])
    with pytest.raises(AssertionError):
        assert_no_fit_leakage(model, dataset.sample_ids[7:])
