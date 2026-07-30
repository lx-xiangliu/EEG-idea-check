import numpy as np

from src.controls import build_controls, label_preserving_derangement


def test_controls_are_matched_and_paired_relation_is_exact():
    rng = np.random.default_rng(1)
    identity = rng.normal(size=(8, 3, 5)).astype(np.float32)
    operated = identity + 2
    noise = identity + 0.1
    labels = np.arange(8) % 2
    controls = build_controls(identity, operated, noise, labels, seed=9)
    assert {value.shape for value in controls.values()} == {identity.shape}
    np.testing.assert_allclose(controls["paired"], 2, atol=5e-7)
    np.testing.assert_allclose(controls["gaussian_noise"], 0.1, atol=1e-6)


def test_label_permutation_preserves_labels_without_fixed_points():
    labels = np.arange(10) % 2
    permutation = label_preserving_derangement(labels, np.random.default_rng(8))
    assert np.all(labels[permutation] == labels)
    assert np.all(permutation != np.arange(len(labels)))
