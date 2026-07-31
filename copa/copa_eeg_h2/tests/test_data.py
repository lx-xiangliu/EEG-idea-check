import numpy as np

from src.data import make_synthetic_eeg


def test_synthetic_is_deterministic_and_subject_grouped():
    first = make_synthetic_eeg(n_subjects=3, epochs_per_subject=6, seed=7)
    second = make_synthetic_eeg(n_subjects=3, epochs_per_subject=6, seed=7)
    np.testing.assert_allclose(first.X, second.X)
    assert np.unique(first.subjects).tolist() == [1, 2, 3]
    assert first.X.shape == (18, 8, 500)
    assert np.unique(first.sample_ids).size == len(first.X)
