import numpy as np

from src.data import make_synthetic_eeg


def test_synthetic_data_is_deterministic_and_grouped():
    first = make_synthetic_eeg(
        n_subjects=3, epochs_per_subject=6, n_channels=8, sfreq=100, seed=7
    )
    second = make_synthetic_eeg(
        n_subjects=3, epochs_per_subject=6, n_channels=8, sfreq=100, seed=7
    )
    first.validate()
    np.testing.assert_allclose(first.X, second.X)
    assert set(np.unique(first.y)) == {0, 1}
    assert np.unique(first.subjects).size == 3
    for subject in np.unique(first.subjects):
        assert set(first.y[first.subjects == subject]) == {0, 1}
