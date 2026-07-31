import numpy as np

from src.data import make_synthetic_eeg
from src.operators import build_operators, apply_operator


def test_operator_views_preserve_pairing_and_shape():
    dataset = make_synthetic_eeg(n_subjects=3, epochs_per_subject=4, seed=3)
    config = {
        "gains": [0.5, 2.0],
        "random_dropout_ratios": [0.25, 0.5],
        "bandpasses": [[4, 30], [8, 30]],
        "resample_target_sfreq": 125,
    }
    operators = build_operators(config)
    assert len(operators) == 10
    assert {item.family for item in operators} >= {"identity", "reference", "gain", "filter", "resampling", "channel/montage"}
    for operator in operators:
        first, metadata = apply_operator(dataset.X, operator, dataset.sfreq, dataset.channel_names, 5)
        second, _ = apply_operator(dataset.X, operator, dataset.sfreq, dataset.channel_names, 5)
        assert first.shape == dataset.X.shape
        assert len(metadata) == len(dataset.X)
        assert np.isfinite(first).all()
        np.testing.assert_allclose(first, second)

