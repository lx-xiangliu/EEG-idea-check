import numpy as np

from src.data import make_synthetic_eeg
from src.operators import apply_operator, build_operators


def test_all_required_operators_preserve_shape_and_are_finite():
    dataset = make_synthetic_eeg(n_subjects=2, epochs_per_subject=4, seed=3)
    operators = build_operators({
        "cz_channel": "Cz",
        "gains": [0.5, 2.0],
        "random_dropout_ratios": [0.25, 0.5],
        "bandpasses": [[4, 30], [8, 30]],
        "resample_target_sfreq": 125,
    })
    assert len(operators) == 9
    for operator in operators:
        first, metadata = apply_operator(
            dataset.X, operator, dataset.sfreq, dataset.channel_names, seed=11
        )
        second, _ = apply_operator(
            dataset.X, operator, dataset.sfreq, dataset.channel_names, seed=11
        )
        assert first.shape == dataset.X.shape
        assert np.isfinite(first).all()
        np.testing.assert_allclose(first, second)
        assert all(item["operator"] == operator.name for item in metadata)
