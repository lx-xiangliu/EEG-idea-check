import numpy as np

from src.operators import ComposeOperator, GainOperator, build_operators


def operator_config():
    return {
        "cz_channel": "Cz",
        "region_channels": ["C3", "Cz", "C4"],
        "bandpasses": [[4, 30], [8, 30]],
        "gains": [0.5, 2.0],
        "random_dropout_ratios": [0.25, 0.5],
        "resample_target_sfreq": 50,
    }


def test_every_operator_preserves_shape_and_metadata():
    rng = np.random.default_rng(1)
    epoch = rng.normal(size=(8, 200))
    channels = ("Fp1", "Fp2", "C3", "Cz", "C4", "P3", "Pz", "P4")
    operators = build_operators(operator_config())
    assert len(operators) == 11
    for index, operator in enumerate(operators):
        result = operator.transform(
            epoch, 100.0, channels, np.random.default_rng(index)
        )
        assert result.data.shape == epoch.shape
        assert np.isfinite(result.data).all()
        assert len(result.metadata["channel_mask"]) == epoch.shape[0]
        assert result.metadata["operator_type"] in {
            "identity",
            "equivalence",
            "lossy",
        }
        for key in (
            "sampling_rate",
            "reference_type",
            "filter_range",
            "gain",
        ):
            assert key in result.metadata


def test_random_dropout_is_seed_deterministic_and_composable():
    epoch = np.arange(800, dtype=float).reshape(8, 100)
    channels = tuple(f"C{i}" for i in range(8))
    dropout = build_operators(operator_config())[5]
    one = dropout.transform(epoch, 100.0, channels, np.random.default_rng(9))
    two = dropout.transform(epoch, 100.0, channels, np.random.default_rng(9))
    np.testing.assert_array_equal(one.data, two.data)
    composition = ComposeOperator([GainOperator(2.0), GainOperator(0.5)])
    composed = composition.transform(
        epoch, 100.0, channels, np.random.default_rng(3)
    )
    np.testing.assert_allclose(composed.data, epoch)
    assert len(composed.metadata["composition_trace"]) == 2
