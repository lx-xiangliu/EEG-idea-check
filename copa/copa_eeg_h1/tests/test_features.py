import numpy as np

from src.features import FeatureExtractor


def feature_config():
    return {
        "bandpower": True,
        "moments": True,
        "covariance": True,
        "correlation": True,
        "covariance_eigenvalues": True,
        "eigenvalue_count": 3,
        "bands": {
            "delta": [1, 4],
            "theta": [4, 8],
            "alpha": [8, 13],
            "beta": [13, 30],
        },
    }


def test_features_are_finite_with_dropped_constant_channel():
    rng = np.random.default_rng(2)
    epochs = rng.normal(size=(4, 5, 200))
    epochs[:, 0] = 0
    result = FeatureExtractor(feature_config()).transform(
        epochs, 100.0, ("C1", "C2", "C3", "C4", "C5")
    )
    assert result.values.shape[0] == 4
    assert result.values.shape[1] == len(result.schema)
    assert np.isfinite(result.values).all()
    assert "bandpower.alpha.C1" in result.schema
    assert "cov.C1.C1" in result.schema
    assert "corr.C1.C1" in result.schema
    assert "cov_eigenvalue.1" in result.schema
