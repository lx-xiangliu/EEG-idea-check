import numpy as np

from src.linalg import explained_by_subspace, spectrum_metrics, top_right_subspace


def test_known_rank_two_matrix_has_stable_spectrum():
    rng = np.random.default_rng(12)
    matrix = rng.normal(size=(80, 2)) @ rng.normal(size=(2, 12))
    metrics = spectrum_metrics(matrix, [1, 2, 4, 8], [0.5, 0.9])
    assert metrics["rho"][2] > 1 - 1e-10
    assert metrics["threshold_ranks"][0.9] <= 2
    assert metrics["effective_rank"] <= 2.01
    basis = top_right_subspace(matrix, 2)
    assert explained_by_subspace(matrix, basis) > 1 - 1e-10


def test_zero_matrix_is_numerically_defined():
    metrics = spectrum_metrics(np.zeros((20, 7)), [1, 2], [0.5, 0.9])
    assert metrics["rho"] == {1: 0.0, 2: 0.0}
    assert metrics["effective_rank"] == 0
    assert metrics["stable_rank"] == 0
