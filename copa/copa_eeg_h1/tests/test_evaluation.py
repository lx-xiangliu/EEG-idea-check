import numpy as np

from src.evaluation import evaluate_cross_operator, evaluate_operator_probe


def test_subject_disjoint_splits_for_both_experiments():
    rng = np.random.default_rng(4)
    subjects = np.repeat(np.arange(1, 4), 8)
    labels = np.tile(np.arange(8) % 2, 3)
    base = rng.normal(size=(24, 6))
    features = {
        "identity": base,
        "gain_2": base * 2 + 0.1,
    }
    probe = evaluate_operator_probe(
        features, subjects, ["logistic_regression"], 3, 10, 0.95, 5, 1
    )
    for fold in probe["models"]["logistic_regression"]["fold_subjects"]:
        assert set(fold["train"]).isdisjoint(fold["test"])
    cross = evaluate_cross_operator(
        features, labels, subjects, "logistic_regression", 10, 0.95, 5, 1
    )
    assert np.asarray(cross["balanced_accuracy_matrix"]).shape == (2, 2)
    for fold in cross["fold_subjects"]:
        assert set(fold["train"]).isdisjoint(fold["test"])
