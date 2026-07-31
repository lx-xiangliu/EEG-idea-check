from src.statistics import holm_correction, paired_cross_operator_test


def test_holm_is_monotone_in_sorted_order():
    raw = [0.04, 0.01, 0.03]
    adjusted = holm_correction(raw)
    assert all(0 <= value <= 1 for value in adjusted)
    assert adjusted[1] <= adjusted[2] <= adjusted[0]


def test_small_subject_count_suppresses_significance():
    result = {
        "operator_names": ["a", "b"],
        "subject_balanced_accuracy": {
            "a__to__a": {"1": 0.8, "2": 0.7, "3": 0.9},
            "a__to__b": {"1": 0.6, "2": 0.6, "3": 0.7},
            "b__to__a": {"1": 0.5, "2": 0.6, "3": 0.6},
            "b__to__b": {"1": 0.8, "2": 0.8, "3": 0.9},
        },
    }
    test = paired_cross_operator_test(result, minimum_subjects=5)
    assert test["status"] == "insufficient_sample_size"
    assert test["p_value"] is None
