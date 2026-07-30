import numpy as np

from src.statistics import holm_adjust, paired_wilcoxon, subject_bootstrap_difference


def test_subject_level_statistics_and_small_sample_guard():
    paired = {subject: 0.8 + subject * 0.001 for subject in range(1, 6)}
    control = {subject: 0.2 + subject * 0.001 for subject in range(1, 6)}
    mean, low, high = subject_bootstrap_difference(paired, control, 50, 0.95, 3)
    assert mean > 0 and low > 0 and high > 0
    p_value, effect, n = paired_wilcoxon(paired, control, minimum_subjects=5)
    assert np.isfinite(p_value) and effect == 1 and n == 5
    guarded, _, _ = paired_wilcoxon(paired, control, minimum_subjects=6)
    assert np.isnan(guarded)
    adjusted = holm_adjust([0.01, 0.03, float("nan")])
    assert adjusted[0] <= adjusted[1]
    assert np.isnan(adjusted[2])
