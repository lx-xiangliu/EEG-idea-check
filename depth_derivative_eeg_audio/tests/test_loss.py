import pytest
import torch

from src.losses import AlignmentObjective, cosine_distance


def test_cosine_loss_zero_for_identical_values() -> None:
    value = torch.randn(8, 6)
    assert float(cosine_distance(value, value)) == pytest.approx(0.0, abs=1e-6)


def test_dda_loss_is_finite() -> None:
    objective = AlignmentObjective("dda_monotonic", 24, 16, 4, 6)
    eeg_states = [torch.randn(5, 8, 24) for _ in range(5)]
    audio_states = [torch.randn(5, 8, 24) for _ in range(7)]
    loss, diagnostics = objective(eeg_states, audio_states)
    assert torch.isfinite(loss)
    assert diagnostics.weights is not None
