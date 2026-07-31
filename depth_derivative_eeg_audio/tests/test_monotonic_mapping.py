import torch

from src.models import MonotonicDepthMapper


def test_mapping_is_monotonic_and_bounded() -> None:
    mapper = MonotonicDepthMapper(eeg_layers=4, audio_layers=6)
    weights, means = mapper.mapping_weights()
    assert weights.shape == (4, 6)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(4))
    assert torch.all(means[1:] >= means[:-1])
    assert float(means.detach().min()) >= 0.0
    assert float(means.detach().max()) <= 5.0
