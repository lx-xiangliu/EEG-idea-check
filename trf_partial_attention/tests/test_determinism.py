import torch

from src.data import generate_synthetic_batch


def test_synthetic_generation_is_deterministic() -> None:
    first = generate_synthetic_batch(7, "acoustic_semantic")
    second = generate_synthetic_batch(7, "acoustic_semantic")
    assert torch.equal(first.eeg, second.eeg)
    assert torch.equal(first.audio, second.audio)

