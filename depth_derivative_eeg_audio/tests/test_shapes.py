import torch

from src.models import AudioTeacher, EEGEncoder


def test_encoder_shapes() -> None:
    eeg = EEGEncoder(input_dim=12, d_model=24, n_layers=4, n_heads=4, max_len=8)
    teacher = AudioTeacher(input_dim=12, d_model=24, n_layers=6)
    x = torch.randn(3, 8, 12)
    eeg_output = eeg(x, return_hidden_states=True)
    audio_output = teacher(x, return_hidden_states=True)
    assert eeg_output["last_hidden_state"].shape == (3, 8, 24)
    assert len(eeg_output["hidden_states"]) == 5
    assert len(audio_output["hidden_states"]) == 7
    assert not any(parameter.requires_grad for parameter in teacher.parameters())
