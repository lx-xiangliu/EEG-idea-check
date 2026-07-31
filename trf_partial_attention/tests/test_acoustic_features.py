import torch

from src.acoustics import AcousticFeatureExtractor


def test_acoustic_feature_shapes_and_pitch() -> None:
    sample_rate = 16000
    time = torch.arange(sample_rate, dtype=torch.float32) / sample_rate
    waveform = torch.sin(2 * torch.pi * 200 * time)
    features = AcousticFeatureExtractor()(waveform, sample_rate)
    lengths = {value.shape for value in features.values()}
    assert len(lengths) == 1
    voiced = features["f0"][features["f0"] > 0]
    assert voiced.numel() > 0
    assert abs(float(voiced.median()) - 200.0) < 12.0
    assert torch.all(features["onset"] >= 0)

