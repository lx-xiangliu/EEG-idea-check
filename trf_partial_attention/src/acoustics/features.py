"""Dependency-light low-level acoustic covariates.

The extractor deliberately uses only PyTorch.  It is intended for diagnostics and
reproducible baselines, not as a drop-in replacement for a validated Praat pitch
tracker.  All returned features share the STFT frame axis.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class AcousticFeatureExtractor:
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    f0_min_hz: float = 60.0
    f0_max_hz: float = 400.0
    eps: float = 1e-8

    def __call__(self, waveform: Tensor, sample_rate: int) -> dict[str, Tensor]:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 2:
            raise ValueError("waveform must have shape [samples] or [batch, samples]")
        frame = max(4, round(sample_rate * self.frame_ms / 1000.0))
        hop = max(1, round(sample_rate * self.hop_ms / 1000.0))
        if waveform.shape[-1] < frame:
            raise ValueError("waveform is shorter than one analysis frame")

        x = waveform.to(dtype=torch.float32)
        frames = x.unfold(-1, frame, hop)
        window = torch.hann_window(frame, dtype=x.dtype, device=x.device)
        windowed = frames * window
        energy = windowed.square().mean(dim=-1).clamp_min(self.eps).sqrt()
        envelope = frames.abs().mean(dim=-1)
        onset = torch.nn.functional.pad(
            (envelope[:, 1:] - envelope[:, :-1]).clamp_min(0.0), (1, 0)
        )

        spectrum = torch.fft.rfft(windowed, dim=-1).abs()
        normalized = spectrum / spectrum.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        flux = torch.nn.functional.pad(
            (normalized[:, 1:] - normalized[:, :-1]).clamp_min(0.0).square().sum(-1).sqrt(),
            (1, 0),
        )
        freqs = torch.fft.rfftfreq(frame, d=1.0 / sample_rate).to(x.device)
        centroid = (normalized * freqs).sum(dim=-1)
        f0 = self._autocorrelation_f0(windowed, sample_rate)
        return {
            "envelope": envelope,
            "onset": onset,
            "f0": f0,
            "energy": energy,
            "spectral_flux": flux,
            "spectral_centroid": centroid,
        }

    def _autocorrelation_f0(self, frames: Tensor, sample_rate: int) -> Tensor:
        centered = frames - frames.mean(dim=-1, keepdim=True)
        n = centered.shape[-1]
        n_fft = 1 << (2 * n - 1).bit_length()
        spec = torch.fft.rfft(centered, n=n_fft, dim=-1)
        ac = torch.fft.irfft(spec * spec.conj(), n=n_fft, dim=-1)[..., :n]
        lag_min = max(1, int(sample_rate / self.f0_max_hz))
        lag_max = min(n - 1, int(sample_rate / self.f0_min_hz))
        if lag_min >= lag_max:
            return torch.zeros(frames.shape[:-1], dtype=frames.dtype, device=frames.device)
        region = ac[..., lag_min : lag_max + 1]
        peak, relative_lag = region.max(dim=-1)
        lag = relative_lag + lag_min
        voiced = peak > (0.1 * ac[..., 0].clamp_min(self.eps))
        return torch.where(voiced, sample_rate / lag.to(frames.dtype), torch.zeros_like(peak))

