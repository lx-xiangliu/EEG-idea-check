from __future__ import annotations

from typing import TypedDict

import torch
from torch import nn

from src.utils.repro import assert_finite


class EncoderOutput(TypedDict, total=False):
    last_hidden_state: torch.Tensor
    hidden_states: list[torch.Tensor]


class EEGEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 24,
        n_layers: int = 4,
        n_heads: int = 4,
        ff_mult: int = 2,
        dropout: float = 0.0,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("EEGEncoder requires at least one layer")
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_model * ff_mult,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        channel_positions: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        return_hidden_states: bool = False,
    ) -> EncoderOutput:
        del channel_positions
        if x.ndim != 3:
            raise ValueError(f"EEG input must be [batch,time,channels], got {tuple(x.shape)}")
        if x.shape[1] > self.position.shape[1]:
            raise ValueError(f"Sequence length {x.shape[1]} exceeds max_len {self.position.shape[1]}")
        hidden = self.input_projection(x) + self.position[:, : x.shape[1]]
        hidden_states = [hidden]
        padding_mask = None
        if attention_mask is not None:
            if attention_mask.shape != x.shape[:2]:
                raise ValueError("attention_mask must have shape [batch,time]")
            padding_mask = ~attention_mask.bool()
        for layer in self.layers:
            hidden = layer(hidden, src_key_padding_mask=padding_mask)
            assert_finite("EEG hidden state", hidden)
            hidden_states.append(hidden)
        last = self.final_norm(hidden)
        result: EncoderOutput = {"last_hidden_state": last}
        if return_hidden_states:
            normalized = hidden_states[:-1] + [last]
            result["hidden_states"] = normalized
        return result


class _FixedResidualBlock(nn.Module):
    def __init__(self, d_model: int, active_slice: slice, gain: float) -> None:
        super().__init__()
        mask = torch.zeros(d_model)
        mask[active_slice] = gain
        self.register_buffer("mask", mask)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        update = torch.tanh(hidden) * self.mask
        return hidden + update


class AudioTeacher(nn.Module):
    """Frozen six-layer teacher with a controlled low/mid/high feature hierarchy."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 24,
        n_layers: int = 6,
        hierarchy: str = "hierarchical",
        seed: int = 1729,
    ) -> None:
        super().__init__()
        if n_layers != 6:
            raise ValueError("The controlled synthetic teacher must have exactly 6 layers")
        generator = torch.Generator().manual_seed(seed)
        self.input_projection = nn.Linear(input_dim, d_model, bias=False)
        with torch.no_grad():
            weight = torch.randn(d_model, input_dim, generator=generator)
            weight /= weight.norm(dim=1, keepdim=True).clamp_min(1e-6)
            self.input_projection.weight.copy_(weight)
        width = d_model // 3
        stages = [0, 0, 1, 1, 2, 2]
        if hierarchy == "flat":
            stages = [0, 0, 0, 0, 0, 0]
        elif hierarchy in {"nonmonotonic", "teacher_shuffled"}:
            stages = [2, 0, 1, 2, 0, 1]
        elif hierarchy == "parallel":
            stages = [0, 1, 0, 2, 1, 2]
        elif hierarchy != "hierarchical":
            raise ValueError(f"Unknown teacher hierarchy: {hierarchy}")
        self.stage_order = tuple(stages)
        self.layers = nn.ModuleList()
        for stage in stages:
            start = stage * width
            stop = d_model if stage == 2 else (stage + 1) * width
            self.layers.append(_FixedResidualBlock(d_model, slice(start, stop), gain=0.55))
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "AudioTeacher":
        del mode
        return super().train(False)

    def forward(
        self,
        waveform: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_hidden_states: bool = True,
    ) -> EncoderOutput:
        del attention_mask
        if waveform.ndim != 3:
            raise ValueError(
                f"Synthetic audio input must be [batch,time,features], got {tuple(waveform.shape)}"
            )
        hidden = self.input_projection(waveform)
        states = [hidden]
        for layer in self.layers:
            hidden = layer(hidden)
            states.append(hidden)
        output: EncoderOutput = {"last_hidden_state": hidden}
        if return_hidden_states:
            output["hidden_states"] = states
        return output
