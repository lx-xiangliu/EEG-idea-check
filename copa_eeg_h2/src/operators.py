"""Shape-preserving EEG observation operators aligned with the H1 project."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt


@dataclass(frozen=True)
class OperatorResult:
    data: np.ndarray
    metadata: dict[str, Any]


class EEGOperator:
    name = "base"

    def transform(self, epoch, sfreq, channel_names, rng) -> OperatorResult:
        raise NotImplementedError

    def result(self, source, output, **metadata) -> OperatorResult:
        if output.shape != source.shape or not np.isfinite(output).all():
            raise ValueError(f"{self.name} produced invalid shape or values")
        return OperatorResult(np.asarray(output, dtype=np.float32), {"operator": self.name, **metadata})


class CAR(EEGOperator):
    name = "car"

    def transform(self, epoch, sfreq, channel_names, rng):
        return self.result(epoch, epoch - epoch.mean(0, keepdims=True), reference="CAR")


class CzReference(EEGOperator):
    name = "cz_reference"

    def __init__(self, channel="Cz"):
        self.channel = channel

    def transform(self, epoch, sfreq, channel_names, rng):
        lowered = [name.lower() for name in channel_names]
        index = lowered.index(self.channel.lower()) if self.channel.lower() in lowered else len(lowered) // 2
        return self.result(epoch, epoch - epoch[index:index + 1], reference=channel_names[index])


class Gain(EEGOperator):
    def __init__(self, value):
        self.value = float(value)
        self.name = f"gain_{self.value:g}"

    def transform(self, epoch, sfreq, channel_names, rng):
        return self.result(epoch, epoch * self.value, gain=self.value)


class ChannelDropout(EEGOperator):
    def __init__(self, ratio):
        self.ratio = float(ratio)
        self.name = f"random_dropout_{self.ratio:g}"

    def transform(self, epoch, sfreq, channel_names, rng):
        count = max(1, round(epoch.shape[0] * self.ratio))
        dropped = np.sort(rng.choice(epoch.shape[0], count, replace=False))
        output = epoch.copy()
        output[dropped] = 0
        return self.result(epoch, output, dropped_channels=dropped.tolist(), drop_ratio=self.ratio)


class Bandpass(EEGOperator):
    def __init__(self, low, high):
        self.low, self.high = float(low), float(high)
        self.name = f"bandpass_{self.low:g}_{self.high:g}"

    def transform(self, epoch, sfreq, channel_names, rng):
        sos = butter(4, (self.low, self.high), btype="bandpass", fs=sfreq, output="sos")
        return self.result(epoch, sosfiltfilt(sos, epoch, axis=-1), band=[self.low, self.high])


class DownUp(EEGOperator):
    def __init__(self, target):
        self.target = float(target)
        self.name = f"resample_{self.target:g}"

    def transform(self, epoch, sfreq, channel_names, rng):
        ratio = Fraction(self.target / sfreq).limit_denominator(1000)
        down = resample_poly(epoch, ratio.numerator, ratio.denominator, axis=-1)
        inverse = Fraction(sfreq / self.target).limit_denominator(1000)
        output = resample_poly(down, inverse.numerator, inverse.denominator, axis=-1)
        if output.shape[-1] < epoch.shape[-1]:
            output = np.pad(output, ((0, 0), (0, epoch.shape[-1] - output.shape[-1])), mode="edge")
        return self.result(epoch, output[..., :epoch.shape[-1]], intermediate_sfreq=self.target)


def build_operators(config: dict[str, Any]) -> list[EEGOperator]:
    operators: list[EEGOperator] = [CAR(), CzReference(config.get("cz_channel", "Cz"))]
    operators.extend(Gain(v) for v in config.get("gains", [0.5, 2.0]))
    operators.extend(ChannelDropout(v) for v in config.get("random_dropout_ratios", [0.25, 0.5]))
    operators.extend(Bandpass(*bounds) for bounds in config.get("bandpasses", [[4, 30], [8, 30]]))
    operators.append(DownUp(config.get("resample_target_sfreq", 125)))
    if len({operator.name for operator in operators}) != len(operators):
        raise ValueError("Operator names must be unique")
    return operators


def apply_operator(
    X: np.ndarray,
    operator: EEGOperator,
    sfreq: float,
    channel_names: tuple[str, ...],
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    output = np.empty_like(X, dtype=np.float32)
    metadata = []
    for index, epoch in enumerate(X):
        # The per-sample seed makes stochastic views reproducible and paired.
        rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
        result = operator.transform(epoch, sfreq, channel_names, rng)
        output[index] = result.data
        metadata.append(result.metadata)
    return output, metadata
