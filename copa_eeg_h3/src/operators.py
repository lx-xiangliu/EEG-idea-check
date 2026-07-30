"""Shape-preserving observation operators with explicit scientific families."""

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
    family = "base"

    def transform(self, epoch, sfreq, channel_names, rng) -> OperatorResult:
        raise NotImplementedError

    def result(self, source, output, **metadata) -> OperatorResult:
        if output.shape != source.shape or not np.isfinite(output).all():
            raise ValueError(f"{self.name} produced invalid output")
        return OperatorResult(
            np.asarray(output, dtype=np.float32),
            {"operator": self.name, "operator_family": self.family, **metadata},
        )


class Identity(EEGOperator):
    name, family = "identity", "identity"

    def transform(self, epoch, sfreq, channel_names, rng):
        return self.result(epoch, epoch.copy(), transform="none")


class CAR(EEGOperator):
    name, family = "car", "reference"

    def transform(self, epoch, sfreq, channel_names, rng):
        return self.result(epoch, epoch - epoch.mean(0, keepdims=True), reference="CAR")


class CzReference(EEGOperator):
    name, family = "cz_reference", "reference"

    def __init__(self, channel="Cz"):
        self.channel = channel

    def transform(self, epoch, sfreq, channel_names, rng):
        lowered = [name.lower() for name in channel_names]
        index = lowered.index(self.channel.lower()) if self.channel.lower() in lowered else len(lowered) // 2
        return self.result(epoch, epoch - epoch[index:index + 1], reference=channel_names[index])


class Gain(EEGOperator):
    family = "gain"

    def __init__(self, value):
        self.value, self.name = float(value), f"gain_{float(value):g}"

    def transform(self, epoch, sfreq, channel_names, rng):
        return self.result(epoch, epoch * self.value, gain=self.value)


class Dropout(EEGOperator):
    family = "channel/montage"

    def __init__(self, ratio):
        self.ratio, self.name = float(ratio), f"random_dropout_{float(ratio):g}"

    def transform(self, epoch, sfreq, channel_names, rng):
        count = max(1, round(epoch.shape[0] * self.ratio))
        dropped = np.sort(rng.choice(epoch.shape[0], count, replace=False))
        output = epoch.copy()
        output[dropped] = 0
        return self.result(epoch, output, drop_ratio=self.ratio, dropped_channels=dropped.tolist())


class Bandpass(EEGOperator):
    family = "filter"

    def __init__(self, low, high):
        self.low, self.high = float(low), float(high)
        self.name = f"bandpass_{self.low:g}_{self.high:g}"

    def transform(self, epoch, sfreq, channel_names, rng):
        sos = butter(4, (self.low, self.high), btype="bandpass", fs=sfreq, output="sos")
        return self.result(epoch, sosfiltfilt(sos, epoch, axis=-1), band=[self.low, self.high])


class DownUp(EEGOperator):
    family = "resampling"

    def __init__(self, target):
        self.target, self.name = float(target), f"resample_{float(target):g}"

    def transform(self, epoch, sfreq, channel_names, rng):
        ratio = Fraction(self.target / sfreq).limit_denominator(1000)
        down = resample_poly(epoch, ratio.numerator, ratio.denominator, axis=-1)
        inverse = Fraction(sfreq / self.target).limit_denominator(1000)
        output = resample_poly(down, inverse.numerator, inverse.denominator, axis=-1)
        if output.shape[-1] < epoch.shape[-1]:
            output = np.pad(output, ((0, 0), (0, epoch.shape[-1] - output.shape[-1])), mode="edge")
        return self.result(epoch, output[..., :epoch.shape[-1]], intermediate_sfreq=self.target)


def build_operators(config: dict[str, Any]) -> list[EEGOperator]:
    result: list[EEGOperator] = [Identity(), CAR(), CzReference(config.get("cz_channel", "Cz"))]
    result.extend(Gain(v) for v in config.get("gains", [0.5, 2.0]))
    result.extend(Dropout(v) for v in config.get("random_dropout_ratios", [0.25, 0.5]))
    result.extend(Bandpass(*bounds) for bounds in config.get("bandpasses", [[4, 30], [8, 30]]))
    result.append(DownUp(config.get("resample_target_sfreq", 125)))
    return result


def apply_operator(X, operator, sfreq, channel_names, seed):
    output, metadata = np.empty_like(X, dtype=np.float32), []
    for index, epoch in enumerate(X):
        rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
        result = operator.transform(epoch, sfreq, channel_names, rng)
        output[index], metadata_item = result.data, result.metadata
        metadata.append(metadata_item)
    return output, metadata

