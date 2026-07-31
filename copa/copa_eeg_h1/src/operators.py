"""Shape-preserving, composable EEG observation operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt


@dataclass(frozen=True)
class OperatorResult:
    data: np.ndarray
    metadata: dict[str, Any]


class EEGOperator:
    name = "base"
    operator_type = "equivalence"

    def transform(
        self,
        epoch: np.ndarray,
        sfreq: float,
        channel_names: tuple[str, ...],
        rng: np.random.Generator,
    ) -> OperatorResult:
        raise NotImplementedError

    def _result(
        self,
        input_epoch: np.ndarray,
        output_epoch: np.ndarray,
        sfreq: float,
        channel_names: tuple[str, ...],
        **metadata: Any,
    ) -> OperatorResult:
        if output_epoch.shape != input_epoch.shape:
            raise ValueError(
                f"{self.name} changed shape from {input_epoch.shape} to {output_epoch.shape}"
            )
        if not np.isfinite(output_epoch).all():
            raise ValueError(f"{self.name} produced NaN or Inf")
        base = {
            "operator": self.name,
            "operator_type": self.operator_type,
            "sampling_rate": float(sfreq),
            "reference_type": "unchanged",
            "filter_range": None,
            "gain": 1.0,
            "channel_mask": [True] * len(channel_names),
        }
        base.update(metadata)
        return OperatorResult(np.asarray(output_epoch, dtype=np.float64), base)


class IdentityOperator(EEGOperator):
    name = "identity"
    operator_type = "identity"

    def transform(self, epoch, sfreq, channel_names, rng):
        return self._result(epoch, epoch.copy(), sfreq, channel_names)


class CommonAverageReference(EEGOperator):
    name = "car"
    operator_type = "equivalence"

    def transform(self, epoch, sfreq, channel_names, rng):
        output = epoch - epoch.mean(axis=0, keepdims=True)
        return self._result(
            epoch, output, sfreq, channel_names, reference_type="common_average"
        )


class CzReference(EEGOperator):
    name = "cz_reference"
    operator_type = "equivalence"

    def __init__(self, channel: str = "Cz"):
        self.channel = channel

    def transform(self, epoch, sfreq, channel_names, rng):
        lowered = [name.lower() for name in channel_names]
        if self.channel.lower() in lowered:
            index = lowered.index(self.channel.lower())
        else:
            index = len(channel_names) // 2
        output = epoch - epoch[index : index + 1]
        return self._result(
            epoch,
            output,
            sfreq,
            channel_names,
            reference_type=channel_names[index],
            reference_channel_index=index,
        )


class GainOperator(EEGOperator):
    operator_type = "equivalence"

    def __init__(self, gain: float):
        if gain <= 0:
            raise ValueError("gain must be positive")
        self.gain = float(gain)
        self.name = f"gain_{self.gain:g}"

    def transform(self, epoch, sfreq, channel_names, rng):
        return self._result(
            epoch, epoch * self.gain, sfreq, channel_names, gain=self.gain
        )


class RandomChannelDropout(EEGOperator):
    operator_type = "lossy"

    def __init__(self, ratio: float):
        if not 0 < ratio < 1:
            raise ValueError("dropout ratio must be in (0, 1)")
        self.ratio = float(ratio)
        self.name = f"random_dropout_{self.ratio:g}"

    def transform(self, epoch, sfreq, channel_names, rng):
        count = max(1, int(round(epoch.shape[0] * self.ratio)))
        dropped = np.sort(rng.choice(epoch.shape[0], size=count, replace=False))
        output = epoch.copy()
        output[dropped] = 0.0
        mask = np.ones(epoch.shape[0], dtype=bool)
        mask[dropped] = False
        return self._result(
            epoch,
            output,
            sfreq,
            channel_names,
            channel_mask=mask.tolist(),
            dropped_channels=[channel_names[i] for i in dropped],
            drop_ratio=self.ratio,
        )


class RegionChannelDropout(EEGOperator):
    name = "region_dropout"
    operator_type = "lossy"

    def __init__(self, region_channels: Iterable[str]):
        self.region_channels = tuple(region_channels)

    def transform(self, epoch, sfreq, channel_names, rng):
        requested = {name.lower() for name in self.region_channels}
        dropped = [i for i, name in enumerate(channel_names) if name.lower() in requested]
        if not dropped:
            center = len(channel_names) // 2
            dropped = sorted(set([max(0, center - 1), center, min(len(channel_names) - 1, center + 1)]))
        output = epoch.copy()
        output[dropped] = 0.0
        mask = np.ones(epoch.shape[0], dtype=bool)
        mask[dropped] = False
        return self._result(
            epoch,
            output,
            sfreq,
            channel_names,
            channel_mask=mask.tolist(),
            dropped_channels=[channel_names[i] for i in dropped],
            region_channels=list(self.region_channels),
        )


class BandpassOperator(EEGOperator):
    operator_type = "lossy"

    def __init__(self, low: float, high: float, order: int = 4):
        if not 0 < low < high:
            raise ValueError("bandpass requires 0 < low < high")
        self.low, self.high, self.order = float(low), float(high), int(order)
        self.name = f"bandpass_{self.low:g}_{self.high:g}"

    def transform(self, epoch, sfreq, channel_names, rng):
        if self.high >= sfreq / 2:
            raise ValueError(f"{self.name}: high cutoff must be below Nyquist")
        sos = butter(self.order, [self.low, self.high], btype="bandpass", fs=sfreq, output="sos")
        output = sosfiltfilt(sos, epoch, axis=-1)
        return self._result(
            epoch,
            output,
            sfreq,
            channel_names,
            filter_range=[self.low, self.high],
        )


class DownUpResampleOperator(EEGOperator):
    operator_type = "lossy"

    def __init__(self, target_sfreq: float):
        self.target_sfreq = float(target_sfreq)
        self.name = f"resample_{self.target_sfreq:g}"

    def transform(self, epoch, sfreq, channel_names, rng):
        from fractions import Fraction

        if not 0 < self.target_sfreq < sfreq:
            raise ValueError("target_sfreq must be positive and below original sfreq")
        down_ratio = Fraction(self.target_sfreq / sfreq).limit_denominator(1000)
        down = resample_poly(epoch, down_ratio.numerator, down_ratio.denominator, axis=-1)
        up_ratio = Fraction(sfreq / self.target_sfreq).limit_denominator(1000)
        output = resample_poly(down, up_ratio.numerator, up_ratio.denominator, axis=-1)
        if output.shape[-1] < epoch.shape[-1]:
            output = np.pad(output, ((0, 0), (0, epoch.shape[-1] - output.shape[-1])), mode="edge")
        output = output[..., : epoch.shape[-1]]
        return self._result(
            epoch,
            output,
            sfreq,
            channel_names,
            sampling_rate=float(sfreq),
            intermediate_sampling_rate=self.target_sfreq,
        )


class ComposeOperator(EEGOperator):
    """Sequential composition retaining a full metadata trace."""

    operator_type = "lossy"

    def __init__(self, operators: Iterable[EEGOperator]):
        self.operators = tuple(operators)
        if not self.operators:
            raise ValueError("ComposeOperator requires at least one operator")
        self.name = "compose__" + "__".join(operator.name for operator in self.operators)
        if all(op.operator_type != "lossy" for op in self.operators):
            self.operator_type = "equivalence"

    def transform(self, epoch, sfreq, channel_names, rng):
        current = epoch
        trace: list[dict[str, Any]] = []
        for operator in self.operators:
            result = operator.transform(current, sfreq, channel_names, rng)
            current = result.data
            trace.append(result.metadata)
        return self._result(
            epoch, current, sfreq, channel_names, composition_trace=trace
        )


def build_operators(config: dict[str, Any]) -> list[EEGOperator]:
    operators: list[EEGOperator] = [
        IdentityOperator(),
        CommonAverageReference(),
        CzReference(config.get("cz_channel", "Cz")),
    ]
    operators.extend(GainOperator(value) for value in config.get("gains", [0.5, 2.0]))
    operators.extend(
        RandomChannelDropout(value)
        for value in config.get("random_dropout_ratios", [0.25, 0.5])
    )
    operators.append(RegionChannelDropout(config.get("region_channels", ["C3", "Cz", "C4"])))
    operators.extend(
        BandpassOperator(float(bounds[0]), float(bounds[1]))
        for bounds in config.get("bandpasses", [[4.0, 30.0], [8.0, 30.0]])
    )
    operators.append(DownUpResampleOperator(config.get("resample_target_sfreq", 125.0)))
    names = [operator.name for operator in operators]
    if len(names) != len(set(names)):
        raise ValueError(f"Operator names must be unique: {names}")
    return operators
