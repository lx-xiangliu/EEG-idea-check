"""Traditional EEG feature extraction with an explicit schema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch
from scipy.stats import kurtosis, skew


@dataclass(frozen=True)
class FeatureSet:
    values: np.ndarray
    schema: tuple[str, ...]


class FeatureExtractor:
    def __init__(self, config: dict[str, Any]):
        self.config = config

    @staticmethod
    def _upper_triangle(matrix: np.ndarray) -> np.ndarray:
        indices = np.triu_indices(matrix.shape[0])
        return matrix[indices]

    def transform_epoch(
        self, epoch: np.ndarray, sfreq: float, channel_names: tuple[str, ...]
    ) -> tuple[np.ndarray, list[str]]:
        if epoch.ndim != 2:
            raise ValueError("epoch must have shape [channels, samples]")
        values: list[np.ndarray] = []
        names: list[str] = []
        if self.config.get("bandpower", True):
            frequencies, power = welch(
                epoch, fs=sfreq, nperseg=min(epoch.shape[-1], max(64, int(sfreq))), axis=-1
            )
            for band_name, limits in self.config["bands"].items():
                low, high = map(float, limits)
                mask = (frequencies >= low) & (frequencies < high)
                if not mask.any():
                    band_values = np.zeros(epoch.shape[0])
                else:
                    band_values = trapezoid(
                        power[:, mask], frequencies[mask], axis=-1
                    )
                values.append(np.log10(np.maximum(band_values, np.finfo(float).tiny)))
                names.extend(f"bandpower.{band_name}.{channel}" for channel in channel_names)
        if self.config.get("moments", True):
            # scipy.stats correctly returns NaN for skew/kurtosis of a constant
            # (dropped) channel. Convert that defined edge case to zero without
            # emitting repeated runtime warnings.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                skew_values = skew(epoch, axis=-1, bias=False)
                kurtosis_values = kurtosis(epoch, axis=-1, bias=False)
            moment_items = [
                ("mean", np.mean(epoch, axis=-1)),
                ("std", np.std(epoch, axis=-1)),
                ("ptp", np.ptp(epoch, axis=-1)),
                (
                    "skew",
                    np.nan_to_num(
                        skew_values,
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    ),
                ),
                (
                    "kurtosis",
                    np.nan_to_num(
                        kurtosis_values,
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    ),
                ),
            ]
            for moment_name, moment_values in moment_items:
                values.append(np.asarray(moment_values))
                names.extend(f"{moment_name}.{channel}" for channel in channel_names)
        covariance = np.cov(epoch)
        covariance = np.atleast_2d(covariance)
        if self.config.get("covariance", True):
            values.append(self._upper_triangle(covariance))
            names.extend(
                f"cov.{channel_names[i]}.{channel_names[j]}"
                for i, j in zip(*np.triu_indices(len(channel_names)))
            )
        if self.config.get("correlation", True):
            # A dropped channel is constant, so its Pearson denominator is zero.
            # We handle that case explicitly and store zero correlation.
            with np.errstate(divide="ignore", invalid="ignore"):
                correlation = np.corrcoef(epoch)
            correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
            values.append(self._upper_triangle(correlation))
            names.extend(
                f"corr.{channel_names[i]}.{channel_names[j]}"
                for i, j in zip(*np.triu_indices(len(channel_names)))
            )
        if self.config.get("covariance_eigenvalues", True):
            count = min(int(self.config.get("eigenvalue_count", 5)), len(channel_names))
            eigenvalues = np.linalg.eigvalsh(covariance)[::-1][:count]
            values.append(eigenvalues)
            names.extend(f"cov_eigenvalue.{index + 1}" for index in range(count))
        feature_vector = np.concatenate([np.ravel(value) for value in values]).astype(float)
        if not np.isfinite(feature_vector).all():
            bad = np.flatnonzero(~np.isfinite(feature_vector))[:10]
            raise ValueError(f"Feature extraction produced NaN/Inf at indices {bad.tolist()}")
        return feature_vector, names

    def transform(
        self, epochs: np.ndarray, sfreq: float, channel_names: tuple[str, ...]
    ) -> FeatureSet:
        rows: list[np.ndarray] = []
        schema: list[str] | None = None
        for epoch in epochs:
            row, row_schema = self.transform_epoch(epoch, sfreq, channel_names)
            if schema is None:
                schema = row_schema
            elif row_schema != schema:
                raise RuntimeError("Feature schema changed between epochs")
            rows.append(row)
        matrix = np.stack(rows)
        if not np.isfinite(matrix).all():
            raise ValueError("Feature matrix contains NaN or Inf")
        return FeatureSet(matrix, tuple(schema or []))
