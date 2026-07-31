from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SplitRecord:
    record_id: str
    subject_id: str
    stimulus_id: str
    start_sample: int
    end_sample: int
    split: str
    eeg_path: str = ""
    audio_path: str = ""

    def __post_init__(self) -> None:
        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split {self.split!r} for {self.record_id}")
        if self.start_sample < 0 or self.end_sample <= self.start_sample:
            raise ValueError(f"Invalid interval for {self.record_id}")


@dataclass(frozen=True)
class DatasetManifest:
    dataset_name: str
    version: str
    sampling_rate: float
    records: tuple[SplitRecord, ...]


def load_manifest(path: str | Path) -> DatasetManifest:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Real-data manifest is missing: {path}. Synthetic data are never used as a silent fallback."
        )
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    manifest = DatasetManifest(
        dataset_name=str(payload["dataset_name"]),
        version=str(payload["version"]),
        sampling_rate=float(payload["sampling_rate"]),
        records=tuple(SplitRecord(**row) for row in payload["records"]),
    )
    validate_no_leakage(manifest.records)
    return manifest


def validate_no_leakage(records: tuple[SplitRecord, ...] | list[SplitRecord]) -> None:
    if not records:
        raise ValueError("Manifest has no records")
    seen_ids: set[str] = set()
    subject_splits: dict[str, set[str]] = {}
    stimulus_splits: dict[str, set[str]] = {}
    by_source: dict[tuple[str, str], list[SplitRecord]] = {}
    for record in records:
        if record.record_id in seen_ids:
            raise ValueError(f"Duplicate record_id: {record.record_id}")
        seen_ids.add(record.record_id)
        subject_splits.setdefault(record.subject_id, set()).add(record.split)
        stimulus_splits.setdefault(record.stimulus_id, set()).add(record.split)
        by_source.setdefault((record.subject_id, record.stimulus_id), []).append(record)
    leaking_subjects = {key: value for key, value in subject_splits.items() if len(value) > 1}
    if leaking_subjects:
        raise ValueError(f"Subject leakage across splits: {leaking_subjects}")
    leaking_stimuli = {key: value for key, value in stimulus_splits.items() if len(value) > 1}
    if leaking_stimuli:
        raise ValueError(f"Stimulus leakage across splits: {leaking_stimuli}")
    for key, group in by_source.items():
        ordered = sorted(group, key=lambda row: row.start_sample)
        for left, right in zip(ordered, ordered[1:]):
            if left.end_sample > right.start_sample and left.split != right.split:
                raise ValueError(f"Overlapping windows cross splits for source {key}")
