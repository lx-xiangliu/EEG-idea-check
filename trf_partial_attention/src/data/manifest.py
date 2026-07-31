from __future__ import annotations

from collections.abc import Iterable


def audit_split_leakage(rows: Iterable[dict[str, object]]) -> list[str]:
    """Return explicit split violations for segment-level manifests."""
    materialized = list(rows)
    errors: list[str] = []
    split_subjects: dict[str, set[object]] = {}
    split_stories: dict[str, set[object]] = {}
    seen_segment: dict[object, str] = {}
    intervals: dict[tuple[object, object], list[tuple[float, float, str]]] = {}
    for row in materialized:
        split = str(row["split"])
        split_subjects.setdefault(split, set()).add(row["subject_id"])
        split_stories.setdefault(split, set()).add(row["story_id"])
        segment = row["segment_id"]
        if segment in seen_segment and seen_segment[segment] != split:
            errors.append(f"duplicate segment {segment} crosses {seen_segment[segment]}/{split}")
        seen_segment[segment] = split
        start = float(row["start_seconds"])
        end = start + float(row["duration_seconds"])
        key = (row["subject_id"], row["story_id"])
        for old_start, old_end, old_split in intervals.setdefault(key, []):
            if old_split != split and max(start, old_start) < min(end, old_end):
                errors.append(f"overlap for subject/story {key} crosses {old_split}/{split}")
        intervals[key].append((start, end, split))
    splits = sorted(split_subjects)
    for i, left in enumerate(splits):
        for right in splits[i + 1 :]:
            common_subjects = split_subjects[left] & split_subjects[right]
            if common_subjects:
                errors.append(f"subjects cross {left}/{right}: {sorted(map(str, common_subjects))}")
            common_stories = split_stories[left] & split_stories[right]
            if common_stories:
                errors.append(f"stories cross {left}/{right}: {sorted(map(str, common_stories))}")
    return errors

