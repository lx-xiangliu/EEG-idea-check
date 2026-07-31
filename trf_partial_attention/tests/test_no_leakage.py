from src.data import audit_split_leakage


def test_detects_subject_story_segment_and_overlap_leakage() -> None:
    rows = [
        {"split": "train", "subject_id": "s1", "story_id": "a", "segment_id": "x", "start_seconds": 0, "duration_seconds": 5},
        {"split": "test", "subject_id": "s1", "story_id": "a", "segment_id": "x", "start_seconds": 4, "duration_seconds": 5},
    ]
    errors = audit_split_leakage(rows)
    assert any("subjects cross" in error for error in errors)
    assert any("stories cross" in error for error in errors)
    assert any("duplicate segment" in error for error in errors)
    assert any("overlap" in error for error in errors)


def test_clean_subject_and_story_disjoint_manifest() -> None:
    rows = [
        {"split": "train", "subject_id": "s1", "story_id": "a", "segment_id": "x", "start_seconds": 0, "duration_seconds": 5},
        {"split": "test", "subject_id": "s2", "story_id": "b", "segment_id": "y", "start_seconds": 0, "duration_seconds": 5},
    ]
    assert audit_split_leakage(rows) == []

