import pytest

from src.data import SplitRecord, validate_no_leakage


def test_clean_subject_and_stimulus_disjoint_manifest() -> None:
    records = [
        SplitRecord("a", "s1", "story1", 0, 100, "train"),
        SplitRecord("b", "s2", "story2", 0, 100, "val"),
        SplitRecord("c", "s3", "story3", 0, 100, "test"),
    ]
    validate_no_leakage(records)


def test_subject_leakage_fails_loudly() -> None:
    records = [
        SplitRecord("a", "s1", "story1", 0, 100, "train"),
        SplitRecord("b", "s1", "story2", 0, 100, "test"),
    ]
    with pytest.raises(ValueError, match="Subject leakage"):
        validate_no_leakage(records)


def test_stimulus_leakage_fails_loudly() -> None:
    records = [
        SplitRecord("a", "s1", "story1", 0, 100, "train"),
        SplitRecord("b", "s2", "story1", 200, 300, "test"),
    ]
    with pytest.raises(ValueError, match="Stimulus leakage"):
        validate_no_leakage(records)
