from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path

import pytest

from nir_stage1.data import (
    DataError,
    VintageDataset,
    release_date,
)
from nir_stage1.prepare import DATA_PATH


def test_old_and_new_as_of_conventions():
    assert release_date(date(2024, 9, 28)) == date(2024, 9, 25)
    assert release_date(date(2025, 9, 24)) == date(2025, 9, 24)


def test_each_fold_uses_one_released_snapshot():
    initial = date(2024, 9, 28)
    updated = date(2024, 10, 5)
    latest = date(2026, 11, 4)
    snapshots = {
        initial: {date(2024, 9, 14): 10, date(2024, 9, 21): 12},
        updated: {
            date(2024, 9, 14): 99,
            date(2024, 9, 21): 13,
            date(2024, 9, 28): 14,
        },
        latest: {date(2024, 10, 5): 20},
    }
    dataset = VintageDataset(snapshots)

    fold = dataset.fold(date(2024, 10, 5))

    assert fold.issue_date == date(2024, 9, 25)
    assert fold.reference_date == date(2024, 9, 28)
    assert fold.as_of == initial
    assert fold.history == (
        (date(2024, 9, 14), 10),
        (date(2024, 9, 21), 12),
    )
    assert fold.truth == 20
    assert all(day < fold.reference_date for day, _ in fold.history)


@pytest.mark.parametrize(
    ("target", "value", "message"),
    [
        ("2024-10-04", "2", "not Saturday"),
        ("2024-10-05", "-1", "negative"),
        ("2024-10-05", "not-an-integer", "invalid row"),
    ],
)
def test_rejects_invalid_source_rows(tmp_path, target, value, message):
    path = tmp_path / "source.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("as_of", "target_end_date", "observation"))
        writer.writerow(("2024-09-28", target, value))

    with pytest.raises(DataError, match=message):
        VintageDataset.from_csv(path)


def test_rejects_duplicate_vintage_target(tmp_path):
    path = tmp_path / "source.csv"
    path.write_text(
        "as_of,target_end_date,observation\n"
        "2024-09-28,2024-09-21,1\n"
        "2024-09-28,2024-09-21,2\n",
        encoding="utf-8",
    )

    with pytest.raises(DataError, match="duplicate"):
        VintageDataset.from_csv(path)


def test_pinned_source_covers_both_complete_seasons_without_lookahead():
    dataset = VintageDataset.from_csv(DATA_PATH)
    for split in ("validation", "test"):
        folds = dataset.folds(split)
        assert len(folds) == 35
        assert all(release_date(fold.as_of) <= fold.issue_date for fold in folds)
        assert all(
            fold.history
            and max(day for day, _ in fold.history)
            <= fold.reference_date - timedelta(days=7)
            for fold in folds
        )
