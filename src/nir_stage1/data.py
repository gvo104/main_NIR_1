"""Read frozen FluSight vintages and construct leakage-free forecast origins."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

TARGET = "wk inc flu hosp"
LOCATION = "US"
FIRST_HISTORY_DATE = date(2022, 10, 1)
AS_OF_CHANGE = date(2025, 7, 5)
SPLITS = {
    "validation": (date(2024, 10, 5), date(2025, 5, 31)),
    "test": (date(2025, 10, 4), date(2026, 5, 30)),
}
EXPECTED_WEEKS = 35
DATA_COLUMNS = ("as_of", "target_end_date", "observation")


class DataError(ValueError):
    """Invalid source data or an unavailable forecast origin."""


def release_date(as_of: date) -> date:
    """Normalize old Saturday reference dates to their Wednesday release."""

    return as_of - timedelta(days=3) if as_of < AS_OF_CHANGE else as_of


@dataclass(frozen=True)
class ForecastFold:
    target_end_date: date
    reference_date: date
    issue_date: date
    as_of: date
    history: tuple[tuple[date, int], ...]
    truth: int


class VintageDataset:
    def __init__(self, snapshots: dict[date, dict[date, int]]) -> None:
        if not snapshots:
            raise DataError("no US influenza hospitalization vintages found")
        self.snapshots = snapshots
        self.ordered_vintages = sorted(
            snapshots, key=lambda as_of: (release_date(as_of), as_of)
        )

    @classmethod
    def from_csv(cls, path: str | Path) -> VintageDataset:
        snapshots: dict[date, dict[date, int]] = {}
        with Path(path).open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if tuple(reader.fieldnames or ()) != DATA_COLUMNS:
                raise DataError(f"expected CSV columns: {', '.join(DATA_COLUMNS)}")
            for line_number, row in enumerate(reader, start=2):
                try:
                    as_of = date.fromisoformat(row["as_of"])
                    target_date = date.fromisoformat(row["target_end_date"])
                    value = int(row["observation"])
                except (TypeError, ValueError) as exc:
                    raise DataError(f"invalid row {line_number}") from exc
                if target_date.weekday() != 5:
                    raise DataError(f"target date is not Saturday at row {line_number}")
                if value < 0:
                    raise DataError(f"negative observation at row {line_number}")
                snapshot = snapshots.setdefault(as_of, {})
                if target_date in snapshot:
                    raise DataError(
                        f"duplicate vintage and target at row {line_number}"
                    )
                snapshot[target_date] = value
        return cls(snapshots)

    def fold(self, target_end_date: date) -> ForecastFold:
        if target_end_date.weekday() != 5:
            raise DataError("target date must be Saturday")
        reference_date = target_end_date - timedelta(days=7)
        issue_date = reference_date - timedelta(days=3)
        eligible = [
            as_of
            for as_of in self.ordered_vintages
            if release_date(as_of) <= issue_date
        ]
        if not eligible:
            raise DataError(f"no released vintage for {issue_date}")
        as_of = eligible[-1]
        history = tuple(
            sorted(
                (target_date, value)
                for target_date, value in self.snapshots[as_of].items()
                if FIRST_HISTORY_DATE <= target_date < reference_date
                and target_date <= issue_date
            )
        )
        if not history:
            raise DataError(f"no historical observations for {issue_date}")
        latest = self.snapshots[self.ordered_vintages[-1]]
        if target_end_date not in latest:
            raise DataError(f"no final truth for {target_end_date}")
        return ForecastFold(
            target_end_date=target_end_date,
            reference_date=reference_date,
            issue_date=issue_date,
            as_of=as_of,
            history=history,
            truth=latest[target_end_date],
        )

    def folds(self, split: str) -> tuple[ForecastFold, ...]:
        try:
            start, end = SPLITS[split]
        except KeyError as exc:
            raise DataError(f"unknown split: {split}") from exc
        folds = tuple(
            self.fold(start + timedelta(weeks=week))
            for week in range((end - start).days // 7 + 1)
        )
        if len(folds) != EXPECTED_WEEKS or folds[-1].target_end_date != end:
            raise DataError(f"{split} must contain {EXPECTED_WEEKS} target weeks")
        return folds
