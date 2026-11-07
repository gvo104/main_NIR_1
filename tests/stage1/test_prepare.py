from __future__ import annotations

import hashlib
import io
from shutil import copy2

import pytest

from nir_stage1.prepare import (
    DATA_PATH,
    extract_us_vintages,
    load_verified_dataset,
)


def test_extract_us_vintages_is_sorted_and_deterministic():
    source = (
        "as_of,target,location,target_end_date,observation\n"
        "2024-10-05,wk inc flu hosp,US,2024-09-28,3\n"
        "2024-09-28,wk inc flu hosp,US,2024-09-21,2\n"
        "2024-09-28,wk inc flu hosp,CA,2024-09-21,999\n"
        "2024-09-28,other,US,2024-09-21,888\n"
    ).encode()

    filtered, digest, count, duplicates = extract_us_vintages(io.BytesIO(source))

    assert filtered.decode() == (
        "as_of,target_end_date,observation\n"
        "2024-09-28,2024-09-21,2\n"
        "2024-10-05,2024-09-28,3\n"
    )
    assert digest == hashlib.sha256(source).hexdigest()
    assert count == 2
    assert duplicates == 0


def test_exact_duplicate_is_counted_but_conflicting_duplicate_fails():
    prefix = "as_of,target,location,target_end_date,observation\n"
    row = "2024-09-28,wk inc flu hosp,US,2024-09-21,2\n"
    _, _, count, duplicates = extract_us_vintages(
        io.BytesIO((prefix + row + row).encode())
    )
    assert count == 1
    assert duplicates == 1

    with pytest.raises(ValueError, match="conflicting observations"):
        extract_us_vintages(
            io.BytesIO((prefix + row + row.replace(",2\n", ",3\n")).encode())
        )


def test_committed_snapshot_matches_recorded_checksum(tmp_path):
    data_path = tmp_path / DATA_PATH.name
    copy2(DATA_PATH, data_path)
    copy2(DATA_PATH.parent / "provenance.json", tmp_path / "provenance.json")
    assert load_verified_dataset(data_path).snapshots

    with data_path.open("a", encoding="utf-8") as file:
        file.write("\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_verified_dataset(data_path)
