"""Extract a compact, immutable US series from a pinned FluSight source."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

from .data import DATA_COLUMNS, LOCATION, TARGET, VintageDataset

SOURCE_COMMIT = "d03574028ece110013c6284d3012ac4bef03ded7"
SOURCE_SHA256 = "2524463c6b02791b4276054986287e747c8a8f6c99dfe59cf6d23ed67812f98d"
FILTERED_SHA256 = "b84c1ec21a1632e18ab0ee381027a0ffa0955afb73ca0357bcb0aaca627a1e3b"
SOURCE_URL = (
    "https://raw.githubusercontent.com/cdcepi/FluSight-forecast-hub/"
    f"{SOURCE_COMMIT}/target-data/time-series.csv"
)
LICENSE_URL = (
    f"https://github.com/cdcepi/FluSight-forecast-hub/blob/{SOURCE_COMMIT}/LICENSE"
)
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/stage1"
DATA_PATH = DATA_DIR / "us_hospital_admissions_vintages.csv"
PROVENANCE_PATH = DATA_DIR / "provenance.json"
TEST_MANIFEST_PATH = DATA_DIR / "test_manifest.json"
REQUIRED_SOURCE_COLUMNS = {
    "as_of",
    "target",
    "target_end_date",
    "location",
    "observation",
}


def extract_us_vintages(
    source: io.BufferedIOBase,
) -> tuple[bytes, str, int, int]:
    source_digest = hashlib.sha256()

    def lines():
        for line in source:
            source_digest.update(line)
            yield line.decode("utf-8-sig")

    reader = csv.DictReader(lines())
    if not REQUIRED_SOURCE_COLUMNS <= set(reader.fieldnames or ()):
        raise ValueError("FluSight source is missing required columns")
    unique: dict[tuple[str, str], str] = {}
    identical_duplicates = 0
    for row in reader:
        if row["location"] != LOCATION or row["target"] != TARGET:
            continue
        key = (row["as_of"], row["target_end_date"])
        value = row["observation"]
        if key in unique:
            if unique[key] != value:
                raise ValueError(f"conflicting observations for {key}")
            identical_duplicates += 1
        else:
            unique[key] = value
    rows = sorted((as_of, target, value) for (as_of, target), value in unique.items())
    if not rows:
        raise ValueError("FluSight source has no US hospitalization rows")

    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(DATA_COLUMNS)
    writer.writerows(rows)
    return (
        output.getvalue().encode("utf-8"),
        source_digest.hexdigest(),
        len(rows),
        identical_duplicates,
    )


def prepare(source: io.BufferedIOBase, *, data_dir: Path = DATA_DIR) -> dict:
    filtered, source_sha256, rows, duplicate_rows = extract_us_vintages(source)
    filtered_sha256 = hashlib.sha256(filtered).hexdigest()
    if source_sha256 != SOURCE_SHA256 or filtered_sha256 != FILTERED_SHA256:
        raise ValueError("pinned FluSight source or filtered output checksum mismatch")
    with tempfile.TemporaryDirectory(prefix="nir-stage1-") as temp_dir:
        temp_path = Path(temp_dir) / DATA_PATH.name
        temp_path.write_bytes(filtered)
        dataset = VintageDataset.from_csv(temp_path)
        validation = dataset.folds("validation")
        test = dataset.folds("test")

    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / DATA_PATH.name).write_bytes(filtered)
    metadata = {
        "source_commit": SOURCE_COMMIT,
        "source_url": SOURCE_URL,
        "source_sha256": source_sha256,
        "filtered_sha256": filtered_sha256,
        "repository_license": "MIT",
        "repository_license_url": LICENSE_URL,
        "target": TARGET,
        "location": LOCATION,
        "rows": rows,
        "source_matching_rows": rows + duplicate_rows,
        "discarded_identical_duplicates": duplicate_rows,
        "vintages": len(dataset.snapshots),
        "validation_weeks": len(validation),
        "test_weeks": len(test),
    }
    (data_dir / PROVENANCE_PATH.name).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    test_manifest = {
        "split": "test",
        "target_end_dates": [fold.target_end_date.isoformat() for fold in test],
        "reference_dates": [fold.reference_date.isoformat() for fold in test],
        "issue_dates": [fold.issue_date.isoformat() for fold in test],
        "history_rows": [len(fold.history) for fold in test],
    }
    (data_dir / TEST_MANIFEST_PATH.name).write_text(
        json.dumps(test_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metadata


def load_verified_dataset(data_path: Path = DATA_PATH) -> VintageDataset:
    manifest_path = data_path.parent / PROVENANCE_PATH.name
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_sha256 = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if (
        metadata.get("source_commit") != SOURCE_COMMIT
        or metadata.get("source_sha256") != SOURCE_SHA256
        or metadata.get("filtered_sha256") != FILTERED_SHA256
        or actual_sha256 != FILTERED_SHA256
    ):
        raise ValueError("stage 1 data provenance or checksum mismatch")
    dataset = VintageDataset.from_csv(data_path)
    if len(dataset.snapshots) != metadata.get("vintages"):
        raise ValueError("stage 1 vintage count mismatch")
    return dataset


def main() -> None:
    request = Request(SOURCE_URL, headers={"User-Agent": "nir-stage1-data-prep"})
    with urlopen(request, timeout=180) as response:
        metadata = prepare(response)
    print(
        f"Prepared {metadata['rows']} US rows from {metadata['vintages']} vintages; "
        f"removed {metadata['discarded_identical_duplicates']} identical source duplicates; "
        f"{metadata['validation_weeks']} validation and "
        f"{metadata['test_weeks']} test weeks."
    )


if __name__ == "__main__":
    main()
