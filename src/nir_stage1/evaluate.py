"""Evaluate a point-forecast program at independent rolling origins."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import uuid
from pathlib import Path

from nir_stage0.docker_executor import DockerExecutor

from .data import ForecastFold, VintageDataset
from .prepare import DATA_PATH, ROOT, load_verified_dataset


class PredictionError(ValueError):
    """A candidate did not produce one valid point forecast."""


def read_prediction(path: Path) -> float:
    if not path.is_file():
        raise PredictionError("prediction.json was not created")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PredictionError("prediction.json is not valid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"prediction"}:
        raise PredictionError("prediction.json must contain only 'prediction'")
    value = payload["prediction"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PredictionError("prediction must be a number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise PredictionError("prediction must be finite") from exc
    if not math.isfinite(result) or result < 0:
        raise PredictionError("prediction must be finite and nonnegative")
    return result


def write_fold_inputs(fold: ForecastFold, workspace: Path) -> None:
    input_dir = workspace / "input"
    working_dir = workspace / "working"
    input_dir.mkdir(parents=True, exist_ok=False)
    working_dir.mkdir(parents=True, exist_ok=False)
    with (input_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(("date", "admissions"))
        writer.writerows(
            (date_value.isoformat(), admissions)
            for date_value, admissions in fold.history
        )
    task = {
        "issue_date": fold.issue_date.isoformat(),
        "reference_date": fold.reference_date.isoformat(),
        "target_end_date": fold.target_end_date.isoformat(),
    }
    (input_dir / "task.json").write_text(
        json.dumps(task, indent=2) + "\n", encoding="utf-8"
    )


def evaluate_validation(
    program: Path,
    dataset: VintageDataset,
    *,
    image: str,
    artifact_root: Path,
    executor_type: type[DockerExecutor] = DockerExecutor,
) -> tuple[dict, Path]:
    code = program.read_text(encoding="utf-8")
    run_dir = artifact_root / f"{program.stem}-{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for fold in dataset.folds("validation"):
        workspace = run_dir / "folds" / fold.target_end_date.isoformat()
        write_fold_inputs(fold, workspace)
        execution = executor_type(workspace, image=image)(code)
        output = "".join(execution.term_out)
        (workspace / "stdout_stderr.txt").write_text(output, encoding="utf-8")
        record = {
            "target_end_date": fold.target_end_date.isoformat(),
            "reference_date": fold.reference_date.isoformat(),
            "issue_date": fold.issue_date.isoformat(),
            "as_of": fold.as_of.isoformat(),
            "status": "PASS",
            "prediction": None,
            "absolute_error": None,
            "execution_seconds": execution.exec_time,
        }
        if execution.exc_type:
            record["status"] = "ERROR"
            record["error_type"] = execution.exc_type
        else:
            try:
                prediction = read_prediction(workspace / "working" / "prediction.json")
            except PredictionError as exc:
                record["status"] = "ERROR"
                record["error_type"] = type(exc).__name__
                record["message"] = str(exc)
            else:
                record["prediction"] = prediction
                record["absolute_error"] = abs(prediction - fold.truth)
        results.append(record)

    successes = sum(item["status"] == "PASS" for item in results)
    errors = len(results) - successes
    report = {
        "split": "validation",
        "program": str(program.resolve()),
        "program_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
        "target": "wk inc flu hosp",
        "location": "US",
        "horizon": 1,
        "total_weeks": len(results),
        "successful_weeks": successes,
        "error_weeks": errors,
        "status": "PASS" if errors == 0 else "ERROR",
        "mae": (
            math.fsum(item["absolute_error"] for item in results) / len(results)
            if errors == 0
            else None
        ),
        "folds": results,
    }
    report_path = run_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--program", type=Path)
    parser.add_argument("--inspect-test", action="store_true")
    parser.add_argument(
        "--artifact-root", type=Path, default=ROOT / "artifacts/stage1/validation"
    )
    parser.add_argument("--image", default="nir-stage0-executor:latest")
    args = parser.parse_args()
    dataset = load_verified_dataset(args.data)
    if args.inspect_test:
        if args.program is not None:
            parser.error("--inspect-test cannot be combined with --program")
        folds = dataset.folds("test")
        print(
            json.dumps(
                {
                    "split": "test",
                    "weeks": len(folds),
                    "first_target": folds[0].target_end_date.isoformat(),
                    "last_target": folds[-1].target_end_date.isoformat(),
                    "minimum_history_rows": min(len(fold.history) for fold in folds),
                },
                ensure_ascii=False,
            )
        )
        return
    if args.program is None:
        parser.error("--program is required for validation")
    report, path = evaluate_validation(
        args.program,
        dataset,
        image=args.image,
        artifact_root=args.artifact_root,
    )
    print(
        f"Validation: {report['successful_weeks']}/{report['total_weeks']} "
        f"successful; MAE={report['mae']}; report={path}"
    )
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
