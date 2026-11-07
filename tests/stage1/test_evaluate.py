from __future__ import annotations

import json
from datetime import date, timedelta

import pytest
from aide.interpreter import ExecutionResult

from nir_stage1.data import VintageDataset
from nir_stage1.evaluate import (
    PredictionError,
    evaluate_validation,
    read_prediction,
    write_fold_inputs,
)


def _dataset() -> VintageDataset:
    validation_start = date(2024, 10, 5)
    test_start = date(2025, 10, 4)
    final_truth = {validation_start + timedelta(weeks=i): 5 for i in range(35)}
    final_truth.update({test_start + timedelta(weeks=i): 9 for i in range(35)})
    return VintageDataset(
        {
            date(2024, 9, 28): {date(2024, 9, 14): 2},
            date(2025, 9, 24): {date(2025, 9, 13): 4},
            date(2026, 11, 4): final_truth,
        }
    )


def test_workspace_has_only_released_inputs_and_no_truth(tmp_path):
    fold = _dataset().fold(date(2025, 10, 4))
    workspace = tmp_path / "workspace"

    write_fold_inputs(fold, workspace)

    files = sorted(
        str(path.relative_to(workspace))
        for path in workspace.rglob("*")
        if path.is_file()
    )
    assert files == ["input/history.csv", "input/task.json"]
    assert '"truth"' not in (workspace / "input/task.json").read_text()
    assert "2025-10-04" not in (workspace / "input/history.csv").read_text()
    assert "2025-09-13,4" in (workspace / "input/history.csv").read_text()


@pytest.mark.parametrize(
    "payload",
    [
        '{"prediction": -1}',
        '{"prediction": true}',
        '{"prediction": "5"}',
        '{"prediction": NaN}',
        '{"prediction": 5, "extra": 0}',
        "not-json",
    ],
)
def test_invalid_predictions_are_rejected(tmp_path, payload):
    path = tmp_path / "prediction.json"
    path.write_text(payload)
    with pytest.raises(PredictionError):
        read_prediction(path)


def test_validation_uses_same_executor_for_baseline_and_candidates(tmp_path):
    program = tmp_path / "candidate.py"
    program.write_text("print('candidate')")

    class FakeExecutor:
        def __init__(self, workspace, *, image):
            assert image == "fixture-image"
            self.workspace = workspace

        def __call__(self, code):
            assert code == "print('candidate')"
            (self.workspace / "working/prediction.json").write_text(
                '{"prediction": 2.5}'
            )
            return ExecutionResult(["ok"], 0.01, None, None, [])

    report, path = evaluate_validation(
        program,
        _dataset(),
        image="fixture-image",
        artifact_root=tmp_path / "artifacts",
        executor_type=FakeExecutor,
    )

    assert report["total_weeks"] == 35
    assert report["successful_weeks"] == 35
    assert report["mae"] == 2.5
    assert json.loads(path.read_text())["mae"] == 2.5


def test_execution_error_invalidates_candidate_mae(tmp_path):
    program = tmp_path / "broken.py"
    program.write_text("raise RuntimeError()")

    class FakeExecutor:
        def __init__(self, workspace, *, image):
            pass

        def __call__(self, code):
            return ExecutionResult(["failure"], 0.01, "RuntimeError", {}, [])

    report, _ = evaluate_validation(
        program,
        _dataset(),
        image="fixture-image",
        artifact_root=tmp_path / "artifacts",
        executor_type=FakeExecutor,
    )
    assert report["error_weeks"] == 35
    assert report["mae"] is None
