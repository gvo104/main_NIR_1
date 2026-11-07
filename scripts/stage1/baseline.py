"""Forecast the last released weekly hospitalization count."""

import csv
import json
from pathlib import Path

history_path = Path("input/history.csv")
with history_path.open(newline="", encoding="utf-8") as file:
    history = list(csv.DictReader(file))
if not history:
    raise ValueError("history is empty")

prediction = int(history[-1]["admissions"])
Path("working/prediction.json").write_text(
    json.dumps({"prediction": prediction}) + "\n", encoding="utf-8"
)
