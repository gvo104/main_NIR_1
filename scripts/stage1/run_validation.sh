#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

if [[ ! -x .venvs/aide/bin/python ]]; then
    echo "AIDE virtual environment is missing; run scripts/stage0/bootstrap.sh first." >&2
    exit 1
fi
if [[ ! -f data/stage1/us_hospital_admissions_vintages.csv ]]; then
    echo "Stage 1 data is missing; run scripts/stage1/prepare.sh first." >&2
    exit 1
fi

program=${1:-scripts/stage1/baseline.py}
image=$(.venvs/aide/bin/python -c 'import tomllib; print(tomllib.load(open("config/stage0.toml", "rb"))["runtime"]["executor_image"])')
if ! docker image inspect "$image" >/dev/null 2>&1; then
    echo "Executor image is missing; run scripts/stage0/run_offline.sh first." >&2
    exit 1
fi

.venvs/aide/bin/python -m nir_stage1.evaluate --program "$program" --image "$image"
