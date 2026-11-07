#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

if [[ ! -x .venvs/aide/bin/python ]]; then
    echo "AIDE virtual environment is missing; run scripts/stage0/bootstrap.sh first." >&2
    exit 1
fi

.venvs/aide/bin/python -m nir_stage1.prepare
