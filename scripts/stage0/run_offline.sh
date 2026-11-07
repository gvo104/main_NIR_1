#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

if [[ ! -x .venvs/era/bin/python || ! -x .venvs/aide/bin/python ]]; then
    echo "Virtual environments are missing; run scripts/stage0/bootstrap.sh first." >&2
    exit 1
fi

executor_image=$(.venvs/era/bin/python -c 'import tomllib; print(tomllib.load(open("config/stage0.toml", "rb"))["runtime"]["executor_image"])')
docker build \
    --file docker/stage0-executor/Dockerfile \
    --tag "$executor_image" \
    .

PYTHONDONTWRITEBYTECODE=1 .venvs/era/bin/python -m pytest \
    tests/stage0/test_config.py \
    tests/stage0/test_llm_client.py \
    tests/stage0/test_futs.py

NIR_EXECUTOR_IMAGE="$executor_image" \
    .venvs/aide/bin/python -m pytest \
    tests/stage0/test_aide_policy.py \
    tests/stage0/test_docker_executor.py \
    tests/stage0/test_live_runner.py
