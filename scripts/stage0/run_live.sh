#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

env_file=../.env
aide_max_tokens=4096

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-file)
            env_file=$2
            shift 2
            ;;
        --aide-max-tokens)
            aide_max_tokens=$2
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ ! -x .venvs/aide/bin/python ]]; then
    echo "AIDE virtual environment is missing; run scripts/stage0/bootstrap.sh first." >&2
    exit 1
fi
if ! docker image inspect nir-stage0-executor:latest >/dev/null 2>&1; then
    echo "Executor image is missing; run scripts/stage0/run_offline.sh first." >&2
    exit 1
fi

RUN_LIVE_LLM=1 \
NIR_ENV_FILE="$env_file" \
NIR_AIDE_MAX_TOKENS="$aide_max_tokens" \
NIR_EXECUTOR_IMAGE=nir-stage0-executor:latest \
    .venvs/aide/bin/python -m pytest \
    -o addopts='' \
    -m live_llm \
    tests/stage0/test_live.py \
    -s
