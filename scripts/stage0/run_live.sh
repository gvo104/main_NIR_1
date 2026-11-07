#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

env_file=../.env
settings_file=config/stage0.toml
aide_max_tokens=
request_timeout=

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-file)
            env_file=$2
            shift 2
            ;;
        --config)
            settings_file=$2
            shift 2
            ;;
        --aide-max-tokens)
            aide_max_tokens=$2
            shift 2
            ;;
        --request-timeout)
            request_timeout=$2
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
if [[ ! -f "$settings_file" ]]; then
    echo "Settings file is missing: $settings_file" >&2
    exit 1
fi
executor_image=$(.venvs/aide/bin/python -c 'import sys, tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["runtime"]["executor_image"])' "$settings_file")
if ! docker image inspect "$executor_image" >/dev/null 2>&1; then
    echo "Executor image is missing; run scripts/stage0/run_offline.sh first." >&2
    exit 1
fi
if [[ -n "$request_timeout" ]] && ! [[ "$request_timeout" =~ ^[1-9][0-9]*$ ]]; then
    echo "--request-timeout must be a positive integer in seconds." >&2
    exit 2
fi
if [[ -n "$aide_max_tokens" ]] && ! [[ "$aide_max_tokens" =~ ^[1-9][0-9]*$ ]]; then
    echo "--aide-max-tokens must be a positive integer." >&2
    exit 2
fi

runner_env=(
    RUN_LIVE_LLM=1
    NIR_ENV_FILE="$env_file"
    NIR_SETTINGS_FILE="$settings_file"
    NIR_EXECUTOR_IMAGE="$executor_image"
)
if [[ -n "$aide_max_tokens" ]]; then
    runner_env+=(NIR_AIDE_MAX_TOKENS="$aide_max_tokens")
fi
if [[ -n "$request_timeout" ]]; then
    runner_env+=(NIR_REQUEST_TIMEOUT="$request_timeout")
fi

env "${runner_env[@]}" .venvs/aide/bin/python -m pytest \
    -o addopts='' \
    -m live_llm \
    tests/stage0/test_live.py \
    -s
