#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

python_bin=${PYTHON_BIN:-python3.12}
command -v "$python_bin" >/dev/null

expected_era=012c2d3957a3a6a249af2bbeeae53bbbd2182558
expected_aide=ad73d227af481bea3de06f884f41b2993dd6453a
ensure_upstream() {
    local path=$1
    local url=$2
    local expected_sha=$3

    # A committed clone has gitlinks in the index; the current uncommitted
    # research setup instead contains ordinary nested repositories.
    if git ls-files --stage -- "$path" | awk '$1 == "160000" { found = 1 } END { exit !found }'; then
        git submodule update --init --depth 1 "$path"
    elif git -C "$path" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        : # Reuse the already-cloned local upstream repository.
    elif [[ ! -e "$path" ]]; then
        git -c http.version=HTTP/1.1 clone --depth 1 "$url" "$path"
    else
        echo "Upstream path exists but is not a Git repository: $path" >&2
        exit 1
    fi

    local actual_sha
    actual_sha=$(git -C "$path" rev-parse HEAD)
    if [[ "$actual_sha" != "$expected_sha" ]]; then
        echo "Commit mismatch for $path: expected $expected_sha, got $actual_sha" >&2
        exit 1
    fi
}

ensure_upstream third_party/era https://github.com/google-research/era.git "$expected_era"
ensure_upstream third_party/aideml https://github.com/WecoAI/aideml.git "$expected_aide"

actual_era=$(git -C third_party/era rev-parse HEAD)
actual_aide=$(git -C third_party/aideml rev-parse HEAD)

if [[ "$actual_era" != "$expected_era" ]]; then
    echo "ERA commit mismatch: expected $expected_era, got $actual_era" >&2
    exit 1
fi
if [[ "$actual_aide" != "$expected_aide" ]]; then
    echo "AIDE commit mismatch: expected $expected_aide, got $actual_aide" >&2
    exit 1
fi

"$python_bin" -m venv .venvs/era
.venvs/era/bin/python -m pip install --upgrade pip
.venvs/era/bin/python -m pip install -e '.[test]'

"$python_bin" -m venv .venvs/aide
.venvs/aide/bin/python -m pip install --upgrade pip
.venvs/aide/bin/python -m pip uninstall --yes aideml >/dev/null 2>&1 || true
.venvs/aide/bin/python -m pip install -e '.[test,aide]'
.venvs/aide/bin/python -m pip install --no-deps -e third_party/aideml

.venvs/era/bin/python -c 'import nir_stage0; print("ERA harness import: OK")'
.venvs/aide/bin/python -c 'import aide, nir_stage0; print("AIDE harness import: OK")'

printf 'Python: %s\n' "$("$python_bin" --version)"
printf 'ERA: %s\n' "$actual_era"
printf 'AIDE: %s\n' "$actual_aide"
