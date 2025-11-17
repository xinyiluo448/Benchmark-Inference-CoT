#!/usr/bin/env bash
# Generic pipeline: run a custom jsonl/json/csv through the benchmark, storing outputs in staging/.
# Usage: scripts/run_custom.sh --data /path/to/file.jsonl [extra bench_framework args...]
# Works with bash; falls back to POSIX set when run under sh.
if [ -n "${BASH_VERSION-}" ]; then
  set -euo pipefail
else
  set -eu
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
STAGING="${ROOT_DIR}/staging"
mkdir -p "${STAGING}"

python3 "${ROOT_DIR}/bench_framework.py" "$@" --save-csv "${STAGING}/custom_run.csv"

echo "Saved csv: ${STAGING}/custom_run.csv"
