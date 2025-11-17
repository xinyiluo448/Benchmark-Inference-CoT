#!/usr/bin/env bash
# Quick pipeline: prepare SST-2 dev set and run benchmark, store intermediates in staging/.
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

OUT_JSONL="${STAGING}/sst2_dev.jsonl"
OUT_CSV="${STAGING}/sst2_dev_run.csv"

python3 "${ROOT_DIR}/data_prep/prepare_sst2.py" --split validation --out "${OUT_JSONL}" "$@"
python3 "${ROOT_DIR}/bench_framework.py" \
  --task classification-sst2 \
  --data "${OUT_JSONL}" \
  --save-csv "${OUT_CSV}" \
  --max-new-tokens 64

echo "Saved jsonl: ${OUT_JSONL}"
echo "Saved csv  : ${OUT_CSV}"
