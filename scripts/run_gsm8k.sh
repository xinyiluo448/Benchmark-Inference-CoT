#!/usr/bin/env bash
# Quick pipeline: prepare GSM8K split and run benchmark, store intermediates in staging/.
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

SPLIT="train"
MAX_SAMPLES=100
OUT_JSONL="${STAGING}/gsm8k_${SPLIT}_${MAX_SAMPLES}.jsonl"
OUT_CSV="${STAGING}/gsm8k_${SPLIT}_${MAX_SAMPLES}_run.csv"

python3 "${ROOT_DIR}/data_prep/prepare_gsm8k.py" --split "${SPLIT}" --max-samples "${MAX_SAMPLES}" --out "${OUT_JSONL}" "$@"
python3 "${ROOT_DIR}/bench_framework.py" \
  --task reasoning-gsm8k \
  --data "${OUT_JSONL}" \
  --save-csv "${OUT_CSV}" \
  --max-new-tokens 512

echo "Saved jsonl: ${OUT_JSONL}"
echo "Saved csv  : ${OUT_CSV}"
