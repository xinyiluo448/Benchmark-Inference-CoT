#!/usr/bin/env bash
# Quick pipeline: prepare XSum split and run benchmark, store intermediates in staging/.
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

SPLIT="validation"
MAX_SAMPLES=100
OUT_JSONL="${STAGING}/xsum_${SPLIT}_${MAX_SAMPLES}.jsonl"
OUT_CSV="${STAGING}/xsum_${SPLIT}_${MAX_SAMPLES}_run.csv"

# Split incoming args: known prep args vs bench args
PREP_ARGS=""
BENCH_ARGS=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --split)
      PREP_ARGS="$PREP_ARGS --split $2"
      shift 2
      ;;
    --max-samples)
      PREP_ARGS="$PREP_ARGS --max-samples $2"
      shift 2
      ;;
    --shuffle-seed)
      PREP_ARGS="$PREP_ARGS --shuffle-seed $2"
      shift 2
      ;;
    *)
      BENCH_ARGS="$BENCH_ARGS $1"
      shift
      ;;
  esac
done

# Prepare data
eval python3 "${ROOT_DIR}/data_prep/prepare_xsum.py" --split "${SPLIT}" --max-samples "${MAX_SAMPLES}" --out "${OUT_JSONL}" ${PREP_ARGS}

# Run benchmark (for summarization, you likely want outputs saved for ROUGE later)
eval python3 "${ROOT_DIR}/bench_framework.py" \
  --task summarization-xsum \
  --data "${OUT_JSONL}" \
  --save-csv "${OUT_CSV}" \
  --save-outputs \
  --max-new-tokens 80 \
  ${BENCH_ARGS}

echo "Saved jsonl: ${OUT_JSONL}"
echo "Saved csv  : ${OUT_CSV}"
