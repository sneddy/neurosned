#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/benchmarks/experiments/01_regression_baselines}"
DATASET="${DATASET:-$ROOT_DIR/data/new_validation/r11_test_5sec.pkl}"
DEVICE="${DEVICE:-auto}"
STARTS="${STARTS:-0.2 0.3 0.4 0.5 0.6 0.7 0.8}"
REFERENCE_START="${REFERENCE_START:-0.5}"

if [[ ! -f "$DATASET" ]]; then
  echo "Missing shifted-eval dataset: $DATASET" >&2
  echo "Build it first:" >&2
  echo "  python benchmarks/preparation/scripts/reload_test_5sec_dataset.py" >&2
  exit 1
fi

echo "Shifted regression eval"
echo "Run root: $RUN_ROOT"
echo "Dataset: $DATASET"
echo "Device: $DEVICE"
echo "Starts: $STARTS"
echo

shopt -s nullglob
for run_dir in "$RUN_ROOT"/*; do
  [[ -d "$run_dir" ]] || continue

  if [[ ! -f "$run_dir/config.yaml" ]]; then
    echo "SKIP $(basename "$run_dir"): missing config.yaml"
    continue
  fi
  if [[ ! -f "$run_dir/best_model.pth" ]]; then
    echo "SKIP $(basename "$run_dir"): missing best_model.pth"
    continue
  fi

  echo "RUN  $(basename "$run_dir")"
  # shellcheck disable=SC2086
  python "$ROOT_DIR/benchmarks/scripts/eval_shifted.py" \
    "$run_dir" \
    --dataset "$DATASET" \
    --device "$DEVICE" \
    --starts $STARTS \
    --reference-start "$REFERENCE_START"
  echo
done
