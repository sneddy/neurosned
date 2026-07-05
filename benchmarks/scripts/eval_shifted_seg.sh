#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/benchmarks/experiments/02_segmentation_ablations}"
DATASET="${DATASET:-$ROOT_DIR/data/new_validation/r11_test_5sec.pkl}"
DEVICE="${DEVICE:-auto}"
STARTS="${STARTS:-0.2 0.3 0.4 0.5 0.6 0.7 0.8}"
REFERENCE_START="${REFERENCE_START:-0.5}"
PATTERNS="${PATTERNS:-unet_deeper_ce_only__* unet_deeper_event_nll__*}"

if [[ ! -f "$DATASET" ]]; then
  echo "Missing shifted-eval dataset: $DATASET" >&2
  echo "Build it first:" >&2
  echo "  python benchmarks/preparation/scripts/reload_test_5sec_dataset.py" >&2
  exit 1
fi

echo "Shifted segmentation eval"
echo "Run root: $RUN_ROOT"
echo "Dataset: $DATASET"
echo "Device: $DEVICE"
echo "Starts: $STARTS"
echo "Patterns: $PATTERNS"
echo

read -r -a pattern_list <<< "$PATTERNS"
shopt -s nullglob
ran=0
for pattern in "${pattern_list[@]}"; do
  for run_dir in "$RUN_ROOT"/$pattern; do
    [[ -d "$run_dir" ]] || continue

    if [[ ! -f "$run_dir/config.yaml" ]]; then
      echo "SKIP $(basename "$run_dir"): missing config.yaml"
      continue
    fi
    if [[ ! -f "$run_dir/best_model.pth" ]]; then
      echo "SKIP $(basename "$run_dir"): missing best_model.pth"
      continue
    fi

    extra_args=()
    temperature_path="$run_dir/calibration/temperature.json"
    if [[ -f "$temperature_path" ]]; then
      temperature="$(python - "$temperature_path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text())
print(data["best_temperature"])
PY
)"
      extra_args+=(--segmentation-temperature "$temperature")
      echo "RUN  $(basename "$run_dir") tau=$temperature"
    else
      echo "RUN  $(basename "$run_dir") tau=config"
    fi

    ran=$((ran + 1))
    # shellcheck disable=SC2086
    python "$ROOT_DIR/benchmarks/scripts/eval_shifted.py" \
      "$run_dir" \
      --dataset "$DATASET" \
      --device "$DEVICE" \
      --starts $STARTS \
      --reference-start "$REFERENCE_START" \
      "${extra_args[@]}"
    echo
  done
done

if [[ "$ran" -eq 0 ]]; then
  echo "No segmentation runs matched PATTERNS under $RUN_ROOT" >&2
  exit 1
fi
