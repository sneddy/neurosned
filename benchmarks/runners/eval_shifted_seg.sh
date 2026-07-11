#!/usr/bin/env sh
set -eu

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

DEVICE="${DEVICE:-auto}"
RUN_ROOTS="${RUN_ROOTS:-benchmarks/experiments/02_segmentation_ablations benchmarks/experiments/03_crop_shift_jitter}"
DATASET="${DATASET:-data/new_validation/r11_test_5sec.pkl}"
STARTS="${STARTS:-0.2 0.3 0.4 0.5 0.6 0.7 0.8}"
REFERENCE_START="${REFERENCE_START:-0.5}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-1000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-2025}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-false}"

case "$SAVE_PREDICTIONS" in
  true|1|yes) SAVE_FLAG="--save-predictions" ;;
  false|0|no) SAVE_FLAG="--no-save-predictions" ;;
  *)
    echo "SAVE_PREDICTIONS must be true or false, got: $SAVE_PREDICTIONS" >&2
    exit 2
    ;;
esac

run_dirs_file="$(mktemp)"
trap 'rm -f "$run_dirs_file"' EXIT

for root in $RUN_ROOTS; do
  if [ -d "$root" ]; then
    find "$root" -mindepth 3 -maxdepth 3 -type f -path '*/seed*/config.yaml' \
      | sort \
      | while IFS= read -r config_path; do
          dirname "$config_path"
        done
  fi
done > "$run_dirs_file"

if [ ! -s "$run_dirs_file" ]; then
  echo "No segmentation seed runs found under: $RUN_ROOTS" >&2
  exit 1
fi

echo "Shifted segmentation eval"
echo "Run roots: $RUN_ROOTS"
echo "Dataset: $DATASET"
echo "Device: $DEVICE"
echo "Starts: $STARTS"
echo "Target support: none; shifted_summary.csv provides all/inside_crop/common_inside masks"
echo "Save predictions: $SAVE_PREDICTIONS"

while IFS= read -r run_dir; do
  echo
  echo "==> Evaluating $run_dir"
  python benchmarks/scripts/eval_shifted.py "$run_dir" \
    --dataset "$DATASET" \
    --starts $STARTS \
    --reference-start "$REFERENCE_START" \
    --device "$DEVICE" \
    --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
    --bootstrap-seed "$BOOTSTRAP_SEED" \
    $SAVE_FLAG
done < "$run_dirs_file"
