#!/usr/bin/env sh
set -eu

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

DEVICE="${DEVICE:-auto}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmarks/experiments}"

run_repeated() {
  config="$1"
  echo
  echo "==> Running $config"
  python benchmarks/scripts/run_repeated.py "$config" \
    --device "$DEVICE" \
    --output-dir "$OUTPUT_DIR"
}

run_repeated benchmarks/configs/03_crop_shift_jitter/ets_unet_ce_shift_jitter.yaml
run_repeated benchmarks/configs/03_crop_shift_jitter/ets_unet_event_nll_shift_jitter.yaml
run_repeated benchmarks/configs/03_crop_shift_jitter/ets_unet_event_nll_mixture_shift_jitter.yaml
run_repeated benchmarks/configs/03_crop_shift_jitter/ets_unet_hazard_event_nll_shift_jitter.yaml
run_repeated benchmarks/configs/03_crop_shift_jitter/ets_unet_time_only_shift_jitter.yaml
run_repeated benchmarks/configs/03_crop_shift_jitter/ets_unet_wasserstein_shift_jitter.yaml
