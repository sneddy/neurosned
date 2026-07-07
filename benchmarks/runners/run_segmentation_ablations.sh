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

run_repeated benchmarks/configs/02_segmentation_ablations/ets_unet_ce.yaml
run_repeated benchmarks/configs/02_segmentation_ablations/ets_unet_ce_time.yaml
run_repeated benchmarks/configs/02_segmentation_ablations/ets_unet_event_nll.yaml
run_repeated benchmarks/configs/02_segmentation_ablations/ets_unet_time_only.yaml
run_repeated benchmarks/configs/02_segmentation_ablations/ets_unet_wasserstein.yaml
run_repeated benchmarks/configs/02_segmentation_ablations/ets_unet_hazard_event_nll.yaml
run_repeated benchmarks/configs/02_segmentation_ablations/ets_unet_event_nll_mixture.yaml
