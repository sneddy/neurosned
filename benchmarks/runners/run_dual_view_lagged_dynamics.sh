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

run_repeated benchmarks/configs/08_dual_view_lagged_dynamics/raw_view_only.yaml
run_repeated benchmarks/configs/08_dual_view_lagged_dynamics/dual_view_covariance_only.yaml
run_repeated benchmarks/configs/08_dual_view_lagged_dynamics/dual_view_full.yaml

python benchmarks/scripts/compare_dual_view_lagged_dynamics.py \
  --matrix-dir "$OUTPUT_DIR/07_lagged_dynamics" \
  --dual-dir "$OUTPUT_DIR/08_dual_view_lagged_dynamics" \
  --output "$OUTPUT_DIR/08_dual_view_lagged_dynamics/comparison.md"
