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

run_repeated benchmarks/configs/07_lagged_dynamics/lagged_dynamics_full.yaml
run_repeated benchmarks/configs/07_lagged_dynamics/lagged_dynamics_covariance_only.yaml
run_repeated benchmarks/configs/07_lagged_dynamics/lagged_dynamics_lagged_only.yaml

python benchmarks/scripts/compare_lagged_dynamics.py \
  --lagged-dir "$OUTPUT_DIR/07_lagged_dynamics" \
  --output "$OUTPUT_DIR/07_lagged_dynamics/comparison.md"
