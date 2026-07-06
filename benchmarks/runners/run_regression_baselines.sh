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

run_repeated benchmarks/configs/01_regression_baselines/msp_cnn.yaml
run_repeated benchmarks/configs/01_regression_baselines/etr_cnn.yaml
run_repeated benchmarks/configs/01_regression_baselines/etr_cnn_large.yaml

run_repeated benchmarks/configs/01_regression_baselines/tidnet_wrapped.yaml
run_repeated benchmarks/configs/01_regression_baselines/eegconformer_wrapped.yaml
run_repeated benchmarks/configs/01_regression_baselines/eegnet_wrapped.yaml
run_repeated benchmarks/configs/01_regression_baselines/deep4net_wrapped.yaml
run_repeated benchmarks/configs/01_regression_baselines/shallowfbcspnet_wrapped.yaml
run_repeated benchmarks/configs/01_regression_baselines/atcnet_wrapped.yaml

run_repeated benchmarks/configs/01_regression_baselines/labram_wrapped.yaml
run_repeated benchmarks/configs/01_regression_baselines/eegpt_wrapped.yaml
run_repeated benchmarks/configs/01_regression_baselines/medformer_wrapped.yaml
