#!/usr/bin/env sh
set -eu

ROOT="benchmarks/experiments/03_crop_shift_jitter"
OUT_ROOT="benchmarks/experiments/03_crop_shift_jitter_canonical_eval"
DEVICE="${DEVICE:-auto}"
TARGET_MIN="${TARGET_MIN:-0.5}"
TARGET_MAX="${TARGET_MAX:-2.5}"

echo "Canonical crop-shift-jitter re-evaluation"
echo "Source root: ${ROOT}"
echo "Output root: ${OUT_ROOT}"
echo "Device: ${DEVICE}"
echo "Target range: [${TARGET_MIN}, ${TARGET_MAX}]"

find "${ROOT}" -mindepth 2 -maxdepth 2 -type d -name 'seed*' | sort | while IFS= read -r run_dir; do
  echo
  echo "=== Re-evaluating ${run_dir} ==="
  python benchmarks/scripts/reeval_canonical.py "${run_dir}" \
    --device "${DEVICE}" \
    --out-root "${OUT_ROOT}" \
    --target-min "${TARGET_MIN}" \
    --target-max "${TARGET_MAX}"
done

echo
echo "Done. Canonical artefacts are under ${OUT_ROOT}"
