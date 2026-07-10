#!/usr/bin/env sh
set -eu

echo "Generating architecture-control posterior figures"

run_one() {
  name="$1"
  experiment_dir="$2"
  shift 2
  output_dir="benchmarks/experiments/paper_figures/architecture_controls/${name}"

  echo ""
  echo "=== ${name} ==="
  python benchmarks/scripts/plot_paper_figures.py "${experiment_dir}" \
    --output-dir "${output_dir}" \
    --figures posterior_geometry posterior_pareto \
    --target-filter representable \
    --near-ms 150 \
    --coverage-levels 0.50 0.60 0.70 0.80 0.90 \
    --formats png svg \
    --dpi 300 \
    "$@"
}

run_one "attnseg" "benchmarks/experiments/04_attnseg" "$@"
run_one "tcn" "benchmarks/experiments/05_tcn" "$@"
run_one "inception_pyramid" "benchmarks/experiments/06_inception_pyramid" "$@"
