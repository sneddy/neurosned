#!/usr/bin/env sh
set -eu

echo "Generating paper-facing figures"
python benchmarks/scripts/plot_paper_figures.py benchmarks/experiments/02_segmentation_ablations \
  --output-dir benchmarks/experiments/paper_figures \
  --formats png svg \
  --dpi 300
