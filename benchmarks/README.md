# EEG Reaction-Time Benchmark

This directory contains the code, configs, and paper-facing artifacts for the
benchmark used in the Neural Computation manuscript:

**Behavioral Latency as Weak Event-Time Supervision for EEG Decoding**

The benchmark compares scalar EEG reaction-time regression with event-time
posterior modeling on the Healthy Brain Network contrast change detection EEG
task. The release is organized so that the paper tables can be inspected
directly, while the training and analysis runs can be regenerated from YAML
configs.

## Scope

The benchmark implements the support-filtered, release-separated protocol used
in the manuscript:

| component | value |
| --- | --- |
| train split | HBN releases R1-R8 |
| development split | HBN releases R9-R10 |
| final holdout | HBN release R11 |
| main input window | fixed 0.5-2.5 s post-stimulus EEG window |
| modeled RT support | 0.5 <= RT <= 2.5 s |
| repeated seeds | 2025, 2026, 2027, 2028, 2029 |
| main scalar metric | nRMSE on R11 |

The code supports four paper-facing experiment blocks:

1. Scalar RT regression baselines.
2. Event-time posterior objective comparisons.
3. Posterior geometry, calibration, and effect-size analyses.
4. Shifted-crop and shift-jitter shortcut-vs-localization diagnostics.

## What Is Included

- `configs/`: YAML configs for all paper-facing model runs.
- `data/`: dataset wrappers for fixed-window, segmentation, and shifted-crop
  training.
- `pkg/`: reusable training, evaluation, model, loss, and plotting code.
- `preparation/`: scripts for downloading, checking, and preparing HBN release
  splits.
- `scripts/`: command-line entrypoints for training, re-evaluation, figures,
  effect sizes, and observation-noise calibration.
- `runners/`: shell scripts that launch the paper-facing experiment groups.
- `experiments/paper_tables/`: compact Markdown tables used by the manuscript.
- `experiments/paper_figures/`: paper figure images and source CSV files.
- `experiments.yaml`: a human-readable map of experiment groups and outputs.
- `EXPERIMENTS.md`: the experiment journal for the current manuscript protocol.

Large runtime artifacts are not part of the code release. Checkpoints,
per-seed directories, logits, dense predictions, raw release data, and prepared
pickle datasets are intentionally ignored by Git and regenerated locally.

## Environment

Create an environment from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r benchmarks/requirements.txt
```

For notebook inspection only:

```bash
pip install -r benchmarks/requirements-dev.txt
```

PyTorch installation can be platform-specific. If the pinned `torch` wheel in
`benchmarks/requirements.txt` does not match your CUDA/CPU setup, install the
appropriate PyTorch build first, then install the remaining requirements.

## Data Preparation

Run commands from the repository root. The preparation scripts materialize raw
release caches under `release_data/` and prepared split pickle files under
`data/new_validation/`; both locations are outside Git.

```bash
python benchmarks/preparation/scripts/download_releases.py
python benchmarks/preparation/scripts/check_releases.py
python benchmarks/preparation/scripts/prepare_splitted_datasets.py
python benchmarks/preparation/scripts/summarize_protocol_counts.py
```

The expected prepared files are:

- `data/new_validation/r1_r8_train.pkl`
- `data/new_validation/r1_r8_train_5sec.pkl`
- `data/new_validation/r9_r10_val.pkl`
- `data/new_validation/r9_r10_val_5sec.pkl`
- `data/new_validation/r11_test.pkl`
- `data/new_validation/r11_test_5sec.pkl`

The manuscript support filter is applied by the benchmark config loader at
dataset construction time; the prepared pickle files themselves are not
rewritten for the filtered protocol.

## Running One Config

Run a single scalar baseline:

```bash
python benchmarks/scripts/run_repeated.py \
  benchmarks/configs/01_regression_baselines/etr_cnn_large.yaml \
  --device cuda \
  --output-dir benchmarks/experiments
```

Run a single event-time posterior model:

```bash
python benchmarks/scripts/run_repeated.py \
  benchmarks/configs/02_segmentation_ablations/ets_unet_event_nll_mixture.yaml \
  --device cuda \
  --output-dir benchmarks/experiments
```

Use `--device auto` to let the runner choose CUDA when available.

Each repeated run creates:

- `seed*/config.yaml`: resolved config snapshot.
- `seed*/best_model.pth`: best validation checkpoint.
- `seed*/summary.json`: scalar, posterior, and shifted-crop summaries.
- `seed*/predictions/`: validation and holdout predictions/logits.
- `repeated_summary.{csv,json}`: aggregate seed-level summary.

These runtime outputs are ignored by Git except for compact paper-facing
summary files that are explicitly committed.

## Reproducing the Paper Runs

Run all scalar baselines:

```bash
DEVICE=cuda sh benchmarks/runners/run_regression_baselines.sh
```

Run all event-time objective comparisons:

```bash
DEVICE=cuda sh benchmarks/runners/run_segmentation_ablations.sh
```

Run all shift-jitter training interventions:

```bash
DEVICE=cuda sh benchmarks/runners/run_crop_shift_jitter.sh
```

Re-evaluate shift-jitter checkpoints on the canonical fixed-window holdout:

```bash
DEVICE=cuda sh benchmarks/runners/reeval_crop_shift_jitter_canonical.sh
```

The full repeated benchmark is GPU-oriented and is not intended as a quick CPU
smoke test.

## Derived Analyses

Regenerate posterior geometry figures:

```bash
sh benchmarks/runners/plot_paper_figures.sh
```

Regenerate the scalar-vs-event-time effect-size table:

```bash
python benchmarks/scripts/reg_vs_seg_effect_size.py
```

Regenerate the observation-noise calibration appendix table:

```bash
python benchmarks/scripts/calibrate_observation_noise.py
```

Run post-hoc shifted-crop evaluation for existing checkpoints:

```bash
DEVICE=cuda sh benchmarks/runners/eval_shifted_regression.sh
DEVICE=cuda sh benchmarks/runners/eval_shifted_seg.sh
```

## Paper Artifacts

The current paper-facing tables are in `experiments/paper_tables/`:

- `main_01_regression_baselines.md`
- `main_02_event_time_objectives.md`
- `main_03_shift_jitter_summary.md`
- `main_04_posterior_geometry.md`
- `main_05_effect_size.md`
- `main_06_experiments_map.md`
- `appendix_01_regression_shifted_crop.md`
- `appendix_02_fixed_shifted_details.md`
- `appendix_03_jitter_shifted_details.md`
- `appendix_04_observation_noise_calibration.md`

The current paper-facing figures and source CSV files are in
`experiments/paper_figures/`.

## Directory Map

```text
benchmarks/
  configs/                 Paper-facing YAML experiment configs.
  data/                    Dataset wrappers.
  experiments/             Compact paper artifacts plus local run outputs.
  pkg/                     Models, losses, training, evaluation, plotting.
  preparation/             HBN release download and split preparation.
  runners/                 Shell launchers for experiment groups.
  scripts/                 Python CLIs for training and derived analyses.
  EXPERIMENTS.md           Current protocol and experiment journal.
  experiments.yaml         Machine-readable-ish experiment map.
  requirements.txt         Runtime dependencies for benchmark reproduction.
```

## Notes for Release Users

- The benchmark assumes commands are run from the repository root.
- The prepared data and checkpoints are intentionally not versioned.
- Foundation-style EEG architectures in this benchmark are trained from scratch
  under the shared protocol; pretrained weights are not used.
- Temperature tuning in the event-time models is a scalar posterior-mean
  readout protocol selected on R9-R10 and applied unchanged to R11.
- Observation-noise calibration is a post-hoc predictive RT interval analysis;
  it does not retrain the EEG model or change the scalar readout.
