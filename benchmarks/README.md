# Benchmarks

This package keeps benchmark experiments reproducible by moving run parameters
from notebooks into YAML configs.

## Running

Run one config from the command line:

```bash
python benchmarks/scripts/run.py benchmarks/configs/0_demo/unet_deeper_demo.yaml
```

Use a different artefact root when needed:

```bash
python benchmarks/scripts/run.py benchmarks/configs/0_demo/unet_deeper_demo.yaml --output-dir /tmp/neurosned-runs
```

Refresh holdout metrics for an existing run from its best checkpoint:

```bash
python benchmarks/scripts/reeval.py benchmarks/experiments/<experiment>/<run_name> --device cuda --enable-temperature
```

For interactive inspection, open `benchmarks/prod.ipynb` and choose one config:

```python
CONFIG_PATH = PROJECT_ROOT / "benchmarks/configs/segmentation/unet_deeper_demo.yaml"
config = load_experiment_config(CONFIG_PATH)
```

The notebook should stay a runner: load config, build datasets, build model,
create an `ArtefactsManager`, initialize trainer, run training. Avoid adding
experiment constants directly to the notebook.

Each run is saved under `benchmarks/experiments/<experiment>/<run_name>/`. The
directory is runtime output and is ignored by Git.

## Configs

Experiment configs live in `benchmarks/configs/`. To start a new experiment,
copy an existing YAML and change only the section you are testing.

Important sections:

- `experiment`: paper-facing experiment group and artifact subdirectory.
- `name`: concrete config variant inside that group.
- `data`: prepared pickle paths and optional dataset wrappers.
- `data.train_dataset.params`: segmentation crop and augmentation settings.
- `loaders`: `DataLoader` batch size, shuffle, workers, and memory options.
- `model`: importable model class via `module_name`, `class_name`, `params`.
- `optimizer`: importable optimizer class and optimizer kwargs.
- `trainer`: trainer class, epochs, checkpoints, monitor, plateau schedule.
- `trainer.params`: task-specific knobs passed directly to the trainer.
- `trainer.stages`: optional main/finetune stages with optimizer, loader,
  dataset, plateau and trainer-param overrides.

Importable objects use this pattern:

```yaml
model:
  module_name: benchmarks.pkg.models.segmentation.sneddy_unet
  class_name: SneddySegUNet1D
  params:
    n_chans: 128
    n_times: 200
    sfreq: 100
```

This also works for dataset wrappers and optimizers. `module_name` must be a
normal Python import path from the project root.

For repeated sweep/final/multiseed configs, use `benchmarks/pkg/multiseed.py`
to generate YAML variants from an existing template. The generated YAML files
then become the configs used by runs.

Multi-stage training is optional. When `trainer.stages` is present, each stage
reuses the same model and output checkpoint. A stage can use `reload: best`,
switch optimizer settings, update train dataset augmentation, and override
trainer params. Without `trainer.stages`, configs run as a single stage.

## Tuning

For segmentation, most controlled changes are usually in:

- `model.params`: architecture size and model-specific options.
- `data.train_dataset.params.sigma`: soft target width.
- `data.train_dataset.params.cropping_offset` and `crop_proba`: crop jitter.
- `data.train_dataset.params.dropout_proba` and `dropout_range`: channel dropout.
- `trainer.params.temperature`: temporal distribution sharpness.
- `trainer.params.lambda_time` and `lambda_ce`: timing vs. distribution loss.
- `trainer.params.mixup_p` and `mixup_alpha`: mixup strength.
- `optimizer.params.lr`: starting learning rate.
- `trainer.plateau.factor`: LR decay after reload-best plateau.

For regression, wrap fixed 2-second pickle datasets with
`benchmarks.data.regression.FixedWindowDataset` when you need channel selection
or per-sample transforms. Keep batch-level mixup in `trainer.params`.

## Outputs

`ArtefactsManager` owns benchmark outputs:

- `benchmarks/experiments/<experiment>/<run_name>/`: one concrete run.
- `config.yaml`: resolved config snapshot and paths used by the run.
- `model.txt`: model architecture and parameter count.
- `best_model.pth`: best validation checkpoint.
- `metrics.csv`: epoch history from the trainer, including epoch timings.
- `summary.json`: per-run summary, including wall time and time-to-best.
- `logs/run.log`: stdout/stderr copied from the run.
- `monitoring/gpu.csv`: GPU utilization and memory sampled during training.
- `figures/gpu_usage.png`: GPU utilization and memory plot.
- `figures/*.png`: optional diagnostic validation plots.
- `predictions/best_val_predictions.csv`: scalar predictions with row ids and metadata.
- `predictions/best_logits.npy`: dense best-epoch segmentation logits in the same row order.
- `benchmarks/experiments/summary.{jsonl,csv,md}`: global run index.

## Experiments Map

The paper-facing benchmark should separate model comparisons from protocol
engineering. The clean structure is:

`benchmarks/experiments.yaml` is a human map of experiment groups, config
locations and artifact locations. It is not loaded by the code.

1. **Protocol calibration**
   - Goal: choose one simple, reproducible training protocol for SneddyUNet.
   - Keep it deliberately plain: one seed group, one optimizer schedule, no
     staged fine-tuning, no restart-heavy tricks.
   - Output: the default protocol used by the main ablations.

2. **Regression baselines**
   - Goal: compare the best SneddyNet-style regression model against standard
     Braindecode architectures under the same simple protocol.
   - Candidates: EEGNet, Deep4Net/ShallowFBCSPNet-style baselines, and the
     strongest local regression model.
   - Output: whether the custom regression family is competitive on fixed
     2-second windows.

3. **Regression vs. segmentation framing**
   - Goal: compare SneddyNet regression with SneddyUNet segmentation under the
     same data split and simple protocol.
   - This answers whether predicting a temporal distribution is better than
     direct scalar regression for reaction time.

4. **Component ablations**
   - Goal: quantify what each modeling/training choice contributes.
   - Run one-factor changes against the simple SneddyUNet protocol:
     segmentation loss weights, soft-label sigma, crop jitter, channel dropout,
     mixup, architecture width/depth, and optional model blocks.
   - Output: a compact ablation table with validation and R11 holdout metrics.

5. **Advanced training protocol**
   - Goal: show the best achievable result when using the full engineering
     recipe.
   - This can include multi-stage training, checkpoint warm starts, reload-best
     LR decay, longer schedules, seed ensembles, or candidate selection.
   - Keep this separate from ablations so the paper does not confuse model
     contribution with optimization budget.

Recommended reporting order:

1. Simple protocol definition.
2. Regression baselines.
3. Regression vs. segmentation.
4. SneddyUNet component ablations.
5. Final advanced protocol result.

For Neural Computation-style reporting, the main claim should rest on the
simple protocol and controlled ablations; the advanced protocol should be
presented as the final optimized benchmark, not as evidence for every design
choice.

## Code Map

- `benchmarks/scripts/run.py`: command-line entrypoint mirroring `prod.ipynb`.
- `benchmarks/scripts/reeval.py`: post-training holdout/calibration refresh from `best_model.pth`.
- `benchmarks/scripts/run_repeated.py`: repeated-seed runner for final evaluation.
- `benchmarks/pkg/config.py`: typed YAML schema and builders.
- `benchmarks/pkg/artefacts_manager.py`: run directories, summaries and saved outputs.
- `benchmarks/pkg/multiseed.py`: sweep/final/multiseed YAML variant helpers.
- `benchmarks/pkg/training/trainers/`: trainer classes.
- `benchmarks/pkg/training/`: scheduling, labels, metrics and plots.
- `benchmarks/data/`: dataset wrappers.
- `benchmarks/preparation/`: data download/check/split preparation scripts.
