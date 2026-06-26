# Benchmarks

This package keeps benchmark experiments reproducible by moving run parameters
from notebooks into YAML configs.

## Running

Open `benchmarks/prod.ipynb` and choose one config:

```python
CONFIG_PATH = PROJECT_ROOT / "benchmarks/configs/segmentation/unet_deeper_demo.yaml"
config = load_experiment_config(CONFIG_PATH)
```

The notebook should stay a runner: load config, build datasets, build model,
create loaders, initialize trainer, run training. Avoid adding experiment
constants directly to the notebook.

## Configs

Experiment configs live in `benchmarks/configs/`. To start a new experiment,
copy an existing YAML and change only the section you are testing.

Important sections:

- `data`: prepared pickle paths and optional dataset wrappers.
- `data.train_dataset.params`: segmentation crop and augmentation settings.
- `loaders`: `DataLoader` batch size, shuffle, workers, and memory options.
- `model`: importable model class via `module_name`, `class_name`, `params`.
- `optimizer`: importable optimizer class and optimizer kwargs.
- `trainer`: trainer class, epochs, checkpoints, monitor, plateau schedule.
- `trainer.params`: task-specific knobs passed directly to the trainer.

Importable objects use this pattern:

```yaml
model:
  module_name: neurosned.models.segmentation.sneddy_unet
  class_name: SneddySegUNet1D
  params:
    n_chans: 128
    n_times: 200
    sfreq: 100
```

This also works for dataset wrappers and optimizers. `module_name` must be a
normal Python import path from the project root.

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

For regression, keep the pickle dataset unwrapped and tune augmentation in
`trainer.params`: channel dropout, cutout, noise, and mixup.

## Code Map

- `benchmarks/config.py`: typed YAML schema and builders.
- `benchmarks/data/`: dataset wrappers.
- `benchmarks/training/`: trainer classes, scheduling, metrics and plots.
- `benchmarks/preparation/`: data download/check/split preparation scripts.
