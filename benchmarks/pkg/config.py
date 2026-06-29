"""Typed experiment configuration loaded from YAML.

An experiment YAML is the source of truth for production benchmark runs:
data pickle paths, optional dataset wrappers, DataLoader arguments, model
constructor, optimizer, trainer class, checkpoints, and trainer-specific
parameters.

Importable objects use the same structure everywhere:

    module_name: neurosned.models.segmentation.sneddy_unet
    class_name: SneddySegUNet1D
    params:
      n_chans: 128
      n_times: 200

`module_name` is a normal Python import path from the project root, and
`params` are passed directly to the configured class.
"""

from __future__ import annotations

import pickle
from importlib import import_module
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class StrictConfig(BaseModel):
    """Base model that rejects unknown YAML fields."""

    model_config = ConfigDict(extra="forbid")


class ObjectConfig(StrictConfig):
    """Importable Python object plus constructor parameters.

    This is used for models, optimizers and dataset wrappers. It deliberately
    avoids a registry: any class in an importable Python module can be selected
    from YAML with `module_name`, `class_name`, and `params`.
    """

    module_name: str
    class_name: str
    params: dict[str, Any] = Field(default_factory=dict)

    def load_class(self):
        """Import and return the configured class."""
        module = import_module(self.module_name)
        return getattr(module, self.class_name)

    def build(self, *args, **kwargs):
        """Instantiate the configured class."""
        params = {**self.params, **kwargs}
        return self.load_class()(*args, **params)


class DatasetConfig(ObjectConfig):
    """Optional wrapper around a loaded pickle dataset."""


class DataConfig(StrictConfig):
    """Dataset pickle paths and optional wrappers.

    `train` and `valid` point to prepared benchmark pickle files. A wrapper can
    be attached when the raw pickle dataset is not what the trainer consumes.
    Segmentation training, for example, wraps the train dataset with
    `TrainCroppingDataset`, while validation stays unwrapped.
    """

    train: Path
    valid: Path
    test: Path | None = None
    train_dataset: DatasetConfig | None = None
    valid_dataset: DatasetConfig | None = None


class LoaderConfig(StrictConfig):
    """Torch DataLoader parameters kept as plain YAML values."""

    batch_size: int
    shuffle: bool = False
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int | None = None
    drop_last: bool = False

    def to_kwargs(self) -> dict[str, Any]:
        """Return kwargs accepted by torch DataLoader."""
        return self.model_dump(exclude_none=True)


class LoadersConfig(StrictConfig):
    """Train/validation DataLoader configs."""

    train: LoaderConfig
    valid: LoaderConfig


class CheckpointConfig(StrictConfig):
    """Optional input checkpoint path relative to the project root.

    Output checkpoints are owned by `ArtefactsManager` and saved inside the run
    directory under `experiments/`.
    """

    input: Path | None = None
    output: Path | None = None


class OptimizerConfig(ObjectConfig):
    """Importable optimizer configuration."""


class PlateauConfig(StrictConfig):
    """Reload-best-on-plateau settings.

    When enabled, the trainer reloads the best checkpoint after patience is
    exhausted, recreates the optimizer, and multiplies the learning rate by
    `factor`.
    """

    enabled: bool = True
    factor: float = 0.5
    max_restarts: int | None = None


class TrainerStageConfig(StrictConfig):
    """Optional stage override for multi-stage training."""

    name: str
    n_epochs: int
    reload: Literal["none", "best"] = "none"
    optimizer: OptimizerConfig | None = None
    train_dataset_params: dict[str, Any] = Field(default_factory=dict)
    train_loader: LoaderConfig | None = None
    early_stopping_patience: int | None = None
    print_batch_stats: bool | None = None
    plateau: PlateauConfig | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class TrainerConfig(StrictConfig):
    """Trainer class and epoch-level settings.

    Common loop settings live as first-class fields. Task-specific knobs, such
    as segmentation loss weights or regression augmentation rates, stay in
    `params` and are passed directly to the configured trainer.
    """

    module_name: str = "benchmarks.pkg.training.trainers"
    class_name: str
    n_epochs: int
    checkpoint: CheckpointConfig
    monitor: str
    minimize: bool = True
    early_stopping_patience: int | None = None
    print_batch_stats: bool = True
    plateau: PlateauConfig = Field(default_factory=PlateauConfig)
    params: dict[str, Any] = Field(default_factory=dict)
    stages: list[TrainerStageConfig] | None = None

    def load_class(self):
        """Import and return the configured trainer class."""
        module = import_module(self.module_name)
        return getattr(module, self.class_name)


class ExperimentConfig(StrictConfig):
    """Complete benchmark experiment config.

    The notebook should only select a config file and call this object to load
    datasets, build models, create loaders and initialize the trainer. To tweak
    an experiment, copy an existing YAML and change the narrow section you are
    testing: `model`, `data.train_dataset.params`, `loaders`, `optimizer`, or
    `trainer.params`.
    """

    experiment: str
    name: str
    task: Literal["segmentation", "regression"]
    seed: int
    data: DataConfig
    loaders: LoadersConfig
    model: ObjectConfig
    optimizer: OptimizerConfig
    trainer: TrainerConfig

    def build_datasets(self, project_root: str | Path):
        """Load train and validation pickle datasets."""
        train_path = resolve_path(self.data.train, project_root)
        valid_path = resolve_path(self.data.valid, project_root)

        with train_path.open("rb") as f:
            train_dataset = pickle.load(f)
        with valid_path.open("rb") as f:
            valid_dataset = pickle.load(f)

        return train_dataset, valid_dataset

    def data_paths(self, project_root: str | Path) -> dict[str, Path | None]:
        """Return resolved dataset paths."""
        return {
            "train": resolve_path(self.data.train, project_root),
            "valid": resolve_path(self.data.valid, project_root),
            "test": resolve_path(self.data.test, project_root),
        }


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment config from YAML."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)
    return ExperimentConfig.model_validate(raw_config)


def resolve_path(path: str | Path | None, project_root: str | Path) -> Path | None:
    """Resolve a config path against the project root."""
    if path is None:
        return None
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(project_root) / path
