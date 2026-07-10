"""Typed experiment configuration loaded from YAML.

An experiment YAML is the source of truth for production benchmark runs:
data pickle paths, optional dataset wrappers, DataLoader arguments, model
constructor, optimizer, trainer class, checkpoints, and trainer-specific
parameters.

Importable objects use the same structure everywhere:

    module_name: benchmarks.pkg.models.segmentation.ets_unet
    class_name: EventTimeUNet1D
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

from benchmarks.data.filtering import apply_target_range_filter


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
    `TrainCroppingDataset`, while validation stays unwrapped. Optional target
    bounds are applied immediately after pickle loading and before any wrappers.
    """

    train: Path
    valid: Path
    test: Path | None = None
    target_min: float | None = None
    target_max: float | None = None
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


class RepeatedRunsConfig(StrictConfig):
    """Planned independent training seeds for final paper evaluation."""

    enabled: bool = False
    seeds: list[int] = Field(default_factory=list)


class ConfidenceIntervalConfig(StrictConfig):
    """Confidence interval settings for saved holdout predictions."""

    enabled: bool = False
    method: Literal["subject_bootstrap"] = "subject_bootstrap"
    n_samples: int = 1000
    resampling_seed: int = 2025


class ShiftedCropEvaluationConfig(StrictConfig):
    """Optional post-training shifted-crop diagnostic."""

    enabled: bool = False
    dataset: Path = Path("data/new_validation/r11_test_5sec.pkl")
    target_min: float | None = None
    target_max: float | None = None
    starts: list[float] = Field(default_factory=lambda: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    reference_start: float = 0.5
    crop_sec: float | None = None
    sfreq: float | None = None
    batch_size: int | None = None
    num_workers: int | None = None
    output_dir: Path | None = None
    segmentation_temperature: float | None = None
    bootstrap_samples: int = 1000
    bootstrap_seed: int = 2025
    save_predictions: bool = False


class TemperatureCalibrationConfig(StrictConfig):
    """Post-hoc temperature calibration for segmentation logits."""

    enabled: bool = False
    min: float = 0.2
    max: float = 3.5
    step: float = 0.05


class CalibrationConfig(StrictConfig):
    """Post-training calibration settings."""

    temperature: TemperatureCalibrationConfig = Field(default_factory=TemperatureCalibrationConfig)


class EvaluationConfig(StrictConfig):
    """Explicit holdout-evaluation settings.

    Calibration configs keep `holdout_eval` false so R11 is not touched during
    protocol search. Frozen paper-facing configs can set it true to run one
    post-training evaluation on `data.test`.
    """

    holdout_eval: bool = False
    holdout_split: str = "test"
    save_predictions: bool = True
    save_logits: bool = False
    repeated_runs: RepeatedRunsConfig = Field(default_factory=RepeatedRunsConfig)
    confidence_interval: ConfidenceIntervalConfig = Field(default_factory=ConfidenceIntervalConfig)
    shifted_crop: ShiftedCropEvaluationConfig = Field(default_factory=ShiftedCropEvaluationConfig)


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
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)

    def build_datasets(self, project_root: str | Path):
        """Load train and validation pickle datasets."""
        return self.build_dataset("train", project_root), self.build_dataset("valid", project_root)

    def build_dataset(self, split: str, project_root: str | Path):
        """Load one configured dataset split from pickle."""
        if split not in {"train", "valid", "test"}:
            raise ValueError(f"Unknown dataset split: {split}")
        path = resolve_path(getattr(self.data, split), project_root)
        if path is None:
            raise ValueError(f"Dataset split has no configured path: {split}")

        with path.open("rb") as f:
            dataset = pickle.load(f)
        return apply_target_range_filter(
            dataset,
            target_min=self.data.target_min,
            target_max=self.data.target_max,
        )

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
