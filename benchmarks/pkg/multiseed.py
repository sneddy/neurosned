"""Helpers for generating repeated benchmark YAML variants."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from benchmarks.pkg.config import ExperimentConfig


def load_template(path: str | Path) -> dict[str, Any]:
    """Load a YAML config template as a plain dictionary."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_variant(template: dict[str, Any], name: str, **recipe: Any) -> dict[str, Any]:
    """Create one config variant from a template and a compact training recipe."""
    config = deepcopy(template)
    config["name"] = name

    experiment = recipe.pop("experiment", None)
    if experiment is not None:
        config["experiment"] = experiment

    _apply_recipe(config, recipe)
    ExperimentConfig.model_validate(config)
    return config


def start_sweep_configs(template: dict[str, Any]) -> list[dict[str, Any]]:
    """Return fast sweep configs matching the old ayana start sweep."""
    common = {
        "experiment": "00_protocol_calibration",
        "n_epochs": 45,
        "early_stopping_patience": 8,
        "plot_last_batch": False,
        "plot_show": False,
        "batch_size": 512,
        "eval_batch_size": 256,
        "seed": 42,
    }
    return [
        make_variant(template, "SneddySegUNet1D_baseline_fast", **common),
        make_variant(template, "SneddySegUNet1D_sig012_fast", **common, sigma=0.12),
        make_variant(template, "SneddySegUNet1D_sig018_fast", **common, sigma=0.18),
        make_variant(template, "SneddySegUNet1D_lr3e4_fast", **common, initial_lr=3e-4),
        make_variant(template, "SneddySegUNet1D_lr7e4_fast", **common, initial_lr=7e-4),
        make_variant(template, "SneddySegUNet1D_tau075_fast", **common, train_temperature=0.75, eval_temperature=0.75),
        make_variant(template, "SneddySegUNet1D_ce02_lt3_fast", **common, lambda_ce=0.2, lambda_time=3.0),
        make_variant(template, "SneddySegUNet1D_ce05_lt2_fast", **common, lambda_ce=0.5, lambda_time=2.0),
        make_variant(template, "SneddySegUNet1D_bs256_fast", **{**common, "batch_size": 256}),
    ]


def final_candidate_configs(template: dict[str, Any]) -> list[dict[str, Any]]:
    """Return full-run final candidate configs matching the old ayana list."""
    common = {
        "experiment": "04_advanced_training_protocol",
        "n_epochs": 120,
        "early_stopping_patience": 15,
        "plot_last_batch": False,
        "plot_show": False,
        "batch_size": 512,
        "eval_batch_size": 256,
        "initial_lr": 0.001,
        "seed": 42,
        "train_temperature": 0.65,
        "eval_temperature": 0.65,
    }
    return [
        make_variant(
            template,
            "SneddySegUNet1D_sig012_full",
            **common,
            sigma=0.12,
            lambda_time=3.0,
            lambda_ce=0.0,
        ),
        make_variant(
            template,
            "SneddySegUNet1D_ce05_lt2_full",
            **common,
            sigma=0.15,
            lambda_time=2.0,
            lambda_ce=0.5,
        ),
    ]


def multiseed_configs(template: dict[str, Any], seeds: tuple[int, ...] = (42, 43, 44)) -> list[dict[str, Any]]:
    """Return full-run multiseed configs for the main final candidates."""
    common = {
        "experiment": "04_advanced_training_protocol",
        "n_epochs": 120,
        "early_stopping_patience": 15,
        "plot_last_batch": False,
        "plot_show": False,
        "batch_size": 512,
        "eval_batch_size": 256,
        "initial_lr": 0.001,
        "train_temperature": 0.65,
        "eval_temperature": 0.65,
    }
    specs = [
        ("SneddySegUNet1D_baseline", {"sigma": 0.15, "lambda_time": 3.0, "lambda_ce": 0.0}),
        ("SneddySegUNet1D_ce05_lt2", {"sigma": 0.15, "lambda_time": 2.0, "lambda_ce": 0.5}),
        ("SneddySegUNet1D_sig012", {"sigma": 0.12, "lambda_time": 3.0, "lambda_ce": 0.0}),
    ]

    configs = []
    for group_name, overrides in specs:
        for seed in seeds:
            configs.append(make_variant(template, f"{group_name}_seed{seed}", **common, **overrides, seed=seed))
    return configs


def write_configs(configs: list[dict[str, Any]], output_dir: str | Path) -> list[Path]:
    """Write generated configs to YAML files and return their paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for config in configs:
        validated = ExperimentConfig.model_validate(config)
        path = output_dir / f"{validated.name}.yaml"
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        paths.append(path)
    return paths


def _apply_recipe(config: dict[str, Any], recipe: dict[str, Any]) -> None:
    if "seed" in recipe:
        config["seed"] = recipe["seed"]

    trainer = config["trainer"]
    trainer_params = trainer.setdefault("params", {})
    train_dataset_params = config["data"]["train_dataset"].setdefault("params", {})

    if "n_epochs" in recipe:
        trainer["n_epochs"] = recipe["n_epochs"]
    if "early_stopping_patience" in recipe:
        trainer["early_stopping_patience"] = recipe["early_stopping_patience"]

    if "batch_size" in recipe:
        config["loaders"]["train"]["batch_size"] = recipe["batch_size"]
    if "eval_batch_size" in recipe:
        config["loaders"]["valid"]["batch_size"] = recipe["eval_batch_size"]
    if "train_num_workers" in recipe:
        config["loaders"]["train"]["num_workers"] = recipe["train_num_workers"]
    if "num_workers" in recipe:
        config["loaders"]["valid"]["num_workers"] = recipe["num_workers"]

    if "initial_lr" in recipe:
        config["optimizer"]["params"]["lr"] = recipe["initial_lr"]
    if "weight_decay" in recipe:
        config["optimizer"]["params"]["weight_decay"] = recipe["weight_decay"]
    if "optimizer" in recipe:
        config["optimizer"]["class_name"] = recipe["optimizer"]

    if "sigma" in recipe:
        train_dataset_params["sigma"] = recipe["sigma"]
        trainer_params["sigma"] = recipe["sigma"]
    for key in ("use_augmentation", "cropping_offset", "crop_proba", "dropout_proba", "dropout_range"):
        if key in recipe:
            train_dataset_params[key] = recipe[key]

    if "train_temperature" in recipe:
        trainer_params["temperature"] = recipe["train_temperature"]
    elif "temperature" in recipe:
        trainer_params["temperature"] = recipe["temperature"]
    if "eval_temperature" in recipe:
        trainer_params["eval_temperature"] = recipe["eval_temperature"]
    elif "temperature" in recipe:
        trainer_params["eval_temperature"] = recipe["temperature"]

    for key in (
        "lambda_time",
        "eval_lambda_time",
        "lambda_ce",
        "lambda_kl",
        "lambda_wass",
        "lambda_entropy",
        "lambda_focal",
        "grad_accum",
        "mixup_p",
        "mixup_alpha",
        "plot_last_batch",
        "plot_show",
    ):
        if key in recipe:
            trainer_params[key] = recipe[key]

    if "plateau_action" in recipe:
        trainer["plateau"]["enabled"] = recipe["plateau_action"] == "halve_lr_reload_best"
    if "lr_decay_factor" in recipe:
        trainer["plateau"]["factor"] = recipe["lr_decay_factor"]
    if "input_checkpoint_path" in recipe:
        trainer["checkpoint"]["input"] = recipe["input_checkpoint_path"]
