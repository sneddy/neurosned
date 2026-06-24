from __future__ import annotations

from copy import deepcopy
from typing import Any


BASE_TRAIN_CONFIG: dict[str, Any] = {
    "n_epochs": 120,
    "early_stopping_patience": 15,
    "use_augmentation": True,
    "initial_lr": 0.001,
    "weight_decay": 0,
    "optimizer": "Adam",
    "batch_size": 512,
    "eval_batch_size": 256,
    "sigma": 0.15,
    "cropping_offset": 0.25,
    "crop_proba": 0.3,
    "dropout_proba": 0.25,
    "dropout_range": 0.8,
    "mixup_p": 0.1,
    "temperature": 0.65,
    "train_temperature": 0.65,
    "eval_temperature": 0.65,
    "lambda_time": 3.0,
    "lambda_ce": 0,
    "train_lambda_ce": 1.0,
    "lambda_wass": 0.0,
    "lambda_kl": 0.0,
    "lambda_entropy": 0.0,
    "lambda_focal": 0.0,
    "grad_accum": 1,
    "save_epoch_plots": True,
    "show_val_epoch_plots": True,
    "run_holdout_test": True,
    "plateau_action": "stop",
    "lr_decay_factor": 0.5,
    "load_initial_checkpoint": False,
    "input_checkpoint_path": None,
    "require_initial_checkpoint": False,
    "seed": 42,
}


FAST_SWEEP_TRAIN_CONFIG: dict[str, Any] = {
    "n_epochs": 45,
    "early_stopping_patience": 8,
    "save_epoch_plots": False,
    "show_val_epoch_plots": False,
    "run_holdout_test": False,
}


def sweep_train_config(**overrides: Any) -> dict[str, Any]:
    return {**FAST_SWEEP_TRAIN_CONFIG, **overrides}


RNN_BASE_MODEL_CONFIG: dict[str, Any] = {
    "n_chans": 126,
    "n_times": 200,
    "sfreq": 100,
    "c0": 32,
    "widen": 1.5,
    "num_stages": 3,
    "depth_per_stage": [1, 1, 1],
    "k": 7,
    "dropout": 0.1,
    "drop_path": 0.0,
    "skip_gating": False,
    "out_channels": 1,
    "rnn_type": "gru",
    "bidirectional": True,
    "rnn_hidden_mult": 1.0,
    "rnn_layers_per_block": 1,
    "bottleneck_type": "rnn",
    "bottleneck_depth": 2,
    "bottleneck_rnn_layers": 1,
    "rnn_dropout": 0.05,
    "use_dwpos": True,
}


SNEDDY_UNET_BASE_MODEL_CONFIG: dict[str, Any] = {
    "n_chans": 128,
    "n_times": 200,
    "sfreq": 100,
    "c0": 96,
    "widen": 2,
    "depth_per_stage": 5,
    "dropout": 0.2,
    "k": 15,
    "out_channels": 1,
}


EXPERIMENTS: dict[str, dict[str, Any]] = {
    "RecurrentSneddyUnet": {
        "name": "RecurrentSneddyUnet",
        "model": "RecurrentSneddyUnet",
        "model_config": RNN_BASE_MODEL_CONFIG,
        "train_config": BASE_TRAIN_CONFIG,
    },
    "SneddySegUNet1D": {
        "name": "SneddySegUNet1D",
        "model": "SneddySegUNet1D",
        "model_config": SNEDDY_UNET_BASE_MODEL_CONFIG,
        "train_config": {
            **BASE_TRAIN_CONFIG,
            "batch_size": 512,
            "eval_batch_size": 256,
        },
    },
}


def get_experiment(name: str, **overrides: Any) -> dict[str, Any]:
    experiment = deepcopy(EXPERIMENTS[name])
    train_overrides = overrides.pop("train_config", {})
    model_overrides = overrides.pop("model_config", {})
    experiment["train_config"].update(train_overrides)
    experiment["model_config"].update(model_overrides)
    experiment.update(overrides)
    return experiment


def make_variant(
    base_name: str,
    variant_name: str,
    *,
    train_config: dict[str, Any] | None = None,
    model_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    experiment = get_experiment(base_name)
    experiment["name"] = variant_name
    if train_config:
        experiment["train_config"].update(train_config)
    if model_config:
        experiment["model_config"].update(model_config)
    return experiment

BEST_SEGMENTATION_MODEL = "SneddySegUNet1D"


def sneddy_seg_unet_notebook_parity(
    *,
    demonstration_mode: bool = True,
    finetune_mode: bool = False,
    seed: int | None = 42,
) -> dict[str, Any]:
    # Match the effective training recipe in notebooks/challenge_1/3_segmentation.ipynb.
    dropout_proba = 0.3 if finetune_mode else 0.25
    train_config = {
        "n_epochs": 10 if demonstration_mode else 100,
        "early_stopping_patience": 20,
        "use_augmentation": True,
        "initial_lr": 1e-4 if finetune_mode else 0.001,
        "weight_decay": 0,
        "optimizer": "SGD" if finetune_mode else "Adam",
        "batch_size": 2000,
        "eval_batch_size": 256,
        "sigma": 0.15,
        "cropping_offset": 0.2 if finetune_mode else 0.25,
        "crop_proba": 0.1 if finetune_mode else 0.3,
        "dropout_proba": dropout_proba,
        # The notebook defines dropout_range separately but passes dropout_proba here.
        "dropout_range": dropout_proba,
        "mixup_p": 0.5 if finetune_mode else 0.1,
        "temperature": 0.75 if finetune_mode else 0.65,
        "train_temperature": 0.75 if finetune_mode else 0.65,
        "eval_temperature": 0.75 if finetune_mode else 0.65,
        "lambda_time": 0 if finetune_mode else 3.0,
        # The notebook variable is 0 in scratch mode, but the call omits lambda_ce,
        # so train_one_epoch uses its default CE weight of 1.0.
        "lambda_ce": 1.0,
        "train_lambda_ce": 1.0,
        "lambda_wass": 0.0,
        "lambda_kl": 0.0,
        "lambda_entropy": 0.0,
        "lambda_focal": 0.0,
        "grad_accum": 1,
        "save_epoch_plots": True,
        "show_val_epoch_plots": True,
        "run_holdout_test": False,
        "plateau_action": "halve_lr_reload_best",
        "lr_decay_factor": 0.5,
        "load_initial_checkpoint": True,
        "input_checkpoint_path": "artefacts/models/challenge_1/unet_deeper.pth",
        "require_initial_checkpoint": False,
        "num_workers": 4,
        "train_num_workers": 8,
        "seed": seed,
    }
    name_suffix = "finetune" if finetune_mode else "scratch"
    mode_suffix = "demo" if demonstration_mode else "full"
    return make_variant(
        BEST_SEGMENTATION_MODEL,
        f"SneddySegUNet1D_notebook_parity_{name_suffix}_{mode_suffix}",
        train_config=train_config,
    )


def sneddy_seg_unet_start_sweep() -> list[dict[str, Any]]:
    base = BEST_SEGMENTATION_MODEL
    return [
        make_variant(base, "SneddySegUNet1D_baseline_fast", train_config=sweep_train_config(seed=42)),
        make_variant(base, "SneddySegUNet1D_sig012_fast", train_config=sweep_train_config(sigma=0.12)),
        make_variant(base, "SneddySegUNet1D_sig018_fast", train_config=sweep_train_config(sigma=0.18)),
        make_variant(base, "SneddySegUNet1D_lr3e4_fast", train_config=sweep_train_config(initial_lr=3e-4)),
        make_variant(base, "SneddySegUNet1D_lr7e4_fast", train_config=sweep_train_config(initial_lr=7e-4)),
        make_variant(
            base,
            "SneddySegUNet1D_tau075_fast",
            train_config=sweep_train_config(temperature=0.75, train_temperature=0.75, eval_temperature=0.75),
        ),
        make_variant(base, "SneddySegUNet1D_ce02_lt3_fast", train_config=sweep_train_config(lambda_ce=0.2, lambda_time=3.0)),
        make_variant(base, "SneddySegUNet1D_ce05_lt2_fast", train_config=sweep_train_config(lambda_ce=0.5, lambda_time=2.0)),
        make_variant(base, "SneddySegUNet1D_bs256_fast", train_config=sweep_train_config(batch_size=256)),
    ]

def sneddy_seg_unet_final_candidates() -> list[dict[str, Any]]:
    base = BEST_SEGMENTATION_MODEL
    final_common = {
        "n_epochs": 120,
        "early_stopping_patience": 15,
        "save_epoch_plots": False,
        "show_val_epoch_plots": False,
        "run_holdout_test": True,
        "seed": 42,
    }
    return [
        make_variant(
            base,
            "SneddySegUNet1D_sig012_full",
            train_config={
                **final_common,
                "sigma": 0.12,
                "initial_lr": 0.001,
                "batch_size": 512,
                "lambda_time": 3.0,
                "lambda_ce": 0,
                "temperature": 0.65,
                "train_temperature": 0.65,
                "eval_temperature": 0.65,
            },
        ),
        make_variant(
            base,
            "SneddySegUNet1D_ce05_lt2_full",
            train_config={
                **final_common,
                "sigma": 0.15,
                "initial_lr": 0.001,
                "batch_size": 512,
                "lambda_time": 2.0,
                "lambda_ce": 0.5,
                "temperature": 0.65,
                "train_temperature": 0.65,
                "eval_temperature": 0.65,
            },
        ),
    ]

def sneddy_seg_unet_multiseed_candidates(seeds: tuple[int, ...] = (42, 43, 44)) -> list[dict[str, Any]]:
    base = BEST_SEGMENTATION_MODEL
    final_common = {
        "n_epochs": 120,
        "early_stopping_patience": 15,
        "save_epoch_plots": False,
        "show_val_epoch_plots": False,
        "run_holdout_test": True,
    }

    specs = [
        (
            "SneddySegUNet1D_baseline",
            {
                "sigma": 0.15,
                "initial_lr": 0.001,
                "batch_size": 512,
                "lambda_time": 3.0,
                "lambda_ce": 0,
                "temperature": 0.65,
                "train_temperature": 0.65,
                "eval_temperature": 0.65,
            },
        ),
        (
            "SneddySegUNet1D_ce05_lt2",
            {
                "sigma": 0.15,
                "initial_lr": 0.001,
                "batch_size": 512,
                "lambda_time": 2.0,
                "lambda_ce": 0.5,
                "temperature": 0.65,
                "train_temperature": 0.65,
                "eval_temperature": 0.65,
            },
        ),
        (
            "SneddySegUNet1D_sig012",
            {
                "sigma": 0.12,
                "initial_lr": 0.001,
                "batch_size": 512,
                "lambda_time": 3.0,
                "lambda_ce": 0,
                "temperature": 0.65,
                "train_temperature": 0.65,
                "eval_temperature": 0.65,
            },
        ),
    ]

    experiments = []
    for group_name, train_overrides in specs:
        for seed in seeds:
            experiment = make_variant(
                base,
                f"{group_name}_seed{seed}",
                train_config={**final_common, **train_overrides, "seed": seed},
            )
            experiment["group"] = group_name
            experiments.append(experiment)
    return experiments

