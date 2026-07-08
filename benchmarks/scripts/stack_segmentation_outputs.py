"""Stack saved segmentation outputs without loading model weights."""

from __future__ import annotations

import argparse
import inspect
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

os.environ["MNE_DONTWRITE_HOME"] = "true"
os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["MPLCONFIGDIR"] = "/tmp/neurosned-matplotlib"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import yaml

from benchmarks.pkg.ensembling.folds import make_subject_balanced_folds
from benchmarks.pkg.ensembling.meta_features import MetaFeatureExtractor
from benchmarks.pkg.ensembling.meta_regressor import HgbMetaRegressor, MetaRegressor, RidgeMetaRegressor
from benchmarks.pkg.ensembling.metrics import regression_metrics
from benchmarks.pkg.ensembling.monotonic import make_monotonic_constraints
from benchmarks.pkg.ensembling.reporting import aggregate_rows, write_markdown_table
from benchmarks.pkg.runtime import choose_device, path_text


REQUIRED_FILES = (
    "predictions/best_val_predictions.csv",
    "predictions/best_logits.npy",
    "predictions/test_predictions.csv",
    "predictions/test_logits.npy",
)

METHOD_ORDER = [
    "Equal-weight scalar RT blend",
    "Equal-weight logits soft-argmax blend",
    "Ridge stacking, RT only",
    "Boosting stacking, RT only",
    "Ridge stacking, posterior meta-features",
    "Boosting stacking, posterior meta-features",
]


@dataclass(frozen=True)
class RunArtifacts:
    """Frozen outputs for one completed segmentation run."""

    run_dir: Path
    experiment: str
    name: str
    seed: int
    valid_predictions: pd.DataFrame
    test_predictions: pd.DataFrame
    valid_logits: np.ndarray
    test_logits: np.ndarray
    sfreq: float
    win_offset: float
    readout: str

    @property
    def model_key(self) -> str:
        return self.name


def build_parser() -> argparse.ArgumentParser:
    """Build the stacking CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_root", type=Path, help="Experiment folder containing repeated seed run directories.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/experiments/07_distribution_aware_stacking"),
        help="Output directory for raw stacking artifacts.",
    )
    parser.add_argument(
        "--paper-table",
        type=Path,
        default=None,
        help="Markdown paper table path. Defaults to canonical path when --out is default, otherwise inside --out.",
    )
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="Optional seed subset.")
    parser.add_argument("--target-min", type=float, default=0.5, help="Minimum accepted target in saved artifacts.")
    parser.add_argument("--target-max", type=float, default=2.5, help="Maximum accepted target in saved artifacts.")
    parser.add_argument(
        "--allow-support-subset",
        action="store_true",
        help=(
            "Allow run config target_min/target_max to differ from requested support. "
            "Use only for explicitly labeled sensitivity analyses."
        ),
    )
    parser.add_argument("--n-folds", type=int, default=5, help="Subject-disjoint OOF folds on the development split.")
    parser.add_argument("--fold-iters", type=int, default=10_000, help="Subject fold balancing random search iterations.")
    parser.add_argument("--fold-seed", type=int, default=42, help="Subject fold random seed.")
    parser.add_argument("--quantiles", type=int, nargs="*", default=[10, 50, 90], help="Posterior quantile features.")
    parser.add_argument(
        "--submit-temperature",
        type=float,
        default=0.92,
        help="Temperature for SubmitWrapper-style equal-logit soft-argmax blending.",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto", help="Accepted for CLI symmetry; unused.")
    return parser


def main() -> None:
    """Run artifact-based stacking."""
    args = build_parser().parse_args()
    _ = choose_device(args.device)  # Validate the same device choices as other benchmark CLIs.
    experiment_root = resolve_path(args.experiment_root)
    out_dir = resolve_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    paper_table = resolve_paper_table(args.paper_table, out_dir)

    run_dirs = discover_run_dirs(experiment_root, seeds=set(args.seeds) if args.seeds else None)
    print(f"Discovered run dirs: {len(run_dirs)}")
    if not run_dirs:
        raise SystemExit(f"No run directories with predictions/ found under {path_text(experiment_root)}")

    missing = missing_artifacts(run_dirs)
    if missing:
        raise SystemExit(format_missing_artifacts(missing))

    try:
        runs = [
            load_run_artifacts(
                run_dir,
                target_min=float(args.target_min),
                target_max=float(args.target_max),
                strict_config_support=not bool(args.allow_support_subset),
            )
            for run_dir in run_dirs
        ]
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    groups = group_by_seed(runs)
    rows: list[dict] = []
    manifest_rows: list[dict] = []
    for seed in sorted(groups):
        seed_runs = sorted(groups[seed], key=lambda r: r.name)
        print(f"\nSeed {seed}: {len(seed_runs)} base models")
        seed_rows, seed_manifest = evaluate_seed_group(
            seed=seed,
            runs=seed_runs,
            n_folds=int(args.n_folds),
            fold_iters=int(args.fold_iters),
            fold_seed=int(args.fold_seed),
            quantiles=tuple(args.quantiles),
            submit_temperature=float(args.submit_temperature),
        )
        rows.extend(seed_rows)
        manifest_rows.extend(seed_manifest)

    summary = pd.DataFrame(rows)
    summary["method"] = pd.Categorical(summary["method"], categories=METHOD_ORDER, ordered=True)
    summary = summary.sort_values(["method", "seed"]).reset_index(drop=True)
    aggregate = aggregate_rows(summary).sort_values("method").reset_index(drop=True)
    aggregate["method"] = pd.Categorical(aggregate["method"], categories=METHOD_ORDER, ordered=True)
    aggregate = aggregate.sort_values("method").reset_index(drop=True)
    manifest = pd.DataFrame(manifest_rows).sort_values(["seed", "name"]).reset_index(drop=True)

    summary_path = out_dir / "stacking_summary.csv"
    aggregate_path = out_dir / "stacking_summary_aggregate.csv"
    manifest_path = out_dir / "run_manifest.csv"
    summary.to_csv(summary_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)
    manifest.to_csv(manifest_path, index=False)
    write_markdown_table(aggregate, paper_table)

    print(f"\nSaved stacking summary: {path_text(summary_path)}")
    print(f"Saved aggregate summary: {path_text(aggregate_path)}")
    print(f"Saved run manifest: {path_text(manifest_path)}")
    print(f"Saved paper table: {path_text(paper_table)}")
    print("\nAggregate")
    print(aggregate[["method", "seeds", "test_nrmse_mean", "test_nrmse_std"]].to_string(index=False))


def evaluate_seed_group(
    *,
    seed: int,
    runs: list[RunArtifacts],
    n_folds: int,
    fold_iters: int,
    fold_seed: int,
    quantiles: tuple[int, ...],
    submit_temperature: float,
) -> tuple[list[dict], list[dict]]:
    """Evaluate all requested stacking rows for one seed."""
    valid_ref, test_ref = validate_alignment(runs)
    y_dev = valid_ref["target"].to_numpy(dtype=np.float64)
    y_test = test_ref["target"].to_numpy(dtype=np.float64)
    folds = make_subject_balanced_folds(
        valid_ref,
        n_folds=n_folds,
        n_iter=fold_iters,
        seed=fold_seed + int(seed),
    )

    scalar_dev, scalar_test = scalar_prediction_matrices(runs)
    logits_blend_dev, logits_blend_test = submit_wrapper_logits_blend_predictions(
        runs,
        temperature=submit_temperature,
    )
    sfreq, win_offset = common_time_readout(runs)
    valid_logits_store = {run.model_key: run.valid_logits for run in runs}
    test_logits_store = {run.model_key: run.test_logits for run in runs}

    ridge_extractor = MetaFeatureExtractor(
        sfreq=sfreq,
        win_offset=win_offset,
        temps=(0.6, 1.0, 0.8),
        q_percentiles=quantiles,
    )
    ridge_posterior_dev = ridge_extractor.build_from_logits_store(valid_logits_store)
    ridge_posterior_test = ridge_extractor.build_from_logits_store(test_logits_store)
    ridge_meta_dev = np.concatenate([scalar_dev, ridge_posterior_dev], axis=1)
    ridge_meta_test = np.concatenate([scalar_test, ridge_posterior_test], axis=1)

    hgb_extractor = MetaFeatureExtractor(
        sfreq=sfreq,
        win_offset=win_offset,
        temps=(0.5, 0.7, 0.8, 1.0),
        q_percentiles=quantiles,
    )
    hgb_posterior_dev, hgb_posterior_names = hgb_extractor.build_from_logits_store(
        valid_logits_store,
        return_names=True,
    )
    hgb_posterior_test = hgb_extractor.build_from_logits_store(test_logits_store)
    scalar_names = [f"scalar_{run.model_key}_t_hard" for run in runs]
    hgb_meta_names = scalar_names + hgb_posterior_names
    hgb_meta_dev = np.concatenate([scalar_dev, hgb_posterior_dev], axis=1)
    hgb_meta_test = np.concatenate([scalar_test, hgb_posterior_test], axis=1)

    best_single = best_dev_selected_single_model(runs, scalar_dev, scalar_test, y_dev, y_test)
    rows: list[dict] = []
    rows.append(
        baseline_row(
            seed=seed,
            method="Equal-weight scalar RT blend",
            predictions_dev=np.mean(scalar_dev, axis=1),
            predictions_test=np.mean(scalar_test, axis=1),
            y_dev=y_dev,
            y_test=y_test,
            n_models=len(runs),
            best_single=best_single,
        )
    )
    rows.append(
        baseline_row(
            seed=seed,
            method="Equal-weight logits soft-argmax blend",
            predictions_dev=logits_blend_dev,
            predictions_test=logits_blend_test,
            y_dev=y_dev,
            y_test=y_test,
            n_models=len(runs),
            best_single=best_single,
        )
    )

    ridge_rt = fit_stacker_row(
        seed=seed,
        method="Ridge stacking, RT only",
        regressor=RidgeMetaRegressor(),
        X_dev=scalar_dev,
        X_test=scalar_test,
        y_dev=y_dev,
        y_test=y_test,
        folds=folds,
        n_models=len(runs),
        best_single=best_single,
    )
    boosting_rt = fit_stacker_row(
        seed=seed,
        method="Boosting stacking, RT only",
        regressor=HgbMetaRegressor(hgb_params=notebook_hgb_params(feature_names=scalar_names)),
        X_dev=scalar_dev,
        X_test=scalar_test,
        y_dev=y_dev,
        y_test=y_test,
        folds=folds,
        n_models=len(runs),
        best_single=best_single,
    )
    rows.extend([ridge_rt, boosting_rt])
    rows.append(
        fit_stacker_row(
            seed=seed,
            method="Ridge stacking, posterior meta-features",
            regressor=RidgeMetaRegressor(),
            X_dev=ridge_meta_dev,
            X_test=ridge_meta_test,
            y_dev=y_dev,
            y_test=y_test,
            folds=folds,
            n_models=len(runs),
            best_single=best_single,
            rt_only_reference=ridge_rt,
        )
    )
    rows.append(
        fit_stacker_row(
            seed=seed,
            method="Boosting stacking, posterior meta-features",
            regressor=HgbMetaRegressor(hgb_params=notebook_hgb_params(feature_names=hgb_meta_names)),
            X_dev=hgb_meta_dev,
            X_test=hgb_meta_test,
            y_dev=y_dev,
            y_test=y_test,
            folds=folds,
            n_models=len(runs),
            best_single=best_single,
            rt_only_reference=boosting_rt,
        )
    )

    manifest = [
        {
            "seed": run.seed,
            "experiment": run.experiment,
            "name": run.name,
            "run_dir": str(run.run_dir),
            "valid_rows": len(run.valid_predictions),
            "test_rows": len(run.test_predictions),
            "readout": run.readout,
            "sfreq": run.sfreq,
            "win_offset": run.win_offset,
            "submit_temperature": submit_temperature,
        }
        for run in runs
    ]
    return rows, manifest


def baseline_row(
    *,
    seed: int,
    method: str,
    predictions_dev: np.ndarray,
    predictions_test: np.ndarray,
    y_dev: np.ndarray,
    y_test: np.ndarray,
    n_models: int,
    best_single: dict,
) -> dict:
    """Build one non-learned ensemble result row."""
    dev_metrics = regression_metrics(y_dev, predictions_dev)
    test_metrics = regression_metrics(y_test, predictions_test)
    return result_row(
        seed=seed,
        method=method,
        dev_metrics=dev_metrics,
        test_metrics=test_metrics,
        n_models=n_models,
        best_single=best_single,
    )


def fit_stacker_row(
    *,
    seed: int,
    method: str,
    regressor: MetaRegressor,
    X_dev: np.ndarray,
    X_test: np.ndarray,
    y_dev: np.ndarray,
    y_test: np.ndarray,
    folds: np.ndarray,
    n_models: int,
    best_single: dict,
    rt_only_reference: dict | None = None,
) -> dict:
    """Fit a meta-model and return its summary row."""
    regressor.fit(X_dev, y_dev, folds)
    if regressor.oof_predictions_ is None:
        raise RuntimeError(f"{method} did not store OOF predictions.")
    test_predictions = regressor.predict(X_test)
    return result_row(
        seed=seed,
        method=method,
        dev_metrics=regression_metrics(y_dev, regressor.oof_predictions_),
        test_metrics=regression_metrics(y_test, test_predictions),
        n_models=n_models,
        best_single=best_single,
        rt_only_reference=rt_only_reference,
    )


def result_row(
    *,
    seed: int,
    method: str,
    dev_metrics: dict[str, float],
    test_metrics: dict[str, float],
    n_models: int,
    best_single: dict,
    rt_only_reference: dict | None = None,
) -> dict:
    """Return a normalized output row."""
    row = {
        "seed": int(seed),
        "method": method,
        "n_models": int(n_models),
        "dev_oof_rmse": dev_metrics["rmse"],
        "dev_oof_mae": dev_metrics["mae"],
        "dev_oof_nrmse": dev_metrics["nrmse"],
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_nrmse": test_metrics["nrmse"],
        "best_single_name": best_single["name"],
        "best_single_dev_nrmse": best_single["dev_nrmse"],
        "best_single_test_nrmse": best_single["test_nrmse"],
        "delta_vs_best_single_nrmse": test_metrics["nrmse"] - best_single["test_nrmse"],
        "delta_vs_rt_only_nrmse": np.nan,
    }
    if rt_only_reference is not None:
        row["delta_vs_rt_only_nrmse"] = test_metrics["nrmse"] - rt_only_reference["test_nrmse"]
    return row


def best_dev_selected_single_model(
    runs: list[RunArtifacts],
    scalar_dev: np.ndarray,
    scalar_test: np.ndarray,
    y_dev: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Select the best base model by development nRMSE and report its test score."""
    dev_scores = [regression_metrics(y_dev, scalar_dev[:, idx])["nrmse"] for idx in range(scalar_dev.shape[1])]
    best_idx = int(np.argmin(dev_scores))
    test_metrics = regression_metrics(y_test, scalar_test[:, best_idx])
    return {
        "name": runs[best_idx].name,
        "dev_nrmse": float(dev_scores[best_idx]),
        "test_nrmse": float(test_metrics["nrmse"]),
    }


def scalar_prediction_matrices(runs: list[RunArtifacts]) -> tuple[np.ndarray, np.ndarray]:
    """Return uncalibrated development and test scalar prediction matrices."""
    valid = np.column_stack([run.valid_predictions["prediction"].to_numpy(dtype=np.float32) for run in runs])
    test = np.column_stack([run.test_predictions["prediction"].to_numpy(dtype=np.float32) for run in runs])
    return valid, test


def submit_wrapper_logits_blend_predictions(
    runs: list[RunArtifacts],
    *,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend segmentation runs like SubmitWrapper: average logits, then soft-argmax."""
    sfreq, win_offset = common_time_readout(runs)
    valid_logits = np.mean(np.stack([run.valid_logits for run in runs], axis=0), axis=0)
    test_logits = np.mean(np.stack([run.test_logits for run in runs], axis=0), axis=0)
    return (
        soft_argmax_times(valid_logits, sfreq=sfreq, win_offset=win_offset, temperature=temperature),
        soft_argmax_times(test_logits, sfreq=sfreq, win_offset=win_offset, temperature=temperature),
    )


def soft_argmax_times(
    logits: np.ndarray,
    *,
    sfreq: float,
    win_offset: float,
    temperature: float,
) -> np.ndarray:
    """Return absolute soft-argmax times from raw segmentation logits."""
    if temperature <= 0:
        raise ValueError(f"Soft-argmax temperature must be positive, got {temperature}.")
    z = np.asarray(logits, dtype=np.float32) / float(temperature)
    z = z - np.max(z, axis=1, keepdims=True)
    p = np.exp(z)
    p = p / np.sum(p, axis=1, keepdims=True)
    t_grid = np.arange(z.shape[1], dtype=np.float32)[None, :] / float(sfreq)
    return np.sum(p * t_grid, axis=1).astype(np.float32) + float(win_offset)


def common_time_readout(runs: list[RunArtifacts]) -> tuple[float, float]:
    """Validate that all stacked models share the same time grid."""
    sfreq = float(runs[0].sfreq)
    win_offset = float(runs[0].win_offset)
    for run in runs[1:]:
        if abs(float(run.sfreq) - sfreq) > 1e-8 or abs(float(run.win_offset) - win_offset) > 1e-8:
            raise ValueError(
                "Kaggle-style MetaFeatureExtractor expects a shared time grid; "
                f"got {runs[0].run_dir} sfreq={sfreq}, win_offset={win_offset}, "
                f"but {run.run_dir} has sfreq={run.sfreq}, win_offset={run.win_offset}."
            )
    return sfreq, win_offset


def notebook_hgb_params(*, feature_names: list[str]) -> dict:
    """Return the HGB settings from the old ensembling notebook."""
    params = {
        "loss": "squared_error",
        "quantile": None,
        "learning_rate": 0.01,
        "max_iter": 2000,
        "max_leaf_nodes": 20,
        "max_depth": 4,
        "min_samples_leaf": 30,
        "l2_regularization": 1.0,
        "max_features": 0.1,
        "max_bins": 100,
        "interaction_cst": "pairwise",
        "early_stopping": "auto",
        "monotonic_cst": make_monotonic_constraints(feature_names, time_dir=1),
        "scoring": "loss",
        "validation_fraction": None,
        "n_iter_no_change": 50,
        "tol": 1e-7,
        "verbose": 0,
        "random_state": 42,
    }
    return supported_hgb_params(params)


def supported_hgb_params(params: dict) -> dict:
    """Drop constructor parameters unsupported by the installed sklearn."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    signature = inspect.signature(HistGradientBoostingRegressor)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return params
    supported = set(signature.parameters)
    dropped = sorted(key for key in params if key not in supported)
    if dropped:
        warnings.warn(
            "Installed sklearn HistGradientBoostingRegressor does not support "
            f"these notebook parameters and they will be dropped: {dropped}",
            RuntimeWarning,
            stacklevel=2,
        )
    return {key: value for key, value in params.items() if key in supported}


def validate_alignment(runs: list[RunArtifacts]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ensure every model in a seed group uses identical rows."""
    if not runs:
        raise ValueError("Cannot validate an empty run group.")
    valid_ref = alignment_frame(runs[0].valid_predictions)
    test_ref = alignment_frame(runs[0].test_predictions)
    for run in runs[1:]:
        assert_alignment(valid_ref, alignment_frame(run.valid_predictions), run.run_dir, "valid")
        assert_alignment(test_ref, alignment_frame(run.test_predictions), run.run_dir, "test")
    return runs[0].valid_predictions.copy(), runs[0].test_predictions.copy()


def assert_alignment(expected: pd.DataFrame, observed: pd.DataFrame, run_dir: Path, split: str) -> None:
    """Raise when saved rows are not aligned across models."""
    if len(expected) != len(observed):
        raise ValueError(f"{split} row count mismatch for {run_dir}: {len(observed)} != {len(expected)}")
    if not expected["row_id"].equals(observed["row_id"]):
        raise ValueError(f"{split} row_id mismatch for {run_dir}")
    if not expected["subject"].equals(observed["subject"]):
        raise ValueError(f"{split} subject mismatch for {run_dir}")
    if not np.allclose(expected["target"], observed["target"], rtol=0.0, atol=1e-7):
        raise ValueError(f"{split} target mismatch for {run_dir}")


def alignment_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return columns used to check artifact row alignment."""
    required = ["row_id", "subject", "target", "prediction"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Prediction CSV is missing required columns: {missing}")
    return pd.DataFrame(
        {
            "row_id": df["row_id"].astype(int),
            "subject": df["subject"].astype(str),
            "target": df["target"].astype(float),
        }
    )


def load_run_artifacts(
    run_dir: Path,
    *,
    target_min: float,
    target_max: float,
    strict_config_support: bool,
) -> RunArtifacts:
    """Load one run's frozen prediction/logit artifacts."""
    snapshot = load_snapshot(run_dir)
    config = snapshot["config"]
    if config.get("task") != "segmentation":
        raise ValueError(f"Only segmentation runs are supported: {run_dir}")
    if strict_config_support:
        validate_config_support(config, target_min=target_min, target_max=target_max, run_dir=run_dir)
    name = str(config["name"])
    seed = int(config["seed"])
    valid_predictions = pd.read_csv(run_dir / "predictions" / "best_val_predictions.csv")
    test_predictions = pd.read_csv(run_dir / "predictions" / "test_predictions.csv")
    validate_targets_inside(valid_predictions, target_min=target_min, target_max=target_max, run_dir=run_dir, split="valid")
    validate_targets_inside(test_predictions, target_min=target_min, target_max=target_max, run_dir=run_dir, split="test")

    valid_logits = np.load(run_dir / "predictions" / "best_logits.npy")
    test_logits = np.load(run_dir / "predictions" / "test_logits.npy")
    if valid_logits.shape[0] != len(valid_predictions):
        raise ValueError(f"best_logits row count mismatch for {run_dir}")
    if test_logits.shape[0] != len(test_predictions):
        raise ValueError(f"test_logits row count mismatch for {run_dir}")

    trainer_params = dict(config.get("trainer", {}).get("params", {}) or {})
    model_params = dict(config.get("model", {}).get("params", {}) or {})
    readout = str(trainer_params.get("temperature_readout", trainer_params.get("readout_distribution", "softmax")))
    sfreq = float(model_params.get("sfreq", trainer_params.get("sfreq", 100.0)))
    win_offset = float(trainer_params.get("win_offset", 0.5))
    return RunArtifacts(
        run_dir=run_dir,
        experiment=str(config["experiment"]),
        name=name,
        seed=seed,
        valid_predictions=valid_predictions,
        test_predictions=test_predictions,
        valid_logits=valid_logits.astype(np.float32, copy=False),
        test_logits=test_logits.astype(np.float32, copy=False),
        readout=readout,
        sfreq=sfreq,
        win_offset=win_offset,
    )


def validate_config_support(config: dict, *, target_min: float, target_max: float, run_dir: Path) -> None:
    """Ensure artifacts were produced under the requested target-support protocol."""
    data = dict(config.get("data", {}) or {})
    cfg_min = data.get("target_min")
    cfg_max = data.get("target_max")
    if cfg_min is None or cfg_max is None:
        raise ValueError(
            f"Run config has no explicit target support for {run_dir}. "
            f"Expected [{target_min}, {target_max}]. Use --allow-support-subset only for labeled diagnostics."
        )
    if abs(float(cfg_min) - float(target_min)) > 1e-8 or abs(float(cfg_max) - float(target_max)) > 1e-8:
        raise ValueError(
            f"Run config support mismatch for {run_dir}: "
            f"config=[{float(cfg_min)}, {float(cfg_max)}], requested=[{target_min}, {target_max}]. "
            "Use the matching experiment root or pass --allow-support-subset for an explicitly labeled diagnostic."
        )


def validate_targets_inside(df: pd.DataFrame, *, target_min: float, target_max: float, run_dir: Path, split: str) -> None:
    """Ensure saved artifacts fit the requested target support."""
    target = df["target"].to_numpy(dtype=np.float64)
    if np.any(target < target_min - 1e-8) or np.any(target > target_max + 1e-8):
        raise ValueError(
            f"{split} targets for {run_dir} are outside requested support "
            f"[{target_min}, {target_max}]. Observed range: [{target.min():.6f}, {target.max():.6f}]."
        )


def discover_run_dirs(experiment_root: Path, *, seeds: set[int] | None) -> list[Path]:
    """Find run directories with prediction artifacts under an experiment root."""
    run_dirs: list[Path] = []
    for predictions_dir in sorted(experiment_root.rglob("predictions")):
        run_dir = predictions_dir.parent
        if not (run_dir / "config.yaml").exists():
            continue
        try:
            snapshot = load_snapshot(run_dir)
            config = snapshot["config"]
            seed = int(config["seed"])
        except Exception:
            continue
        if config.get("task") != "segmentation":
            continue
        if seeds is not None and seed not in seeds:
            continue
        run_dirs.append(run_dir)
    seen: set[Path] = set()
    unique: list[Path] = []
    for run_dir in run_dirs:
        resolved = run_dir.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(run_dir)
    validate_no_duplicate_seed_names(unique)
    return unique


def validate_no_duplicate_seed_names(run_dirs: list[Path]) -> None:
    """Fail when multiple runs would create the same seed/model feature key."""
    keys: dict[tuple[int, str], Path] = {}
    for run_dir in run_dirs:
        snapshot = load_snapshot(run_dir)
        config = snapshot["config"]
        key = (int(config["seed"]), str(config["name"]))
        if key in keys:
            raise ValueError(f"Duplicate run for seed/name {key}: {keys[key]} and {run_dir}")
        keys[key] = run_dir


def group_by_seed(runs: list[RunArtifacts]) -> dict[int, list[RunArtifacts]]:
    """Group loaded runs by training seed."""
    groups: dict[int, list[RunArtifacts]] = {}
    for run in runs:
        groups.setdefault(run.seed, []).append(run)
    return groups


def missing_artifacts(run_dirs: list[Path]) -> dict[Path, list[Path]]:
    """Return missing required artifact paths for each run."""
    missing: dict[Path, list[Path]] = {}
    for run_dir in run_dirs:
        absent = [run_dir / relative for relative in REQUIRED_FILES if not (run_dir / relative).exists()]
        if absent:
            missing[run_dir] = absent
    return missing


def format_missing_artifacts(missing: dict[Path, list[Path]]) -> str:
    """Format an actionable missing-artifact error."""
    lines = ["Missing required stacking artifacts:"]
    for run_dir, paths in missing.items():
        lines.append(f"- {path_text(run_dir)}")
        for path in paths:
            lines.append(f"  - {path_text(path)}")
        lines.append(f"  refresh with: python benchmarks/scripts/reeval.py {path_text(run_dir)} --enable-temperature")
    return "\n".join(lines)


def load_snapshot(run_dir: Path) -> dict:
    """Load a benchmark run config snapshot."""
    path = run_dir / "config.yaml"
    with path.open("r", encoding="utf-8") as f:
        snapshot = yaml.safe_load(f)
    if not isinstance(snapshot, dict) or "config" not in snapshot:
        raise ValueError(f"Invalid run config snapshot: {path}")
    return snapshot


def resolve_path(path: Path) -> Path:
    """Resolve a CLI path relative to the project root."""
    path = Path(path)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def resolve_paper_table(value: Path | None, out_dir: Path) -> Path:
    """Choose the Markdown table path."""
    if value is not None:
        return resolve_path(value)
    canonical_out = (PROJECT_ROOT / "benchmarks/experiments/07_distribution_aware_stacking").resolve()
    if out_dir.resolve() == canonical_out:
        return (PROJECT_ROOT / "benchmarks/experiments/paper_tables/appendix_04_distribution_aware_stacking.md").resolve()
    return out_dir / "appendix_04_distribution_aware_stacking.md"


if __name__ == "__main__":
    main()
