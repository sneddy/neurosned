"""Notebook-style stacking replica from frozen segmentation artifacts.

This is a diagnostic script, not the canonical paper pipeline. It intentionally
uses the old challenge_1 meta-feature and meta-regressor classes directly, and
only replaces notebook model inference with loading saved benchmark logits.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
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

from neurosned.wrappers.challenge_1.meta_features import MetaFeatureExtractor
from neurosned.wrappers.challenge_1.meta_regressor import HgbMetaRegressor, RidgeMetaRegressor


REQUIRED_FILES = (
    "predictions/best_val_predictions.csv",
    "predictions/best_logits.npy",
    "predictions/test_predictions.csv",
    "predictions/test_logits.npy",
)

METHOD_ORDER = [
    "Best single model",
    "Equal scalar blend",
    "Equal logits soft-argmax blend",
    "Notebook Ridge posterior features",
    "Notebook HGB posterior features",
]


@dataclass(frozen=True)
class RunArtifacts:
    run_dir: Path
    name: str
    seed: int
    valid_predictions: pd.DataFrame
    test_predictions: pd.DataFrame
    valid_logits: np.ndarray
    test_logits: np.ndarray
    sfreq: float
    win_offset: float

    @property
    def key(self) -> str:
        return self.name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/experiments/ensembling_tmp_notebook_replica"),
        help="Diagnostic output directory.",
    )
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--target-min", type=float, default=0.5)
    parser.add_argument("--target-max", type=float, default=2.5)
    parser.add_argument("--fold-iters", type=int, default=10_000)
    parser.add_argument("--fold-seed", type=int, default=42)
    parser.add_argument("--submit-temperature", type=float, default=0.92)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = resolve_path(args.experiment_root)
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    run_dirs = discover_run_dirs(root, seeds=set(args.seeds) if args.seeds else None)
    if not run_dirs:
        raise SystemExit(f"No segmentation prediction folders found under {root}")
    missing = missing_artifacts(run_dirs)
    if missing:
        raise SystemExit(format_missing(missing))

    runs = [
        load_run(
            run_dir,
            target_min=float(args.target_min),
            target_max=float(args.target_max),
        )
        for run_dir in run_dirs
    ]
    groups = group_by_seed(runs)

    rows: list[dict] = []
    manifest: list[dict] = []
    for seed in sorted(groups):
        seed_runs = sorted(groups[seed], key=lambda run: run.name)
        print(f"\nSeed {seed}: {len(seed_runs)} models")
        seed_rows, seed_manifest = evaluate_seed(
            seed=seed,
            runs=seed_runs,
            fold_iters=int(args.fold_iters),
            fold_seed=int(args.fold_seed),
            submit_temperature=float(args.submit_temperature),
        )
        rows.extend(seed_rows)
        manifest.extend(seed_manifest)

    summary = pd.DataFrame(rows)
    summary["method"] = pd.Categorical(summary["method"], categories=METHOD_ORDER, ordered=True)
    summary = summary.sort_values(["method", "seed"]).reset_index(drop=True)
    aggregate = aggregate_summary(summary)
    manifest_df = pd.DataFrame(manifest).sort_values(["seed", "name"]).reset_index(drop=True)

    summary_path = out / "notebook_replica_summary.csv"
    aggregate_path = out / "notebook_replica_aggregate.csv"
    manifest_path = out / "notebook_replica_manifest.csv"
    markdown_path = out / "notebook_replica_summary.md"
    summary.to_csv(summary_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)
    manifest_df.to_csv(manifest_path, index=False)
    write_markdown(aggregate, markdown_path)

    print(f"\nSaved summary: {summary_path}")
    print(f"Saved aggregate: {aggregate_path}")
    print(f"Saved manifest: {manifest_path}")
    print(f"Saved markdown: {markdown_path}")
    print("\nAggregate")
    print(aggregate[["method", "seeds", "test_nrmse_mean", "test_nrmse_std", "test_mae_mean", "test_mae_std"]].to_string(index=False))


def evaluate_seed(
    *,
    seed: int,
    runs: list[RunArtifacts],
    fold_iters: int,
    fold_seed: int,
    submit_temperature: float,
) -> tuple[list[dict], list[dict]]:
    valid_ref, test_ref = validate_alignment(runs)
    y_dev = valid_ref["target"].to_numpy(dtype=np.float64)
    y_test = test_ref["target"].to_numpy(dtype=np.float64)
    meta_information_valid = valid_ref[["row_id", "subject", "target"]].copy()
    folds, fold_std = notebook_subject_folds(
        meta_information_valid,
        n_folds=5,
        n_iter=fold_iters,
        seed=fold_seed,
    )
    meta_information_valid["fold"] = folds
    print(f"Best permutation found | std of fold target means: {fold_std:.6f}")
    print(meta_information_valid.groupby("fold").target.mean().to_string())

    logits_store = {run.key: run.valid_logits for run in runs}
    test_logits_store = {run.key: run.test_logits for run in runs}
    scalar_dev = np.column_stack([run.valid_predictions["prediction"].to_numpy(dtype=np.float64) for run in runs])
    scalar_test = np.column_stack([run.test_predictions["prediction"].to_numpy(dtype=np.float64) for run in runs])
    best_single = best_dev_selected_single(runs, scalar_dev, scalar_test, y_dev, y_test)

    rows = [
        result_row(
            seed=seed,
            method="Best single model",
            y_dev=y_dev,
            pred_dev=scalar_dev[:, best_single["index"]],
            y_test=y_test,
            pred_test=scalar_test[:, best_single["index"]],
            n_models=len(runs),
            best_single=best_single,
        ),
        result_row(
            seed=seed,
            method="Equal scalar blend",
            y_dev=y_dev,
            pred_dev=np.mean(scalar_dev, axis=1),
            y_test=y_test,
            pred_test=np.mean(scalar_test, axis=1),
            n_models=len(runs),
            best_single=best_single,
        ),
    ]

    sfreq, win_offset = common_time_readout(runs)
    pred_dev_logits = equal_logits_softargmax(
        [run.valid_logits for run in runs],
        sfreq=sfreq,
        win_offset=win_offset,
        temperature=submit_temperature,
    )
    pred_test_logits = equal_logits_softargmax(
        [run.test_logits for run in runs],
        sfreq=sfreq,
        win_offset=win_offset,
        temperature=submit_temperature,
    )
    rows.append(
        result_row(
            seed=seed,
            method="Equal logits soft-argmax blend",
            y_dev=y_dev,
            pred_dev=pred_dev_logits,
            y_test=y_test,
            pred_test=pred_test_logits,
            n_models=len(runs),
            best_single=best_single,
        )
    )

    fx_ridge = MetaFeatureExtractor(sfreq=sfreq, win_offset=win_offset, temps=(0.6, 1.0, 0.8))
    X_dev_ridge = fx_ridge.build_from_logits_store(logits_store, cls_outputs_store=None)
    X_test_ridge = fx_ridge.build_from_logits_store(test_logits_store, cls_outputs_store=None)
    ridge = RidgeMetaRegressor()
    ridge.fit(X_dev_ridge, y_dev, folds=folds)
    rows.append(
        result_row(
            seed=seed,
            method="Notebook Ridge posterior features",
            y_dev=y_dev,
            pred_dev=oof_predictions(ridge, X_dev_ridge, folds),
            y_test=y_test,
            pred_test=ridge.predict(X_test_ridge),
            n_models=len(runs),
            best_single=best_single,
        )
    )

    fx_hgb = MetaFeatureExtractor(sfreq=sfreq, win_offset=win_offset, temps=(0.5, 0.7, 0.8, 1.0))
    X_dev_hgb, colnames = fx_hgb.build_from_logits_store(logits_store, cls_outputs_store=None, return_names=True)
    X_test_hgb = fx_hgb.build_from_logits_store(test_logits_store, cls_outputs_store=None)
    defaults_hgbr = notebook_hgbr_params(colnames)
    hgb = HgbMetaRegressor(hgb_params=defaults_hgbr)
    hgb.fit(X_dev_hgb, y_dev, folds=folds)
    rows.append(
        result_row(
            seed=seed,
            method="Notebook HGB posterior features",
            y_dev=y_dev,
            pred_dev=oof_predictions(hgb, X_dev_hgb, folds),
            y_test=y_test,
            pred_test=hgb.predict(X_test_hgb),
            n_models=len(runs),
            best_single=best_single,
        )
    )

    manifest = [
        {
            "seed": run.seed,
            "name": run.name,
            "run_dir": str(run.run_dir),
            "valid_rows": len(run.valid_predictions),
            "test_rows": len(run.test_predictions),
            "sfreq": run.sfreq,
            "win_offset": run.win_offset,
            "fold_target_std": fold_std,
            "submit_temperature": submit_temperature,
        }
        for run in runs
    ]
    return rows, manifest


def notebook_subject_folds(
    metadata: pd.DataFrame,
    *,
    n_folds: int,
    n_iter: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    subject_means = metadata.groupby("subject").target.mean()
    subjects = subject_means.index.values
    n_subjects = len(subjects)
    base_fold_assignments = np.arange(n_subjects) % int(n_folds)

    best_std = float("inf")
    best_subject_to_fold = None
    rng = np.random.default_rng(seed=seed)
    for _ in range(int(n_iter)):
        permuted_assignments = np.copy(base_fold_assignments)
        rng.shuffle(permuted_assignments)
        subject_to_fold = dict(zip(subjects, permuted_assignments))
        fold_assignments = metadata["subject"].map(subject_to_fold).values
        mean_per_fold = metadata.assign(fold=fold_assignments).groupby("fold").target.mean()
        fold_std = float(mean_per_fold.std())
        if fold_std < best_std:
            best_std = fold_std
            best_subject_to_fold = subject_to_fold.copy()

    if best_subject_to_fold is None:
        raise RuntimeError("Failed to build notebook fold assignment.")
    folds = metadata["subject"].map(best_subject_to_fold).values
    return folds.astype(int), best_std


def notebook_hgbr_params(feature_names: list[str]) -> dict:
    return {
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


def make_monotonic_constraints(feature_names, time_dir=1):
    cst = []
    for name in feature_names:
        is_time = (
            name.endswith("t_hard")
            or ("t_abs_temp" in name)
            or bool(re.search(r"q\d+_temp", name))
        )
        cst.append(time_dir if is_time else 0)
    return np.asarray(cst, dtype=int)


def oof_predictions(regressor, X: np.ndarray, folds: np.ndarray) -> np.ndarray:
    preds = np.empty(len(folds), dtype=np.float64)
    for model, fold in zip(regressor.models_, np.unique(folds)):
        idx = np.where(folds == fold)[0]
        preds[idx] = model.predict(X[idx])
    return preds


def result_row(
    *,
    seed: int,
    method: str,
    y_dev: np.ndarray,
    pred_dev: np.ndarray,
    y_test: np.ndarray,
    pred_test: np.ndarray,
    n_models: int,
    best_single: dict,
) -> dict:
    dev = metrics(y_dev, pred_dev)
    test = metrics(y_test, pred_test)
    return {
        "seed": int(seed),
        "method": method,
        "n_models": int(n_models),
        "dev_rmse": dev["rmse"],
        "dev_mae": dev["mae"],
        "dev_nrmse": dev["nrmse"],
        "test_rmse": test["rmse"],
        "test_mae": test["mae"],
        "test_nrmse": test["nrmse"],
        "best_single_name": best_single["name"],
        "best_single_test_nrmse": best_single["test_nrmse"],
        "delta_vs_best_single_nrmse": test["nrmse"] - best_single["test_nrmse"],
    }


def best_dev_selected_single(
    runs: list[RunArtifacts],
    scalar_dev: np.ndarray,
    scalar_test: np.ndarray,
    y_dev: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    dev_scores = [metrics(y_dev, scalar_dev[:, idx])["nrmse"] for idx in range(scalar_dev.shape[1])]
    best_idx = int(np.argmin(dev_scores))
    return {
        "index": best_idx,
        "name": runs[best_idx].name,
        "dev_nrmse": float(dev_scores[best_idx]),
        "test_nrmse": metrics(y_test, scalar_test[:, best_idx])["nrmse"],
    }


def equal_logits_softargmax(
    logits_list: list[np.ndarray],
    *,
    sfreq: float,
    win_offset: float,
    temperature: float,
) -> np.ndarray:
    logits = np.mean(np.stack(logits_list, axis=0), axis=0)
    z = logits.astype(np.float32) / float(temperature)
    z = z - np.max(z, axis=1, keepdims=True)
    p = np.exp(z)
    p = p / np.sum(p, axis=1, keepdims=True)
    t_grid = np.arange(z.shape[1], dtype=np.float32)[None, :] / float(sfreq)
    return np.sum(p * t_grid, axis=1) + float(win_offset)


def metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    return {
        "rmse": rmse,
        "mae": float(np.mean(np.abs(err))),
        "nrmse": rmse / float(np.std(y_true)),
    }


def aggregate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    records = []
    for method, group in summary.groupby("method", sort=False, observed=False):
        record = {
            "method": method,
            "seeds": int(group["seed"].nunique()),
            "n_models_mean": float(group["n_models"].mean()),
        }
        for column in (
            "dev_nrmse",
            "dev_mae",
            "test_nrmse",
            "test_mae",
            "delta_vs_best_single_nrmse",
        ):
            values = pd.to_numeric(group[column], errors="coerce")
            record[f"{column}_mean"] = float(values.mean())
            record[f"{column}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        records.append(record)
    aggregate = pd.DataFrame(records)
    aggregate["method"] = pd.Categorical(aggregate["method"], categories=METHOD_ORDER, ordered=True)
    return aggregate.sort_values("method").reset_index(drop=True)


def write_markdown(aggregate: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Notebook Replica Stacking Diagnostic",
        "",
        "This table uses the old `neurosned.wrappers.challenge_1` feature extractor and meta-regressors directly.",
        "",
        "| Method | Seeds | R11 nRMSE | R11 MAE | Delta vs best single |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in aggregate.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    str(int(row["seeds"])),
                    fmt_pm(row, "test_nrmse"),
                    fmt_pm(row, "test_mae"),
                    fmt_pm(row, "delta_vs_best_single_nrmse", signed=True),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt_pm(row: pd.Series, column: str, *, signed: bool = False) -> str:
    mean = float(row[f"{column}_mean"])
    std = float(row[f"{column}_std"])
    prefix = "+" if signed and mean > 0 else ""
    return f"{prefix}{mean:.4f} +/- {std:.4f}"


def validate_alignment(runs: list[RunArtifacts]) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_ref = alignment_frame(runs[0].valid_predictions)
    test_ref = alignment_frame(runs[0].test_predictions)
    for run in runs[1:]:
        assert_alignment(valid_ref, alignment_frame(run.valid_predictions), run.run_dir, "valid")
        assert_alignment(test_ref, alignment_frame(run.test_predictions), run.run_dir, "test")
    return valid_ref.copy(), test_ref.copy()


def alignment_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = ["row_id", "subject", "target", "prediction"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")
    return pd.DataFrame(
        {
            "row_id": df["row_id"].astype(int),
            "subject": df["subject"].astype(str),
            "target": df["target"].astype(float),
            "prediction": df["prediction"].astype(float),
        }
    )


def assert_alignment(expected: pd.DataFrame, observed: pd.DataFrame, run_dir: Path, split: str) -> None:
    if len(expected) != len(observed):
        raise ValueError(f"{split} row count mismatch for {run_dir}")
    if not expected["row_id"].equals(observed["row_id"]):
        raise ValueError(f"{split} row_id mismatch for {run_dir}")
    if not expected["subject"].equals(observed["subject"]):
        raise ValueError(f"{split} subject mismatch for {run_dir}")
    if not np.allclose(expected["target"], observed["target"], rtol=0.0, atol=1e-7):
        raise ValueError(f"{split} target mismatch for {run_dir}")


def common_time_readout(runs: list[RunArtifacts]) -> tuple[float, float]:
    sfreq = float(runs[0].sfreq)
    win_offset = float(runs[0].win_offset)
    for run in runs[1:]:
        if abs(run.sfreq - sfreq) > 1e-8 or abs(run.win_offset - win_offset) > 1e-8:
            raise ValueError(f"Time readout mismatch between {runs[0].run_dir} and {run.run_dir}")
    return sfreq, win_offset


def load_run(run_dir: Path, *, target_min: float, target_max: float) -> RunArtifacts:
    config = load_config(run_dir)
    if config.get("task") != "segmentation":
        raise ValueError(f"Only segmentation runs are supported: {run_dir}")
    valid_predictions = pd.read_csv(run_dir / "predictions" / "best_val_predictions.csv")
    test_predictions = pd.read_csv(run_dir / "predictions" / "test_predictions.csv")
    validate_targets(valid_predictions, target_min=target_min, target_max=target_max, run_dir=run_dir, split="valid")
    validate_targets(test_predictions, target_min=target_min, target_max=target_max, run_dir=run_dir, split="test")
    valid_logits = np.load(run_dir / "predictions" / "best_logits.npy")
    test_logits = np.load(run_dir / "predictions" / "test_logits.npy")
    if valid_logits.shape[0] != len(valid_predictions) or test_logits.shape[0] != len(test_predictions):
        raise ValueError(f"Logit row count mismatch for {run_dir}")

    trainer_params = dict(config.get("trainer", {}).get("params", {}) or {})
    model_params = dict(config.get("model", {}).get("params", {}) or {})
    return RunArtifacts(
        run_dir=run_dir,
        name=str(config["name"]),
        seed=int(config["seed"]),
        valid_predictions=valid_predictions,
        test_predictions=test_predictions,
        valid_logits=valid_logits.astype(np.float32, copy=False),
        test_logits=test_logits.astype(np.float32, copy=False),
        sfreq=float(model_params.get("sfreq", trainer_params.get("sfreq", 100.0))),
        win_offset=float(trainer_params.get("win_offset", 0.5)),
    )


def validate_targets(df: pd.DataFrame, *, target_min: float, target_max: float, run_dir: Path, split: str) -> None:
    target = df["target"].to_numpy(dtype=np.float64)
    if np.any(target < target_min - 1e-8) or np.any(target > target_max + 1e-8):
        raise ValueError(
            f"{split} targets outside [{target_min}, {target_max}] for {run_dir}: "
            f"[{target.min():.6f}, {target.max():.6f}]"
        )


def discover_run_dirs(root: Path, *, seeds: set[int] | None) -> list[Path]:
    dirs = []
    for predictions_dir in sorted(root.rglob("predictions")):
        run_dir = predictions_dir.parent
        config_path = run_dir / "config.yaml"
        if not config_path.exists():
            continue
        try:
            config = load_config(run_dir)
        except Exception:
            continue
        if config.get("task") != "segmentation":
            continue
        if seeds is not None and int(config["seed"]) not in seeds:
            continue
        dirs.append(run_dir)
    validate_no_duplicate_seed_names(dirs)
    return dirs


def validate_no_duplicate_seed_names(run_dirs: list[Path]) -> None:
    seen: dict[tuple[int, str], Path] = {}
    for run_dir in run_dirs:
        config = load_config(run_dir)
        key = (int(config["seed"]), str(config["name"]))
        if key in seen:
            raise ValueError(f"Duplicate seed/name {key}: {seen[key]} and {run_dir}")
        seen[key] = run_dir


def load_config(run_dir: Path) -> dict:
    with (run_dir / "config.yaml").open("r", encoding="utf-8") as f:
        snapshot = yaml.safe_load(f)
    if not isinstance(snapshot, dict) or "config" not in snapshot:
        raise ValueError(f"Invalid config snapshot: {run_dir / 'config.yaml'}")
    return snapshot["config"]


def group_by_seed(runs: list[RunArtifacts]) -> dict[int, list[RunArtifacts]]:
    groups: dict[int, list[RunArtifacts]] = {}
    for run in runs:
        groups.setdefault(run.seed, []).append(run)
    return groups


def missing_artifacts(run_dirs: list[Path]) -> dict[Path, list[Path]]:
    missing = {}
    for run_dir in run_dirs:
        absent = [run_dir / rel for rel in REQUIRED_FILES if not (run_dir / rel).exists()]
        if absent:
            missing[run_dir] = absent
    return missing


def format_missing(missing: dict[Path, list[Path]]) -> str:
    lines = ["Missing required artifacts:"]
    for run_dir, paths in missing.items():
        lines.append(f"- {run_dir}")
        for path in paths:
            lines.append(f"  - {path}")
    return "\n".join(lines)


def resolve_path(path: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


if __name__ == "__main__":
    main()
