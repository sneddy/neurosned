"""Meta-regressors for artifact-based stacking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np

from benchmarks.pkg.ensembling.metrics import regression_metrics


@dataclass
class StackerResult:
    """Predictions and metrics from one stacker fit."""

    name: str
    oof_predictions: np.ndarray
    test_predictions: np.ndarray
    dev_metrics: dict[str, float]
    test_metrics: dict[str, float]


class BaseStacker:
    """Small CV stacker interface used by the paper ablation script."""

    name = "base"

    def fit_predict(self, X_dev, y_dev, folds, X_test, y_test) -> StackerResult:
        """Fit fold-specific meta-models and return OOF/test predictions."""
        X_dev = np.asarray(X_dev, dtype=np.float32)
        X_test = np.asarray(X_test, dtype=np.float32)
        y_dev = np.asarray(y_dev, dtype=np.float64)
        y_test = np.asarray(y_test, dtype=np.float64)
        folds = np.asarray(folds)
        if X_dev.shape[0] != y_dev.shape[0] or folds.shape[0] != y_dev.shape[0]:
            raise ValueError("Development features, targets, and folds must have matching lengths.")
        if X_test.shape[0] != y_test.shape[0]:
            raise ValueError("Test features and targets must have matching lengths.")

        oof = np.empty_like(y_dev, dtype=np.float64)
        test_fold_predictions: list[np.ndarray] = []
        for fold in np.unique(folds):
            train_idx = np.where(folds != fold)[0]
            valid_idx = np.where(folds == fold)[0]
            model = self._fit_fold(X_dev, y_dev, train_idx, valid_idx)
            oof[valid_idx] = model.predict(X_dev[valid_idx])
            test_fold_predictions.append(np.asarray(model.predict(X_test), dtype=np.float64))

        test_predictions = np.mean(np.column_stack(test_fold_predictions), axis=1)
        return StackerResult(
            name=self.name,
            oof_predictions=oof,
            test_predictions=test_predictions,
            dev_metrics=regression_metrics(y_dev, oof),
            test_metrics=regression_metrics(y_test, test_predictions),
        )

    def _fit_fold(self, X, y, train_idx: np.ndarray, valid_idx: np.ndarray):
        raise NotImplementedError


class RidgeStacker(BaseStacker):
    """Standardized Ridge stacker with alpha selected inside each fold."""

    name = "Ridge"

    def __init__(self, alphas: np.ndarray | None = None):
        self.alphas = np.logspace(-4, 2, 25) if alphas is None else np.asarray(alphas, dtype=float)

    def _fit_fold(self, X, y, train_idx: np.ndarray, valid_idx: np.ndarray):
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_squared_error
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        try:
            from scipy.linalg import LinAlgWarning
        except Exception:  # pragma: no cover - scipy is present in the benchmark env.
            LinAlgWarning = RuntimeWarning

        best_model = None
        best_mse = float("inf")
        for alpha in self.alphas:
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("ridge", Ridge(alpha=float(alpha))),
                ]
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", LinAlgWarning)
                model.fit(X[train_idx], y[train_idx])
            mse = mean_squared_error(y[valid_idx], model.predict(X[valid_idx]))
            if mse < best_mse:
                best_mse = float(mse)
                best_model = model
        if best_model is None:
            raise RuntimeError("Failed to fit Ridge stacker.")
        return best_model


class BoostingStacker(BaseStacker):
    """Histogram gradient boosting stacker."""

    name = "Boosting"

    def __init__(self, params: dict[str, Any] | None = None):
        defaults = {
            "loss": "squared_error",
            "learning_rate": 0.03,
            "max_iter": 1000,
            "max_leaf_nodes": 20,
            "max_depth": 4,
            "min_samples_leaf": 30,
            "l2_regularization": 1.0,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 50,
            "random_state": 42,
        }
        if params:
            defaults.update(params)
        self.params = defaults

    def _fit_fold(self, X, y, train_idx: np.ndarray, valid_idx: np.ndarray):
        from sklearn.ensemble import HistGradientBoostingRegressor

        model = HistGradientBoostingRegressor(**self.params)
        model.fit(X[train_idx], y[train_idx])
        return model
