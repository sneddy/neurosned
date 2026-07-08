"""Kaggle-style meta-regressors for artifact-based stacking."""

from __future__ import annotations

import inspect
import pickle
import warnings
from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

import numpy as np


class MetaRegressor(ABC):
    """Minimal fold-driven meta-regressor with old challenge-1 semantics."""

    KIND = "base"

    def __init__(self, random_state: int = 42, agg: str = "mean"):
        self.random_state = int(random_state)
        self.agg = agg
        self.model = None
        self.models_: list = []
        self.oof_predictions_: np.ndarray | None = None
        self.oof_metrics_: dict[str, float] | None = None

    @abstractmethod
    def fit_fold(
        self,
        X,
        y,
        tr_idx: np.ndarray,
        va_idx: np.ndarray,
        iters_grid: Optional[Sequence[int]] = None,
    ):
        """Fit one fold-specific model."""

    def fit(self, X, y, folds: np.ndarray) -> None:
        """Fit one model per fold and store OOF predictions."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=float)
        folds = np.asarray(folds)
        if len(folds) != len(y):
            raise ValueError("folds and targets must have matching lengths.")
        if len(X) != len(y):
            raise ValueError("features and targets must have matching lengths.")

        self.models_, y_oof = [], np.empty_like(y, dtype=float)

        for fold, (tr, va) in self._splits_from_folds(folds):
            model = self.fit_fold(X, y, tr, va)
            self.models_.append(model)
            y_va = model.predict(X[va])
            y_oof[va] = y_va

            rmse_f = float(np.sqrt(_mean_squared_error(y[va], y_va)))
            nrmse_f = float(rmse_f / np.std(y))
            print(f"Fold {fold}: RMSE {rmse_f:.6f} | NRMSE {nrmse_f:.6f} | n={len(va)}")

        rmse = float(np.sqrt(_mean_squared_error(y, y_oof)))
        nrmse = float(rmse / np.std(y))
        print(f"OOF  : RMSE {rmse:.6f} | NRMSE {nrmse:.6f} | n={len(y)}")
        self.oof_predictions_ = y_oof
        self.oof_metrics_ = {"rmse": rmse, "nrmse": nrmse}

    def _aggregate(self, preds: np.ndarray) -> np.ndarray:
        return np.median(preds, axis=1) if self.agg == "median" else np.mean(preds, axis=1)

    def predict(self, X):
        """Predict with the mean/median ensemble of fold models."""
        X = np.asarray(X, dtype=np.float32)
        if self.models_:
            preds = np.column_stack([model.predict(X) for model in self.models_])
            return self._aggregate(preds)
        if self.model is None:
            raise RuntimeError("Model is not fitted/loaded.")
        return self.model.predict(X)

    def save(self, path: str):
        """Persist sklearn fold models."""
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "kind": self.KIND,
                    "models": self.models_,
                    "model": self.model,
                    "random_state": self.random_state,
                    "agg": self.agg,
                },
                f,
            )

    @staticmethod
    def load(path: str) -> "MetaRegressor":
        """Load a persisted meta-regressor."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        kind = obj["kind"]
        cls = RidgeMetaRegressor if kind == "ridge" else HgbMetaRegressor if kind == "hgb" else None
        if cls is None:
            raise ValueError(f"Unknown kind {kind!r}")
        inst = cls(random_state=obj.get("random_state", 42), agg=obj.get("agg", "mean"))
        inst.model = obj.get("model", None)
        inst.models_ = obj.get("models", []) or []
        return inst

    @staticmethod
    def _splits_from_folds(folds: np.ndarray):
        unique = np.unique(folds)
        for fold in unique:
            va = np.where(folds == fold)[0]
            tr = np.where(folds != fold)[0]
            yield fold, (tr, va)


class RidgeMetaRegressor(MetaRegressor):
    """Standardized Ridge stacker with fold-local alpha selection."""

    KIND = "ridge"

    def __init__(
        self,
        ridge_alphas: Optional[np.ndarray] = None,
        random_state: int = 42,
        agg: str = "mean",
    ):
        super().__init__(random_state, agg)
        self.alphas = np.logspace(-4, 2, 25) if ridge_alphas is None else np.asarray(ridge_alphas, dtype=float)

    def fit_fold(self, X, y, tr_idx: np.ndarray, va_idx: np.ndarray, iters_grid: Optional[Sequence[int]] = None):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        best_pipe, best_mse = None, np.inf
        for alpha in self.alphas:
            pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=float(alpha)))])
            pipe.fit(X[tr_idx], y[tr_idx])
            mse = _mean_squared_error(y[va_idx], pipe.predict(X[va_idx]))
            if mse < best_mse:
                best_mse, best_pipe = mse, pipe
        return best_pipe


class HgbMetaRegressor(MetaRegressor):
    """Histogram gradient boosting stacker."""

    KIND = "hgb"

    def __init__(self, hgb_params: Optional[dict[str, Any]] = None, random_state: int = 42, agg: str = "mean"):
        super().__init__(random_state, agg)
        params = dict(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.0,
            n_iter_no_change=50,
            random_state=random_state,
        )
        if hgb_params:
            params.update(hgb_params)
        self.hgb_params = params

    def fit_fold(self, X, y, tr_idx: np.ndarray, va_idx: np.ndarray, iters_grid: Optional[Sequence[int]] = None):
        from sklearn.ensemble import HistGradientBoostingRegressor

        model = HistGradientBoostingRegressor(**self.hgb_params)
        return fit_hgb_with_fold_validation(model, X, y, tr_idx, va_idx)


def _mean_squared_error(y_true, y_pred) -> float:
    """Small local MSE helper to keep sklearn imports lazy."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_pred - y_true) ** 2))


def fit_hgb_with_fold_validation(model, X, y, tr_idx: np.ndarray, va_idx: np.ndarray):
    """Fit HGB using explicit fold validation when sklearn supports it."""
    fit_signature = inspect.signature(model.fit)
    if "X_val" in fit_signature.parameters and "y_val" in fit_signature.parameters:
        return model.fit(X[tr_idx], y[tr_idx], X_val=X[va_idx], y_val=y[va_idx])

    warnings.warn(
        "Installed sklearn HistGradientBoostingRegressor.fit does not support "
        "X_val/y_val; falling back to internal validation_fraction behavior.",
        RuntimeWarning,
        stacklevel=2,
    )
    return model.fit(X[tr_idx], y[tr_idx])
