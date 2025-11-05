"""Minimal ridge/HGB CV-ensemble with fold-driven training and per-fold logging."""
import numpy as np
import pickle
import inspect
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Sequence
from copy import deepcopy
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error


class MetaRegressor(ABC):
    KIND = "base"

    def __init__(self, random_state: int = 42, agg: str = "mean"):
        self.random_state = int(random_state)
        self.agg = agg
        self.model = None
        self.models_: list = []

    @abstractmethod
    def fit_fold(self, X, y, tr_idx: np.ndarray, va_idx: np.ndarray,
                 iters_grid: Optional[Sequence[int]] = None):
        ...

    def fit(self, X, y, folds: np.ndarray) -> None:
        folds = np.asarray(folds)
        assert len(folds) == len(y)
        self.models_, y_oof = [], np.empty_like(y, dtype=float)

        for f, (tr, va) in self._splits_from_folds(folds):
            m = self.fit_fold(X, y, tr, va)
            self.models_.append(m)
            y_va = m.predict(X[va])
            y_oof[va] = y_va
            
            rmse_f = float(np.sqrt(mean_squared_error(y[va], y_va)))
            nrmse_f = float(rmse_f / np.std(y))
            print(f"Fold {f}: RMSE {rmse_f:.6f} | NRMSE {nrmse_f:.6f} | n={len(va)}")
        rmse = float(np.sqrt(mean_squared_error(y, y_oof)))
        nrmse = float(rmse / np.std(y))
        print(f"OOF  : RMSE {rmse:.6f} | NRMSE {nrmse:.6f} | n={len(y)}")

    def _aggregate(self, preds: np.ndarray) -> np.ndarray:
        return np.median(preds, axis=1) if self.agg == "median" else np.mean(preds, axis=1)

    def predict(self, X):
        if self.models_:
            preds = np.column_stack([m.predict(X) for m in self.models_])
            return self._aggregate(preds)
        if self.model is None:
            raise RuntimeError("Model is not fitted/loaded.")
        return self.model.predict(X)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(
                {"kind": self.KIND, "models": self.models_, "model": self.model,
                 "random_state": self.random_state, "agg": self.agg},
                f,
            )

    @staticmethod
    def load(path: str) -> "MetaRegressor":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        kind = obj["kind"]
        cls = RidgeMetaRegressor if kind == "ridge" else HgbMetaRegressor if kind == "hgb" else None
        if cls is None:
            raise ValueError(f"Unknown kind '{kind}'")
        inst = cls(random_state=obj.get("random_state", 42), agg=obj.get("agg", "mean"))
        inst.model = obj.get("model", None)
        inst.models_ = obj.get("models", []) or []
        return inst

    @staticmethod
    def _splits_from_folds(folds: np.ndarray):
        u = np.unique(folds)
        for f in u:
            va = np.where(folds == f)[0]
            tr = np.where(folds != f)[0]
            yield f, (tr, va)


class RidgeMetaRegressor(MetaRegressor):
    KIND = "ridge"

    def __init__(self, ridge_alphas: Optional[np.ndarray] = None, random_state: int = 42, agg: str = "mean"):
        super().__init__(random_state, agg)
        self.alphas = np.logspace(-4, 2, 25)

    def fit_fold(self, X, y, tr_idx: np.ndarray, va_idx: np.ndarray):
        best_pipe, best_mse = None, np.inf
        for a in self.alphas:
            pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=float(a)))])
            pipe.fit(X[tr_idx], y[tr_idx])
            mse = mean_squared_error(y[va_idx], pipe.predict(X[va_idx]))
            if mse < best_mse:
                best_mse, best_pipe = mse, pipe
        return best_pipe


class HgbMetaRegressor(MetaRegressor):
    KIND = "hgb"

    def __init__(self, hgb_params: Optional[Dict[str, Any]] = None, random_state: int = 42, agg: str = "mean"):
        super().__init__(random_state, agg)
        params = dict(loss="squared_error", learning_rate=0.05, max_iter=2000,
                      early_stopping=True, validation_fraction=0., n_iter_no_change=50,
                      random_state=random_state)
        if hgb_params:
            params.update(hgb_params)
        self.hgb_params = params

    def fit_fold(self, X, y, tr_idx: np.ndarray, va_idx: np.ndarray):
        m = HistGradientBoostingRegressor(**{**self.hgb_params, "early_stopping": True})
        m.fit(X[tr_idx], y[tr_idx], X_val=X[va_idx], y_val=y[va_idx])
        return m
