import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

class KnnModel(nn.Module):
    """
    Torch-модель для предсказаний на заранее вычисленных EEG фичах.
    Использует k-ближайших соседей (knn).
    Интерфейс и сигнатуры совместимы с RidgeModel.
    """
    def __init__(self, usecols: list[str], n_neighbors: int = 5, weights: str = 'uniform'):
        super().__init__()
        self.usecols = usecols
        self.n_neighbors = n_neighbors
        self.weights = weights

        self.register_buffer("feat_means", torch.empty(len(usecols), dtype=torch.float32))
        self.register_buffer("feat_stds", torch.empty(len(usecols), dtype=torch.float32))

        # knn model fitted in .make_checkpoint; loaded as attribute here
        self.knn = None  # Will be set on load

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        feats: (B, N) — признаки (возможно полный вектор)
        """
        # Normalize features as in training
        feats = (feats - self.feat_means) / self.feat_stds
        feats = feats[:, :len(self.usecols)]  # In case feats has more columns
        feats_np = feats.detach().cpu().numpy()
        y_pred = self._knn_predict(feats_np)
        return torch.from_numpy(y_pred.astype(np.float32)).to(feats.device).unsqueeze(1)

    def _knn_predict(self, feats_np: np.ndarray) -> np.ndarray:
        if self.knn is None:
            raise RuntimeError("KNN model has not been set. Use make_checkpoint and/or load_state_dict.")
        preds = self.knn.predict(feats_np)
        return preds.reshape(-1)

    @classmethod
    def make_checkpoint(cls,
                        X_train: pd.DataFrame,
                        y_train: pd.Series | np.ndarray,
                        usecols: list[str],
                        n_neighbors: int = 5,
                        weights: str = 'uniform',
                        save_path: str = "knn_checkpoint.pt"):
        """
        Train KNN regressor on X_train[usecols], save only per-usecols normalization stats and training points.
        Args:
            X_train: DataFrame with all features.
            y_train: Series/array of targets.
            usecols: feature names to train on (order matters).
            n_neighbors: Number of neighbors for KNN.
            weights: Neighbor weighting ('uniform' or 'distance').
            save_path: path to save the checkpoint.
        """
        X_sub = X_train[usecols].to_numpy(dtype=np.float32)
        y_arr = np.asarray(y_train, dtype=np.float32).reshape(-1)
        assert X_sub.shape[0] == y_arr.shape[0], "X_train and y_train length mismatch"

        feat_means = X_sub.mean(axis=0)
        feat_stds  = X_sub.std(axis=0)
        feat_stds[feat_stds == 0] = 1.0
        print('Means: ', feat_means)
        print('stds:  ', feat_stds)

        Xn = (X_sub - feat_means) / feat_stds
        knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
        knn.fit(Xn, y_arr)

        # Save everything in state_dict and meta!
        state_dict = {
            "Xn": torch.tensor(Xn, dtype=torch.float32),  # shape: (n_samples, n_features)
            "y_arr": torch.tensor(y_arr, dtype=torch.float32), # shape: (n_samples,)
            "feat_means": torch.tensor(feat_means, dtype=torch.float32),
            "feat_stds": torch.tensor(feat_stds, dtype=torch.float32),
            "n_neighbors": n_neighbors,
            "weights": weights,
            "usecols": usecols,
        }
        meta = {
            "usecols": usecols,
            "n_neighbors": n_neighbors,
            "weights": weights,
        }
        torch.save({"state_dict": state_dict, "meta": meta}, save_path)
        print(f"✅ KNN checkpoint saved to {save_path}")

        # 5) Return a ready-to-use model bound to usecols/params
        model = cls(usecols, n_neighbors=n_neighbors, weights=weights)
        model.load_state_dict(state_dict, strict=False)
        model._init_knn_from_data(
            Xn=state_dict["Xn"].numpy(),
            y_arr=state_dict["y_arr"].numpy()
        )
        model.eval()
        return model

    def _init_knn_from_data(self, Xn, y_arr):
        # Used for loading/checkpoint interoperability
        self.knn = KNeighborsRegressor(n_neighbors=self.n_neighbors, weights=self.weights)
        self.knn.fit(Xn, y_arr)

    def load_state_dict(self, state_dict, strict=True):
        # Overridden for full interoperability: loads model params usedcols, weights, n_neighbors if available
        # Accept both checkpoint dicts and bare state_dicts
        # State dict could be torch.load()['state_dict'] (used in existing notebooks)
        _update_params = False
        # Try to extract params from state_dict if present; else do nothing
        if "usecols" in state_dict:
            self.usecols = list(state_dict["usecols"])
            _update_params = True
        if "n_neighbors" in state_dict:
            n_neighbors = state_dict["n_neighbors"]
            if isinstance(n_neighbors, torch.Tensor):
                n_neighbors = int(n_neighbors.item())
            self.n_neighbors = int(n_neighbors)
            _update_params = True
        if "weights" in state_dict:
            weights = state_dict["weights"]
            if isinstance(weights, torch.Tensor):
                # Will be string, not tensor, but just in case
                weights = str(weights)
            self.weights = weights
            _update_params = True

        # Set normalization vectors to right size if usecols have changed
        if _update_params:
            # Remove existing buffers if needed
            try:
                del self._buffers['feat_means']
                del self._buffers['feat_stds']
            except Exception:
                pass
            self.register_buffer("feat_means", torch.empty(len(self.usecols), dtype=torch.float32))
            self.register_buffer("feat_stds", torch.empty(len(self.usecols), dtype=torch.float32))

        # Filter out things that are not buffers
        base_dict = {k: v for k, v in state_dict.items() if k not in ("Xn", "y_arr", "n_neighbors", "weights", "usecols")}
        super().load_state_dict(base_dict, strict=strict)
        if "Xn" in state_dict and "y_arr" in state_dict:
            Xn = state_dict["Xn"]
            y_arr = state_dict["y_arr"]
            if isinstance(Xn, torch.Tensor):
                Xn = Xn.cpu().numpy()
            if isinstance(y_arr, torch.Tensor):
                y_arr = y_arr.cpu().numpy()
            self._init_knn_from_data(Xn, y_arr)
        return None

    def forward_df(self, X: pd.DataFrame, clip_low=None, clip_high=None):
        """
        Compute forward prediction (as numpy) from pandas DataFrame (using self.usecols).
        """
        X_full = X[self.usecols].to_numpy(dtype=np.float32)
        feats = torch.from_numpy(X_full)
        preds_t = self.forward(feats).squeeze(1)
        preds = preds_t.detach().cpu().numpy()
        # Apply clipping if clip_low or clip_high is not None
        if clip_low is not None or clip_high is not None:
            preds = np.clip(preds, a_min=clip_low, a_max=clip_high)
        return preds

    def eval_df(self, X: pd.DataFrame, y, clip_low=None, clip_high=None) -> dict:
        preds = self.forward_df(X, clip_low=clip_low, clip_high=clip_high)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        mse = float(np.mean((y - preds) ** 2))
        nrmse = float(np.sqrt(mse) / np.std(y))
        return nrmse
