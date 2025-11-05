import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits


class BoostingRegressorModel(nn.Module):
    """
    Wrapper for HistGradientBoostingRegressor trained on precomputed EEG features.
    Matches the interface of RandomForestModel (state_dict I/O via torch).
    Single-threaded fit/predict is enforced via threadpoolctl (equivalent to n_jobs=1).
    """
    def __init__(self, usecols: list[str]):
        super().__init__()
        self.usecols = usecols
        self.register_buffer("feat_means", torch.empty(len(usecols), dtype=torch.float32))
        self.register_buffer("feat_stds", torch.empty(len(usecols), dtype=torch.float32))
        self.gb_params = None   # Save hyperparams for reproducibility
        self.gb_state = None    # Pickled sklearn estimator

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        feats: (B, N) torch tensor of features.
        Normalizes using stored stats, converts to NumPy, and predicts with sklearn HGBR.
        """
        feats_np = (
            (feats.detach().cpu().numpy() - self.feat_means.cpu().numpy())
            / self.feat_stds.cpu().numpy()
        )
        # Force single-threaded prediction
        with threadpool_limits(limits=1):
            preds = self.gb_state.predict(feats_np)
        return torch.tensor(preds, dtype=torch.float32, device=feats.device).unsqueeze(1)

    @classmethod
    def make_checkpoint(cls,
                        X_train: pd.DataFrame,
                        y_train: pd.Series | np.ndarray,
                        usecols: list[str],
                        gb_kwargs: dict = None,
                        save_path: str = "boosting_regressor_checkpoint.pt"):
        """
        Trains a HistGradientBoostingRegressor on (X_train[usecols], y_train), saves model
        and normalization stats via torch.save, and returns an initialized model.
        """
        X_sub = X_train[usecols].to_numpy(dtype=np.float32)
        y_arr = np.asarray(y_train, dtype=np.float32).reshape(-1)
        assert X_sub.shape[0] == y_arr.shape[0], "X_train and y_train length mismatch"

        # Normalization (per-feature)
        feat_means = X_sub.mean(axis=0)
        feat_stds  = X_sub.std(axis=0)
        feat_stds[feat_stds == 0] = 1.0

        # Normalize features
        Xn = (X_sub - feat_means) / feat_stds

        # Default params; HGBR has no n_jobs, so we enforce single-threading below
        gb_kwargs = gb_kwargs or dict(
            loss="squared_error",
            learning_rate=0.1,
            max_iter=300,
            max_depth=None,
            max_leaf_nodes=31,
            random_state=42
        )

        gb = HistGradientBoostingRegressor(**gb_kwargs)

        # Fit single-threaded (equivalent to n_jobs=1)
        with threadpool_limits(limits=1):
            gb.fit(Xn, y_arr)
        print("✅ Trained HistGradientBoostingRegressor (single-threaded)")

        # Save checkpoint: normalization and estimator
        state_dict = {
            "feat_means": torch.tensor(feat_means, dtype=torch.float32),
            "feat_stds" : torch.tensor(feat_stds, dtype=torch.float32),
            "gb_params": gb.get_params(),
            "gb_state": pickle.dumps(gb),
        }
        meta = {"usecols": usecols}
        torch.save({"state_dict": state_dict, "meta": meta}, save_path)
        print(f"✅ Boosting checkpoint saved to {save_path}")

        # Return initialized model
        model = cls(usecols)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model

    def load_state_dict(self, state_dict, strict=True):
        # Load normalization and estimator
        self.feat_means.copy_(state_dict["feat_means"])
        self.feat_stds.copy_(state_dict["feat_stds"])
        self.gb_params = state_dict.get("gb_params", None)
        self.gb_state = pickle.loads(state_dict["gb_state"])
        return

    def forward_df(self, X: pd.DataFrame, clip_low=None, clip_high=None):
        """
        Predict from pandas DataFrame using self.usecols.
        """
        X_full = X[self.usecols].to_numpy(dtype=np.float32)
        feats = torch.from_numpy(X_full)
        preds_t = self.forward(feats).squeeze(1)
        preds = preds_t.detach().cpu().numpy()
        if clip_low is not None or clip_high is not None:
            preds = np.clip(preds, a_min=clip_low, a_max=clip_high)
        return preds

    def eval_df(self, X: pd.DataFrame, y, clip_low=None, clip_high=None) -> dict:
        preds = self.forward_df(X, clip_low=clip_low, clip_high=clip_high)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        mse = float(np.mean((y - preds) ** 2))
        # Guard against zero std
        y_std = float(np.std(y)) if float(np.std(y)) > 0 else 1.0
        nrmse = float(np.sqrt(mse) / y_std)
        return nrmse
