import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

class RandomForestModel(nn.Module):
    """
    Wrapper for RandomForestRegressor trained on precomputed EEG features.
    Supports state_dict loading/saving via torch.
    """
    def __init__(self, usecols: list[str]):
        super().__init__()
        self.usecols = usecols
        self.register_buffer("feat_means", torch.empty(len(usecols), dtype=torch.float32))
        self.register_buffer("feat_stds", torch.empty(len(usecols), dtype=torch.float32))
        self.rf_params = None    # Used for saving hyperparams
        self.rf_state = None     # Contains actual scikit-learn RandomForestRegressor parameters

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        feats: (B, N) — features (as torch tensor)
        Will convert to numpy & go through sklearn RF.
        """
        feats_np = (
            (feats.detach().cpu().numpy() - self.feat_means.cpu().numpy())
            / self.feat_stds.cpu().numpy()
        )
        # If in torchscript, need onnx model for prod, but for "offline" just use sklearn object here.
        preds = self.rf_state.predict(feats_np)
        return torch.tensor(preds, dtype=torch.float32, device=feats.device).unsqueeze(1)

    @classmethod
    def make_checkpoint(cls,
                        X_train: pd.DataFrame,
                        y_train: pd.Series | np.ndarray,
                        usecols: list[str],
                        rf_kwargs: dict = None,
                        save_path: str = "random_forest_checkpoint.pt"):
        """
        Trains a RandomForestRegressor on (X_train[usecols], y_train), saves model,
        and normalization stats using torch.save.
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

        # Train random forest
        rf_kwargs = rf_kwargs or dict(n_estimators=100, n_jobs=-1, random_state=42)
        rf = RandomForestRegressor(**rf_kwargs)
        rf.fit(Xn, y_arr)
        print("✅ Trained RandomForestRegressor")

        # Save checkpoint: normalization and RF state
        state_dict = {
            "feat_means": torch.tensor(feat_means, dtype=torch.float32),
            "feat_stds" : torch.tensor(feat_stds, dtype=torch.float32),
            "rf_params": rf.get_params(),
            "rf_state": pickle.dumps(rf),    # Pickle the whole estimator with pickle, not joblib
        }
        meta = {"usecols": usecols}
        torch.save({"state_dict": state_dict, "meta": meta}, save_path)
        print(f"✅ RandomForest checkpoint saved to {save_path}")

        # Return initialized model
        model = cls(usecols)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model

    def load_state_dict(self, state_dict, strict=True):
        # Loads normalization stats
        self.feat_means.copy_(state_dict["feat_means"])
        self.feat_stds.copy_(state_dict["feat_stds"])
        self.rf_params = state_dict.get("rf_params", None)
        self.rf_state = pickle.loads(state_dict["rf_state"])
        return

    def forward_df(self, X: pd.DataFrame, clip_low=None, clip_high=None):
        """
        Predict from pandas dataframe (using self.usecols).
        """
        X_full = X[self.usecols].to_numpy(dtype=np.float32)
        feats = torch.from_numpy(X_full)
        preds_t = self.forward(feats).squeeze(1)
        preds = preds_t.detach().cpu().numpy()
        # Apply clipping
        if clip_low is not None or clip_high is not None:
            preds = np.clip(preds, a_min=clip_low, a_max=clip_high)
        return preds

    def eval_df(self, X: pd.DataFrame, y, clip_low=None, clip_high=None) -> dict:
        preds = self.forward_df(X, clip_low=clip_low, clip_high=clip_high)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        mse = float(np.mean((y - preds) ** 2))
        nrmse = float(np.sqrt(mse) / np.std(y))
        return nrmse
