import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

class RidgeModel(nn.Module):
    """
    Torch model for predictions on precomputed EEG features.
    Uses simple linear regression (ridge).
    """
    def __init__(self, usecols: list[str]):
        super().__init__()
        self.usecols = usecols
        self.register_buffer("coef_", torch.empty(len(usecols), dtype=torch.float32))
        self.register_buffer("bias_", torch.tensor(0.0, dtype=torch.float32))  # scalar
        self.register_buffer("feat_means", torch.empty(len(usecols), dtype=torch.float32))
        self.register_buffer("feat_stds", torch.empty(len(usecols), dtype=torch.float32))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        feats: (B, N) — features (may be a full vector)
        """
        feats = (feats - self.feat_means) / self.feat_stds
        feats = feats[:, :len(self.usecols)]
        pred = feats @ self.coef_ + self.bias_
        return pred.unsqueeze(1)

    @classmethod
    def make_checkpoint(cls,
                        X_train: pd.DataFrame,
                        y_train: pd.Series | np.ndarray,
                        usecols: list[str],
                        C: float = 1.0,
                        force_bias = None,
                        save_path: str = "ridge_checkpoint.pt"):
        """
        Train ridge on X_train[usecols], save only normalization stats for usecols.

        Args:
            X_train: DataFrame with all features.
            y_train: Series/array of targets.
            usecols: feature names to train on (order matters).
            C: ridge inverse regularization (alpha = 1/C).
            save_path: path to save the checkpoint.
        """
        # 1) Select subset in the specified order
        X_sub = X_train[usecols].to_numpy(dtype=np.float32)
        y_arr = np.asarray(y_train, dtype=np.float32).reshape(-1)
        assert X_sub.shape[0] == y_arr.shape[0], "X_train and y_train length mismatch"

        # 2) Compute normalization stats ONLY for usecols
        feat_means = X_sub.mean(axis=0)
        feat_stds  = X_sub.std(axis=0)
        feat_stds[feat_stds == 0] = 1.0
        print('Means: ', feat_means)
        print('stds:  ', feat_stds)

        # 3) Normalize and train ridge
        Xn = (X_sub - feat_means) / feat_stds
        ridge = Ridge(alpha=1.0 / C, fit_intercept=True)
        ridge.fit(Xn, y_arr)

        # 4) Build and save checkpoint (only usecols-sized tensors)
        bias = force_bias if force_bias is not None else ridge.intercept_
        state_dict = {
            "coef_": torch.tensor(ridge.coef_.ravel(), dtype=torch.float32),
            "bias_" : torch.tensor(float(bias), dtype=torch.float32),  # scalar
            "feat_means": torch.tensor(feat_means, dtype=torch.float32),
            "feat_stds" : torch.tensor(feat_stds, dtype=torch.float32),
        }
        meta = {"usecols": usecols}
        torch.save({"state_dict": state_dict, "meta": meta}, save_path)
        print(f"✅ Ridge checkpoint saved to {save_path}")

        # 5) Return a ready-to-use model bound to usecols
        model = cls(usecols)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model

    def save(self, save_path: str):
        """
        Save the current model's state dict and usecols to the given path.
        """
        state_dict = {
            "coef_": self.coef_.detach().cpu(),
            "bias_": self.bias_.detach().cpu(),
            "feat_means": self.feat_means.detach().cpu(),
            "feat_stds": self.feat_stds.detach().cpu(),
        }
        meta = {"usecols": self.usecols}
        torch.save({"state_dict": state_dict, "meta": meta}, save_path)
        print(f"✅ RidgeModel saved to {save_path}")

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
