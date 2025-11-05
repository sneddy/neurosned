import numpy as np
import torch
import torch.nn as nn
from collections import deque


class ExternalizingPredictorsBlend(nn.Module):
    """
    Wraps: feature_extractor (numpy/torch-агностичный) + list of RidgeModel.
    Forward принимает сырой EEG: (B, C, T), возвращает (B, 1).
    Supports ensemble of ridge models with optional weights.
    Optionally applies median sliding average smoothing over previous predictions.
    """
    def __init__(
        self, 
        feature_extractor, 
        ridge_models: list, 
        ridge_weights: list[float] | None = None,
        history_size: int = 0,
        history_weight: float = 0,
        clip_min: float | None = None, 
        clip_max: float | None = None,
    ):
        super().__init__()
        assert isinstance(ridge_models, list) and len(ridge_models) > 0, "ridge_models must be a non-empty list"
        self.fe = feature_extractor
        self.ridge_models = ridge_models

        self.history_size = history_size
        self.history_weight = history_weight
        self.history = deque(maxlen=history_size if history_size > 0 else None)

        # If weights aren't given, use uniform weights
        if ridge_weights is None:
            self.ridge_weights = [1.0/len(ridge_models)] * len(ridge_models)
        else:
            self.ridge_weights = ridge_weights
        # For reference, gather all feature keys needed (union of all used)
        self.all_feature_keys = sorted({k for model in ridge_models for k in model.usecols})
        self.clip_min = clip_min
        self.clip_max = clip_max

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        """
        x_raw: (B, C, T) raw EEG
        Applies ensemble prediction then (if enabled) median sliding average smoothing.
        If history_size > 0 and history_weight > 0, smoothes output as:
            final_pred = (1 - history_weight) * current_pred + history_weight * median(history + [current_pred])
        Otherwise, just returns current_pred.
        """
        B = x_raw.size(0)
        # Extract features per example
        batch_feats_all = []

        # Extract all needed features (union of all used)
        for b in range(B):
            eeg_np = x_raw[b].detach().cpu().numpy() 
            feats_dict, _ = self.fe.extract_features(eeg_np)
            batch_feats_all.append(feats_dict)

        all_preds = []

        for model_idx, ridge in enumerate(self.ridge_models):
            try:
                ridge_device = next(ridge.buffers()).device
            except StopIteration:
                ridge_device = x_raw.device

            batch_feats = []
            for b in range(B):
                vec = np.array([batch_feats_all[b][k] for k in ridge.usecols], dtype=np.float32)
                batch_feats.append(vec)
            F = torch.from_numpy(np.stack(batch_feats, axis=0)).to(device=ridge_device, dtype=torch.float32) 
            preds = ridge(F) # B
            all_preds.append(preds * self.ridge_weights[model_idx])

        # Weighted sum of predictions
        preds = torch.stack(all_preds, dim=0).sum(dim=0)  # shape: (B,)

        # Optionally smooth predictions with median sliding average
        if self.history_size > 0 and self.history_weight > 0:
            preds_list = preds.detach().cpu().numpy()
            smoothed_preds = []
            for pred_item in preds_list:
                if len(self.history) > 0:
                    sliding_median = float(np.median(self.history))
                else:
                    sliding_median = float(pred_item)
                smoothed_pred_item = float(sliding_median) * self.history_weight + float(pred_item) * (1 - self.history_weight)
                smoothed_preds.append(smoothed_pred_item)
                self.history.append(float(pred_item))
            preds = torch.tensor(smoothed_preds, dtype=preds.dtype, device=preds.device).unsqueeze(1)
        if self.clip_min is not None:
            preds = torch.clamp_min(preds, self.clip_min)
        if self.clip_max is not None:
            preds = torch.clamp_max(preds, self.clip_max)
        return preds