import numpy as np
import torch
import torch.nn as nn


class ExternalizingPredictor(nn.Module):
    """
    Wraps: feature_extractor (numpy/torch-agnostic) + RidgeModel.
    The forward method accepts raw EEG input with shape (B, C, T) and returns predictions with shape (B, 1).
    """
    def __init__(self, feature_extractor, ridge_model,
        clip_min: float | None = None, clip_max: float | None = None):
        super().__init__()
        self.fe = feature_extractor    
        self.ridge = ridge_model
        self.all_feature_keys = ridge_model.usecols
        self.clip_min = clip_min
        self.clip_max = clip_max

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        """
        x_raw: (B, C, T) raw EEG
        """
        B = x_raw.size(0)
        batch_feats = []

        for b in range(B):
            eeg_np = x_raw[b].detach().cpu().numpy() 
            feats_dict, _ = self.fe.extract_features(eeg_np)  
            vec = np.array([feats_dict[k] for k in self.ridge.usecols], dtype=np.float32)
            batch_feats.append(vec)
        try:
            ridge_device = next(self.ridge.buffers()).device
        except StopIteration:
            ridge_device = x_raw.device

        F = torch.from_numpy(np.stack(batch_feats, axis=0)).to(device=ridge_device, dtype=torch.float32)
        preds = self.ridge(F)
        if self.clip_min is not None:
            preds = torch.clamp_min(preds, self.clip_min)
        if self.clip_max is not None:
            preds = torch.clamp_max(preds, self.clip_max)
        return preds