import torch
import torch.nn as nn
import torch.nn.functional as F

class SubmitWrapper(nn.Module):
    def __init__(
        self, segmentation_models=None, regression_models=None, 
        seg_weights=None, abs_seg_weights=None, reg_weights=None, use_channels=None,
        device='cuda', temperature:float=1.,
        # per_model_taus=None, # new
    ):
        super().__init__()
        self.segmentation_models = segmentation_models
        self.regression_models = regression_models
        self.seg_weights = seg_weights if seg_weights is not None else []
        self.abs_seg_weights = abs_seg_weights if abs_seg_weights is not None else []
        self.reg_weights = reg_weights if reg_weights is not None else []
        self.use_channels = use_channels
        self.device = device
        self.temperature = temperature
        # self.per_model_taus = per_model_taus 

        self.dt = 1 / 100
        self.win_offset = 0.5

        self.reg_weights_sum = sum(self.reg_weights) if self.reg_weights else 0
        self.seg_weights_sum = sum(self.seg_weights) if self.seg_weights else 0
        self.abs_seg_weights_sum = sum(self.abs_seg_weights) if self.abs_seg_weights else 0
        if abs(self.reg_weights_sum + self.seg_weights_sum + self.abs_seg_weights_sum - 1.0) > 1e-6:
            raise ValueError('Weight not sumarazing to 1')
        if self.segmentation_models is not None and len(seg_weights) != len(self.segmentation_models):
            raise ValueError("Missing weights or models for segmentation")
        if self.regression_models is not None and len(self.reg_weights) != len(self.regression_models):
            raise ValueError("Missing weights or models for regression")
        # if self.per_model_taus is not None and len(self.per_model_taus) != len(self.segmentation_models):
            # raise ValueError("per_model_taus length must match segmentation_models length")


    def _get_soft_argmax(self, seg_logits: torch.tensor):
        B, T = seg_logits.shape
        t_grid = (torch.arange(T, device=self.device, dtype=seg_logits.dtype) * self.dt)[None, :]
        p = torch.softmax(seg_logits / self.temperature, dim=-1)
        t_hat_rel = (p * t_grid).sum(dim=-1)          
        return t_hat_rel

    def forward(self, x: torch.tensor):
        B, C, T = x.shape
        if self.use_channels is not None:
            x = x[:, self.use_channels, :]

        t_hat_abs = 0
        regression_prediction = 0
        segmentation_prediction = 0
        reg_logits_by_model = {}

        if self.segmentation_models is not None and len(self.segmentation_models) > 0:
            weighted_logits = torch.zeros((B,T), device=self.device, dtype=x.dtype)
            seg_logits_by_model = {}
            segmentation_prediction = torch.zeros(B, device=self.device, dtype=x.dtype)
            for idx, weight in enumerate(self.seg_weights):
                seg_logits_by_model[idx] = self.segmentation_models[idx](x).squeeze(1) # (B,T)
                # current_tau = self.per_model_taus[idx] if self.per_model_taus is not None else 1
                # weighted_logits += (weight / self.seg_weights_sum) * (seg_logits_by_model[idx] / current_tau) 
                weighted_logits += (weight / self.seg_weights_sum) * (seg_logits_by_model[idx]) 
            t_hat_rel = self._get_soft_argmax(weighted_logits) # (B,)
            t_hat_abs = t_hat_rel + self.win_offset            # (B,)

            for idx, weight in enumerate(self.abs_seg_weights):
                reg_logits_by_model[idx] = self._get_soft_argmax(seg_logits_by_model[idx])
                segmentation_prediction += (reg_logits_by_model[idx] + self.win_offset) * weight

        if self.regression_models is not None and len(self.regression_models) > 0:
            regression_prediction = torch.zeros(B, device=self.device, dtype=x.dtype)
            for idx, weight in enumerate(self.reg_weights):
                regression_prediction += self.regression_models[idx](x).view(-1) * weight

        return (t_hat_abs * self.seg_weights_sum + segmentation_prediction + regression_prediction).unsqueeze(1)

