import numpy as np
import torch
import torch.nn as nn

class MetaWrapper(nn.Module):
    """Torch wrapper: runs base models, extracts numpy features, applies sklearn meta-regressor."""
    def __init__(self, seg_models: list[nn.Module] | None,
                 cls_models: list[nn.Module] | None,
                 feature_extractor,
                 meta_regressor,
                 use_channels=None,
                 device="cuda"):
        super().__init__()
        self.seg_models = nn.ModuleList(seg_models or [])
        self.cls_models = nn.ModuleList(cls_models or [])
        for m in self.seg_models: m.eval()
        for m in self.cls_models: m.eval()
        self.fx = feature_extractor
        self.reg = meta_regressor
        self.use_channels = use_channels
        self.device = device

    @torch.no_grad()
    def _infer_seg_logits(self, x: torch.Tensor):
        Zs = []
        for m in self.seg_models:
            z = m(x).squeeze(1)              # (B,T)
            Zs.append(z.detach().cpu().numpy())
        return Zs

    @torch.no_grad()
    def _infer_cls_outputs(self, x: torch.Tensor):
        Ys = []
        for m in self.cls_models:
            y = m(x)                         # (B,K) or (B,1)
            if isinstance(y, (tuple, list)):
                y = y[0]
            y = y.squeeze(-1) if y.ndim == 3 else y
            Ys.append(y.detach().cpu().numpy())
        return Ys

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        if self.use_channels is not None:
            x = x[:, self.use_channels, :]
        x = x.to(self.device).float()
        seg_logits = self._infer_seg_logits(x)
        cls_outputs = self._infer_cls_outputs(x) if len(self.cls_models) > 0 else None
        X = self.fx.build_from_batch(seg_logits, cls_outputs)
        y_hat = self.reg.predict(X).astype(np.float32)             # (B,)
        return torch.from_numpy(y_hat).to(x.device).unsqueeze(1)   # (B,1)
