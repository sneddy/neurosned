import torch
from typing import Optional


def reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    pred/target: (B, P, C, T)
    mask: optional bool/float mask over patches (B, P). If provided, loss is computed only on mask==1.
    reduction: 'mean' or 'sum'.
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {tuple(pred.shape)} vs target {tuple(target.shape)}.")

    loss = (pred - target) ** 2  # (B, P, C, T)

    if mask is not None:
        if mask.shape != pred.shape[:2]:
            raise ValueError(f"Mask shape should be (B, P), got {tuple(mask.shape)}.")
        mask_f = mask.to(dtype=loss.dtype, device=loss.device).unsqueeze(-1).unsqueeze(-1)  # (B,P,1,1)
        loss = loss * mask_f
        denom = mask_f.sum()
    else:
        denom = torch.tensor(loss.numel(), device=loss.device, dtype=loss.dtype)

    if reduction == "mean":
        denom_safe = denom.clamp_min(1.0)
        return loss.sum() / denom_safe
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"Unsupported reduction: {reduction}")