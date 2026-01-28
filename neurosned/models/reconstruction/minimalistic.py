import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sinusoidal_pos_enc(seq_len: int, dim: int, device=None, dtype=None) -> torch.Tensor:
    """Standard sine/cos positional encoding (seq_len, dim)."""
    position = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)  # (L,1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / dim))
    pe = torch.zeros(seq_len, dim, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def make_block_mask(num_patches: int, mask_ratio: float = 0.7, block_size: int = 4, device=None) -> torch.Tensor:
    """
    Build a contiguous block mask over patches.
    mask_ratio: fraction of patches to mask.
    block_size: approximate length of each contiguous block (in patches).
    Returns: bool mask of shape (num_patches,) where True = masked.
    """
    num_mask = max(1, int(round(num_patches * mask_ratio)))
    mask = torch.zeros(num_patches, dtype=torch.bool, device=device)
    remaining = num_mask
    rng = torch.randint(0, 2**31 - 1, (1,), device=device).item()
    gen = torch.Generator(device=device)
    gen.manual_seed(rng)

    while remaining > 0:
        start = int(torch.randint(0, num_patches, (1,), generator=gen, device=device))
        span = min(block_size, remaining, num_patches - start)
        mask[start:start + span] = True
        remaining -= span

    # If we overshoot due to overlap, trim to exact count
    if mask.sum() > num_mask:
        idx = mask.nonzero(as_tuple=False).squeeze(1)
        drop = idx[: int(mask.sum().item() - num_mask)]
        mask[drop] = False
    return mask


class MinimalistReconstructor(nn.Module):
    """
    Lightweight patch-level reconstructor for EEG.

    Input:  patches shaped (B, P, C, T_patch)
    Output: dict with reconstructed patches and embeddings.
    """

    def __init__(
        self,
        n_channels: int,
        patch_size: int = 20,
        embed_dim: int = 128,
        depth: int = 2,
        n_heads: int = 4,
        ff_mult: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_channels = int(n_channels)
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)

        patch_dim = self.n_channels * self.patch_size

        self.patch_proj = nn.Linear(patch_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_mult * embed_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.recon_proj = nn.Linear(embed_dim, patch_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(
        self,
        patches: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        """
        patches: (B, P, C, T_patch)
        mask: optional bool mask over patches (B, P), True = masked (reconstruct only these, hide content).
        attn_mask: optional padding mask for transformer (bool, shape (B, P), True=pad).
        """
        if patches.ndim != 4:
            raise ValueError(f"Expected patches shaped (B, P, C, T), got {tuple(patches.shape)}.")

        B, P, C, T = patches.shape
        if C != self.n_channels or T != self.patch_size:
            raise ValueError(f"Expected channels={self.n_channels}, patch_size={self.patch_size}, got {(C, T)}.")

        x = patches.reshape(B, P, -1)              # (B, P, C*T)
        x = self.patch_proj(x)                     # (B, P, D)
        pe = _sinusoidal_pos_enc(P, self.embed_dim, device=x.device, dtype=x.dtype).unsqueeze(0)  # (1,P,D)
        pe = pe.expand(B, -1, -1)

        if mask is not None:
            mask = mask.to(dtype=torch.bool, device=x.device)
            mask_f = mask.unsqueeze(-1).type_as(x)
            # replace masked tokens with mask_token (no EEG content)
            x = x * (1.0 - mask_f) + (self.mask_token.expand(B, P, -1)) * mask_f

        x = x + pe                                  # add position

        z = self.encoder(x, src_key_padding_mask=attn_mask)  # (B, P, D)
        recon = self.recon_proj(z).view(B, P, C, T)          # (B, P, C, T)
        z_global = z.mean(dim=1)                             # (B, D)

        if return_dict:
            return {"recon": recon, "z_tokens": z, "z_global": z_global}
        return recon


