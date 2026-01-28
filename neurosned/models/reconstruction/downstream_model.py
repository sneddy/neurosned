import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftArgmaxRTHead(nn.Module):
    """
    z_tokens: (B, P, D)
    starts_sec: (B, P)  (время начала каждого патча)
    returns: (B,) predicted RT in seconds (или в тех же единицах, что starts_sec)
    """
    def __init__(self, dim: int, temperature: float = 1.0, learn_affine: bool = True):
        super().__init__()
        self.score = nn.Linear(dim, 1, bias=False)
        self.temperature = temperature
        self.learn_affine = learn_affine
        if learn_affine:
            self.alpha = nn.Parameter(torch.tensor(1.0))
            self.beta  = nn.Parameter(torch.tensor(0.0))

    def forward(self, z_tokens, starts_sec, attn_mask=None):
        logits = self.score(z_tokens).squeeze(-1)  # (B,P)
        if attn_mask is not None:
            logits = logits.masked_fill(attn_mask, float("-inf"))

        p = F.softmax(logits / self.temperature, dim=1)  # (B,P)
        t_hat = (p * starts_sec).sum(dim=1)              # (B,)

        if self.learn_affine:
            t_hat = self.alpha * t_hat + self.beta
        return t_hat
