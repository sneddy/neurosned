# attention_sneddy_unet.py - point event detection as 1D segmentation
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union

# ------------------------- utils -------------------------

class DropPath(nn.Module):
    """Stochastic depth per-sample."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep) / keep
        return x * mask


class StdPerSample(nn.Module):
    """Пер-семпловая нормализация по времени."""
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
    def forward(self, x):  # x:(B,C,T)
        mu = x.mean(dim=-1, keepdim=True)
        sd = x.std(dim=-1, keepdim=True).clamp_min(self.eps)
        return (x - mu) / sd


class DSConv1d(nn.Module):
    """Depthwise separable 1D conv."""
    def __init__(self, ch, k=7, stride=1, dilation=1, bias=False):
        super().__init__()
        pad = ((k - 1) // 2) * dilation
        self.dw = nn.Conv1d(ch, ch, k, stride=stride, padding=pad,
                            dilation=dilation, groups=ch, bias=bias)
        self.pw = nn.Conv1d(ch, ch, 1, bias=bias)
    def forward(self, x):
        return self.pw(self.dw(x))


class ResBlock(nn.Module):
    """Лёгкий residual-блок: DSConv → GN → GELU → Dropout → DSConv → GN + DropPath + skip."""
    def __init__(self, ch, k=7, dropout=0.0, dilation=1, drop_path: float = 0.0):
        super().__init__()
        self.conv1 = DSConv1d(ch, k=k, dilation=dilation)
        self.gn1   = nn.GroupNorm(1, ch)
        self.conv2 = DSConv1d(ch, k=k, dilation=1)
        self.gn2   = nn.GroupNorm(1, ch)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        self.drop_path = DropPath(drop_path)
    def forward(self, x):
        r = x
        x = self.conv1(x)
        x = self.act(self.gn1(x))
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.gn2(x)
        x = r + self.drop_path(x)
        x = self.act(x)
        return x


class ChannelSqueeze(nn.Module):
    """Сужение по электродам: 1x1 C_in→C_out."""
    def __init__(self, c_in, c_out):
        super().__init__()
        self.proj = nn.Conv1d(c_in, c_out, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity="linear")
    def forward(self, x):
        return self.proj(x)


class TimeDown(nn.Module):
    """Антиалиас depthwise + AvgPool(stride=2)."""
    def __init__(self, ch, k=5):
        super().__init__()
        pad = (k - 1) // 2
        self.aa = nn.Conv1d(ch, ch, k, groups=ch, padding=pad, bias=False)
        with torch.no_grad():
            w = torch.tensor([1, 2, 3, 2, 1], dtype=torch.float32)
            w = (w / w.sum()).view(1, 1, -1).repeat(ch, 1, 1)
            self.aa.weight.copy_(w)
        for p in self.aa.parameters():
            p.requires_grad_(False)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
    def forward(self, x):
        return self.pool(self.aa(x))

# ------------------------- attention blocks -------------------------

class MHSABlock(nn.Module):
    """
    MHSA по времени на бутылочном уровне + позиционная depthwise-конволька.
    Формат: (B, C, T) → (B, C, T)
    """
    def __init__(
        self,
        ch: int,
        n_heads: int = 4,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        ff_mult: float = 2.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.pos = nn.Conv1d(ch, ch, kernel_size=3, padding=1, groups=ch, bias=True)
        self.ln1 = nn.LayerNorm(ch)
        self.attn = nn.MultiheadAttention(embed_dim=ch, num_heads=n_heads,
                                          dropout=attn_dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(ch)
        hidden = int(ch * ff_mult)
        self.ff = nn.Sequential(
            nn.Linear(ch, hidden),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(hidden, ch),
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x):  # x: (B,C,T)
        x = x + self.pos(x)                    # positional bias (local)
        xt = x.transpose(1, 2)                 # (B,T,C)
        h, _ = self.attn(self.ln1(xt), self.ln1(xt), self.ln1(xt), need_weights=False)
        xt = xt + self.drop_path(h)
        h = self.ff(self.ln2(xt))
        xt = xt + self.drop_path(h)
        return xt.transpose(1, 2)              # (B,C,T)

# ------------------------- encoder / decoder -------------------------

class Encoder1D(nn.Module):
    """Sneddy-style encoder с параметризуемым числом стадий."""
    def __init__(
        self,
        num_stages: int = 3,
        c0: int = 32,
        widen: float = 2.0,
        depth_per_stage: Union[int, List[int]] = 2,
        k: int = 7,
        dropout: float = 0.1,
        drop_path: float = 0.0,
    ):
        super().__init__()
        assert 3 <= num_stages <= 5, "num_stages должен быть в диапазоне [3,5]"
        # каналы геометрической прогрессией, совместимо с 32,64,128 при widen=2
        self.chs = [int(round(c0 * (widen ** i))) for i in range(num_stages)]
        if isinstance(depth_per_stage, int):
            dps = [depth_per_stage] * num_stages
        else:
            assert len(depth_per_stage) == num_stages
            dps = list(depth_per_stage)

        enc, downs = [], []
        in_c = self.chs[0]
        # первый уровень предполагает вход уже сжатый до c0 извне (ChannelSqueeze)
        for stage_idx, out_c in enumerate(self.chs):
            stage = []
            if stage_idx == 0:
                # in_c == out_c == c0, без проекции
                pass
            else:
                stage += [nn.Conv1d(in_c, out_c, 1, bias=False), nn.GroupNorm(1, out_c), nn.GELU()]
            for _ in range(dps[stage_idx]):
                stage.append(ResBlock(out_c, k=k, dropout=dropout, dilation=1, drop_path=drop_path))
            enc.append(nn.Sequential(*stage))
            downs.append(TimeDown(out_c))
            in_c = out_c

        self.encoder_blocks = nn.ModuleList(enc)
        self.downs = nn.ModuleList(downs)

    def forward(self, x):
        skips = []
        h = x
        for enc, down in zip(self.encoder_blocks, self.downs):
            h = enc(h)
            skips.append(h)
            h = down(h)
        return h, skips  # h на самом низком масштабе, skips от мелкого к глубокому (r0..r{S-1})


class UpBlock(nn.Module):
    """Linear upsample + (optional gated) skip + fuse + DS refinement."""
    def __init__(self, in_ch, skip_ch, out_ch, k=7, dropout=0.0, drop_path: float = 0.0, skip_gating: bool = False):
        super().__init__()
        self.skip_gating = skip_gating
        if skip_gating:
            self.skip_gate = nn.Conv1d(skip_ch, skip_ch, kernel_size=1, bias=True)
        self.fuse = nn.Conv1d(in_ch + skip_ch, out_ch, kernel_size=1, bias=False)
        self.gn   = nn.GroupNorm(1, out_ch)
        self.act  = nn.GELU()
        self.refine = ResBlock(out_ch, k=k, dropout=dropout, dilation=1, drop_path=drop_path)

    def forward(self, x_low, x_skip):
        x = F.interpolate(x_low, size=x_skip.shape[-1], mode='linear', align_corners=False)
        if self.skip_gating:
            gate = torch.sigmoid(self.skip_gate(x_skip))
            x_skip = x_skip * gate
        x = torch.cat([x, x_skip], dim=1)
        x = self.act(self.gn(self.fuse(x)))
        x = self.refine(x)
        return x


class Decoder1D(nn.Module):
    """U-Net-like decoder с произвольным числом уровней."""
    def __init__(
        self,
        chs: List[int],
        k: int = 7,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        skip_gating: bool = False,
    ):
        super().__init__()
        # chs: [c0, c1, ..., cS-1]
        steps = []
        # идём сверху вниз: r{S-1}->r{S-2}->...->r0
        for i in range(len(chs) - 1, -1, -1):
            in_ch   = chs[i] if i == len(chs) - 1 else steps[-1][2]  # предыдущий out_ch
            skip_ch = chs[i]
            out_ch  = chs[i - 1] if i - 1 >= 0 else chs[0]
            steps.append((in_ch, skip_ch, out_ch))
        # первый шаг в списке соответствует глубочайшему апсемплу
        self.upblocks = nn.ModuleList([
            UpBlock(in_ch, skip_ch, out_ch, k=k, dropout=dropout, drop_path=drop_path, skip_gating=skip_gating)
            for (in_ch, skip_ch, out_ch) in steps
        ])
        self.out_ch = chs[0]

    def forward(self, h, skips):
        # skips: [r0, r1, ..., r{S-1}]
        for up, skip in zip(self.upblocks, reversed(skips)):
            h = up(h, skip)
        return h  # r0

# ------------------------- bottlenecks -------------------------

class DilatedBottleneck(nn.Module):
    """Глубокий бутылочный блок на дилатациях: растёт поле зрения."""
    def __init__(self, ch: int, depth: int = 2, k: int = 7, dropout: float = 0.1, drop_path: float = 0.0):
        super().__init__()
        # экспоненциальная лестница дилатаций: 1,2,4,8,...
        blocks = []
        for i in range(depth):
            dil = 2 ** i
            blocks.append(ResBlock(ch, k=k, dropout=dropout, dilation=dil, drop_path=drop_path))
        self.net = nn.Sequential(*blocks)
    def forward(self, x):
        return self.net(x)


class AttentionBottleneck(nn.Module):
    """Бутылочный блок на MHSA по времени (глобальный контекст)."""
    def __init__(
        self,
        ch: int,
        depth: int = 2,
        attn_heads: int = 4,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        ff_mult: float = 2.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            MHSABlock(ch, n_heads=attn_heads, attn_dropout=attn_dropout,
                      ffn_dropout=ffn_dropout, ff_mult=ff_mult, drop_path=drop_path)
            for _ in range(depth)
        ])
    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


class HybridBottleneck(nn.Module):
    """Чередование MHSA и дилатаций."""
    def __init__(
        self,
        ch: int,
        depth: int = 4,
        k: int = 7,
        dropout: float = 0.1,
        attn_heads: int = 4,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        ff_mult: float = 2.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        blocks = []
        for i in range(depth):
            if i % 2 == 0:
                blocks.append(MHSABlock(ch, n_heads=attn_heads, attn_dropout=attn_dropout,
                                        ffn_dropout=ffn_dropout, ff_mult=ff_mult, drop_path=drop_path))
            else:
                dil = 2 ** (i // 2)  # 1,2,4...
                blocks.append(ResBlock(ch, k=k, dropout=dropout, dilation=dil, drop_path=drop_path))
        self.blocks = nn.Sequential(*blocks)
    def forward(self, x):
        return self.blocks(x)

# ------------------------- model -------------------------

class AttentionSneddyUnet(nn.Module):
    """
    Encoder/Decoder 1D сегментация с расширяемой глубиной и настраиваемым bottleneck.
    Совместим по интерфейсу с исходной версией.
    """
    def __init__(
        self,
        n_chans: int,
        n_times: int,
        sfreq: int,
        c0: int = 32,
        widen: float = 2.0,
        depth_per_stage: Union[int, List[int]] = 2,
        dropout: float = 0.1,
        k: int = 7,
        out_channels: int = 1,
        # новинки:
        num_stages: int = 3,                 # 3..5
        bottleneck_type: str = "dilated",    # "dilated" | "mhsa" | "hybrid"
        bottleneck_depth: int = 2,
        attn_heads: int = 4,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        ff_mult: float = 2.0,
        drop_path: float = 0.0,
        skip_gating: bool = False,
        use_norm: bool = True,
    ):
        super().__init__()
        self.n_chans = n_chans
        self.n_times = n_times
        self.sfreq   = sfreq
        self.out_channels = out_channels
        self.use_norm = use_norm

        self.norm = StdPerSample()
        self.c_squeeze = ChannelSqueeze(n_chans, c0)

        # Encoder
        self.encoder = Encoder1D(
            num_stages=num_stages, c0=c0, widen=widen,
            depth_per_stage=depth_per_stage, k=k, dropout=dropout, drop_path=drop_path
        )

        # Bottleneck (каналы == последний уровень энкодера)
        bottleneck_ch = self.encoder.chs[-1]
        if bottleneck_type == "dilated":
            self.bottleneck = DilatedBottleneck(bottleneck_ch, depth=bottleneck_depth, k=k,
                                                dropout=dropout, drop_path=drop_path)
        elif bottleneck_type == "mhsa":
            self.bottleneck = AttentionBottleneck(bottleneck_ch, depth=bottleneck_depth,
                                                  attn_heads=attn_heads, attn_dropout=attn_dropout,
                                                  ffn_dropout=ffn_dropout, ff_mult=ff_mult, drop_path=drop_path)
        elif bottleneck_type == "hybrid":
            self.bottleneck = HybridBottleneck(bottleneck_ch, depth=bottleneck_depth, k=k, dropout=dropout,
                                               attn_heads=attn_heads, attn_dropout=attn_dropout,
                                               ffn_dropout=ffn_dropout, ff_mult=ff_mult, drop_path=drop_path)
        else:
            raise ValueError("bottleneck_type must be 'dilated' | 'mhsa' | 'hybrid'.")

        # Decoder
        self.decoder = Decoder1D(self.encoder.chs, k=k, dropout=dropout, drop_path=drop_path, skip_gating=skip_gating)

        # Head
        self.head = nn.Conv1d(self.decoder.out_ch, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        B, C, T = x.shape
        if self.use_norm:
            x = self.norm(x)
        x = self.c_squeeze(x)
        h_low, skips = self.encoder(x)
        h = self.bottleneck(h_low)
        h = self.decoder(h, skips)
        if h.shape[-1] != T:
            h = F.interpolate(h, size=T, mode='linear', align_corners=False)
        logits = self.head(h)  # (B, out_channels, T)
        return logits

    @torch.no_grad()
    def predict(
        self,
        x,
        mode: str = "argmax",          # "argmax" | "softargmax"
        temperature: float = 1.0,      # used for softargmax
        window_sec: float = 2.0,       # [0.5..2.5]s длина окна
        return_var: bool = False
    ):
        """
        Returns:
          - t_hat_sec: (B,) time in seconds relative to window start
          - var_sec (optional): (B,) predictive variance
        """
        logits = self.forward(x)
        if self.out_channels != 1:
            raise ValueError("predict() assumes out_channels==1 for time readout.")
        B, _, T = logits.shape
        dt = window_sec / T
        z = logits.squeeze(1)  # (B, T)

        if mode == "argmax":
            idx = torch.argmax(z, dim=-1)                       # (B,)
            t_hat = idx.to(z.dtype) * dt
            if not return_var:
                return t_hat
            var = torch.full_like(t_hat, fill_value=(dt**2))
            return t_hat, var

        elif mode == "softargmax":
            p = F.softmax(z / temperature, dim=-1)              # (B, T)
            grid = torch.arange(T, device=z.device, dtype=z.dtype)[None, :]
            t_idx = (p * grid).sum(dim=-1)                      # (B,)
            t_hat = t_idx * dt
            if not return_var:
                return t_hat
            var = (p * ((grid * dt - t_hat[:, None])**2)).sum(dim=-1)
            return t_hat, var

        else:
            raise ValueError("mode must be 'argmax' or 'softargmax'.")

    @torch.no_grad()
    def predict_mask(self, x, temperature: float = 1.0):
        """Пер-временные вероятности (B, T) из логитов (softmax по времени)."""
        logits = self.forward(x)
        if self.out_channels != 1:
            raise ValueError("predict_mask() assumes out_channels==1.")
        z = logits.squeeze(1)
        return F.softmax(z / temperature, dim=-1)
