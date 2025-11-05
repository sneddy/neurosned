# recurrent_sneddy_unet.py — recurrent 1D "segmentation"
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union


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
    """Per-sample normalization over time: (B,C,T) -> (B,C,T)."""
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
    def forward(self, x):
        mu = x.mean(dim=-1, keepdim=True)
        sd = x.std(dim=-1, keepdim=True).clamp_min(self.eps)
        return (x - mu) / sd

class DSConv1d(nn.Module):
    """Depthwise separable 1D convolution."""
    def __init__(self, ch, k=7, stride=1, dilation=1, bias=False):
        super().__init__()
        pad = ((k - 1) // 2) * dilation
        self.dw = nn.Conv1d(ch, ch, k, stride=stride, padding=pad, dilation=dilation, groups=ch, bias=bias)
        self.pw = nn.Conv1d(ch, ch, 1, bias=bias)
    def forward(self, x):
        return self.pw(self.dw(x))

class ResBlock(nn.Module):
    """Lightweight residual block on DSConv (for local channel mixing)."""
    def __init__(self, ch, k=7, dropout=0.0, dilation=1, drop_path: float = 0.0):
        super().__init__()
        self.conv1 = DSConv1d(ch, k=k, dilation=dilation)
        self.gn1 = nn.GroupNorm(1, ch)
        self.conv2 = DSConv1d(ch, k=k, dilation=1)
        self.gn2 = nn.GroupNorm(1, ch)
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
    """Channel squeeze: 1x1, C_in→C_out."""
    def __init__(self, c_in, c_out):
        super().__init__()
        self.proj = nn.Conv1d(c_in, c_out, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity="linear")
    def forward(self, x):
        return self.proj(x)

class TimeDown(nn.Module):
    """Antialias depthwise + AvgPool(stride=2)."""
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

# ------------------------- recurrent blocks -------------------------
class RNNBlock1D(nn.Module):
    """
    Recurrent block that preserves channel size:
    (B,C,T) -> LN -> {GRU/LSTM} over time -> Linear->C + DropPath + skip -> GELU.
    Optionally adds depthwise 'positional' conv before RNN.
    """
    def __init__(
        self,
        ch: int,
        rnn_type: str = "gru",       
        bidirectional: bool = True,
        hidden_mult: float = 1.0,      
        num_layers: int = 1,
        rnn_dropout: float = 0.0,      
        use_dwpos: bool = True,
        k: int = 3,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.use_dwpos = use_dwpos
        if use_dwpos:
            self.pos = nn.Conv1d(ch, ch, kernel_size=k, padding=k // 2, groups=ch, bias=True)

        self.ln = nn.LayerNorm(ch)
        hidden_per_dir = max(1, int(round(ch * hidden_mult / (2 if bidirectional else 1))))
        rnn_cls = {"gru": nn.GRU, "lstm": nn.LSTM}[rnn_type.lower()]
        self.rnn = rnn_cls(
            input_size=ch,
            hidden_size=hidden_per_dir,
            num_layers=num_layers,
            dropout=rnn_dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        out_dim = hidden_per_dir * (2 if bidirectional else 1)
        self.proj = nn.Linear(out_dim, ch)
        self.drop_path = DropPath(drop_path)
        self.act = nn.GELU()

    def forward(self, x):
        # x: (B,C,T)
        if self.use_dwpos:
            x = x + self.pos(x)
        xt = x.transpose(1, 2)          # (B,T,C)
        h, _ = self.rnn(self.ln(xt))    # (B,T,H*)
        xt = xt + self.drop_path(self.proj(h))
        xt = self.act(xt)
        return xt.transpose(1, 2)       # (B,C,T)

# ------------------------- encoder / decoder -------------------------
class RNNEncoder1D(nn.Module):
    """Sneddy-style encoder, but each level is a stack of recurrent blocks."""
    def __init__(
        self,
        num_stages: int = 3,
        c0: int = 32,
        widen: float = 2.0,
        depth_per_stage: Union[int, List[int]] = 2,
        k: int = 7,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        rnn_type: str = "gru",
        bidirectional: bool = True,
        rnn_layers_per_block: int = 1,
        rnn_hidden_mult: float = 1.0,
        rnn_dropout: float = 0.0,
        use_dwpos: bool = True,
    ):
        super().__init__()
        assert 3 <= num_stages <= 5, "num_stages must be in [3,5]"
        self.chs = [int(round(c0 * (widen ** i))) for i in range(num_stages)]

        if isinstance(depth_per_stage, int):
            dps = [depth_per_stage] * num_stages
        else:
            assert len(depth_per_stage) == num_stages
            dps = list(depth_per_stage)

        enc, downs = [], []
        in_c = self.chs[0]
        for stage_idx, out_c in enumerate(self.chs):
            blocks = []
            if stage_idx > 0:
                blocks += [nn.Conv1d(in_c, out_c, 1, bias=False), nn.GroupNorm(1, out_c), nn.GELU()]
            # stack of recurrent blocks
            for _ in range(dps[stage_idx]):
                blocks.append(
                    RNNBlock1D(
                        out_c,
                        rnn_type=rnn_type,
                        bidirectional=bidirectional,
                        hidden_mult=rnn_hidden_mult,
                        num_layers=rnn_layers_per_block,
                        rnn_dropout=rnn_dropout,
                        use_dwpos=use_dwpos,
                        k=3,
                        drop_path=drop_path,
                    )
                )
                blocks.append(ResBlock(out_c, k=k, dropout=dropout, dilation=1, drop_path=drop_path))
            enc.append(nn.Sequential(*blocks))
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
        return h, skips  

class RecurrentUpBlock(nn.Module):
    """Linear upsample + (optional gated) skip + fuse + RNN-refine."""
    def __init__(
        self,
        in_ch,
        skip_ch,
        out_ch,
        k=7,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        skip_gating: bool = False,
        rnn_type: str = "gru",
        bidirectional: bool = True,
        rnn_layers_per_block: int = 1,
        rnn_hidden_mult: float = 1.0,
        rnn_dropout: float = 0.0,
        use_dwpos: bool = True,
    ):
        super().__init__()
        self.skip_gating = skip_gating
        if skip_gating:
            self.skip_gate = nn.Conv1d(skip_ch, skip_ch, kernel_size=1, bias=True)
        self.fuse = nn.Conv1d(in_ch + skip_ch, out_ch, kernel_size=1, bias=False)
        self.gn = nn.GroupNorm(1, out_ch)
        self.act = nn.GELU()
        self.refine_rnn = RNNBlock1D(
            out_ch,
            rnn_type=rnn_type,
            bidirectional=bidirectional,
            hidden_mult=rnn_hidden_mult,
            num_layers=rnn_layers_per_block,
            rnn_dropout=rnn_dropout,
            use_dwpos=use_dwpos,
            k=3,
            drop_path=drop_path,
        )
        self.refine_local = ResBlock(out_ch, k=k, dropout=dropout, dilation=1, drop_path=drop_path)

    def forward(self, x_low, x_skip):
        x = F.interpolate(x_low, size=x_skip.shape[-1], mode="linear", align_corners=False)
        if self.skip_gating:
            gate = torch.sigmoid(self.skip_gate(x_skip))
            x_skip = x_skip * gate
        x = torch.cat([x, x_skip], dim=1)
        x = self.act(self.gn(self.fuse(x)))
        x = self.refine_rnn(x)
        x = self.refine_local(x)
        return x

class RNNDecoder1D(nn.Module):
    """U-Net-like decoder, but with RNN-refine on each upsample."""
    def __init__(
        self,
        chs: List[int],
        k: int = 7,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        skip_gating: bool = False,
        rnn_type: str = "gru",
        bidirectional: bool = True,
        rnn_layers_per_block: int = 1,
        rnn_hidden_mult: float = 1.0,
        rnn_dropout: float = 0.0,
        use_dwpos: bool = True,
    ):
        super().__init__()
        steps = []
        for i in range(len(chs) - 1, -1, -1):
            in_ch = chs[i] if i == len(chs) - 1 else steps[-1][2]
            skip_ch = chs[i]
            out_ch = chs[i - 1] if i - 1 >= 0 else chs[0]
            steps.append((in_ch, skip_ch, out_ch))
        self.upblocks = nn.ModuleList(
            [
                RecurrentUpBlock(
                    in_ch, skip_ch, out_ch,
                    k=k, dropout=dropout, drop_path=drop_path, skip_gating=skip_gating,
                    rnn_type=rnn_type, bidirectional=bidirectional,
                    rnn_layers_per_block=rnn_layers_per_block,
                    rnn_hidden_mult=rnn_hidden_mult,
                    rnn_dropout=rnn_dropout,
                    use_dwpos=use_dwpos,
                )
                for (in_ch, skip_ch, out_ch) in steps
            ]
        )
        self.out_ch = chs[0]

    def forward(self, h, skips):
        for up, skip in zip(self.upblocks, reversed(skips)):
            h = up(h, skip)
        return h


class RNNBottleneck(nn.Module):
    """Stack of recurrent blocks in the bottleneck."""
    def __init__(
        self,
        ch: int,
        depth: int = 2,
        rnn_type: str = "gru",
        bidirectional: bool = True,
        rnn_layers: int = 1,
        rnn_hidden_mult: float = 1.0,
        rnn_dropout: float = 0.0,
        use_dwpos: bool = True,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                RNNBlock1D(
                    ch, rnn_type=rnn_type, bidirectional=bidirectional,
                    hidden_mult=rnn_hidden_mult, num_layers=rnn_layers,
                    rnn_dropout=rnn_dropout, use_dwpos=use_dwpos, k=3, drop_path=drop_path
                )
                for _ in range(depth)
            ]
        )
    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x

class DilatedBottleneck(nn.Module):
    def __init__(self, ch: int, depth: int = 2, k: int = 7, dropout: float = 0.1, drop_path: float = 0.0):
        super().__init__()
        blocks = []
        for i in range(depth):
            dil = 2 ** i
            blocks.append(ResBlock(ch, k=k, dropout=dropout, dilation=dil, drop_path=drop_path))
        self.net = nn.Sequential(*blocks)
    def forward(self, x):
        return self.net(x)

class HybridBottleneck(nn.Module):
    """Alternating RNN and dilated blocks."""
    def __init__(
        self, ch: int, depth: int = 4, k: int = 7, dropout: float = 0.1,
        rnn_type: str = "gru", bidirectional: bool = True,
        rnn_layers: int = 1, rnn_hidden_mult: float = 1.0, rnn_dropout: float = 0.0,
        use_dwpos: bool = True, drop_path: float = 0.0,
    ):
        super().__init__()
        blocks = []
        for i in range(depth):
            if i % 2 == 0:
                blocks.append(
                    RNNBlock1D(
                        ch, rnn_type=rnn_type, bidirectional=bidirectional,
                        hidden_mult=rnn_hidden_mult, num_layers=rnn_layers,
                        rnn_dropout=rnn_dropout, use_dwpos=use_dwpos, k=3, drop_path=drop_path
                    )
                )
            else:
                dil = 2 ** (i // 2)
                blocks.append(ResBlock(ch, k=k, dropout=dropout, dilation=dil, drop_path=drop_path))
        self.blocks = nn.Sequential(*blocks)
    def forward(self, x):
        return self.blocks(x)

# ------------------------- model -------------------------
class RecurrentSneddyUnet(nn.Module):
    """
    Encoder/Decoder 1D segmentation with recurrent "optic".
    Interface compatible with AttentionSneddyUnet + rnn-specific parameters.

    Args (main ones match your model):
        n_chans, n_times, sfreq, c0, widen, depth_per_stage, dropout, k, out_channels,
        num_stages, bottleneck_type, bottleneck_depth, drop_path, skip_gating, use_norm

    New parameters:
        rnn_type: "gru" | "lstm"
        bidirectional: bool
        rnn_layers_per_block: int
        rnn_hidden_mult: float
        rnn_dropout: float
        bottleneck_rnn_layers: int
        use_dwpos: bool
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
        # familiar args:
        num_stages: int = 3,
        bottleneck_type: str = "rnn",     # "rnn" | "dilated" | "hybrid"
        bottleneck_depth: int = 2,
        drop_path: float = 0.0,
        skip_gating: bool = False,
        use_norm: bool = True,
        # new rnn-parameters:
        rnn_type: str = "gru",
        bidirectional: bool = True,
        rnn_layers_per_block: int = 1,
        bottleneck_rnn_layers: int = 1,
        rnn_hidden_mult: float = 1.0,
        rnn_dropout: float = 0.0,
        use_dwpos: bool = True,
    ):
        super().__init__()
        self.n_chans = n_chans
        self.n_times = n_times
        self.sfreq = sfreq
        self.out_channels = out_channels
        self.use_norm = use_norm

        self.norm = StdPerSample()
        self.c_squeeze = ChannelSqueeze(n_chans, c0)

        # Encoder
        self.encoder = RNNEncoder1D(
            num_stages=num_stages,
            c0=c0, widen=widen, depth_per_stage=depth_per_stage,
            k=k, dropout=dropout, drop_path=drop_path,
            rnn_type=rnn_type, bidirectional=bidirectional,
            rnn_layers_per_block=rnn_layers_per_block,
            rnn_hidden_mult=rnn_hidden_mult,
            rnn_dropout=rnn_dropout,
            use_dwpos=use_dwpos,
        )

        bottleneck_ch = self.encoder.chs[-1]
        if bottleneck_type == "rnn":
            self.bottleneck = RNNBottleneck(
                bottleneck_ch, depth=bottleneck_depth,
                rnn_type=rnn_type, bidirectional=bidirectional,
                rnn_layers=bottleneck_rnn_layers,
                rnn_hidden_mult=rnn_hidden_mult, rnn_dropout=rnn_dropout,
                use_dwpos=use_dwpos, drop_path=drop_path,
            )
        elif bottleneck_type == "dilated":
            self.bottleneck = DilatedBottleneck(bottleneck_ch, depth=bottleneck_depth, k=k, dropout=dropout, drop_path=drop_path)
        elif bottleneck_type == "hybrid":
            self.bottleneck = HybridBottleneck(
                bottleneck_ch, depth=bottleneck_depth, k=k, dropout=dropout,
                rnn_type=rnn_type, bidirectional=bidirectional,
                rnn_layers=bottleneck_rnn_layers, rnn_hidden_mult=rnn_hidden_mult,
                rnn_dropout=rnn_dropout, use_dwpos=use_dwpos, drop_path=drop_path,
            )
        else:
            raise ValueError("bottleneck_type must be 'rnn' | 'dilated' | 'hybrid'.")

        # Decoder
        self.decoder = RNNDecoder1D(
            self.encoder.chs, k=k, dropout=dropout, drop_path=drop_path, skip_gating=skip_gating,
            rnn_type=rnn_type, bidirectional=bidirectional,
            rnn_layers_per_block=rnn_layers_per_block,
            rnn_hidden_mult=rnn_hidden_mult, rnn_dropout=rnn_dropout, use_dwpos=use_dwpos,
        )

        # Head
        self.head = nn.Conv1d(self.decoder.out_ch, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        # x: (B,C,T)
        hard_use_channels = [idx for idx in range(128) if idx not in (9, 121)]
        if x.shape[1] > 126:
            x = x[:, hard_use_channels, :]
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

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
