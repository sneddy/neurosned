# sneddy_net.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class StdPerSample(nn.Module):
    """Пер-семпловая нормализация: вычитаем среднее по времени и делим на std по времени.
    Без зависимости от датасета/батча. Для устойчивого старта."""
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
    def forward(self, x):  # x:(B,C,T)
        mu = x.mean(dim=-1, keepdim=True)
        sd = x.std(dim=-1, keepdim=True).clamp_min(self.eps)
        return (x - mu) / sd


class DSConv1d(nn.Module):
    """Depthwise separable 1D: depthwise (по каналам признаков) + pointwise 1x1."""
    def __init__(self, ch, k=7, stride=1, dilation=1):
        super().__init__()
        pad = ((k - 1) // 2) * dilation
        self.dw = nn.Conv1d(ch, ch, k, stride=stride, padding=pad,
                            dilation=dilation, groups=ch, bias=False)
        self.pw = nn.Conv1d(ch, ch, 1, bias=False)
    def forward(self, x):
        return self.pw(self.dw(x))


class ResBlock(nn.Module):
    """Лёгкий residual-блок: DSConv → GN → GELU → DSConv → GN + skip."""
    def __init__(self, ch, k=7, dropout=0.0, dilation=1):
        super().__init__()
        self.conv1 = DSConv1d(ch, k=k, dilation=dilation)
        self.gn1   = nn.GroupNorm(1, ch)
        self.conv2 = DSConv1d(ch, k=k, dilation=1)
        self.gn2   = nn.GroupNorm(1, ch)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
    def forward(self, x):
        r = x
        x = self.conv1(x)
        x = self.act(self.gn1(x))
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.act(self.gn2(x) + r)
        return x


class ChannelSqueeze(nn.Module):
    """Сужение по электродам: 1x1 свёртка C_in→C_out (учебная смесь каналов)."""
    def __init__(self, c_in, c_out):
        super().__init__()
        self.proj = nn.Conv1d(c_in, c_out, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity="linear")
    def forward(self, x):
        return self.proj(x)


class TimeDown(nn.Module):
    """Сужение по времени: антиалиас (легкий depthwise) + AvgPool(stride=2)."""
    def __init__(self, ch, k=5):
        super().__init__()
        pad = (k - 1) // 2
        self.aa = nn.Conv1d(ch, ch, k, groups=ch, padding=pad, bias=False)
        with torch.no_grad():
            # треугольное ядро сглаживания
            w = torch.tensor([1, 2, 3, 2, 1], dtype=torch.float32)
            w = (w / w.sum()).view(1, 1, -1).repeat(ch, 1, 1)
            self.aa.weight.copy_(w)
        for p in self.aa.parameters():
            p.requires_grad_(False)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
    def forward(self, x):
        return self.pool(self.aa(x))


class SegmentStatPool(nn.Module):
    """Сегментный пуллинг: разбивает время на S сегментов и даёт mean/max по каждому.
    Это выделяет 'где' (начало/конец/середина)."""
    def __init__(self, segments=(2, 4)):
        super().__init__()
        self.segments = segments
    def forward(self, x):  # x:(B,F,T)
        B, F, T = x.shape
        outs = []
        for S in self.segments:
            seg_len = max(1, T // S)
            trimmed_T = seg_len * S
            if trimmed_T != T:
                x_cut = x[..., :trimmed_T]
            else:
                x_cut = x
            x_resh = x_cut.view(B, F, S, seg_len)  # (B,F,S,L)
            m = x_resh.mean(dim=-1)                # (B,F,S)
            a = x_resh.amax(dim=-1)                # (B,F,S)
            outs.append(m)
            outs.append(a)
        y = torch.cat([o.flatten(start_dim=1) for o in outs], dim=1)  # (B, F*sum(2*S_i))
        return y


class SneddyNet(nn.Module):
    """
    Простой baseline под (B, C, T), акцент на 'где изменение' (начало/конец):
      - StdPerSample нормализация
      - Channel squeeze (C -> c0)
      - 3 ступени: [ResBlock x2 + TimeDown] с увеличением фич и сужением времени
      - SegmentStatPool на последней карте признаков (S=2,4)
      - MLP-голова -> n_outputs
    Интерфейс совместим с braindecode-моделями.
    """
    def __init__(
        self,
        n_chans: int,
        n_times: int,
        sfreq: int,
        n_outputs: int = 1,
        c0: int = 32,          # стартовая ширина после сжатия каналов
        widen: int = 2,        # коэффициент расширения фич по ступеням
        depth_per_stage: int = 2,
        dropout: float = 0.1,
        k: int = 7,
        use_norm: bool = True,
    ):
        super().__init__()
        self.n_chans = n_chans
        self.n_times = n_times
        self.sfreq   = sfreq
        self.n_outputs = n_outputs
        self.use_norm = use_norm

        self.norm = StdPerSample()

        # Channel squeeze: 129 -> 32 (по умолчанию)
        self.c_squeeze = ChannelSqueeze(n_chans, c0)

        # Ступени пирамиды: time //= 2 на каждой
        chs = [c0, c0 * widen, c0 * widen * 2]
        stages = []
        in_c = c0
        for out_c in chs:
            # расширяем число признаков в начале ступени
            if out_c != in_c:
                stages.append(nn.Conv1d(in_c, out_c, kernel_size=1, bias=False))
                stages.append(nn.GroupNorm(1, out_c))
                stages.append(nn.GELU())
            # residual блоки
            for d in range(depth_per_stage):
                stages.append(ResBlock(out_c, k=k, dropout=dropout, dilation=1))
            # сужаем время
            stages.append(TimeDown(out_c))  # stride=2
            in_c = out_c
        self.backbone = nn.Sequential(*stages)

        # Сегментный пуллинг (чётко подсветить "раньше/позже")
        self.segpool = SegmentStatPool(segments=(2, 4))  # даёт mean/max по сегментам

        # Подсчитаем размер признаков после спины для головы
        with torch.no_grad():
            dummy = torch.zeros(1, n_chans, n_times)
            z = self.forward_features(dummy)
            feat_dim = z.shape[1]

        self.head = nn.Sequential(
            nn.Linear(feat_dim, max(64, feat_dim // 2)),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(max(64, feat_dim // 2), n_outputs),
        )

    def forward_features(self, x):
        if self.use_norm:
            x = self.norm(x)          # (B,C,T)
        x = self.c_squeeze(x)         # (B,c0,T)
        x = self.backbone(x)          # (B,F,T')
        z = self.segpool(x)           # (B, F * (2*sum(segments)))
        return z

    def forward(self, x):             # (B,C,T) -> (B,n_outputs)
        z = self.forward_features(x)
        y = self.head(z)
        return y
