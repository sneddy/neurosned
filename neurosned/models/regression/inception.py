import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==== базовые элементы из вашего стека ====

class StdPerSample(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
    def forward(self, x):  # (B,C,T)
        mu = x.mean(dim=-1, keepdim=True)
        sd = x.std(dim=-1, keepdim=True).clamp_min(self.eps)
        return (x - mu) / sd

class DSConv1d(nn.Module):
    def __init__(self, ch, k=7, stride=1, dilation=1):
        super().__init__()
        pad = ((k - 1) // 2) * dilation
        self.dw = nn.Conv1d(ch, ch, k, stride=stride, padding=pad,
                            dilation=dilation, groups=ch, bias=False)
        self.pw = nn.Conv1d(ch, ch, 1, bias=False)
    def forward(self, x):
        return self.pw(self.dw(x))

class ResBlock(nn.Module):
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


# ==== Inception-подобные ветки для 1D EEG ====

class InceptionTemporalSpatialBranch(nn.Module):
    """
    Ветка: (темпоральный depthwise по каждому электроду) -> (пространственная 1x1 смесь каналов).
    Идея как в EEGInceptionERP: сначала время, потом каналы. Без изменения T.
    """
    def __init__(self, n_chans, out_ch, k_time, dropout=0.1, bn_momentum=0.01, activation=nn.ELU):
        super().__init__()
        pad = (k_time - 1) // 2
        # Temporal depthwise: по каждому электроду свой фильтр во времени; каналов столько же
        self.t_dw = nn.Conv1d(n_chans, n_chans, kernel_size=k_time, padding=pad,
                              groups=n_chans, bias=False)
        self.t_bn = nn.BatchNorm1d(n_chans, momentum=bn_momentum)
        self.act  = activation()
        self.do   = nn.Dropout(dropout)
        # Spatial 1x1 mix: обучаемая смесь электродов -> out_ch
        self.s_pw = nn.Conv1d(n_chans, out_ch, kernel_size=1, bias=False)
        self.s_bn = nn.BatchNorm1d(out_ch, momentum=bn_momentum)

    def forward(self, x):   # x: (B,C,T)
        x = self.t_dw(x)
        x = self.act(self.t_bn(x))
        x = self.do(x)
        x = self.s_pw(x)
        x = self.act(self.s_bn(x))
        x = self.do(x)
        return x             # (B, out_ch, T)


class InceptionStage1(nn.Module):
    """
    Первая Inception-ступень: 3 масштаба по времени + конкат по каналам.
    """
    def __init__(self, n_chans, branch_out, scales_samples, dropout=0.1, bn_momentum=0.01, activation=nn.ELU):
        super().__init__()
        self.b1 = InceptionTemporalSpatialBranch(n_chans, branch_out, scales_samples[0],
                                                 dropout, bn_momentum, activation)
        self.b2 = InceptionTemporalSpatialBranch(n_chans, branch_out, scales_samples[1],
                                                 dropout, bn_momentum, activation)
        self.b3 = InceptionTemporalSpatialBranch(n_chans, branch_out, scales_samples[2],
                                                 dropout, bn_momentum, activation)
    def forward(self, x):
        x1 = self.b1(x)
        x2 = self.b2(x)
        x3 = self.b3(x)
        return torch.cat([x1, x2, x3], dim=1)  # (B, 3*branch_out, T)


class InceptionStage2(nn.Module):
    """
    Вторая Inception-ступень: работает уже на признаках (после конката веток),
    более короткие ядра (как в оригинале).
    """
    def __init__(self, in_ch, branch_out, k_list, dropout=0.1, bn_momentum=0.01, activation=nn.ELU):
        super().__init__()
        def conv_block(k):
            pad = (k - 1) // 2
            return nn.Sequential(
                nn.Conv1d(in_ch, branch_out, kernel_size=k, padding=pad, bias=False),
                nn.BatchNorm1d(branch_out, momentum=bn_momentum),
                activation(),
                nn.Dropout(dropout),
            )
        self.c4 = conv_block(k_list[0])
        self.c5 = conv_block(k_list[1])
        self.c6 = conv_block(k_list[2])

    def forward(self, x):
        z4 = self.c4(x)
        z5 = self.c5(x)
        z6 = self.c6(x)
        return torch.cat([z4, z5, z6], dim=1)  # (B, 3*branch_out, T)


# ==== Энкодер/декодер в стиле вашего Sneddy ====

class InceptionEncoder1D(nn.Module):
    """
    Encoder: Stage1 (multi-scale) -> (опц) лёгкое сжатие T -> Stage2 (короткие масштабы) -> bottleneck.
    Даунсэмпл по времени опционален и очень мягкий, чтобы не терять точность пика.
    """
    def __init__(self, n_chans, sfreq, branch_out=16, scales_s=(0.5, 0.25, 0.125),
                 pooling_sizes=(1, 1),  # (p1, p2) по времени; 1=без пуллинга
                 dropout=0.1, bn_momentum=0.01, activation=nn.ELU):
        super().__init__()
        # переведём секунды в сэмплы от частоты дискретизации
        scales_samples = tuple(max(3, int(round(s * sfreq))//2*2 + 1) for s in scales_s)  # гарантируем нечётное k
        # Stage 1
        self.stage1 = InceptionStage1(n_chans, branch_out, scales_samples,
                                      dropout, bn_momentum, activation)
        self.pool1  = None if pooling_sizes[0] == 1 else nn.AvgPool1d(kernel_size=pooling_sizes[0], stride=pooling_sizes[0])
        # Stage 2: ядра в 4 раза короче (как у оригинала)
        k_list = [max(3, s//4//2*2 + 1) for s in scales_samples]
        self.stage2 = InceptionStage2(in_ch=3*branch_out, branch_out=branch_out,
                                      k_list=k_list, dropout=dropout,
                                      bn_momentum=bn_momentum, activation=activation)
        self.pool2  = None if pooling_sizes[1] == 1 else nn.AvgPool1d(kernel_size=pooling_sizes[1], stride=pooling_sizes[1])

        self.bottleneck = ResBlock(3*branch_out, k=7, dropout=dropout, dilation=2)

        self.out_ch = 3*branch_out

    def forward(self, x):  # (B,C,T)
        skips = []
        h = self.stage1(x)          # (B, 3*branch_out, T)
        skips.append(h)
        if self.pool1 is not None:
            h = self.pool1(h)       # мягкий даунсэмпл времени
        h = self.stage2(h)          # (B, 3*branch_out, T' или T/p1)
        skips.append(h)
        if self.pool2 is not None:
            h = self.pool2(h)
        h = self.bottleneck(h)      # (B, out_ch, T'')
        return h, skips


class InceptionDecoder1D(nn.Module):
    """
    Decoder: линейный апсемпл до размерностей скипов + слияние + лёгкая дообработка.
    """
    def __init__(self, ch, dropout=0.1, k=7):
        super().__init__()
        self.refine2 = ResBlock(ch, k=k, dropout=dropout, dilation=1)  # после слияния со skip2
        self.refine1 = ResBlock(ch, k=k, dropout=dropout, dilation=1)  # после слияния со skip1
        self.gn = nn.GroupNorm(1, ch)
        self.act = nn.GELU()

    def forward(self, h, skips):
        # skips: [after stage1 (high-res), after stage2 (mid-res)]
        s1, s2 = skips[0], skips[1]
        # вверх до s2
        if h.shape[-1] != s2.shape[-1]:
            h = F.interpolate(h, size=s2.shape[-1], mode='linear', align_corners=False)
        h = self.refine2(h + s2)
        # вверх до s1
        if h.shape[-1] != s1.shape[-1]:
            h = F.interpolate(h, size=s1.shape[-1], mode='linear', align_corners=False)
        h = self.refine1(h + s1)
        return h  # (B, ch, T)


# ==== Итоговая модель в «нашем» формате ====

class EEGInceptionSeg1D(nn.Module):
    """
    Inception-подобная сегментация/локализация для EEG:
      StdPerSample -> Encoder(Inception1/2, мягкий pool опционально) -> Decoder(upsample+res) -> 1x1 head
    Выход: логиты (B, out_channels, T) без потери разрешения (если pooling_sizes=(1,1)).
    Совместимые методы: predict / predict_mask.
    """
    def __init__(
        self,
        n_chans: int,
        n_times: int,
        sfreq: int,
        # ключевые «крутилки»:
        branch_out: int = 16,                 # сколько каналов на ветку
        scales_samples_s = (0.5, 0.25, 0.125),# временные масштабы (сек)
        pooling_sizes = (1, 1),               # мягкий pool по времени для Stage1/2; 1=без пула
        dropout: float = 0.1,
        bn_momentum: float = 0.01,
        activation: nn.Module = nn.ELU,
        out_channels: int = 1,
        head_kernel: int = 1,                 # 1x1 по умолчанию
    ):
        super().__init__()
        self.n_chans = n_chans
        self.n_times = n_times
        self.sfreq   = sfreq
        self.out_channels = out_channels

        self.norm = StdPerSample()

        self.encoder = InceptionEncoder1D(
            n_chans=n_chans, sfreq=sfreq,
            branch_out=branch_out,
            scales_s=scales_samples_s,
            pooling_sizes=pooling_sizes,
            dropout=dropout, bn_momentum=bn_momentum, activation=activation
        )
        self.decoder = InceptionDecoder1D(self.encoder.out_ch, dropout=dropout, k=7)

        pad = (head_kernel - 1) // 2
        self.head = nn.Conv1d(self.encoder.out_ch, out_channels, kernel_size=head_kernel, padding=pad, bias=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d,)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                if getattr(m, "weight", None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):  # x: (B,C,T)
        B, C, T = x.shape
        x = self.norm(x)
        h_low, skips = self.encoder(x)
        h = self.decoder(h_low, skips)
        if h.shape[-1] != T:
            h = F.interpolate(h, size=T, mode='linear', align_corners=False)
        logits = self.head(h)  # (B, out_channels, T)
        return logits

    @torch.no_grad()
    def predict(
        self,
        x,
        mode: str = "argmax",     # "argmax" | "softargmax"
        temperature: float = 1.0,
        window_sec: float = 2.0,
        return_var: bool = False
    ):
        logits = self.forward(x)
        if self.out_channels != 1:
            raise ValueError("predict() assumes out_channels==1 for time readout.")
        B, _, T = logits.shape
        dt = window_sec / T
        z = logits.squeeze(1)  # (B, T)

        if mode == "argmax":
            idx = torch.argmax(z, dim=-1)
            t_hat = idx.to(z.dtype) * dt
            if not return_var:
                return t_hat
            var = torch.full_like(t_hat, fill_value=dt**2)
            return t_hat, var

        elif mode == "softargmax":
            p = F.softmax(z / temperature, dim=-1)
            grid = torch.arange(T, device=z.device, dtype=z.dtype)[None, :]
            t_idx = (p * grid).sum(dim=-1)
            t_hat = t_idx * dt
            if not return_var:
                return t_hat
            var = (p * ((grid * dt - t_hat[:, None])**2)).sum(dim=-1)
            return t_hat, var

        else:
            raise ValueError("mode must be 'argmax' or 'softargmax'.")

    @torch.no_grad()
    def predict_mask(self, x, temperature: float = 1.0):
        logits = self.forward(x)
        if self.out_channels != 1:
            raise ValueError("predict_mask() assumes out_channels==1.")
        z = logits.squeeze(1)
        return F.softmax(z / temperature, dim=-1)
