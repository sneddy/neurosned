# --- доп. импорты остаются прежними ---
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- вспомогательные блоки из вашей версии (оставлены как есть) ----------
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
    """Depthwise separable 1D: depthwise + pointwise."""
    def __init__(self, ch, k=7, stride=1, dilation=1):
        super().__init__()
        pad = ((k - 1) // 2) * dilation
        self.dw = nn.Conv1d(ch, ch, k, stride=stride, padding=pad,
                            dilation=dilation, groups=ch, bias=False)
        self.pw = nn.Conv1d(ch, ch, 1, bias=False)
    def forward(self, x):
        return self.pw(self.dw(x))

class ResBlock(nn.Module):
    """Лёгкий residual-блок с DSConv."""
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
    """Смешивание электродов: 1x1 C_in→C_out."""
    def __init__(self, c_in, c_out):
        super().__init__()
        self.proj = nn.Conv1d(c_in, c_out, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity="linear")
    def forward(self, x):
        return self.proj(x)

class TimeDown(nn.Module):
    """Антиалиас + AvgPool(stride=2) по времени."""
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

class UpBlock(nn.Module):
    """Линейный апсемпл + skip concat + 1x1 fuse + ResBlock."""
    def __init__(self, in_ch, skip_ch, out_ch, k=7, dropout=0.0):
        super().__init__()
        self.fuse = nn.Conv1d(in_ch + skip_ch, out_ch, kernel_size=1, bias=False)
        self.gn   = nn.GroupNorm(1, out_ch)
        self.act  = nn.GELU()
        self.refine = ResBlock(out_ch, k=k, dropout=dropout, dilation=1)

    def forward(self, x_low, x_skip):
        x = F.interpolate(x_low, size=x_skip.shape[-1], mode='linear', align_corners=False)
        x = torch.cat([x, x_skip], dim=1)
        x = self.act(self.gn(self.fuse(x)))
        x = self.refine(x)
        return x


# ---------- новый: факторизованные кросс-канальные взаимодействия ----------
class FactorizedCrossChannel(nn.Module):
    """
    FM-подобный слой для кросс-канальных взаимодействий.
    На вход x: (B, C, T). Возвращает (B, F, T) — по факторам.
    Реализует 0.5 * sum_f[ (sum_c v_cf x_c)^2 - sum_c (v_cf^2 x_c^2) ] в разложенном виде.
    """
    def __init__(self, n_chans: int, n_factors: int, dropout: float = 0.0, use_gn: bool = True):
        super().__init__()
        self.n_chans = n_chans
        self.n_factors = n_factors
        # Матрица факторов V (C x F)
        self.V = nn.Parameter(torch.empty(n_chans, n_factors))
        nn.init.xavier_normal_(self.V)  # стабильная инициализация
        self.in_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.out_norm = nn.GroupNorm(1, n_factors) if use_gn else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):  # x: (B, C, T)
        x = self.in_drop(x)
        # s = x^T V  -> (B, F, T)
        s = torch.einsum('bct,cf->bft', x, self.V)
        # sum_sq = (sum_c v_cf x_c)^2 ; sq_sum = sum_c (v_cf^2 x_c^2)
        sum_sq = s * s
        sq_sum = torch.einsum('bct,cf->bft', x * x, self.V * self.V)
        inter = 0.5 * (sum_sq - sq_sum)  # (B, F, T)
        return self.act(self.out_norm(inter))


# ---------- новый: «гибкий» энкодер/декодер на произвольное число стейджей ----------
class Encoder1DFlex(nn.Module):
    """
    Sneddy-style encoder на N стейджей: [enc(stage)->skip->down] * N, затем bottleneck.
    Опционально можно вставить FM на выходе каждого стейджа и слить его с признаками.
    """
    def __init__(self, c0=32, widen=2, n_stages=3, depth_per_stage=2, k=7, dropout=0.1,
                 use_stage_fm: bool = False, fm_factors: int = 32, fm_dropout: float = 0.0):
        super().__init__()
        assert n_stages >= 1, "n_stages должно быть >=1"
        # канальные размеры стейджей (до даунсамплинга)
        chs = [c0]
        for _ in range(1, n_stages):
            chs.append(int(chs[-1] * widen))
        self.chs = chs

        enc, downs = [], []
        stage_fm, stage_fuse, stage_norm, stage_act = [], [], [], []
        in_c = c0
        for out_c in chs:
            stage = []
            if out_c != in_c:
                stage += [nn.Conv1d(in_c, out_c, 1, bias=False), nn.GroupNorm(1, out_c), nn.GELU()]
            for _ in range(depth_per_stage):
                stage.append(ResBlock(out_c, k=k, dropout=dropout, dilation=1))
            enc.append(nn.Sequential(*stage))
            # опциональная FM-вставка в конце стейджа
            if use_stage_fm:
                stage_fm.append(FactorizedCrossChannel(out_c, fm_factors, dropout=fm_dropout, use_gn=True))
                stage_fuse.append(nn.Conv1d(out_c + fm_factors, out_c, kernel_size=1, bias=False))
                stage_norm.append(nn.GroupNorm(1, out_c))
                stage_act.append(nn.GELU())
            else:
                stage_fm.append(None)
                stage_fuse.append(None)
                stage_norm.append(None)
                stage_act.append(None)
            downs.append(TimeDown(out_c))
            in_c = out_c

        self.encoder_blocks = nn.ModuleList(enc)
        self.downs = nn.ModuleList(downs)

        # слои для FM-вставки по стейджам
        self.use_stage_fm = use_stage_fm
        if use_stage_fm:
            self.stage_fm = nn.ModuleList(stage_fm)
            self.stage_fuse = nn.ModuleList(stage_fuse)
            self.stage_norm = nn.ModuleList(stage_norm)
            self.stage_act = nn.ModuleList(stage_act)

        # bottleneck на самом низком разрешении
        bottleneck_ch = chs[-1]
        self.bottleneck = nn.Sequential(
            ResBlock(bottleneck_ch, k=k, dropout=dropout, dilation=1),
            ResBlock(bottleneck_ch, k=k, dropout=dropout, dilation=2),
        )

    def forward(self, x):
        skips = []
        h = x
        for i, (enc, down) in enumerate(zip(self.encoder_blocks, self.downs)):
            h = enc(h)
            if self.use_stage_fm:
                fm = self.stage_fm[i](h)  # (B, F, T_i)
                h = self.stage_act[i](self.stage_norm[i](self.stage_fuse[i](torch.cat([h, fm], dim=1))))
            skips.append(h)
            h = down(h)
        h = self.bottleneck(h)
        return h, skips  # h на самом низком слое, skips от мелкого к глубокому


class Decoder1DFlex(nn.Module):
    """U-Net-подобный decoder на N стейджей (симметрично энкодеру)."""
    def __init__(self, chs, k=7, dropout=0.1):
        super().__init__()
        steps = []
        in_ch = chs[-1]  # входной канал из bottleneck
        # поднимаемся с deepest skip к shallow
        for i in reversed(range(len(chs))):
            skip_ch = chs[i]
            out_ch  = chs[i]  # держим те же каналы, что и skip
            steps.append(UpBlock(in_ch, skip_ch, out_ch, k=k, dropout=dropout))
            in_ch = out_ch
        self.upblocks = nn.ModuleList(steps)
        self.out_ch = chs[0]

    def forward(self, h, skips):
        for up, skip in zip(self.upblocks, reversed(skips)):
            h = up(h, skip)
        return h


# ---------- фронт-энд со склейкой FM и линейного смешивания каналов ----------
class FactorizedFrontEnd(nn.Module):
    """
    1) ChannelSqueeze C->c0 (линейное смешивание электродов),
    2) FactorizedCrossChannel C->F (кросс-канальные взаимодействия),
    3) fuse: concat по каналам -> 1x1 conv обратно в c0.
    """
    def __init__(self, n_chans: int, c0: int, fm_factors: int, fm_dropout: float = 0.0):
        super().__init__()
        self.lin = ChannelSqueeze(n_chans, c0)
        self.fm  = FactorizedCrossChannel(n_chans, fm_factors, dropout=fm_dropout, use_gn=True)
        self.fuse = nn.Conv1d(c0 + fm_factors, c0, kernel_size=1, bias=False)
        self.gn = nn.GroupNorm(1, c0)
        self.act = nn.GELU()

    def forward(self, x):  # x: (B, C, T)
        a = self.lin(x)        # (B, c0, T)
        b = self.fm(x)         # (B, F,  T)
        y = torch.cat([a, b], dim=1)
        y = self.act(self.gn(self.fuse(y)))  # (B, c0, T)
        return y


# ---------- основная модель ----------
class FactorizationSneddyUnet(nn.Module):
    """
    U-Net 1D с FM-взаимодействиями между каналами.
    Параметры позволяют углублять сеть (n_stages, depth_per_stage) и расширять (widen, c0).
    """
    def __init__(
        self,
        n_chans: int,
        n_times: int,
        sfreq: int,
        c0: int = 32,
        widen: int = 2,
        n_stages: int = 3,
        depth_per_stage: int = 2,
        dropout: float = 0.1,
        k: int = 7,
        out_channels: int = 1,
        # FM-конфигурация
        fm_factors_front: int = 32,
        fm_dropout_front: float = 0.0,
        use_stage_fm: bool = False,
        fm_factors_stage: int = 16,
        fm_dropout_stage: float = 0.0,
    ):
        super().__init__()
        self.n_chans = n_chans
        self.n_times = n_times
        self.sfreq   = sfreq
        self.out_channels = out_channels

        self.norm = StdPerSample()
        self.front = FactorizedFrontEnd(n_chans, c0, fm_factors=fm_factors_front, fm_dropout=fm_dropout_front)

        self.encoder = Encoder1DFlex(
            c0=c0, widen=widen, n_stages=n_stages, depth_per_stage=depth_per_stage,
            k=k, dropout=dropout,
            use_stage_fm=use_stage_fm, fm_factors=fm_factors_stage, fm_dropout=fm_dropout_stage
        )
        self.decoder = Decoder1DFlex(self.encoder.chs, k=k, dropout=dropout)

        self.head = nn.Conv1d(self.decoder.out_ch, out_channels, kernel_size=1, bias=True)

    def forward(self, x):  # x:(B,C,T)
        B, C, T = x.shape
        x = self.norm(x)
        x = self.front(x)                # (B, c0, T)
        h_low, skips = self.encoder(x)   # h_low at lowest time scale
        h = self.decoder(h_low, skips)   # (B, c0, ~T)
        if h.shape[-1] != T:
            h = F.interpolate(h, size=T, mode='linear', align_corners=False)
        logits = self.head(h)            # (B, out_channels, T)
        return logits

    @torch.no_grad()
    def predict(
        self,
        x,
        mode: str = "argmax",          # "argmax" | "softargmax"
        temperature: float = 1.0,      # used for softargmax
        window_sec: float = 2.0,       # длина окна соответствующая входу
        return_var: bool = False
    ):
        """
        Возвращает:
          - t_hat_sec: (B,) время в секундах относительно начала окна
          - var_sec (опционально): (B,) простая прокси дисперсии
        """
        logits = self.forward(x)
        if self.out_channels != 1:
            raise ValueError("predict() предполагает out_channels==1.")
        B, _, T = logits.shape
        dt = window_sec / T
        z = logits.squeeze(1)  # (B, T)

        if mode == "argmax":
            idx = torch.argmax(z, dim=-1)
            t_hat = idx.to(z.dtype) * dt
            if not return_var:
                return t_hat
            var = torch.full_like(t_hat, fill_value=(dt**2))
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
            raise ValueError("mode must be 'argmax' или 'softargmax'.")

    @torch.no_grad()
    def predict_mask(self, x, temperature: float = 1.0):
        """Пер-временные вероятности (B, T) из логитов (softmax по времени)."""
        logits = self.forward(x)
        if self.out_channels != 1:
            raise ValueError("predict_mask() предполагает out_channels==1.")
        z = logits.squeeze(1)
        return F.softmax(z / temperature, dim=-1)
