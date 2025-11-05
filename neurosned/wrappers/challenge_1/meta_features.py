import numpy as np


class MetaFeatureExtractor:
    """Builds per-model temporal features from segmentation logits and appends classification outputs."""
    def __init__(self, sfreq=100.0, win_offset=0.5, temps=(0.7, 1.0, 1.3), q_percentiles=(10, 50, 90)):
        self.sfreq = float(sfreq)
        self.dt = 1.0 / self.sfreq
        self.win_offset = float(win_offset)
        self.temps = tuple(temps)
        self.qs = tuple(q_percentiles)

    def _softmax(self, z, t, axis=-1):
        z = z / float(t)
        z = z - np.max(z, axis=axis, keepdims=True)
        ez = np.exp(z)
        return ez / np.sum(ez, axis=axis, keepdims=True)

    def _time_grid(self, T):
        return np.arange(T, dtype=np.float32)[None, :] * self.dt

    def _softargmax_time(self, p, tg):
        return np.sum(p * tg, axis=1)

    def _entropy(self, p, eps=1e-12):
        p = np.clip(p, eps, 1.0)
        return -np.sum(p * np.log(p), axis=1)

    def _top2_margin(self, x, axis=-1):
        part = np.partition(x, -2, axis=axis)
        top2 = part[..., -2:]
        return top2[..., -1] - top2[..., -2]

    def _quantile_times(self, p, tg):
        cdf = np.cumsum(p, axis=1)
        out = []
        for q in self.qs:
            thr = q / 100.0
            idx = np.argmax(cdf >= thr, axis=1)
            out.append(tg[0, idx])
        return np.stack(out, axis=1)

    def _seg_feats_single(self, Z, return_names=False, prefix=None):  # Z:(N,T)
        N, T = Z.shape
        tg = self._time_grid(T)
        feats = []
        names = []

        idx = np.argmax(Z, axis=1)
        t_hard = idx * self.dt + self.win_offset
        z_max = np.max(Z, axis=1)
        z_margin = self._top2_margin(Z, axis=1)
        feats += [t_hard[:, None], z_max[:, None], z_margin[:, None]]
        names += [
            "t_hard",
            "z_max",
            "z_margin"
        ]

        for t in self.temps:
            p = self._softmax(Z, t, axis=1)
            t_rel = self._softargmax_time(p, tg)
            t_abs = t_rel + self.win_offset
            ent = self._entropy(p)
            pmax = np.max(p, axis=1)
            pmargin = self._top2_margin(p, axis=1)
            tvar = np.sum(p * (tg - t_rel[:, None])**2, axis=1)
            qs = self._quantile_times(p, tg)
            feats += [t_abs[:, None], ent[:, None], pmax[:, None], pmargin[:, None], tvar[:, None], qs]
            names += [
                f"t_abs_temp{t}",
                f"ent_temp{t}",
                f"pmax_temp{t}",
                f"pmargin_temp{t}",
                f"tvar_temp{t}",
            ] + [f"q{q}_temp{t}" for q in self.qs]
        concat_feats = np.concatenate(feats, axis=1)
        if prefix is not None:
            names = [f"{prefix}_{n}" for n in names]
        if return_names:
            return concat_feats, names
        return concat_feats

    def _cls_feats_single(self, Y, return_names=False, prefix=None):  # Y:(N,K) or (N,)
        Y = np.asarray(Y)
        names = []
        if Y.ndim == 1:
            feats = Y[:, None]
            names = ["cls"]
        elif Y.shape[1] > 1:
            K = Y.shape[1]
            Y_sm = self._softmax(Y, t=1.0, axis=1)
            feats = np.concatenate([Y, Y_sm], axis=1)
            names = [f"cls_logits_{k}" for k in range(K)] + [f"cls_sm_{k}" for k in range(K)]
        else:
            feats = Y
            names = ["cls"]
        if prefix is not None:
            names = [f"{prefix}_{n}" for n in names]
        if return_names:
            return feats, names
        return feats

    def build_from_logits_store(self, seg_logits_store: dict, cls_outputs_store: dict | None = None, return_names: bool = False):
        """Builds features from dicts: seg[name]->(N,T), cls[name]->(N, K or 1). Returns (X, names) if return_names."""
        Xs = []
        names = []
        if seg_logits_store:
            for name in seg_logits_store.keys():
                Z = np.asarray(seg_logits_store[name], dtype=np.float32)
                feats, nms = self._seg_feats_single(Z, return_names=True, prefix=f"seg_{name}")
                Xs.append(feats)
                names += nms
        if cls_outputs_store:
            for name in cls_outputs_store.keys():
                Y = np.asarray(cls_outputs_store[name], dtype=np.float32)
                feats, nms = self._cls_feats_single(Y, return_names=True, prefix=f"cls_{name}")
                Xs.append(feats)
                names += nms
        X = np.concatenate(Xs, axis=1) if len(Xs) > 1 else Xs[0]
        if return_names:
            return X, names
        return X

    def build_from_batch(self, seg_logits_list: list[np.ndarray], cls_outputs_list: list[np.ndarray] | None = None, return_names: bool = False):
        """Builds features from lists aligned across models: each seg item is (B,T), each cls item is (B,K or 1). Returns (X, names) if return_names."""
        Xs = []
        names = []
        for idx, Z in enumerate(seg_logits_list):
            feats, nms = self._seg_feats_single(np.asarray(Z, dtype=np.float32), return_names=True, prefix=f"seg{idx}")
            Xs.append(feats)
            names += nms
        if cls_outputs_list:
            for idx, Y in enumerate(cls_outputs_list):
                feats, nms = self._cls_feats_single(np.asarray(Y, dtype=np.float32), return_names=True, prefix=f"cls{idx}")
                Xs.append(feats)
                names += nms
        X = np.concatenate(Xs, axis=1) if len(Xs) > 1 else Xs[0]
        if return_names:
            return X, names
        return X