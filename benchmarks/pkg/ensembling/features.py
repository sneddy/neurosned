"""Temporal posterior feature extraction for segmentation stacking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class RunReadout:
    """Readout metadata needed to convert temporal logits to probabilities."""

    readout: str = "softmax"
    sfreq: float = 100.0
    win_offset: float = 0.5
    hazard_condition_inside: bool = True


class TemporalFeatureExtractor:
    """Build scalar posterior features from per-time logits."""

    def __init__(
        self,
        *,
        temps: tuple[float, ...] = (0.5, 0.7, 0.8, 1.0),
        q_percentiles: tuple[int, ...] = (10, 50, 90),
    ):
        self.temps = tuple(float(t) for t in temps)
        self.qs = tuple(int(q) for q in q_percentiles)

    def build_from_logits_store(
        self,
        logits_store: Mapping[str, np.ndarray],
        readout_store: Mapping[str, RunReadout],
        *,
        return_names: bool = False,
    ):
        """Return concatenated posterior/logit features for all models."""
        Xs: list[np.ndarray] = []
        names: list[str] = []
        for model_name in sorted(logits_store):
            logits = np.asarray(logits_store[model_name], dtype=np.float32)
            readout = readout_store[model_name]
            feats, feat_names = self._single_model_features(logits, readout, prefix=model_name)
            Xs.append(feats)
            names.extend(feat_names)
        if not Xs:
            raise ValueError("No logits were provided.")
        X = np.concatenate(Xs, axis=1)
        return (X, names) if return_names else X

    def mode_times(self, logits: np.ndarray, readout: RunReadout, *, temperature: float = 1.0) -> np.ndarray:
        """Return posterior mode times in absolute seconds."""
        p = probabilities_from_logits(logits, readout, temperature=temperature)
        idx = np.argmax(p, axis=1)
        return idx.astype(np.float32) / float(readout.sfreq) + float(readout.win_offset)

    def mean_times(self, logits: np.ndarray, readout: RunReadout, *, temperature: float = 1.0) -> np.ndarray:
        """Return posterior mean times in absolute seconds."""
        p = probabilities_from_logits(logits, readout, temperature=temperature)
        tg = _time_grid(p.shape[1], readout)
        return np.sum(p * tg, axis=1).astype(np.float32, copy=False)

    def _single_model_features(self, logits: np.ndarray, readout: RunReadout, *, prefix: str) -> tuple[np.ndarray, list[str]]:
        if logits.ndim != 2:
            raise ValueError(f"Expected logits with shape (n_samples, n_times), got {logits.shape}.")

        feats: list[np.ndarray] = []
        names: list[str] = []
        z_max = np.max(logits, axis=1)
        z_margin = _top2_margin(logits)
        feats.extend([z_max[:, None], z_margin[:, None]])
        names.extend([f"{prefix}_z_max", f"{prefix}_z_margin"])

        for temperature in self.temps:
            p = probabilities_from_logits(logits, readout, temperature=temperature)
            tg = _time_grid(logits.shape[1], readout)
            t_abs = np.sum(p * tg, axis=1)
            entropy = _entropy(p)
            pmax = np.max(p, axis=1)
            pmargin = _top2_margin(p)
            tvar = np.sum(p * (tg - t_abs[:, None]) ** 2, axis=1)
            mode = np.argmax(p, axis=1).astype(np.float32) / float(readout.sfreq) + float(readout.win_offset)
            quantiles = _quantile_times(p, tg, self.qs)
            label = _temp_label(temperature)
            feats.extend(
                [
                    t_abs[:, None],
                    mode[:, None],
                    entropy[:, None],
                    pmax[:, None],
                    pmargin[:, None],
                    tvar[:, None],
                    quantiles,
                ]
            )
            names.extend(
                [
                    f"{prefix}_t_mean_tau{label}",
                    f"{prefix}_t_mode_tau{label}",
                    f"{prefix}_entropy_tau{label}",
                    f"{prefix}_pmax_tau{label}",
                    f"{prefix}_pmargin_tau{label}",
                    f"{prefix}_tvar_tau{label}",
                ]
                + [f"{prefix}_q{q}_tau{label}" for q in self.qs]
            )

        return np.concatenate(feats, axis=1).astype(np.float32, copy=False), names


def probabilities_from_logits(logits: np.ndarray, readout: RunReadout, *, temperature: float) -> np.ndarray:
    """Convert logits to event-time probabilities for softmax or hazard readouts."""
    name = str(readout.readout).lower().replace("-", "_")
    if name in {"softmax", "soft_argmax", "event_softmax"}:
        return _softmax(logits / float(temperature), axis=1)
    if name in {"hazard", "survival", "event_hazard"}:
        return _hazard_pmf(
            logits,
            temperature=float(temperature),
            condition_inside=bool(readout.hazard_condition_inside),
        )
    raise ValueError(f"Unsupported readout: {readout.readout!r}")


def _time_grid(n_times: int, readout: RunReadout) -> np.ndarray:
    return np.arange(n_times, dtype=np.float32)[None, :] / float(readout.sfreq) + float(readout.win_offset)


def _softmax(x: np.ndarray, *, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def _hazard_pmf(logits: np.ndarray, *, temperature: float, condition_inside: bool) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float32) / float(temperature)
    log_hazard = -np.logaddexp(0.0, -z)
    log_not_hazard = -np.logaddexp(0.0, z)
    log_survival_through = np.cumsum(log_not_hazard, axis=1)
    log_survival_before = np.concatenate(
        [np.zeros((z.shape[0], 1), dtype=np.float32), log_survival_through[:, :-1]],
        axis=1,
    )
    log_pmf = log_survival_before + log_hazard
    if condition_inside:
        log_inside = _logsumexp(log_pmf, axis=1, keepdims=True)
        log_pmf = log_pmf - log_inside
    return np.exp(log_pmf).astype(np.float32, copy=False)


def _logsumexp(x: np.ndarray, *, axis: int, keepdims: bool) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)


def _entropy(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    clipped = np.clip(p, eps, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=1)


def _top2_margin(x: np.ndarray) -> np.ndarray:
    if x.shape[1] < 2:
        return np.zeros(x.shape[0], dtype=np.float32)
    top2 = np.partition(x, -2, axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0]


def _quantile_times(p: np.ndarray, tg: np.ndarray, qs: tuple[int, ...]) -> np.ndarray:
    cdf = np.cumsum(p, axis=1)
    values = []
    for q in qs:
        threshold = float(q) / 100.0
        idx = np.argmax(cdf >= threshold, axis=1)
        values.append(tg[0, idx])
    return np.stack(values, axis=1)


def _temp_label(value: float) -> str:
    return str(float(value)).replace(".", "p")
