"""Dual-view EEG regressors combining raw waveforms and local dynamics."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from benchmarks.pkg.models.layers import ChannelSqueeze, ResBlock, StdPerSample
from benchmarks.pkg.models.regression.lagged_dynamics import LaggedDynamicsRegressor


def _repeat_schedule(values: Sequence[int], depth: int) -> tuple[int, ...]:
    """Repeat a positive dilation schedule to a fixed depth."""
    if depth < 0:
        raise ValueError("depth must be non-negative.")
    if depth == 0:
        return ()
    values = tuple(int(value) for value in values)
    if not values or any(value <= 0 for value in values):
        raise ValueError("dilations must contain positive integers.")
    return tuple(values[index % len(values)] for index in range(depth))


class RawSegmentEncoder(nn.Module):
    """Encode full-resolution EEG and pool features into aligned segment tokens.

    Temporal convolutions run before segment pooling, so waveform order, phase,
    and cross-segment continuity bypass the lossy covariance transformation.
    The final attention pooling uses exactly the same overlapping intervals as
    the matrix branch.
    """

    def __init__(
        self,
        *,
        n_chans: int,
        n_times: int,
        segment_samples: int,
        segment_stride: int,
        raw_width: int,
        token_dim: int,
        raw_depth: int,
        raw_dilations: Sequence[int],
        raw_kernel: int,
        dropout: float,
    ):
        super().__init__()
        if raw_width <= 0 or token_dim <= 0:
            raise ValueError("raw_width and token_dim must be positive.")
        if raw_kernel <= 0 or raw_kernel % 2 == 0:
            raise ValueError("raw_kernel must be a positive odd integer.")
        if not 1 <= segment_samples <= n_times:
            raise ValueError("segment_samples must lie in [1, n_times].")
        if segment_stride <= 0:
            raise ValueError("segment_stride must be positive.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must lie in [0, 1).")

        self.n_chans = int(n_chans)
        self.n_times = int(n_times)
        self.segment_samples = int(segment_samples)
        self.segment_stride = int(segment_stride)
        self.raw_width = int(raw_width)
        self.token_dim = int(token_dim)
        self.num_segments = 1 + (self.n_times - self.segment_samples) // self.segment_stride
        schedule = _repeat_schedule(raw_dilations, int(raw_depth))

        self.stem = ChannelSqueeze(self.n_chans, self.raw_width)
        self.temporal_backbone = nn.Sequential(
            *[
                ResBlock(self.raw_width, k=raw_kernel, dropout=dropout, dilation=dilation)
                for dilation in schedule
            ]
        )
        self.sample_gate = nn.Conv1d(self.raw_width, 1, kernel_size=1)
        nn.init.zeros_(self.sample_gate.weight)
        nn.init.zeros_(self.sample_gate.bias)
        self.token_projection = nn.Sequential(
            nn.Linear(2 * self.raw_width, self.token_dim),
            nn.LayerNorm(self.token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return the raw feature map, within-segment attention, and tokens."""
        if x.ndim != 3 or x.shape[1] != self.n_chans or x.shape[2] != self.n_times:
            raise ValueError(
                f"Expected input shape (*, {self.n_chans}, {self.n_times}), got {tuple(x.shape)}."
            )
        feature_map = self.temporal_backbone(self.stem(x))
        feature_segments = feature_map.unfold(
            -1, self.segment_samples, self.segment_stride
        ).permute(0, 2, 1, 3)
        sample_logits = self.sample_gate(feature_map).unfold(
            -1, self.segment_samples, self.segment_stride
        ).squeeze(1)
        sample_attention = torch.softmax(sample_logits, dim=-1)
        attended = torch.sum(feature_segments * sample_attention.unsqueeze(-2), dim=-1)
        mean_pooled = feature_segments.mean(dim=-1)
        tokens = self.token_projection(torch.cat((attended, mean_pooled), dim=-1))
        return {
            "raw_feature_map": feature_map,
            "raw_segment_attention": sample_attention,
            "raw_segment_tokens": tokens,
        }


class DualViewLaggedDynamicsRegressor(LaggedDynamicsRegressor):
    """Fuse raw temporal tokens with covariance and lagged-dynamics tokens."""

    def __init__(
        self,
        n_chans: int,
        n_times: int,
        sfreq: float,
        n_outputs: int = 1,
        segment_samples: int = 50,
        segment_stride: int = 25,
        projection_dim: int = 24,
        lags: Sequence[int] = (5, 10, 20),
        include_covariance: bool = True,
        include_cross_correlation: bool = True,
        include_transition: bool = True,
        cov_hidden: int = 256,
        operator_hidden: int = 128,
        lag_hidden: int = 256,
        token_dim: int = 384,
        temporal_depth: int = 6,
        temporal_dilations: Sequence[int] = (1, 2, 4),
        temporal_kernel: int = 3,
        raw_width: int = 128,
        raw_depth: int = 4,
        raw_dilations: Sequence[int] = (1, 2, 4, 8),
        raw_kernel: int = 7,
        dropout: float = 0.15,
        cov_shrinkage_init: float = 0.10,
        ridge_init: float = 0.05,
        matrix_eps: float = 1e-4,
        use_norm: bool = True,
    ):
        super().__init__(
            n_chans=n_chans,
            n_times=n_times,
            sfreq=sfreq,
            n_outputs=n_outputs,
            segment_samples=segment_samples,
            segment_stride=segment_stride,
            projection_dim=projection_dim,
            lags=lags,
            include_covariance=include_covariance,
            include_cross_correlation=include_cross_correlation,
            include_transition=include_transition,
            cov_hidden=cov_hidden,
            operator_hidden=operator_hidden,
            lag_hidden=lag_hidden,
            token_dim=token_dim,
            temporal_depth=temporal_depth,
            temporal_dilations=temporal_dilations,
            temporal_kernel=temporal_kernel,
            dropout=dropout,
            cov_shrinkage_init=cov_shrinkage_init,
            ridge_init=ridge_init,
            matrix_eps=matrix_eps,
            use_norm=use_norm,
        )
        self.raw_encoder = RawSegmentEncoder(
            n_chans=self.n_chans,
            n_times=self.n_times,
            segment_samples=self.segment_samples,
            segment_stride=self.segment_stride,
            raw_width=raw_width,
            token_dim=token_dim,
            raw_depth=raw_depth,
            raw_dilations=raw_dilations,
            raw_kernel=raw_kernel,
            dropout=dropout,
        )

        self.matrix_token_norm = nn.LayerNorm(token_dim)
        self.raw_token_norm = nn.LayerNorm(token_dim)
        gate_hidden = max(token_dim // 2, 32)
        self.modality_gate = nn.Sequential(
            nn.Linear(2 * token_dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 2),
        )
        nn.init.zeros_(self.modality_gate[-1].weight)
        nn.init.zeros_(self.modality_gate[-1].bias)

        self.fusion_update = nn.Sequential(
            nn.Linear(3 * token_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, token_dim),
        )
        nn.init.zeros_(self.fusion_update[-1].weight)
        nn.init.zeros_(self.fusion_update[-1].bias)
        self.fusion_norm = nn.LayerNorm(token_dim)

    def _matrix_segment_features(
        self, x: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor | None], torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Return inherited matrix diagnostics and pre-TCN matrix tokens."""
        matrices = self.extract_matrix_sequence(x)
        branches = []
        covariance_vector = None
        operator_attention = None

        if matrices["covariance"] is not None:
            covariance_vector = self._log_covariance_vector(matrices["covariance"])
            branches.append(self.covariance_encoder(covariance_vector))
        if self.num_operator_slots:
            lag_features, operator_attention = self._encode_operators(
                matrices["lagged_correlation"], matrices["transition"]
            )
            branches.append(lag_features)

        matrix_tokens = self.segment_fuse(torch.cat(branches, dim=-1))
        return matrices, matrix_tokens, covariance_vector, operator_attention

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor | None]:
        """Return both views, their fusion, and the shared temporal readout."""
        matrices, matrix_tokens, covariance_vector, operator_attention = (
            self._matrix_segment_features(x)
        )
        raw_input = self.input_norm(x) if self.use_norm else x
        raw_features = self.raw_encoder(raw_input)

        matrix_normalized = self.matrix_token_norm(matrix_tokens)
        raw_normalized = self.raw_token_norm(raw_features["raw_segment_tokens"])
        paired = torch.cat((matrix_normalized, raw_normalized), dim=-1)
        modality_attention = torch.softmax(self.modality_gate(paired), dim=-1)
        mixture = (
            modality_attention[..., 0, None] * matrix_normalized
            + modality_attention[..., 1, None] * raw_normalized
        )
        interaction_input = torch.cat(
            (matrix_normalized, raw_normalized, matrix_normalized * raw_normalized), dim=-1
        )
        fused_tokens = self.fusion_norm(mixture + self.fusion_update(interaction_input))

        positioned_tokens = fused_tokens + self.position_embedding.to(dtype=fused_tokens.dtype)
        temporal_tokens = self.temporal_backbone(positioned_tokens.transpose(1, 2)).transpose(1, 2)
        segment_attention = torch.softmax(self.segment_gate(temporal_tokens).squeeze(-1), dim=-1)
        attended = torch.sum(temporal_tokens * segment_attention.unsqueeze(-1), dim=1)
        pooled = torch.cat((attended, temporal_tokens.mean(dim=1)), dim=-1)

        return {
            **matrices,
            **raw_features,
            "covariance_vector": covariance_vector,
            "operator_attention": operator_attention,
            "matrix_segment_tokens": matrix_tokens,
            "modality_attention": modality_attention,
            "segment_tokens": fused_tokens,
            "temporal_tokens": temporal_tokens,
            "segment_attention": segment_attention,
            "pooled_features": pooled,
        }


class RawTemporalRegressor(nn.Module):
    """Raw-waveform control with the same segment-level temporal readout."""

    def __init__(
        self,
        n_chans: int,
        n_times: int,
        sfreq: float,
        n_outputs: int = 1,
        segment_samples: int = 50,
        segment_stride: int = 25,
        raw_width: int = 128,
        raw_depth: int = 4,
        raw_dilations: Sequence[int] = (1, 2, 4, 8),
        raw_kernel: int = 7,
        token_dim: int = 384,
        temporal_depth: int = 6,
        temporal_dilations: Sequence[int] = (1, 2, 4),
        temporal_kernel: int = 3,
        dropout: float = 0.15,
        matrix_eps: float = 1e-4,
        use_norm: bool = True,
    ):
        super().__init__()
        self.n_chans = int(n_chans)
        self.n_times = int(n_times)
        self.sfreq = float(sfreq)
        self.n_outputs = int(n_outputs)
        self.segment_samples = int(segment_samples)
        self.segment_stride = int(segment_stride)
        self.use_norm = bool(use_norm)
        if self.n_chans <= 0 or self.n_times <= 0 or self.n_outputs <= 0 or self.sfreq <= 0:
            raise ValueError("n_chans, n_times, n_outputs, and sfreq must be positive.")
        if temporal_kernel <= 0 or temporal_kernel % 2 == 0:
            raise ValueError("temporal_kernel must be a positive odd integer.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must lie in [0, 1).")
        if matrix_eps <= 0:
            raise ValueError("matrix_eps must be positive.")

        self.input_norm = StdPerSample(eps=matrix_eps)
        self.raw_encoder = RawSegmentEncoder(
            n_chans=self.n_chans,
            n_times=self.n_times,
            segment_samples=self.segment_samples,
            segment_stride=self.segment_stride,
            raw_width=raw_width,
            token_dim=token_dim,
            raw_depth=raw_depth,
            raw_dilations=raw_dilations,
            raw_kernel=raw_kernel,
            dropout=dropout,
        )
        self.num_segments = self.raw_encoder.num_segments
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_segments, token_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        schedule = _repeat_schedule(temporal_dilations, int(temporal_depth))
        self.temporal_backbone = nn.Sequential(
            *[
                ResBlock(token_dim, k=temporal_kernel, dropout=dropout, dilation=dilation)
                for dilation in schedule
            ]
        )
        self.segment_gate = nn.Linear(token_dim, 1)
        self.head = nn.Sequential(
            nn.Linear(2 * token_dim, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, self.n_outputs),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return raw segment tokens and shared sequence-level features."""
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B, C, T), got {tuple(x.shape)}.")
        if x.shape[1] != self.n_chans or x.shape[2] != self.n_times:
            raise ValueError(
                f"Expected input shape (*, {self.n_chans}, {self.n_times}), got {tuple(x.shape)}."
            )
        raw_input = self.input_norm(x) if self.use_norm else x
        raw_features = self.raw_encoder(raw_input)
        segment_tokens = raw_features["raw_segment_tokens"]
        positioned_tokens = segment_tokens + self.position_embedding.to(dtype=segment_tokens.dtype)
        temporal_tokens = self.temporal_backbone(positioned_tokens.transpose(1, 2)).transpose(1, 2)
        segment_attention = torch.softmax(self.segment_gate(temporal_tokens).squeeze(-1), dim=-1)
        attended = torch.sum(temporal_tokens * segment_attention.unsqueeze(-1), dim=1)
        pooled = torch.cat((attended, temporal_tokens.mean(dim=1)), dim=-1)
        return {
            **raw_features,
            "segment_tokens": segment_tokens,
            "temporal_tokens": temporal_tokens,
            "segment_attention": segment_attention,
            "pooled_features": pooled,
        }

    def forward(self, x: torch.Tensor, return_dict: bool = False):
        """Return scalar predictions, optionally with raw-view diagnostics."""
        features = self.forward_features(x)
        prediction = self.head(features["pooled_features"])
        if return_dict:
            return {"prediction": prediction, **features}
        return prediction


__all__ = ["DualViewLaggedDynamicsRegressor", "RawSegmentEncoder", "RawTemporalRegressor"]
