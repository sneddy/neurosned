"""Two-scale lagged-dynamics network for scalar EEG regression.

The model turns a fixed EEG window into a sequence of overlapping local
second-order representations.  A zero-lag branch encodes regularized
covariance matrices in log-Cholesky coordinates, while lagged branches encode
cross-correlation and differentiable ridge-transition operators.  A small TCN
then models how those local dynamics evolve across the input window.

This module deliberately keeps the benchmark-facing regression contract:
``(B, channels, time) -> (B, n_outputs)``.
"""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch
import torch.nn.functional as F
from torch import nn

from benchmarks.pkg.models.layers import ResBlock, StdPerSample


def _repeat_schedule(values: Sequence[int], depth: int) -> tuple[int, ...]:
    """Repeat a positive dilation schedule to ``depth`` entries."""
    if depth < 0:
        raise ValueError("temporal_depth must be non-negative.")
    if depth == 0:
        return ()
    values = tuple(int(value) for value in values)
    if not values or any(value <= 0 for value in values):
        raise ValueError("temporal_dilations must contain positive integers.")
    return tuple(values[index % len(values)] for index in range(depth))


def _inverse_softplus(value: float) -> float:
    """Return an unconstrained scalar whose softplus is ``value``."""
    if value <= 0:
        raise ValueError("ridge_init must be positive.")
    return math.log(math.expm1(value))


def _logit(value: float) -> float:
    """Return the logit of a probability strictly between zero and one."""
    if not 0 < value < 1:
        raise ValueError("cov_shrinkage_init must lie strictly between 0 and 1.")
    return math.log(value / (1.0 - value))


class LaggedDynamicsRegressor(nn.Module):
    """Regress a scalar target from a trajectory of local EEG dynamics.

    Two temporal scales are represented explicitly:

    * within-segment lags encode delayed channel dependence;
    * the segment TCN encodes changes in those matrices across the window.

    The zero-lag covariance branch is optional so that covariance-only and
    lagged-only controls can be run with the same implementation.
    """

    def __init__(
        self,
        n_chans: int,
        n_times: int,
        sfreq: float,
        n_outputs: int = 1,
        segment_samples: int = 100,
        segment_stride: int = 25,
        projection_dim: int = 24,
        lags: Sequence[int] = (5, 10, 25),
        include_covariance: bool = True,
        include_cross_correlation: bool = True,
        include_transition: bool = True,
        cov_hidden: int = 96,
        operator_hidden: int = 48,
        lag_hidden: int = 96,
        token_dim: int = 128,
        temporal_depth: int = 3,
        temporal_dilations: Sequence[int] = (1, 2),
        temporal_kernel: int = 3,
        dropout: float = 0.15,
        cov_shrinkage_init: float = 0.10,
        ridge_init: float = 0.05,
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
        self.projection_dim = int(projection_dim)
        self.lags = tuple(int(lag) for lag in lags)
        self.include_covariance = bool(include_covariance)
        self.include_cross_correlation = bool(include_cross_correlation)
        self.include_transition = bool(include_transition)
        self.matrix_eps = float(matrix_eps)
        self.use_norm = bool(use_norm)

        self._validate_configuration(
            cov_hidden=cov_hidden,
            operator_hidden=operator_hidden,
            lag_hidden=lag_hidden,
            token_dim=token_dim,
            temporal_kernel=temporal_kernel,
            dropout=dropout,
        )
        self.num_segments = 1 + (self.n_times - self.segment_samples) // self.segment_stride
        self.dilation_schedule = _repeat_schedule(temporal_dilations, int(temporal_depth))

        self.input_norm = StdPerSample(eps=self.matrix_eps)
        self.projected_norm = StdPerSample(eps=self.matrix_eps)
        self.spatial_projection = nn.Parameter(torch.empty(self.projection_dim, self.n_chans))
        nn.init.orthogonal_(self.spatial_projection)

        identity = torch.eye(self.projection_dim)
        self.register_buffer("matrix_identity", identity, persistent=False)
        tril = torch.tril_indices(self.projection_dim, self.projection_dim)
        self.register_buffer("tril_indices", tril, persistent=False)
        self.register_buffer("tril_diagonal", tril[0] == tril[1], persistent=False)

        branch_dims: list[int] = []
        if self.include_covariance:
            self.raw_cov_shrinkage = nn.Parameter(torch.tensor(_logit(cov_shrinkage_init)))
            covariance_input_dim = self.projection_dim * (self.projection_dim + 1) // 2 + 1
            self.covariance_encoder = nn.Sequential(
                nn.Linear(covariance_input_dim, cov_hidden),
                nn.LayerNorm(cov_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(cov_hidden, cov_hidden),
                nn.GELU(),
            )
            branch_dims.append(cov_hidden)
        else:
            self.register_parameter("raw_cov_shrinkage", None)
            self.covariance_encoder = None

        self.operator_labels = tuple(
            [f"corr_lag{lag}" for lag in self.lags] if self.include_cross_correlation else []
        ) + tuple([f"transition_lag{lag}" for lag in self.lags] if self.include_transition else [])
        self.num_operator_slots = len(self.operator_labels)

        matrix_size = self.projection_dim * self.projection_dim
        if self.include_cross_correlation:
            self.correlation_encoder = self._operator_encoder(matrix_size, operator_hidden, dropout)
        else:
            self.correlation_encoder = None
        if self.include_transition:
            self.transition_encoder = self._operator_encoder(matrix_size, operator_hidden, dropout)
            ridge_raw = torch.full((len(self.lags),), _inverse_softplus(ridge_init))
            self.raw_ridge = nn.Parameter(ridge_raw)
        else:
            self.transition_encoder = None
            self.register_parameter("raw_ridge", None)

        if self.num_operator_slots:
            self.operator_slot_embedding = nn.Parameter(
                torch.zeros(1, 1, self.num_operator_slots, operator_hidden)
            )
            nn.init.trunc_normal_(self.operator_slot_embedding, std=0.02)
            self.operator_gate = nn.Linear(operator_hidden, 1)
            lag_fuse_input = operator_hidden * (self.num_operator_slots + 1)
            self.lag_fuse = nn.Sequential(
                nn.Linear(lag_fuse_input, lag_hidden),
                nn.LayerNorm(lag_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            branch_dims.append(lag_hidden)
        else:
            self.register_parameter("operator_slot_embedding", None)
            self.operator_gate = None
            self.lag_fuse = None

        self.segment_fuse = nn.Sequential(
            nn.Linear(sum(branch_dims), token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_segments, token_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

        self.temporal_backbone = nn.Sequential(
            *[
                ResBlock(token_dim, k=temporal_kernel, dropout=dropout, dilation=dilation)
                for dilation in self.dilation_schedule
            ]
        )
        self.segment_gate = nn.Linear(token_dim, 1)
        self.head = nn.Sequential(
            nn.Linear(2 * token_dim, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, self.n_outputs),
        )
        # Let the scalar readout adapt before a large first Adam update reaches
        # every matrix and temporal block. This keeps the baseline learning
        # rate stable without imposing a target-specific output bias.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    @staticmethod
    def _operator_encoder(matrix_size: int, hidden: int, dropout: float) -> nn.Sequential:
        """Build a shared nonlinear encoder for one family of lagged matrices."""
        return nn.Sequential(
            nn.Linear(matrix_size, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )

    def _validate_configuration(
        self,
        *,
        cov_hidden: int,
        operator_hidden: int,
        lag_hidden: int,
        token_dim: int,
        temporal_kernel: int,
        dropout: float,
    ) -> None:
        """Reject shape combinations that cannot produce valid local matrices."""
        if self.n_chans <= 0 or self.n_times <= 0 or self.n_outputs <= 0:
            raise ValueError("n_chans, n_times, and n_outputs must be positive.")
        if self.sfreq <= 0:
            raise ValueError("sfreq must be positive.")
        if not 1 <= self.segment_samples <= self.n_times:
            raise ValueError("segment_samples must lie in [1, n_times].")
        if self.segment_stride <= 0:
            raise ValueError("segment_stride must be positive.")
        if not 1 <= self.projection_dim <= self.n_chans:
            raise ValueError("projection_dim must lie in [1, n_chans].")
        if self.matrix_eps <= 0:
            raise ValueError("matrix_eps must be positive.")
        if temporal_kernel <= 0 or temporal_kernel % 2 == 0:
            raise ValueError("temporal_kernel must be a positive odd integer.")
        if any(value <= 0 for value in (cov_hidden, operator_hidden, lag_hidden, token_dim)):
            raise ValueError("All hidden dimensions must be positive.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must lie in [0, 1).")
        if not (self.include_covariance or self.include_cross_correlation or self.include_transition):
            raise ValueError("At least one matrix branch must be enabled.")
        if self.include_cross_correlation or self.include_transition:
            if not self.lags:
                raise ValueError("At least one lag is required for lagged matrix branches.")
            if any(lag <= 0 or lag >= self.segment_samples for lag in self.lags):
                raise ValueError("Every lag must be positive and smaller than segment_samples.")
            if len(set(self.lags)) != len(self.lags):
                raise ValueError("lags must not contain duplicates.")

    @property
    def covariance_shrinkage(self) -> torch.Tensor | None:
        """Return the learned covariance shrinkage coefficient."""
        if self.raw_cov_shrinkage is None:
            return None
        return torch.sigmoid(self.raw_cov_shrinkage)

    @property
    def ridge_coefficients(self) -> torch.Tensor | None:
        """Return positive scale-normalized ridge coefficients for all lags."""
        if self.raw_ridge is None:
            return None
        return F.softplus(self.raw_ridge)

    def _project_and_segment(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize, project, and unfold EEG into overlapping local segments."""
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B, C, T), got {tuple(x.shape)}.")
        if x.shape[1] != self.n_chans or x.shape[2] != self.n_times:
            raise ValueError(
                f"Expected input shape (*, {self.n_chans}, {self.n_times}), got {tuple(x.shape)}."
            )
        if self.use_norm:
            x = self.input_norm(x)
        weight = F.normalize(self.spatial_projection, dim=-1, eps=self.matrix_eps)
        projected = torch.einsum("dc,bct->bdt", weight, x)
        projected = self.projected_norm(projected)
        segments = projected.unfold(-1, self.segment_samples, self.segment_stride)
        segments = segments.permute(0, 2, 1, 3).contiguous()
        return projected, segments

    def _regularized_covariance(self, segments: torch.Tensor) -> torch.Tensor:
        """Return shrinkage covariance matrices for local segments."""
        centered = segments - segments.mean(dim=-1, keepdim=True)
        covariance = centered @ centered.transpose(-1, -2)
        covariance = covariance / float(max(self.segment_samples - 1, 1))
        scale = covariance.diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp_min(self.matrix_eps)
        identity = self.matrix_identity.to(dtype=covariance.dtype)
        alpha = self.covariance_shrinkage.to(dtype=covariance.dtype)
        covariance = (1.0 - alpha) * covariance + alpha * scale[..., None, None] * identity
        covariance = covariance + self.matrix_eps * scale[..., None, None] * identity
        return covariance

    def _log_covariance_vector(self, covariance: torch.Tensor) -> torch.Tensor:
        """Map SPD matrices to stable, scale-aware log-Cholesky vectors.

        A direct eigendecomposition is fragile on short-window EEG covariance
        batches: repeated or tightly clustered eigenvalues can make CUDA
        ``eigh`` fail and its eigenvector gradients are ill-conditioned.  The
        log-Cholesky map is smooth on the SPD cone, remains one-to-one, and
        avoids eigenvectors entirely.
        """
        scale = covariance.diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp_min(self.matrix_eps)
        covariance_shape = covariance / scale[..., None, None]
        covariance_shape = 0.5 * (covariance_shape + covariance_shape.transpose(-1, -2))
        cholesky = torch.linalg.cholesky(covariance_shape)
        row, col = self.tril_indices
        vector = cholesky[..., row, col]
        diagonal = self.tril_diagonal
        vector = torch.where(diagonal, vector.clamp_min(self.matrix_eps).log(), vector)
        return torch.cat((vector, scale.log().unsqueeze(-1)), dim=-1)

    def _lagged_matrices(self, segments: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return per-lag correlation and ridge-transition matrix sequences."""
        if not (self.include_cross_correlation or self.include_transition):
            return None, None

        correlations = []
        transitions = []
        identity = self.matrix_identity.to(dtype=segments.dtype)
        ridge_coefficients = self.ridge_coefficients

        for lag_index, lag in enumerate(self.lags):
            current = segments[..., :-lag]
            future = segments[..., lag:]
            current = current - current.mean(dim=-1, keepdim=True)
            future = future - future.mean(dim=-1, keepdim=True)
            denominator = float(max(current.shape[-1] - 1, 1))
            cxx = (current @ current.transpose(-1, -2)) / denominator
            cyy = (future @ future.transpose(-1, -2)) / denominator
            cxy = (future @ current.transpose(-1, -2)) / denominator

            if self.include_cross_correlation:
                var_current = cxx.diagonal(dim1=-2, dim2=-1).clamp_min(self.matrix_eps)
                var_future = cyy.diagonal(dim1=-2, dim2=-1).clamp_min(self.matrix_eps)
                normalizer = torch.sqrt(var_future[..., :, None] * var_current[..., None, :])
                correlations.append((cxy / normalizer.clamp_min(self.matrix_eps)).clamp(-1.0, 1.0))

            if self.include_transition:
                scale = cxx.diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp_min(self.matrix_eps)
                ridge = ridge_coefficients[lag_index].to(dtype=segments.dtype)
                regularized = cxx + (ridge + self.matrix_eps) * scale[..., None, None] * identity
                transition_t = torch.linalg.solve(regularized, cxy.transpose(-1, -2))
                transitions.append(transition_t.transpose(-1, -2))

        correlation_tensor = torch.stack(correlations, dim=2) if correlations else None
        transition_tensor = torch.stack(transitions, dim=2) if transitions else None
        return correlation_tensor, transition_tensor

    def extract_matrix_sequence(self, x: torch.Tensor) -> dict[str, torch.Tensor | None]:
        """Return projected EEG and all enabled per-segment matrix representations."""
        projected, segments = self._project_and_segment(x)
        covariance = self._regularized_covariance(segments) if self.include_covariance else None
        correlation, transition = self._lagged_matrices(segments)
        return {
            "projected": projected,
            "segments": segments,
            "covariance": covariance,
            "lagged_correlation": correlation,
            "transition": transition,
        }

    def _encode_operators(
        self,
        correlation: torch.Tensor | None,
        transition: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode, mix, and return lagged operator features and attention weights."""
        slot_features = []
        if correlation is not None:
            for lag_index in range(len(self.lags)):
                matrix = correlation[:, :, lag_index].flatten(start_dim=-2)
                slot_features.append(self.correlation_encoder(matrix))
        if transition is not None:
            signed_log_transition = torch.sign(transition) * torch.log1p(transition.abs())
            for lag_index in range(len(self.lags)):
                matrix = signed_log_transition[:, :, lag_index].flatten(start_dim=-2)
                slot_features.append(self.transition_encoder(matrix))

        slots = torch.stack(slot_features, dim=2)
        slots = slots + self.operator_slot_embedding.to(dtype=slots.dtype)
        attention = torch.softmax(self.operator_gate(slots).squeeze(-1), dim=-1)
        context = torch.sum(slots * attention.unsqueeze(-1), dim=2)
        fused = self.lag_fuse(torch.cat((slots.flatten(start_dim=2), context), dim=-1))
        return fused, attention

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor | None]:
        """Return matrix representations, segment tokens, and pooled features."""
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

        segment_tokens = self.segment_fuse(torch.cat(branches, dim=-1))
        positioned_tokens = segment_tokens + self.position_embedding.to(dtype=segment_tokens.dtype)
        temporal_tokens = self.temporal_backbone(positioned_tokens.transpose(1, 2)).transpose(1, 2)
        segment_attention = torch.softmax(self.segment_gate(temporal_tokens).squeeze(-1), dim=-1)
        attended = torch.sum(temporal_tokens * segment_attention.unsqueeze(-1), dim=1)
        pooled = torch.cat((attended, temporal_tokens.mean(dim=1)), dim=-1)

        return {
            **matrices,
            "covariance_vector": covariance_vector,
            "operator_attention": operator_attention,
            "segment_tokens": segment_tokens,
            "temporal_tokens": temporal_tokens,
            "segment_attention": segment_attention,
            "pooled_features": pooled,
        }

    def forward(self, x: torch.Tensor, return_dict: bool = False):
        """Return scalar predictions, optionally with matrix-level diagnostics."""
        features = self.forward_features(x)
        prediction = self.head(features["pooled_features"])
        if return_dict:
            return {"prediction": prediction, **features}
        return prediction


__all__ = ["LaggedDynamicsRegressor"]
