"""Multiscale Segment Attention CNN for direct RT regression."""

from __future__ import annotations

import torch
from torch import nn

from benchmarks.pkg.models.layers import ChannelSqueeze, ResBlock, StdPerSample, TimeDown


class MultiscaleSegmentAttentionHead(nn.Module):
    """Pool multiscale temporal segment tokens with learned attention."""

    def __init__(
        self,
        in_ch: int,
        n_outputs: int = 1,
        segments=(2, 4, 8),
        attn_dim: int = 256,
        head_hidden: int = 128,
        dropout: float = 0.1,
        pooling: str = "score",
        num_heads: int = 4,
        attn_layers: int = 0,
        attn_ff_dim: int | None = None,
        attn_dropout: float = 0.1,
        pool_queries: int = 1,
        use_segment_embeddings: bool = False,
    ):
        super().__init__()
        self.segments = tuple(segments)
        self.pooling = pooling
        self.pool_queries = pool_queries
        self.token_proj = nn.Sequential(
            nn.Linear(in_ch * 2, attn_dim),
            nn.GELU(),
            nn.LayerNorm(attn_dim),
        )

        token_count = sum(self.segments)
        scale_ids = []
        position_ids = []
        for scale_index, segment_count in enumerate(self.segments):
            scale_ids.extend([scale_index] * segment_count)
            position_ids.extend(range(segment_count))
        self.register_buffer("scale_ids", torch.tensor(scale_ids, dtype=torch.long), persistent=False)
        self.register_buffer("position_ids", torch.tensor(position_ids, dtype=torch.long), persistent=False)

        self.use_segment_embeddings = use_segment_embeddings
        if use_segment_embeddings:
            self.scale_embedding = nn.Embedding(len(self.segments), attn_dim)
            self.position_embedding = nn.Embedding(max(self.segments), attn_dim)
        else:
            self.scale_embedding = None
            self.position_embedding = None

        if attn_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=attn_dim,
                nhead=num_heads,
                dim_feedforward=attn_ff_dim or attn_dim * 2,
                dropout=attn_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.token_encoder = nn.TransformerEncoder(encoder_layer, num_layers=attn_layers)
        else:
            self.token_encoder = nn.Identity()

        if pooling == "query":
            self.query = nn.Parameter(torch.randn(pool_queries, attn_dim) * 0.02)
            self.pool = nn.MultiheadAttention(
                embed_dim=attn_dim,
                num_heads=num_heads,
                dropout=attn_dropout,
                batch_first=True,
            )
            pooled_dim = attn_dim * pool_queries
            self.score = None
        elif pooling == "score":
            self.query = None
            self.pool = None
            pooled_dim = attn_dim
            self.score = nn.Linear(attn_dim, 1)
        else:
            raise ValueError("pooling must be 'score' or 'query'.")

        assert len(scale_ids) == token_count
        self.head = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, n_outputs),
        )

    def make_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Return segment tokens of shape (batch, tokens, 2 * channels)."""
        batch_size, features, time = x.shape
        tokens = []
        for segment_count in self.segments:
            segment_len = max(1, time // segment_count)
            trimmed_time = segment_len * segment_count
            x_cut = x[..., :trimmed_time]
            x_reshaped = x_cut.view(batch_size, features, segment_count, segment_len)
            mean = x_reshaped.mean(dim=-1).transpose(1, 2)
            max_value = x_reshaped.amax(dim=-1).transpose(1, 2)
            tokens.append(torch.cat([mean, max_value], dim=-1))
        return torch.cat(tokens, dim=1)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """Return regression output, optionally with segment attention weights."""
        tokens = self.token_proj(self.make_tokens(x))
        if self.use_segment_embeddings:
            embeddings = self.scale_embedding(self.scale_ids) + self.position_embedding(self.position_ids)
            tokens = tokens + embeddings.unsqueeze(0)
        tokens = self.token_encoder(tokens)

        if self.pooling == "query":
            queries = self.query.unsqueeze(0).expand(tokens.size(0), -1, -1)
            pooled_tokens, weights = self.pool(queries, tokens, tokens, need_weights=True)
            pooled = pooled_tokens.flatten(start_dim=1)
            if weights.size(1) == 1:
                weights = weights.squeeze(1)
        else:
            weights = torch.softmax(self.score(tokens).squeeze(-1), dim=-1)
            pooled = torch.einsum("btd,bt->bd", tokens, weights)

        output = self.head(pooled)
        if return_attention:
            return output, weights
        return output


class MultiscaleSegmentGatedHead(nn.Module):
    """Pool multiscale temporal segment tokens with sigmoid gates."""

    def __init__(
        self,
        in_ch: int,
        n_outputs: int = 1,
        segments=(2, 4),
        token_dim: int = 256,
        head_hidden: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.segments = tuple(segments)
        self.token_proj = nn.Sequential(
            nn.Linear(in_ch * 2, token_dim),
            nn.GELU(),
            nn.LayerNorm(token_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(token_dim, token_dim // 2),
            nn.GELU(),
            nn.Linear(token_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, n_outputs),
        )

    def make_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Return segment tokens of shape (batch, tokens, 2 * channels)."""
        batch_size, features, time = x.shape
        tokens = []
        for segment_count in self.segments:
            segment_len = max(1, time // segment_count)
            trimmed_time = segment_len * segment_count
            x_cut = x[..., :trimmed_time]
            x_reshaped = x_cut.view(batch_size, features, segment_count, segment_len)
            mean = x_reshaped.mean(dim=-1).transpose(1, 2)
            max_value = x_reshaped.amax(dim=-1).transpose(1, 2)
            tokens.append(torch.cat([mean, max_value], dim=-1))
        return torch.cat(tokens, dim=1)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """Return regression output, optionally with normalized gate weights."""
        tokens = self.token_proj(self.make_tokens(x))
        gates = self.gate(tokens).squeeze(-1)
        weights = gates / gates.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        pooled = torch.einsum("btd,bt->bd", tokens, weights)
        output = self.head(pooled)
        if return_attention:
            return output, weights
        return output


class _SegmentBackbone(nn.Module):
    """Shared CNN backbone for segment-token regression heads."""

    def __init__(
        self,
        n_chans: int,
        n_times: int,
        sfreq: int,
        n_outputs: int,
        c0: int,
        widen: int,
        depth_per_stage: int,
        dropout: float,
        k: int,
        use_norm: bool,
    ):
        super().__init__()
        self.n_chans = n_chans
        self.n_times = n_times
        self.sfreq = sfreq
        self.n_outputs = n_outputs
        self.use_norm = use_norm

        self.norm = StdPerSample()
        self.c_squeeze = ChannelSqueeze(n_chans, c0)

        chs = [c0, c0 * widen, c0 * widen * 2]
        stages = []
        in_ch = c0
        for out_ch in chs:
            if out_ch != in_ch:
                stages.append(nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False))
                stages.append(nn.GroupNorm(1, out_ch))
                stages.append(nn.GELU())
            for _ in range(depth_per_stage):
                stages.append(ResBlock(out_ch, k=k, dropout=dropout, dilation=1))
            stages.append(TimeDown(out_ch))
            in_ch = out_ch
        self.backbone = nn.Sequential(*stages)
        self.out_ch = in_ch

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the final temporal feature map."""
        if self.use_norm:
            x = self.norm(x)
        x = self.c_squeeze(x)
        return self.backbone(x)


class MSACNN(nn.Module):
    """Direct RT regression CNN with multiscale segment attention pooling."""

    def __init__(
        self,
        n_chans: int,
        n_times: int,
        sfreq: int,
        n_outputs: int = 1,
        c0: int = 32,
        widen: int = 2,
        depth_per_stage: int = 2,
        dropout: float = 0.1,
        k: int = 7,
        use_norm: bool = True,
        segments=(2, 4, 8),
        attn_dim: int = 256,
        head_hidden: int = 128,
        head_dropout: float = 0.1,
        pooling: str = "score",
        num_heads: int = 4,
        attn_layers: int = 0,
        attn_ff_dim: int | None = None,
        attn_dropout: float = 0.1,
        pool_queries: int = 1,
        use_segment_embeddings: bool = False,
    ):
        super().__init__()
        self.backbone = _SegmentBackbone(
            n_chans=n_chans,
            n_times=n_times,
            sfreq=sfreq,
            n_outputs=n_outputs,
            c0=c0,
            widen=widen,
            depth_per_stage=depth_per_stage,
            dropout=dropout,
            k=k,
            use_norm=use_norm,
        )
        self.n_chans = self.backbone.n_chans
        self.n_times = self.backbone.n_times
        self.sfreq = self.backbone.sfreq
        self.n_outputs = self.backbone.n_outputs
        self.attention_pool = MultiscaleSegmentAttentionHead(
            in_ch=self.backbone.out_ch,
            n_outputs=n_outputs,
            segments=segments,
            attn_dim=attn_dim,
            head_hidden=head_hidden,
            dropout=head_dropout,
            pooling=pooling,
            num_heads=num_heads,
            attn_layers=attn_layers,
            attn_ff_dim=attn_ff_dim,
            attn_dropout=attn_dropout,
            pool_queries=pool_queries,
            use_segment_embeddings=use_segment_embeddings,
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the final temporal feature map."""
        return self.backbone.forward_features(x)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """Return direct scalar prediction."""
        features = self.forward_features(x)
        return self.attention_pool(features, return_attention=return_attention)


class MSGCNN(nn.Module):
    """Direct RT regression CNN with multiscale segment gated pooling."""

    def __init__(
        self,
        n_chans: int,
        n_times: int,
        sfreq: int,
        n_outputs: int = 1,
        c0: int = 32,
        widen: int = 2,
        depth_per_stage: int = 2,
        dropout: float = 0.1,
        k: int = 7,
        use_norm: bool = True,
        segments=(2, 4),
        token_dim: int = 256,
        head_hidden: int = 128,
        head_dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = _SegmentBackbone(
            n_chans=n_chans,
            n_times=n_times,
            sfreq=sfreq,
            n_outputs=n_outputs,
            c0=c0,
            widen=widen,
            depth_per_stage=depth_per_stage,
            dropout=dropout,
            k=k,
            use_norm=use_norm,
        )
        self.n_chans = self.backbone.n_chans
        self.n_times = self.backbone.n_times
        self.sfreq = self.backbone.sfreq
        self.n_outputs = self.backbone.n_outputs
        self.gated_pool = MultiscaleSegmentGatedHead(
            in_ch=self.backbone.out_ch,
            n_outputs=n_outputs,
            segments=segments,
            token_dim=token_dim,
            head_hidden=head_hidden,
            dropout=head_dropout,
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the final temporal feature map."""
        return self.backbone.forward_features(x)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """Return direct scalar prediction."""
        features = self.forward_features(x)
        return self.gated_pool(features, return_attention=return_attention)
