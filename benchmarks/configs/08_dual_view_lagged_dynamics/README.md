# 08 Dual-View Lagged Dynamics

Independent regression experiments that preserve a conventional raw EEG view
alongside the local covariance and lagged-dynamics view from experiment 07.
Nothing in `07_lagged_dynamics` is modified by this experiment group.

## Architecture

The two paths split immediately after per-channel input normalization:

- **raw path:** a full-resolution temporal CNN processes all 128 EEG channels
  before attention pooling over the seven overlapping 0.5 s intervals;
- **matrix path:** the existing learned 24-dimensional projection, shrinkage
  covariance, lagged correlation, and ridge-transition operators produce one
  token for each matching interval.

Both paths produce `(B, 7, 384)` tokens. A learned two-way modality gate is
initialized to equal weights, followed by a residual interaction layer and the
same segment-level TCN/readout used by the matrix-only architecture. Detailed
forward output exposes raw sample attention, modality attention, matrix tokens,
raw tokens, and fused tokens.

## Prespecified comparisons

| config | views | purpose | parameters |
| --- | --- | --- | ---: |
| `raw_view_only.yaml` | raw EEG | Ordinary temporal-network control. | 2.35M |
| `dual_view_covariance_only.yaml` | raw + covariance | Tests whether local zero-lag SPD structure adds information to raw waveforms. | 3.34M |
| `dual_view_full.yaml` | raw + covariance + lagged correlation + transition | Tests whether delayed dynamics add information after the raw view is retained. | 3.85M |

The existing `07_lagged_dynamics/lagged_dynamics_full.yaml` remains the
matrix-only counterpart. Together the groups support the main contrasts:

1. dual full versus matrix-only full: incremental value of raw waveforms;
2. dual full versus raw-only: incremental value of matrix dynamics;
3. dual full versus dual covariance-only: incremental value of selected lags;
4. dual covariance-only versus raw-only: incremental value of zero-lag SPD
   structure.

All configs retain the same release-separated data, augmentation, optimizer,
trainer, five seeds, holdout evaluation, and shifted-crop diagnostic as the
completed scalar-regression baseline protocol.

## Run

```bash
DEVICE=cuda sh benchmarks/runners/run_dual_view_lagged_dynamics.sh
```

After completion, the runner creates
`benchmarks/experiments/08_dual_view_lagged_dynamics/comparison.md` with the
available baseline, matrix-only, raw-only, and dual-view rows plus paired-seed
contrasts.
