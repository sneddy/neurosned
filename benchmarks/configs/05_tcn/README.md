# 05 TCN

ETS-TCN event-time segmentation configs for architecture-control runs. The data
protocol, optimizer, augmentation, evaluation, and repeated-seed settings mirror
`02_segmentation_ablations`. Temperature calibration uses the wider `0.2..3.5`
readout grid to avoid architecture-specific clipping.

## Configs

| config | role |
| --- | --- |
| `ets_tcn_ce.yaml` | Soft-label event-time CE objective. |
| `ets_tcn_time_only.yaml` | Scalar soft-argmax loss control without distributional supervision. |
| `ets_tcn_event_nll_mixture.yaml` | Two-scale Gaussian observation-kernel EventNLL objective. |

## Model

The TCN model keeps the segmentation contract `(B, C, T) -> (B, 1, T)` but
replaces the U-Net encoder-decoder with a full-resolution stack of dilated
temporal residual blocks. The configured `c0=384, depth=10` setting is
capacity-matched to the main ETS-U-Net configuration at roughly three million
trainable parameters.
