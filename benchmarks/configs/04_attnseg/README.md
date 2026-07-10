# 04 AttnSeg

ETS-AttnSeg event-time segmentation configs for architecture-control runs.
The data protocol, optimizer, augmentation, evaluation, and repeated-seed
settings mirror `02_segmentation_ablations`. Temperature calibration uses the
wider `0.2..3.5` readout grid to avoid architecture-specific clipping.

## Configs

| config | role |
| --- | --- |
| `ets_attnseg_ce.yaml` | Soft-label event-time CE objective. |
| `ets_attnseg_time_only.yaml` | Scalar soft-argmax loss control without distributional supervision. |
| `ets_attnseg_event_nll_mixture.yaml` | Two-scale Gaussian observation-kernel EventNLL objective. |

## Model

The AttnSeg model keeps the segmentation contract `(B, C, T) -> (B, 1, T)`
but replaces the U-Net encoder-decoder with full-resolution temporal
self-attention and depthwise convolution blocks. The configured `c0=128,
depth=10` setting is capacity-matched to the main ETS-U-Net configuration
at roughly three million trainable parameters.
