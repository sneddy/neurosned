# 06 Inception Pyramid

ETS-InceptionPyramid event-time segmentation configs for architecture-control
runs. The data protocol, optimizer, augmentation, evaluation, and repeated-seed
settings mirror `02_segmentation_ablations`. Temperature calibration uses the
wider `0.2..3.5` readout grid to avoid architecture-specific clipping.

## Configs

| config | role |
| --- | --- |
| `ets_inception_pyramid_ce.yaml` | Soft-label event-time CE objective. |
| `ets_inception_pyramid_time_only.yaml` | Scalar soft-argmax loss control without distributional supervision. |
| `ets_inception_pyramid_event_nll_mixture.yaml` | Two-scale Gaussian observation-kernel EventNLL objective. |

## Model

The InceptionPyramid model keeps the segmentation contract
`(B, C, T) -> (B, 1, T)` but replaces the U-Net encoder-decoder with a
full-resolution temporal scale bank. Each block applies parallel temporal
filters at multiple receptive-field scales, fuses them at the original time
resolution, and refines the result with residual temporal convolutions. The
configured `c0=288, branch_ch=96, depth=6` setting is intended as a competitive
multi-scale architecture-control run without temporal pooling or decoder skips.
