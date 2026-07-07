# 02 Segmentation Ablations

Paper-facing ETS-U-Net event-time segmentation configs for the support-filtered
protocol. Exploratory and negative kernel probes from the earlier unfiltered
protocol remain archived under `benchmarks/archive/unfiltered_protocol/`.

## Configs

| config | paper name | role |
| --- | --- | --- |
| `ets_unet_ce.yaml` | ETS-U-Net CE | Soft-label event-time CE baseline. |
| `ets_unet_event_nll.yaml` | ETS-U-Net EventNLL | Latent event-time likelihood with a Gaussian observation kernel. |
| `ets_unet_event_nll_mixture.yaml` | ETS-U-Net mixture EventNLL | Two-scale Gaussian observation-kernel extension. |
| `ets_unet_hazard_event_nll.yaml` | ETS-U-Net hazard EventNLL | Hazard/survival posterior parameterization with continuous EventNLL. |
| `ets_unet_time_only.yaml` | ETS-U-Net time-only | Scalar soft-argmax loss control without distributional supervision. |
| `ets_unet_wasserstein.yaml` | ETS-U-Net Wasserstein | CDF-distance control objective. |

Laplace, Student-t, heteroscedastic EventNLL, and exact-bin hazard NLL remain
appendix/internal probes unless they become necessary for reviewer-facing
robustness.

## Protocol

All configs use:

| component | value |
| --- | --- |
| train split | R1-R8 fixed 2 s windows |
| valid split | R9-R10 fixed 2 s windows |
| test split | R11 fixed 2 s windows |
| target support | `0.5 <= RT <= 2.5` |
| main training window | fixed `0.5-2.5 s` post-stimulus window |
| train batch size | 128 |
| optimizer | Adam, lr `1e-3`, weight decay `0` |
| early stopping | validation NRMSE, patience 20 |
| repeated seeds | 2025, 2026, 2027, 2028, 2029 |
| holdout evaluation | enabled |
| temperature calibration | enabled on R9-R10 logits |
| shifted-crop diagnostic | enabled on `r11_test_5sec.pkl` |
| shifted-crop subset | `0.8 <= RT <= 2.2` common-inside trials |
| shifted-crop starts | `0.2, 0.3, ..., 0.8` s |
| shifted-crop predictions | summary only by default; per-trial CSV disabled |

These configs are intended to answer the focused paper question: which
event-time supervision objective gives the best scalar readout and posterior
geometry under the same ETS-U-Net backbone and data protocol.
