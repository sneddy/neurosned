# Benchmark Experiments

Current paper-facing experiment journal for the support-filtered protocol. The
previous unfiltered-protocol journal is archived at
`benchmarks/archive/unfiltered_protocol/EXPERIMENTS.md`.

Status date: 2026-07-07.

## Current Scope

The current benchmark is aligned with the manuscript draft in
`writing/overleaf/release_v3/main_revised_v3.tex`.

Main paper story:

1. Establish scalar RT baselines on fixed 2 s EEG windows.
2. Reframe RT prediction as event-time distribution modeling.
3. Compare ETS-U-Net event-time objectives under one backbone and protocol.
4. Analyze posterior geometry to show that scalar nRMSE hides output semantics.
5. Use shifted-crop inference as a shortcut-vs-localization diagnostic.
6. Add seed robustness as reliability support, not as a separate claim.
7. Keep stacking and shift-jitter training as optional add-ons until current
   filtered-protocol artifacts exist.

## Protocol

| component | value |
| --- | --- |
| train split | R1-R8 |
| development split | R9-R10 |
| final labeled holdout | R11 |
| target support | `0.5 <= RT <= 2.5` |
| metric | nRMSE, normalized within split |
| repeated seeds | 2025, 2026, 2027, 2028, 2029 |
| confidence interval | subject bootstrap for per-run holdout predictions |

The target-support filter is applied at dataset construction through
`ExperimentConfig.build_dataset()`. The raw pickle files are not regenerated for
this protocol.

## Config Groups

| group | path | role | status |
| --- | --- | --- | --- |
| Regression baselines | `benchmarks/configs/01_regression_baselines/` | Scalar RT and wrapped external EEG baselines. | Complete: 12/12 configs, 5 seeds each. |
| Segmentation ablations | `benchmarks/configs/02_segmentation_ablations/` | ETS-U-Net event-time objective comparison. | Running: CE and EventNLL complete; time-only started. |

## Artifact Registry

| artifact | canonical path | role | status |
| --- | --- | --- | --- |
| Regression repeated runs | `benchmarks/experiments/01_regression_baselines/` | Main scalar baseline table. | Complete: 12/12 configs, 5 seeds each. |
| Regression leaderboard | `benchmarks/experiments/regression_leaderboard.md` | Camera-ready scalar baseline table. | Complete. |
| Segmentation repeated runs | `benchmarks/experiments/02_segmentation_ablations/` | Event-time objective table and shifted-crop summaries. | Running: CE and EventNLL complete; time-only started. |
| Segmentation leaderboard | `benchmarks/experiments/segmentation_leaderboard.md` | Current event-time objective table with scalar and shifted-crop metrics. | Running. |
| Old unfiltered protocol | `benchmarks/archive/unfiltered_protocol/` | Historical reference only. | Archived. |

## 01 Regression Baselines

Runner:

```bash
sh /home/sneddy/sneddy_projects/neurosned/benchmarks/runners/run_regression_baselines.sh
```

Paper-facing configs:

| config | paper role |
| --- | --- |
| `msp_cnn.yaml` | Compact scalar regression baseline. |
| `etr_cnn.yaml` | Temporal-readout scalar regression baseline. |
| `etr_cnn_large.yaml` | Capacity ablation for ETR-CNN. |
| `tidnet_wrapped.yaml` | Strong wrapped external supervised baseline. |
| `eegconformer_wrapped.yaml` | Transformer-style supervised EEG baseline. |
| `eegnet_wrapped.yaml` | Canonical compact EEG baseline. |
| `deep4net_wrapped.yaml` | Classical convolutional EEG baseline. |
| `shallowfbcspnet_wrapped.yaml` | Classical shallow FBCSP-style baseline. |
| `atcnet_wrapped.yaml` | Supervised conv/attention/TCN baseline. |
| `labram_wrapped.yaml` | Foundation-style architecture from scratch. |
| `eegpt_wrapped.yaml` | Foundation-style architecture from scratch. |
| `medformer_wrapped.yaml` | Larger transformer/time-series baseline. |

Current repeated-run snapshot is also maintained as a standalone leaderboard at
`benchmarks/experiments/regression_leaderboard.md`.

Final 5-seed rows:

| model | seeds | valid nRMSE mean +/- std | R11 nRMSE mean +/- std | R11 range |
| --- | ---: | ---: | ---: | ---: |
| `etr_cnn_large` | 5/5 | 0.8972 +/- 0.0040 | 0.8928 +/- 0.0042 | 0.8873-0.8977 |
| `etr_cnn` | 5/5 | 0.9008 +/- 0.0060 | 0.8977 +/- 0.0068 | 0.8922-0.9085 |
| `msp_cnn` | 5/5 | 0.9006 +/- 0.0051 | 0.8998 +/- 0.0080 | 0.8927-0.9112 |
| `tidnet_wrapped` | 5/5 | 0.9235 +/- 0.0024 | 0.9192 +/- 0.0027 | 0.9146-0.9219 |
| `deep4net_wrapped` | 5/5 | 0.9269 +/- 0.0045 | 0.9260 +/- 0.0044 | 0.9210-0.9309 |
| `eegconformer_wrapped` | 5/5 | 0.9188 +/- 0.0026 | 0.9287 +/- 0.0057 | 0.9225-0.9342 |
| `labram_wrapped` | 5/5 | 0.9304 +/- 0.0061 | 0.9327 +/- 0.0086 | 0.9247-0.9462 |
| `eegnet_wrapped` | 5/5 | 0.9350 +/- 0.0054 | 0.9335 +/- 0.0028 | 0.9292-0.9370 |
| `shallowfbcspnet_wrapped` | 5/5 | 0.9324 +/- 0.0016 | 0.9343 +/- 0.0024 | 0.9316-0.9372 |
| `eegpt_wrapped` | 5/5 | 0.9616 +/- 0.0201 | 0.9584 +/- 0.0185 | 0.9420-0.9878 |
| `medformer_wrapped` | 5/5 | 0.9623 +/- 0.0046 | 0.9585 +/- 0.0051 | 0.9542-0.9674 |
| `atcnet_wrapped` | 5/5 | 0.9686 +/- 0.0183 | 0.9666 +/- 0.0151 | 0.9533-0.9920 |

Interpretation so far:

- `etr_cnn_large` is the strongest completed scalar baseline.
- `etr_cnn` improves over `msp_cnn`, but the gap is modest relative to seed
  variation.
- Wrapped external architectures train meaningfully under the fixed
  normalization protocol but are currently weaker than the compact scalar
  models.
- LaBraM is the strongest foundation-style wrapped baseline in this filtered
  protocol, while EEGPT and Medformer remain weaker than the compact
  task-specific scalar baselines.

## 02 Segmentation Ablations

Paper-facing configs:

| config | paper name | role |
| --- | --- | --- |
| `ets_unet_ce.yaml` | ETS-U-Net CE | Soft-label event-time CE baseline. |
| `ets_unet_ce_time.yaml` | ETS-U-Net CE+time | Hybrid CE plus soft-argmax time loss. |
| `ets_unet_event_nll.yaml` | ETS-U-Net EventNLL | Latent event-time likelihood with Gaussian observation kernel. |
| `ets_unet_event_nll_mixture.yaml` | ETS-U-Net mixture EventNLL | Two-scale Gaussian observation-kernel extension. |
| `ets_unet_hazard_event_nll.yaml` | ETS-U-Net hazard EventNLL | Hazard/survival posterior parameterization with EventNLL. |
| `ets_unet_time_only.yaml` | ETS-U-Net time-only | Scalar soft-argmax control without distributional supervision. |
| `ets_unet_wasserstein.yaml` | ETS-U-Net Wasserstein | CDF-distance control objective. |

All segmentation configs currently include:

| component | value |
| --- | --- |
| train pickle | `data/new_validation/r1_r8_train.pkl` |
| valid pickle | `data/new_validation/r9_r10_val.pkl` |
| test pickle | `data/new_validation/r11_test.pkl` |
| target support | `0.5 <= RT <= 2.5` |
| repeated seeds | 2025-2029 |
| temperature calibration | enabled |
| shifted-crop diagnostic | enabled |
| shifted-crop dataset | `data/new_validation/r11_test_5sec.pkl` |
| shifted-crop subset | `0.8 <= RT <= 2.2` |
| shifted-crop starts | `0.2, 0.3, ..., 0.8` |
| shifted-crop per-trial predictions | disabled by default |

Current repeated-run snapshot is also maintained as a standalone leaderboard at
`benchmarks/experiments/segmentation_leaderboard.md`.

Current rows:

| model | seeds | valid nRMSE mean +/- std | R11 nRMSE mean +/- std | R11 tau nRMSE mean +/- std | shift slope mean +/- std | localizer-like mean +/- std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ets_unet_ce` | 5/5 | 0.8763 +/- 0.0044 | 0.8774 +/- 0.0044 | 0.8753 +/- 0.0039 | -0.339 +/- 0.017 | 0.209 +/- 0.031 |
| `ets_unet_event_nll` | 5/5 | 0.8769 +/- 0.0030 | 0.8805 +/- 0.0021 | 0.8772 +/- 0.0018 | -0.344 +/- 0.014 | 0.221 +/- 0.026 |
| `ets_unet_time_only` | 0/5 (+1 marked running) | - | - | - | - | - |
| `ets_unet_ce_time` | pending | - | - | - | - | - |
| `ets_unet_wasserstein` | pending | - | - | - | - | - |
| `ets_unet_event_nll_mixture` | pending | - | - | - | - | - |
| `ets_unet_hazard_event_nll` | pending | - | - | - | - | - |

Interpretation so far:

- CE and EventNLL both outperform the strongest completed scalar regression
  baseline on R11 after temperature calibration.
- CE is currently the best scalar readout row; EventNLL is very close in scalar
  error and slightly more localizer-like in the shifted-crop diagnostic.
- Both completed event-time rows move in a more localizer-like direction than a
  crop-invariant shortcut, but neither is close to ideal shift equivariance.
- Time-only, Wasserstein, CE+time, mixture EventNLL, and hazard EventNLL are
  still needed before the loss-ablation story is complete.

Expected paper use:

- Main segmentation table: CE, CE+time, EventNLL, time-only, Wasserstein,
  mixture EventNLL, hazard EventNLL.
- Appendix/internal only unless needed: Laplace EventNLL, Student-t EventNLL,
  heteroscedastic EventNLL, exact-bin hazard NLL. These remain in the archived
  unfiltered protocol and should not be mixed into the new main table without
  rerunning under the filtered protocol.

## Posterior Geometry

Posterior-geometry figures and tables must be regenerated after the filtered
segmentation reruns. Old figures under the archive remain useful only for
design and wording.

Paper-facing metrics to preserve:

- scalar nRMSE and MAE;
- CRPS;
- fixed-kernel EventNLL under a common Gaussian observation kernel;
- posterior Width80;
- target-aligned mass within +/-150 ms;
- mode-mean gap;
- empirical interval coverage and coverage MAE.

## Shifted-Crop Diagnostic

The shifted-crop diagnostic is now configured directly inside the segmentation
configs. It should also be rerun for the final scalar baselines after regression
training finishes.

Main interpretation to preserve:

- Ideal crop-relative localizer: raw shift slope near `-1`.
- Crop-invariant shortcut: raw shift slope near `0`.
- Current claim should remain diagnostic, not solved localization.

## Optional Add-Ons

Keep these out of the main experimental spine until filtered-protocol artifacts
exist:

| add-on | purpose | current status |
| --- | --- | --- |
| Shift-jitter training | Test whether random crop shifts improve localization/equivariance. | Planned. |
| Distribution-aware stacking | Test whether posterior/logit features add reusable information beyond scalar predictions. | Planned. |
| Two-stage checkpoint reload recipe | Potential training-stability/final recipe. | Code path exists, not a current main claim. |

## Immediate Next Steps

1. Add a segmentation runner for `benchmarks/configs/02_segmentation_ablations/`
   or run the seven configs manually with `run_repeated.py`.
2. Recompute posterior geometry from filtered segmentation predictions/logits.
3. Recompute shifted-crop comparison from filtered scalar and segmentation
   runs.
4. Update `writing/overleaf/release_v4/main_revised_v4.tex` only after the
   corresponding filtered-protocol artifacts are complete.
