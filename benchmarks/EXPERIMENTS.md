# Benchmark Experiments

Current paper-facing experiment journal for the support-filtered protocol. The
previous unfiltered-protocol journal is archived at
`benchmarks/archive/unfiltered_protocol/EXPERIMENTS.md`.

Status date: 2026-07-07.

## Current Scope

The current benchmark is aligned with the manuscript draft in
`writing/overleaf/release_v4/main_revised_v4.tex`.

Main paper story:

1. Establish scalar RT baselines on fixed 2 s EEG windows.
2. Reframe RT prediction as event-time distribution modeling.
3. Compare ETS-U-Net event-time objectives under one backbone and protocol.
4. Analyze posterior geometry to show that scalar nRMSE hides output semantics.
5. Use shifted-crop inference as a shortcut-vs-localization diagnostic.
6. Test whether shift-jitter training can turn the diagnostic into a stronger
   localization result.
7. Keep stacking and final two-stage recipes as optional add-ons until current
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

Metric conventions:

- `nRMSE`: RMSE divided by the target standard deviation inside the evaluated
  split and support filter.
- `tau nRMSE`: nRMSE after validation-tuned softmax temperature calibration for
  event-time posteriors; this is evaluation-time calibration, not retraining.
- `shift slope`: slope of raw crop-relative prediction versus crop start in the
  shifted-crop diagnostic. A crop-relative localizer should be near `-1`; a
  crop-invariant shortcut should be near `0`.
- `localizer-like`: fraction of trials with per-trial raw shift slope in
  `[-1.25, -0.75]`.
- Posterior metrics are interpreted as output-geometry diagnostics, not only as
  scalar prediction quality.

## Config Groups

| group | path | role | status |
| --- | --- | --- | --- |
| Data protocol | `benchmarks/experiments/00_data_protocol/` | Split, support-filter, subject-disjointness, and package-version record. | Complete. |
| Regression baselines | `benchmarks/configs/01_regression_baselines/` | Scalar RT and wrapped external EEG baselines. | Complete: 12/12 configs, 5 seeds each. |
| Segmentation ablations | `benchmarks/configs/02_segmentation_ablations/` | ETS-U-Net event-time objective comparison. | Complete for selected paper-facing objectives. |
| Posterior geometry | generated from `02_segmentation_ablations` outputs | Distributional output semantics beyond scalar nRMSE. | Pending filtered-protocol regeneration. |
| Shifted-crop diagnostic | `evaluation.shifted_crop` in run configs plus post-hoc regression runner | Shortcut-vs-localization stress test. | Complete for scalar baselines; running for segmentation. |
| Shift-jitter training | `benchmarks/configs/03_crop_shift_jitter/` | Training-time removal of fixed-window shortcuts. | Configs ready, not run. |
| Final training recipe | planned config group | Optional final model recipe after ablations are fixed. | Planned. |
| Distribution-aware stacking | planned artifacts under stacking experiments | Test whether posterior/logit features add reusable information. | Planned. |

## Artifact Registry

| artifact | canonical path | role | status |
| --- | --- | --- | --- |
| Data protocol summary | `benchmarks/experiments/00_data_protocol/protocol_summary.md` | Canonical split/support table for the filtered protocol. | Complete. |
| Regression repeated runs | `benchmarks/experiments/01_regression_baselines/` | Main scalar baseline table. | Complete: 12/12 configs, 5 seeds each. |
| Regression leaderboard | `benchmarks/experiments/paper_tables/regression_leaderboard.md` | Camera-ready scalar baseline table. | Complete. |
| Regression shifted-crop table | `benchmarks/experiments/paper_tables/regression_shifted_crop.md` | Scalar baseline shortcut/localization diagnostic. | Complete: 60/60 seed-runs. |
| Segmentation repeated runs | `benchmarks/experiments/02_segmentation_ablations/` | Event-time objective table and shifted-crop summaries. | Complete for selected paper-facing objectives. |
| Segmentation leaderboard | `benchmarks/experiments/paper_tables/segmentation_leaderboard.md` | Current event-time objective table with scalar and shifted-crop metrics. | Complete for selected paper-facing objectives. |
| Segmentation shifted-crop table | `benchmarks/experiments/paper_tables/segmentation_shifted_crop.md` | Event-time shortcut/localization diagnostic. | Running: 30 seed-runs available. |
| Shift-jitter repeated runs | `benchmarks/experiments/03_crop_shift_jitter/` | Jitter-trained event-time localization test. | Planned. |
| Posterior geometry figures | TBD under `benchmarks/experiments/02_segmentation_ablations/figures/` | Camera-ready posterior profile panels. | Pending filtered-protocol regeneration. |
| Shifted-crop summaries | per-run `shifted_eval/` folders | Crop-start robustness and localization diagnostics. | Complete for scalar baselines; running for segmentation. |
| Stacking artifacts | TBD | Competition-style ensemble/calibration reproduction. | Planned. |

## 00 Data Protocol

Canonical artifact:
`benchmarks/experiments/00_data_protocol/protocol_summary.md`.

Narrative role:

- Defines the release-separated protocol before any model comparison.
- Shows that train, development, and R11 holdout are subject-disjoint.
- Makes the `0.5 <= RT <= 2.5` support filter explicit, matching the modeled
  fixed-window event-time support.
- Separates raw pickle preparation from analysis filtering: the raw files stay
  unchanged, while `ExperimentConfig.build_dataset()` applies the paper-facing
  target filter.

Paper-facing metrics:

| metric | use |
| --- | --- |
| prepared trials | Raw available CCD windows before support filtering. |
| analyzed trials | Trials included in model fitting/evaluation after support filtering. |
| subject overlap | Leakage check across release-separated splits. |
| shifted-crop common-inside subset | Trials where all shifted 2 s crops can contain the RT. |

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
`benchmarks/experiments/paper_tables/regression_leaderboard.md`.

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

Runner:

```bash
sh /home/sneddy/sneddy_projects/neurosned/benchmarks/runners/run_segmentation_ablations.sh
```

Paper-facing configs:

| config | paper name | role |
| --- | --- | --- |
| `ets_unet_ce.yaml` | ETS-U-Net CE | Soft-label event-time CE baseline. |
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
`benchmarks/experiments/paper_tables/segmentation_leaderboard.md`.

Current rows:

| model | seeds | valid nRMSE mean +/- std | R11 nRMSE mean +/- std | R11 tau nRMSE mean +/- std | shift slope mean +/- std | localizer-like mean +/- std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ets_unet_ce` | 5/5 | 0.8763 +/- 0.0044 | 0.8774 +/- 0.0044 | 0.8753 +/- 0.0039 | -0.339 +/- 0.017 | 0.209 +/- 0.031 |
| `ets_unet_event_nll` | 5/5 | 0.8769 +/- 0.0030 | 0.8805 +/- 0.0021 | 0.8772 +/- 0.0018 | -0.344 +/- 0.014 | 0.221 +/- 0.026 |
| `ets_unet_event_nll_mixture` | 5/5 | 0.8744 +/- 0.0018 | 0.8785 +/- 0.0047 | 0.8745 +/- 0.0053 | -0.355 +/- 0.024 | 0.246 +/- 0.042 |
| `ets_unet_hazard_event_nll` | 5/5 | 0.8755 +/- 0.0027 | 0.8776 +/- 0.0031 | 0.8778 +/- 0.0041 | -0.328 +/- 0.020 | 0.196 +/- 0.042 |
| `ets_unet_time_only` | 5/5 | 0.8944 +/- 0.0048 | 0.8943 +/- 0.0025 | 0.8917 +/- 0.0046 | -0.301 +/- 0.043 | 0.160 +/- 0.066 |
| `ets_unet_wasserstein` | 5/5 | 0.8997 +/- 0.0035 | 0.8995 +/- 0.0078 | 0.8896 +/- 0.0033 | -0.337 +/- 0.044 | 0.271 +/- 0.065 |

Interpretation so far:

- CE and EventNLL both outperform the strongest completed scalar regression
  baseline on R11 after temperature calibration.
- Mixture EventNLL has the best calibrated scalar readout so far
  (`test_tau_nRMSE = 0.8745`) and the most localizer-like shift slope among the
  likelihood-style objectives.
- CE remains the simplest strong row; hazard EventNLL is competitive but does
  not beat mixture/CE cleanly.
- Time-only is clearly weaker than CE/EventNLL, which supports the claim that
  distributional event-time supervision matters beyond the soft-argmax scalar
  readout.
- Wasserstein has the highest completed localizer-like fraction so far, but it
  pays for that with worse scalar nRMSE and worse shifted-crop nRMSE.

Expected paper use:

- Main segmentation table: CE, EventNLL, mixture EventNLL, hazard EventNLL,
  time-only, Wasserstein.
- Appendix/internal only unless needed: Laplace EventNLL, Student-t EventNLL,
  heteroscedastic EventNLL, exact-bin hazard NLL. These remain in the archived
  unfiltered protocol and should not be mixed into the new main table without
  rerunning under the filtered protocol.

## 03 Posterior Geometry

Posterior-geometry figures and tables must be regenerated after the filtered
segmentation reruns. Old figures under the archive remain useful only for
design and wording.

Narrative role:

- Show that similar scalar RT errors can come from very different temporal
  posterior profiles.
- Make the event-time formulation visibly different from scalar regression:
  the output is not only a number, but a probability distribution over when the
  response-relevant event occurs.
- Separate localization-like evidence from broad calibrated uncertainty.
  EventNLL is expected to be sharper; CE is expected to be broader and often
  better calibrated; time-only is expected to recover the scalar readout with a
  weaker distributional semantics.

Paper-facing metrics:

| metric | use |
| --- | --- |
| scalar nRMSE / MAE | Scalar readout quality from the posterior. |
| CRPS | Distributional score sensitive to both location and spread. |
| fixed-kernel EventNLL | Common likelihood diagnostic under one Gaussian observation kernel. |
| posterior Width80 | Concentration of the central posterior mass. |
| target-aligned mass within +/-150 ms | How much posterior mass lands near the observed RT. |
| mode-mean gap | Multimodality/asymmetry diagnostic. |
| empirical interval coverage | Calibration of posterior credible intervals. |
| coverage MAE | Distance from nominal coverage across intervals. |

Expected artifact shape:

- target-aligned average posterior curves by objective;
- representative trial posteriors;
- posterior width and coverage tables;
- trial-sorted posterior raster if the filtered predictions/logits are saved.

## 04 Shifted-Crop Diagnostic

The shifted-crop diagnostic is now configured directly inside the segmentation
configs and has been run post-hoc for all completed scalar regression baselines
with `benchmarks/runners/eval_shifted_regression.sh`.

Narrative role:

- Test whether models truly locate response-related evidence inside the window
  or exploit a fixed stimulus-locked crop shortcut.
- A scalar regressor can perform well when the crop always starts at the same
  time after stimulus, because it can learn subject/trial slowness and
  stimulus-locked structure without being forced to localize.
- A genuine crop-relative localizer should move its predicted time when the
  crop start moves.

Paper-facing metrics:

| metric | expected meaning |
| --- | --- |
| per-start nRMSE / MAE | Whether scalar accuracy collapses for shifted crops. |
| raw shift slope | `-1` means crop-relative localization; `0` means crop-invariant shortcut. |
| corrected shift slope | Slope after converting crop-relative prediction back to absolute time; `0` is ideal. |
| MAE to localizer slope | Distance from ideal raw slope `-1`. |
| MAE to stable corrected slope | Distance from ideal absolute-time stability. |
| localizer-like fraction | Fraction of trials with raw slope in `[-1.25, -0.75]`. |
| invariant-like fraction | Fraction of trials with raw slope in `[-0.25, 0.25]`. |
| wrong-direction fraction | Fraction of trials where raw slope is positive. |

Current interpretation constraints:

- Ideal crop-relative localizer: raw shift slope near `-1`.
- Crop-invariant shortcut: raw shift slope near `0`.
- Current claim should remain diagnostic, not solved localization, unless
  shift-jitter training materially improves the slope metrics.

Segmentation shifted-crop snapshot:
`benchmarks/experiments/paper_tables/segmentation_shifted_crop.md`.

| model | seeds | ref nRMSE @0.5 | mean shifted nRMSE | worst shifted nRMSE | raw shift slope | localizer-like | invariant-like |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ets_unet_ce` | 5/5 | 0.8593 +/- 0.0071 | 0.9470 +/- 0.0087 | 1.0655 +/- 0.0135 | -0.339 +/- 0.017 | 0.209 +/- 0.031 | 0.286 +/- 0.016 |
| `ets_unet_event_nll` | 5/5 | 0.8656 +/- 0.0049 | 0.9514 +/- 0.0063 | 1.0616 +/- 0.0129 | -0.344 +/- 0.014 | 0.221 +/- 0.026 | 0.292 +/- 0.016 |
| `ets_unet_event_nll_mixture` | 5/5 | 0.8623 +/- 0.0070 | 0.9520 +/- 0.0077 | 1.0678 +/- 0.0151 | -0.355 +/- 0.024 | 0.246 +/- 0.042 | 0.290 +/- 0.026 |
| `ets_unet_hazard_event_nll` | 5/5 | 0.8605 +/- 0.0065 | 0.9434 +/- 0.0086 | 1.0466 +/- 0.0152 | -0.328 +/- 0.020 | 0.196 +/- 0.042 | 0.305 +/- 0.024 |
| `ets_unet_time_only` | 5/5 | 0.8802 +/- 0.0034 | 0.9563 +/- 0.0113 | 1.0531 +/- 0.0224 | -0.301 +/- 0.043 | 0.160 +/- 0.066 | 0.323 +/- 0.046 |
| `ets_unet_wasserstein` | 5/5 | 0.8851 +/- 0.0169 | 0.9810 +/- 0.0217 | 1.1083 +/- 0.0327 | -0.337 +/- 0.044 | 0.271 +/- 0.065 | 0.270 +/- 0.045 |

Segmentation interpretation:

- CE/EventNLL/mixture/hazard are the strongest completed scalar event-time rows
  at the canonical crop and remain better than the scalar regression baselines.
- Segmentation is more localizer-like than regression overall, but it is still
  far from the ideal raw shift slope of `-1`.
- Time-only underperforms CE/EventNLL, supporting the value of distributional
  event-time supervision.
- Mixture EventNLL gives the strongest likelihood-style shift slope; Wasserstein
  improves localizer-like fraction, but worsens scalar and shifted nRMSE.

Regression shifted-crop snapshot:
`benchmarks/experiments/paper_tables/regression_shifted_crop.md`.

| model | seeds | ref nRMSE @0.5 | mean shifted nRMSE | worst shifted nRMSE | raw shift slope | localizer-like | invariant-like |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `etr_cnn_large` | 5/5 | 0.8734 +/- 0.0046 | 0.9503 +/- 0.0131 | 1.0574 +/- 0.0263 | -0.294 +/- 0.029 | 0.139 +/- 0.048 | 0.316 +/- 0.054 |
| `etr_cnn` | 5/5 | 0.8799 +/- 0.0095 | 0.9536 +/- 0.0120 | 1.0475 +/- 0.0163 | -0.277 +/- 0.011 | 0.118 +/- 0.025 | 0.330 +/- 0.027 |
| `msp_cnn` | 5/5 | 0.8891 +/- 0.0129 | 0.9636 +/- 0.0218 | 1.0670 +/- 0.0386 | -0.299 +/- 0.029 | 0.149 +/- 0.063 | 0.315 +/- 0.043 |
| `tidnet_wrapped` | 5/5 | 0.9106 +/- 0.0042 | 0.9654 +/- 0.0028 | 1.0438 +/- 0.0098 | -0.210 +/- 0.014 | 0.075 +/- 0.014 | 0.466 +/- 0.023 |
| `eegnet_wrapped` | 5/5 | 0.9159 +/- 0.0034 | 0.9638 +/- 0.0045 | 1.0370 +/- 0.0126 | -0.173 +/- 0.009 | 0.043 +/- 0.011 | 0.511 +/- 0.009 |
| `deep4net_wrapped` | 5/5 | 0.9177 +/- 0.0058 | 0.9714 +/- 0.0056 | 1.0420 +/- 0.0094 | -0.208 +/- 0.018 | 0.083 +/- 0.016 | 0.462 +/- 0.019 |
| `eegconformer_wrapped` | 5/5 | 0.9242 +/- 0.0107 | 0.9799 +/- 0.0166 | 1.0621 +/- 0.0201 | -0.218 +/- 0.030 | 0.041 +/- 0.025 | 0.365 +/- 0.053 |
| `shallowfbcspnet_wrapped` | 5/5 | 0.9243 +/- 0.0018 | 0.9677 +/- 0.0044 | 1.0249 +/- 0.0116 | -0.168 +/- 0.017 | 0.038 +/- 0.020 | 0.509 +/- 0.033 |
| `labram_wrapped` | 5/5 | 0.9287 +/- 0.0124 | 0.9800 +/- 0.0066 | 1.0529 +/- 0.0117 | -0.187 +/- 0.017 | 0.048 +/- 0.016 | 0.439 +/- 0.031 |
| `eegpt_wrapped` | 5/5 | 0.9503 +/- 0.0245 | 0.9804 +/- 0.0125 | 1.0272 +/- 0.0255 | -0.111 +/- 0.058 | 0.023 +/- 0.026 | 0.654 +/- 0.199 |
| `medformer_wrapped` | 5/5 | 0.9532 +/- 0.0079 | 0.9810 +/- 0.0041 | 1.0199 +/- 0.0114 | -0.095 +/- 0.021 | 0.005 +/- 0.007 | 0.634 +/- 0.095 |
| `atcnet_wrapped` | 5/5 | 0.9633 +/- 0.0169 | 0.9837 +/- 0.0075 | 1.0266 +/- 0.0209 | -0.087 +/- 0.043 | 0.008 +/- 0.015 | 0.771 +/- 0.144 |

Regression interpretation:

- Scalar regression is best at the canonical crop start `0.5` and degrades for
  earlier/later crops.
- Raw shift slopes are mostly between `-0.1` and `-0.3`, much closer to
  crop-invariant behavior than to the localizer ideal of `-1`.
- The completed segmentation rows are also not solved localizers, but CE and
  EventNLL are currently more localizer-like than the regression baselines.

## 05 Shift-Jitter Training

Runner:

```bash
sh /home/sneddy/sneddy_projects/neurosned/benchmarks/runners/run_crop_shift_jitter.sh
```

Planned role:

- Turn the shifted-crop diagnostic into a training intervention.
- Randomize crop starts during training while keeping the target inside the
  crop, so the model cannot rely on one fixed stimulus-aligned window.
- Test the hypothesis that event-time supervision should benefit more from
  jittered crops than pure scalar regression, because the target is explicitly a
  crop-relative event location.

Expected story if it works:

- Scalar nRMSE may stay similar or improve modestly, but shifted-crop slope
  should move closer to the localizer regime.
- Event-time objectives should show stronger gains in raw shift slope,
  corrected slope stability, and localizer-like fraction than scalar-only
  controls.
- This would support the claim that the event-time formulation is useful when
  the protocol actually demands temporal localization.

Paper-facing configs:

| config | paper role |
| --- | --- |
| `ets_unet_ce_shift_jitter.yaml` | Soft-label event-time CE under shift-jitter training. |
| `ets_unet_event_nll_shift_jitter.yaml` | Latent EventNLL under shift-jitter training. |
| `ets_unet_event_nll_mixture_shift_jitter.yaml` | Best current likelihood-style objective under shift-jitter training. |
| `ets_unet_hazard_event_nll_shift_jitter.yaml` | Hazard/survival event-time parameterization under shift-jitter training. |
| `ets_unet_time_only_shift_jitter.yaml` | Soft-argmax scalar control under shift-jitter training. |
| `ets_unet_wasserstein_shift_jitter.yaml` | Wasserstein geometry control under shift-jitter training. |

Protocol details:

| component | value |
| --- | --- |
| train pickle | `data/new_validation/r1_r8_train_5sec.pkl` |
| train wrapper | `TrainCroppingDataset` |
| train/eval support | `0.8 <= RT <= 2.2` |
| crop duration | `2.0` s |
| sampled crop starts | `0.2 <= start <= 0.8` |
| canonical valid/test windows | `data/new_validation/r9_r10_val.pkl`, `data/new_validation/r11_test.pkl` |
| shifted-crop eval | enabled on `data/new_validation/r11_test_5sec.pkl` |

Metrics:

| metric | use |
| --- | --- |
| R11 nRMSE / MAE | Check that jitter does not destroy ordinary scalar accuracy. |
| shifted-crop raw slope | Main localization improvement target. |
| corrected slope | Absolute-time stability after undoing crop start. |
| localizer-like / invariant-like fractions | Interpretable trial-level behavior classes. |
| posterior geometry metrics | Check whether jitter changes sharpness/calibration. |

## 06 Final Training Recipe

Planned role:

- Keep the controlled ablations as the main evidence.
- Only after the main story is fixed, optionally define a final engineering
  recipe using the best objective, checkpoint reload, and any validated
  two-stage training details.
- This section should not replace the ablation logic; it can be a final-model
  recipe or appendix if it gives a clean improvement.

Metrics:

| metric | use |
| --- | --- |
| final R11 nRMSE / MAE | Final scalar performance claim. |
| validation-to-R11 gap | Overfitting/generalization check. |
| calibration temperature and tau nRMSE | Whether final posterior readout benefits from calibration. |
| posterior geometry summary | Confirm final recipe does not only improve scalar readout. |
| shifted-crop diagnostics | Confirm final recipe does not worsen localization behavior. |

## 07 Distribution-Aware Stacking

Planned role:

- Reproduce the competition-style stacking idea after base filtered-protocol
  models are fixed.
- Test whether event-time posterior/logit features contain reusable information
  beyond the scalar mean prediction.
- This should be framed as a downstream utility check, not as the core method.

Candidate feature families:

| feature family | examples |
| --- | --- |
| scalar base predictions | direct regression outputs, posterior mean/mode. |
| posterior shape | width, entropy, mode-mean gap, aligned mass. |
| logits/posterior samples | compact posterior summaries or learned meta-features. |
| calibration features | temperature-calibrated readouts and uncertainty measures. |

Metrics:

| metric | use |
| --- | --- |
| out-of-fold validation nRMSE | Honest meta-model selection. |
| R11 nRMSE / MAE | Final stacking comparison against best single model. |
| delta vs best base model | Whether stacking adds value beyond model selection. |
| feature ablation | Whether posterior features matter beyond scalar predictions. |

## Planned/Optional Scope

Keep these out of the main experimental spine until filtered-protocol artifacts
exist:

| add-on | purpose | current status |
| --- | --- | --- |
| Shift-jitter training | Test whether random crop shifts improve localization/equivariance. | Planned. |
| Distribution-aware stacking | Test whether posterior/logit features add reusable information beyond scalar predictions. | Planned. |
| Two-stage checkpoint reload recipe | Potential training-stability/final recipe. | Code path exists, not a current main claim. |

## Immediate Next Steps

1. Let the remaining segmentation ablations finish through
   `benchmarks/runners/run_segmentation_ablations.sh`.
2. Refresh `benchmarks/experiments/paper_tables/segmentation_leaderboard.md` after each
   completed repeated run.
3. Recompute posterior geometry from filtered segmentation predictions/logits.
4. Recompute shifted-crop comparison for the final scalar regression baselines.
5. Decide whether shift-jitter training is strong enough to become a main
   paper block or should stay as appendix/planned work.
6. Update `writing/overleaf/release_v4/main_revised_v4.tex` only after the
   corresponding filtered-protocol artifacts are complete.
