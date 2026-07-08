# Benchmark Experiments

Current paper-facing experiment journal for the support-filtered protocol. The
previous unfiltered-protocol journal is archived at
`benchmarks/archive/unfiltered_protocol/EXPERIMENTS.md`.

Status date: 2026-07-08.

## Current Scope

The current benchmark is aligned with the manuscript draft in
`writing/overleaf/release_v7/main_revised_v7.tex`.

Main paper story:

1. Establish scalar RT baselines on fixed 2 s EEG windows.
2. Reframe RT prediction as event-time distribution modeling.
3. Compare ETS-U-Net event-time objectives under one backbone and protocol.
4. Analyze posterior geometry to show that scalar nRMSE hides output semantics.
5. Use shifted-crop inference as a shortcut-vs-localization diagnostic.
6. Evaluate shift-jitter training as a training-time intervention that reduces
   the fixed-window shortcut but does not fully solve crop-relative
   localization.
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
- `shifted-crop rel nRMSE`: pooled shifted-crop pseudo-validation RMSE divided
  by `std(target_abs - crop_start)` for valid shifted-crop examples. For the
  paper presentation, this should be used as the shifted-crop accuracy metric,
  while shift-tracking metrics should be computed on matched trial-level crop
  pairs. Shifted-crop rel nRMSE is not numerically interchangeable with ordinary
  Holdout nRMSE.
- `sensitivity`: fraction of the imposed crop shift reflected in the
  prediction; `1` is ideal crop-relative localization and `0` is crop-invariant
  behavior.
- `direction`: fraction of shifted examples whose prediction moves in the
  expected localizer direction.
- `shift error`: absolute error between the observed prediction shift and the
  ideal crop-relative shift.
- Posterior metrics are interpreted as output-geometry diagnostics, not only as
  scalar prediction quality.

## Config Groups

| group | path | role | status |
| --- | --- | --- | --- |
| Data protocol | `benchmarks/experiments/00_data_protocol/` | Split, support-filter, subject-disjointness, and package-version record. | Complete. |
| Regression baselines | `benchmarks/configs/01_regression_baselines/` | Scalar RT and wrapped external EEG baselines. | Complete: 12/12 configs, 5 seeds each. |
| Segmentation ablations | `benchmarks/configs/02_segmentation_ablations/` | ETS-U-Net event-time objective comparison. | Complete for selected paper-facing objectives. |
| Posterior geometry | generated from `02_segmentation_ablations` outputs | Distributional output semantics beyond scalar nRMSE. | Pending filtered-protocol regeneration. |
| Shifted-crop diagnostic | `evaluation.shifted_crop` in run configs plus post-hoc runners | Shortcut-vs-localization stress test. | Complete for regression and original segmentation; shift-jitter summaries updated in paper tables. |
| Shift-jitter training | `benchmarks/configs/03_crop_shift_jitter/` | Training-time removal of fixed-window shortcuts. | Complete: 6/6 configs, 5 seeds each. |
| Final training recipe | planned config group | Optional final model recipe after ablations are fixed. | Planned. |
| Distribution-aware stacking | planned artifacts under stacking experiments | Test whether posterior/logit features add reusable information. | Planned. |

## Artifact Registry

| artifact | canonical path | role | status |
| --- | --- | --- | --- |
| Data protocol summary | `benchmarks/experiments/00_data_protocol/protocol_summary.md` | Canonical split/support table for the filtered protocol. | Complete. |
| Regression repeated runs | `benchmarks/experiments/01_regression_baselines/` | Main scalar baseline table. | Complete: 12/12 configs, 5 seeds each. |
| Main regression table | `benchmarks/experiments/paper_tables/main_01_regression_baselines.md` | Camera-ready scalar baseline table. | Complete. |
| Main event-time objective table | `benchmarks/experiments/paper_tables/main_02_event_time_objectives.md` | Camera-ready ETS-U-Net objective table. | Complete. |
| Main shift-jitter summary table | `benchmarks/experiments/paper_tables/main_03_shift_jitter_summary.md` | Fixed-vs-jitter comparison of holdout accuracy, shifted-crop robustness, and localization behavior. | Complete. |
| Segmentation repeated runs | `benchmarks/experiments/02_segmentation_ablations/` | Event-time objective table and shifted-crop summaries. | Complete for selected paper-facing objectives. |
| Regression shifted-crop appendix table | `benchmarks/experiments/paper_tables/appendix_01_regression_shifted_crop.md` | Appendix diagnostic for all scalar baselines. | Complete: 60/60 seed-runs. |
| Fixed-window segmentation shifted-crop appendix table | `benchmarks/experiments/paper_tables/appendix_02_fixed_shifted_details.md` | Detailed diagnostic for fixed-window ETS-U-Net objectives. | Complete: 30/30 seed-runs. |
| Shift-jitter repeated runs | `benchmarks/experiments/03_crop_shift_jitter/` | Jitter-trained event-time localization test. | Complete: 6/6 configs, 5 seeds each. |
| Shift-jitter canonical holdout re-eval | `benchmarks/experiments/03_crop_shift_jitter_canonical_eval/` | Canonical `0.5 <= RT <= 2.5` holdout scores for jitter-trained checkpoints. | Complete: 30/30 seed-runs. |
| Shift-jitter training appendix table | `benchmarks/experiments/paper_tables/appendix_03_jitter_shifted_details.md` | Detailed diagnostic after shift-jitter training. | Complete. |
| Posterior geometry figures | TBD under `benchmarks/experiments/02_segmentation_ablations/figures/` | Camera-ready posterior profile panels. | Pending filtered-protocol regeneration. |
| Shifted-crop summaries | per-run `shifted_eval/` folders | Crop-start robustness and localization diagnostics. | New pooled-summary format complete for regression, original segmentation, and shift-jitter runs. |
| Stacking artifacts | TBD | Competition-style ensemble/calibration reproduction. | Planned. |

## 00 Data Protocol

Canonical artifact:
`benchmarks/experiments/00_data_protocol/protocol_summary.md`.

Narrative role:

- Defines the release-separated protocol before any model comparison.
- Shows that train, development, and holdout are subject-disjoint.
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

Current repeated-run snapshot is also maintained as a paper-facing table at
`benchmarks/experiments/paper_tables/main_01_regression_baselines.md`.

Final 5-seed rows:

| model | seeds | valid nRMSE mean +/- std | Holdout nRMSE mean +/- std | Holdout range |
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
| `ets_unet_time_only.yaml` | ETS-U-Net soft-argmax RT loss | Scalar RT loss on posterior expectation, without distributional supervision. |
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
| shifted-crop paper rows | accuracy on valid crop examples; shift tracking on the matched common-inside trial subset |
| shifted-crop starts | `0.2, 0.3, ..., 0.8` |
| shifted-crop per-trial predictions | disabled by default |

Current repeated-run snapshot is also maintained as a paper-facing table at
`benchmarks/experiments/paper_tables/main_02_event_time_objectives.md`.

Current rows:

| model | paper name | seeds | valid nRMSE mean +/- std | Holdout nRMSE mean +/- std | Holdout tau nRMSE mean +/- std |
| --- | --- | ---: | ---: | ---: | ---: |
| `ets_unet_ce` | ETS-U-Net CE | 5/5 | 0.8763 +/- 0.0044 | 0.8774 +/- 0.0044 | 0.8753 +/- 0.0039 |
| `ets_unet_event_nll` | ETS-U-Net EventNLL | 5/5 | 0.8769 +/- 0.0030 | 0.8805 +/- 0.0021 | 0.8772 +/- 0.0018 |
| `ets_unet_event_nll_mixture` | ETS-U-Net mixture EventNLL | 5/5 | 0.8744 +/- 0.0018 | 0.8785 +/- 0.0047 | 0.8745 +/- 0.0053 |
| `ets_unet_hazard_event_nll` | ETS-U-Net hazard EventNLL | 5/5 | 0.8755 +/- 0.0027 | 0.8776 +/- 0.0031 | 0.8778 +/- 0.0041 |
| `ets_unet_time_only` | ETS-U-Net soft-argmax RT loss | 5/5 | 0.8944 +/- 0.0048 | 0.8943 +/- 0.0025 | 0.8917 +/- 0.0046 |
| `ets_unet_wasserstein` | ETS-U-Net Wasserstein | 5/5 | 0.8997 +/- 0.0035 | 0.8995 +/- 0.0078 | 0.8896 +/- 0.0033 |

Interpretation so far:

- CE and EventNLL both outperform the strongest completed scalar regression
  baseline on holdout after temperature calibration.
- Mixture EventNLL has the best calibrated scalar readout so far
  (`test_tau_nRMSE = 0.8745`) and the best original-segmentation shifted
  pseudo-validation rel nRMSE.
- CE remains the simplest strong row; hazard EventNLL is competitive but does
  not beat mixture/CE cleanly.
- Soft-argmax RT loss is clearly weaker than CE/EventNLL, which supports the claim that
  distributional event-time supervision matters beyond the soft-argmax scalar
  readout.
- Wasserstein has the strongest shift sensitivity among original segmentation
  controls, but it pays for that with worse scalar nRMSE and worse shifted-crop
  rel nRMSE.

Expected paper use:

- Main segmentation table: CE, EventNLL, mixture EventNLL, hazard EventNLL,
  soft-argmax RT loss, Wasserstein.
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
  better calibrated; soft-argmax RT loss is expected to recover the scalar readout with a
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
| rel nRMSE | Pooled pseudo-validation nRMSE on valid crop-relative targets `target_abs - crop_start`. |
| RMSE / MAE, seconds | Crop-relative prediction error in physical units, using the same valid shifted-crop examples as rel nRMSE. |
| shift error | Absolute error between observed prediction shift and ideal localizer shift. |
| sensitivity | Prediction-shift magnitude divided by crop-shift magnitude; `1` is ideal localizer, `0` is invariant shortcut. |
| direction | Fraction of shifted examples moving in the correct direction. |

Current interpretation constraints:

- Paper-facing presentation separates shifted-crop accuracy from shift
  tracking.
- Accuracy metrics (`rel nRMSE`, RMSE, MAE) are computed only on crop examples
  where the behavioral response remains observable inside the evaluated 2 s
  window. This keeps the shifted-crop accuracy score comparable to the standard
  fixed-window evaluation.
- Shift-tracking metrics (`shift error`, `sensitivity`, `direction`) are
  computed on the matched common-inside trial subset, where the response remains
  inside every evaluated crop start. This makes prediction changes attributable
  to the imposed temporal displacement rather than to a changing trial set.
- `rel nRMSE` is normalized by `std(target_abs - crop_start)` and should not be
  numerically compared to ordinary holdout nRMSE.
- Current claim: fixed-window regression and fixed-window event-time models show
  partial crop sensitivity, but neither family solves crop-relative
  localization without an explicit training-time intervention.

Main shift-jitter summary table:
`benchmarks/experiments/paper_tables/main_03_shift_jitter_summary.md`.

Detailed fixed-window segmentation shifted-crop appendix:
`benchmarks/experiments/paper_tables/appendix_02_fixed_shifted_details.md`.

| model | seeds | rel nRMSE | RMSE, s | MAE, s | shift error, s | sensitivity | direction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ets_unet_event_nll_mixture` | 5/5 | 0.866 +/- 0.004 | 0.332 +/- 0.001 | 0.234 +/- 0.003 | 0.130 +/- 0.004 | 0.602 +/- 0.048 | 0.773 +/- 0.006 |
| `ets_unet_ce` | 5/5 | 0.868 +/- 0.002 | 0.333 +/- 0.001 | 0.237 +/- 0.002 | 0.133 +/- 0.003 | 0.581 +/- 0.035 | 0.778 +/- 0.004 |
| `ets_unet_event_nll` | 5/5 | 0.869 +/- 0.002 | 0.333 +/- 0.001 | 0.237 +/- 0.002 | 0.132 +/- 0.002 | 0.584 +/- 0.029 | 0.774 +/- 0.006 |
| `ets_unet_hazard_event_nll` | 5/5 | 0.869 +/- 0.003 | 0.334 +/- 0.001 | 0.238 +/- 0.003 | 0.135 +/- 0.004 | 0.562 +/- 0.038 | 0.769 +/- 0.009 |
| `ets_unet_time_only` | 5/5 | 0.886 +/- 0.007 | 0.340 +/- 0.003 | 0.247 +/- 0.005 | 0.140 +/- 0.008 | 0.538 +/- 0.059 | 0.759 +/- 0.018 |
| `ets_unet_wasserstein` | 5/5 | 0.893 +/- 0.003 | 0.343 +/- 0.001 | 0.239 +/- 0.004 | 0.133 +/- 0.008 | 0.668 +/- 0.071 | 0.783 +/- 0.020 |

Segmentation interpretation:

- CE/EventNLL/mixture/hazard remain close in shifted pseudo-validation
  performance, with mixture EventNLL slightly best by rel nRMSE.
- Segmentation is somewhat more shift-sensitive than many scalar baselines, but
  it is still far from the ideal sensitivity of `1.0`.
- Soft-argmax RT loss underperforms CE/EventNLL, supporting the value of distributional
  event-time supervision.
- Wasserstein gives the largest shift sensitivity, but worsens scalar and
  shifted rel nRMSE.

Regression shifted-crop snapshot:
`benchmarks/experiments/paper_tables/appendix_01_regression_shifted_crop.md`.

| model | seeds | rel nRMSE | RMSE, s | MAE, s | shift error, s | sensitivity | direction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `etr_cnn_large` | 5/5 | 0.851 +/- 0.008 | 0.326 +/- 0.003 | 0.245 +/- 0.002 | 0.142 +/- 0.005 | 0.533 +/- 0.057 | 0.773 +/- 0.013 |
| `etr_cnn` | 5/5 | 0.853 +/- 0.008 | 0.327 +/- 0.003 | 0.246 +/- 0.002 | 0.145 +/- 0.002 | 0.517 +/- 0.036 | 0.761 +/- 0.004 |
| `msp_cnn` | 5/5 | 0.857 +/- 0.014 | 0.329 +/- 0.005 | 0.247 +/- 0.003 | 0.143 +/- 0.004 | 0.553 +/- 0.076 | 0.761 +/- 0.010 |
| `tidnet_wrapped` | 5/5 | 0.858 +/- 0.002 | 0.329 +/- 0.001 | 0.252 +/- 0.001 | 0.168 +/- 0.001 | 0.516 +/- 0.047 | 0.640 +/- 0.004 |
| `eegnet_wrapped` | 5/5 | 0.863 +/- 0.003 | 0.331 +/- 0.001 | 0.255 +/- 0.001 | 0.167 +/- 0.001 | 0.365 +/- 0.010 | 0.664 +/- 0.006 |
| `shallowfbcspnet_wrapped` | 5/5 | 0.864 +/- 0.003 | 0.331 +/- 0.001 | 0.254 +/- 0.001 | 0.167 +/- 0.002 | 0.348 +/- 0.031 | 0.672 +/- 0.005 |
| `deep4net_wrapped` | 5/5 | 0.865 +/- 0.004 | 0.332 +/- 0.002 | 0.253 +/- 0.001 | 0.166 +/- 0.003 | 0.481 +/- 0.016 | 0.652 +/- 0.010 |
| `labram_wrapped` | 5/5 | 0.870 +/- 0.004 | 0.334 +/- 0.002 | 0.253 +/- 0.001 | 0.170 +/- 0.002 | 0.549 +/- 0.067 | 0.660 +/- 0.007 |
| `eegconformer_wrapped` | 5/5 | 0.871 +/- 0.010 | 0.334 +/- 0.004 | 0.252 +/- 0.002 | 0.157 +/- 0.006 | 0.436 +/- 0.054 | 0.759 +/- 0.007 |
| `medformer_wrapped` | 5/5 | 0.873 +/- 0.003 | 0.335 +/- 0.001 | 0.258 +/- 0.001 | 0.182 +/- 0.004 | 0.326 +/- 0.060 | 0.623 +/- 0.008 |
| `eegpt_wrapped` | 5/5 | 0.874 +/- 0.008 | 0.335 +/- 0.003 | 0.259 +/- 0.004 | 0.179 +/- 0.011 | 0.333 +/- 0.095 | 0.610 +/- 0.041 |
| `atcnet_wrapped` | 5/5 | 0.875 +/- 0.006 | 0.336 +/- 0.002 | 0.261 +/- 0.003 | 0.185 +/- 0.006 | 0.313 +/- 0.110 | 0.589 +/- 0.022 |

Regression interpretation:

- Scalar regression also learns partial crop sensitivity; it is not a pure
  invariant shortcut.
- The strongest scalar baselines reach sensitivity around `0.5`, which is still
  far from solved crop-relative localization.
- Event-time models provide explicit posterior outputs for this diagnostic, but
  the diagnostic itself shows partial rather than complete localization.

## 05 Shift-Jitter Training

Runner:

```bash
sh /home/sneddy/sneddy_projects/neurosned/benchmarks/runners/run_crop_shift_jitter.sh
```

Role:

- Turn the shifted-crop diagnostic into a training intervention.
- Randomize crop starts during training while keeping the target inside the
  crop, so the model cannot rely on one fixed stimulus-aligned window.
- Test the hypothesis that event-time supervision should benefit more from
  jittered crops than pure scalar regression, because the target is explicitly a
  crop-relative event location.

Current shift-jitter comparison:

Main shift-jitter summary table:
`benchmarks/experiments/paper_tables/main_03_shift_jitter_summary.md`.

Detailed shift-jitter appendix:
`benchmarks/experiments/paper_tables/appendix_03_jitter_shifted_details.md`.

| model | seeds | rel nRMSE | RMSE, s | MAE, s | shift error, s | sensitivity | direction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ets_unet_ce_shift_jitter` | 5/5 | 0.857 +/- 0.006 | 0.329 +/- 0.002 | 0.235 +/- 0.004 | 0.126 +/- 0.004 | 0.583 +/- 0.041 | 0.792 +/- 0.006 |
| `ets_unet_event_nll_mixture_shift_jitter` | 5/5 | 0.858 +/- 0.004 | 0.329 +/- 0.002 | 0.231 +/- 0.003 | 0.122 +/- 0.004 | 0.625 +/- 0.031 | 0.792 +/- 0.008 |
| `ets_unet_event_nll_shift_jitter` | 5/5 | 0.859 +/- 0.003 | 0.330 +/- 0.001 | 0.232 +/- 0.002 | 0.123 +/- 0.003 | 0.617 +/- 0.022 | 0.794 +/- 0.007 |
| `ets_unet_hazard_event_nll_shift_jitter` | 5/5 | 0.861 +/- 0.005 | 0.330 +/- 0.002 | 0.235 +/- 0.003 | 0.126 +/- 0.004 | 0.589 +/- 0.026 | 0.792 +/- 0.007 |
| `ets_unet_time_only_shift_jitter` | 5/5 | 0.861 +/- 0.003 | 0.330 +/- 0.001 | 0.239 +/- 0.003 | 0.126 +/- 0.003 | 0.578 +/- 0.036 | 0.795 +/- 0.005 |
| `ets_unet_wasserstein_shift_jitter` | 5/5 | 0.877 +/- 0.010 | 0.337 +/- 0.004 | 0.233 +/- 0.004 | 0.123 +/- 0.005 | 0.685 +/- 0.042 | 0.803 +/- 0.012 |

Interpretation:

- Canonical holdout re-evaluation shows that shift-jitter does not produce a
  broad ordinary fixed-window gain under `0.5 <= RT <= 2.5`.
- Shift-jitter consistently improves shifted-crop rel nRMSE, so the main effect
  is robustness to crop placement rather than ordinary holdout improvement.
- Direction improves across objectives and sensitivity improves modestly, but
  sensitivity remains far below the ideal localizer value of `1.0`.
- Wasserstein has the strongest sensitivity/direction but worse shifted-crop
  error, so it is a geometry-control row rather than the best predictor.

Paper-facing configs:

| config | paper role |
| --- | --- |
| `ets_unet_ce_shift_jitter.yaml` | Soft-label event-time CE under shift-jitter training. |
| `ets_unet_event_nll_shift_jitter.yaml` | Latent EventNLL under shift-jitter training. |
| `ets_unet_event_nll_mixture_shift_jitter.yaml` | Best current likelihood-style objective under shift-jitter training. |
| `ets_unet_hazard_event_nll_shift_jitter.yaml` | Hazard/survival event-time parameterization under shift-jitter training. |
| `ets_unet_time_only_shift_jitter.yaml` | Soft-argmax RT loss under shift-jitter training. |
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
| holdout nRMSE / MAE | Check that jitter does not destroy ordinary scalar accuracy. |
| shifted-crop rel nRMSE / RMSE / MAE | Pooled crop-relative pseudo-validation performance. |
| sensitivity / direction | Main shortcut-vs-localization behavior diagnostics. |
| shift error | Distance from ideal crop-relative shift tracking. |
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
| final holdout nRMSE / MAE | Final scalar performance claim. |
| validation-to-holdout gap | Overfitting/generalization check. |
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
- The intended claim is not that stacking is needed for the single-model
  event-time result. The intended claim is narrower: posterior-producing models
  expose distributional features that are useful for post-hoc alignment and
  ensembling when different objectives have different confidence and posterior
  shapes.

Candidate feature families:

| feature family | examples |
| --- | --- |
| scalar base predictions | direct regression outputs, posterior mean/mode. |
| posterior shape | width, entropy, mode-mean gap, aligned mass. |
| logits/posterior samples | compact posterior summaries or learned meta-features. |
| calibration features | temperature-calibrated readouts and uncertainty measures. |

Expected table format:

| row | feature/input type | learner | expected interpretation |
| --- | --- | --- | --- |
| Equal-weight scalar RT blend | One scalar RT prediction per base model. | Fixed equal weights | Basic ensemble reference; tests whether averaging base readouts is enough. |
| Equal-weight logits soft-argmax blend | Raw segmentation logits from each model. | Fixed equal weights | SubmitWrapper-style segmentation blend: average logits first, then apply softmax/soft-argmax. |
| Ridge stacking, RT only | Scalar RT predictions only. | Ridge | Linear learned reweighting of models; controls for simple model selection/blending. |
| Boosting stacking, RT only | Scalar RT predictions only. | HistGradientBoosting / GBDT | Nonlinear learned reweighting using only point predictions. |
| Ridge stacking, posterior meta-features | RT predictions plus posterior/logit summaries. | Ridge | Tests whether posterior features help even under a constrained linear meta-model. |
| Boosting stacking, posterior meta-features | RT predictions plus posterior/logit summaries. | HistGradientBoosting / GBDT | Main utility row; tests whether distributional outputs add reusable nonlinear reliability signals. |

Expected result columns:

| column | meaning |
| --- | --- |
| R11 nRMSE | Final release-separated holdout score after the stacker design is fixed. |
| R11 MAE | Final physical-unit holdout score. |
| delta vs best single model | Practical value beyond selecting the best base model. |
| delta vs RT-only stacker | Direct evidence that posterior/meta-features add information beyond scalar readouts. |

Raw stacker artifacts may still store development OOF nRMSE/MAE for audit and
debugging, but the paper-facing table should not report them. The only
paper-facing score is the release-separated R11 result after the stacker design
has been fixed.

Caption/narrative context to preserve:

- Base models must be trained before stacking and must not use R11 for any
  decision.
- The stacker should be selected on R9-R10 only, preferably with
  subject-disjoint out-of-fold predictions inside the development split.
- R11 is used only once for final reporting after the feature set and
  meta-learner are fixed.
- Scalar stacking should use the uncalibrated fixed-window predictions
  (`best_val_predictions.csv` and `test_predictions.csv`). Temperature
  calibration is deliberately left to the stacker rather than applied before
  feature construction.
- The equal-weight segmentation baseline follows the original
  `SubmitWrapper` logic: average model logits, then compute a softmax
  distribution and soft-argmax readout with temperature `0.92`. It should not
  be implemented as an average of hard posterior modes.
- Posterior meta-feature stacking should use the ported challenge-style
  `MetaFeatureExtractor`, `RidgeMetaRegressor`, `HgbMetaRegressor`, and
  monotonic-constraint helpers in `benchmarks/pkg/ensembling`.
- Feature pruning should be based on the saved feature-audit artifacts from
  `stack_segmentation_outputs.py`: Ridge stores fold-wise standardized
  coefficients, while HGB stores validation-fold permutation MSE increases.
  These diagnostics are development-only and should not use R11 for deciding
  which features to keep.
- The most important comparison is not equal blending versus stacking; it is
  `RT-only stacking` versus `posterior meta-feature stacking`.
- A positive result supports the statement that event-time posteriors contain
  reusable information discarded by scalar RT predictions. It should be
  described as downstream ensemble utility, not as the main evidence for the
  event-time formulation.
- If posterior meta-feature stacking does not improve over RT-only stacking,
  keep this section as an appendix/control rather than a main claim.

Expected artifact:

| artifact | path |
| --- | --- |
| Camera-ready stacking ablation table | `benchmarks/experiments/paper_tables/appendix_04_distribution_aware_stacking.md` |
| Raw stacker results | `benchmarks/experiments/07_distribution_aware_stacking/` |

Metrics:

| metric | use |
| --- | --- |
| out-of-fold validation nRMSE | Honest meta-model selection. |
| holdout nRMSE / MAE | Final stacking comparison against best single model. |
| delta vs best base model | Whether stacking adds value beyond model selection. |
| feature ablation | Whether posterior features matter beyond scalar predictions. |

## Planned/Optional Scope

Keep these out of the main experimental spine until filtered-protocol artifacts
exist:

| add-on | purpose | current status |
| --- | --- | --- |
| Distribution-aware stacking | Test whether posterior/logit features add reusable information beyond scalar predictions. | Planned. |
| Two-stage checkpoint reload recipe | Potential training-stability/final recipe. | Code path exists, not a current main claim. |

## Immediate Next Steps

1. Recompute posterior geometry from filtered segmentation predictions/logits.
2. Decide final manuscript placement for `main_03_shift_jitter_summary.md`.
3. Reproduce distribution-aware stacking under the filtered protocol if it is
   kept in the paper narrative.
4. Define the optional final training recipe only after the ablation and
   diagnostic tables are frozen.
5. Update `writing/overleaf/release_v7/main_revised_v7.tex` only after the
   corresponding filtered-protocol artifacts are complete.
