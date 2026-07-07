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
| Regression baselines | `benchmarks/configs/01_regression_baselines/` | Scalar RT and wrapped external EEG baselines. | Running: 9/12 configs complete; LaBraM partial. |
| Segmentation ablations | `benchmarks/configs/02_segmentation_ablations/` | ETS-U-Net event-time objective comparison. | Configs ready. |

## Artifact Registry

| artifact | canonical path | role | status |
| --- | --- | --- | --- |
| Regression repeated runs | `benchmarks/experiments/01_regression_baselines/` | Main scalar baseline table. | Running: 9/12 configs complete; LaBraM 3/5 finished. |
| Regression leaderboard | `benchmarks/experiments/regression_leaderboard.md` | Maintained current scalar baseline ranking. | Running: completed rows plus partial LaBraM state. |
| Segmentation repeated runs | `benchmarks/experiments/02_segmentation_ablations/` | Event-time objective table and shifted-crop summaries. | Pending rerun. |
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

Completed 5/5 rows:

| model | seeds | valid nRMSE mean +/- std | R11 nRMSE mean +/- std | R11 range |
| --- | ---: | ---: | ---: | ---: |
| `etr_cnn_large` | 5/5 | 0.8972 +/- 0.0040 | 0.8928 +/- 0.0042 | 0.8873-0.8977 |
| `etr_cnn` | 5/5 | 0.9008 +/- 0.0060 | 0.8977 +/- 0.0068 | 0.8922-0.9085 |
| `msp_cnn` | 5/5 | 0.9006 +/- 0.0051 | 0.8998 +/- 0.0080 | 0.8927-0.9112 |
| `tidnet_wrapped` | 5/5 | 0.9235 +/- 0.0024 | 0.9192 +/- 0.0027 | 0.9146-0.9219 |
| `deep4net_wrapped` | 5/5 | 0.9269 +/- 0.0045 | 0.9260 +/- 0.0044 | 0.9210-0.9309 |
| `eegconformer_wrapped` | 5/5 | 0.9188 +/- 0.0026 | 0.9287 +/- 0.0057 | 0.9225-0.9342 |
| `eegnet_wrapped` | 5/5 | 0.9350 +/- 0.0054 | 0.9335 +/- 0.0028 | 0.9292-0.9370 |
| `shallowfbcspnet_wrapped` | 5/5 | 0.9324 +/- 0.0016 | 0.9343 +/- 0.0024 | 0.9316-0.9372 |
| `atcnet_wrapped` | 5/5 | 0.9686 +/- 0.0183 | 0.9666 +/- 0.0151 | 0.9533-0.9920 |

Partial / pending rows:

| model | state | partial valid nRMSE | partial R11 nRMSE | notes |
| --- | --- | ---: | ---: | --- |
| `labram_wrapped` | 3/5 finished, seed 2028 running | 0.9313 +/- 0.0083 | 0.9357 +/- 0.0100 | Not ranked until all 5 seeds finish. |
| `eegpt_wrapped` | pending | - | - | Starts after LaBraM. |
| `medformer_wrapped` | pending | - | - | Starts after EEGPT. |

Interpretation so far:

- `etr_cnn_large` is the strongest completed scalar baseline.
- `etr_cnn` improves over `msp_cnn`, but the gap is modest relative to seed
  variation.
- Wrapped external architectures train meaningfully under the fixed
  normalization protocol but are currently weaker than the compact scalar
  models.
- Partial `labram_wrapped` results are in the same range as the weaker wrapped
  CNN baselines, but should not be used as a final ranked row yet.
- Do not update manuscript regression tables until the runner finishes all
  listed configs.

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
| train pickle | `data/new_validation/r1_r8_train_5sec.pkl` |
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

1. Let `run_regression_baselines.sh` finish LaBraM, EEGPT, and Medformer.
2. Add a segmentation runner for `benchmarks/configs/02_segmentation_ablations/`
   or run the seven configs manually with `run_repeated.py`.
3. Recompute posterior geometry from filtered segmentation predictions/logits.
4. Recompute shifted-crop comparison from filtered scalar and segmentation
   runs.
5. Update `writing/overleaf/release_v3/main_revised_v3.tex` only after the
   corresponding filtered-protocol artifacts are complete.
