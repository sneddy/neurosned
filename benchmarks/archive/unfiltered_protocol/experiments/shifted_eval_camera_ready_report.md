# Shifted-crop evaluation report

## Protocol

Models were evaluated on the same R11 trials using 5 s EEG windows. At test time, a 2 s crop was extracted with start time

`s in {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}`.

The reference crop is `s = 0.5`, matching the original training protocol. The analysis below uses only `common_inside` trials, where the observed RT remains inside every shifted crop (`0.8 <= RT <= 2.2`). This avoids mixing shift sensitivity with target censoring. The common-inside set contains 14,183 trials. Confidence intervals are 95% subject-level bootstrap intervals with 1,000 resamples.

For a true event-time localizer, shifting the crop start by `Delta s` should shift the predicted time within the crop by `-Delta s`. Therefore the ideal raw shift slope is `-1`. A crop-invariant scalar RT regressor has raw shift slope near `0`.

## Main table

| Model | Ref nRMSE | Ref MAE, s | Worst delta nRMSE | Raw shift slope | Error to ideal slope -1 | Localizer-like trials | Invariant-like trials |
|---|---:|---:|---:|---:|---:|---:|---:|
| Best regression: SneddyRTNet | 0.874 | 0.206 [0.199, 0.215] | 0.174 | -0.253 [-0.270, -0.235] | 0.747 | 10.1% [9.0, 11.1] | 34.5% |
| CE segmentation | 0.856 | 0.201 [0.193, 0.209] | 0.193 | -0.287 [-0.306, -0.267] | 0.713 | 14.8% [13.3, 16.3] | 34.4% |
| EventNLL segmentation | 0.859 | 0.201 [0.193, 0.210] | 0.198 | -0.292 [-0.311, -0.272] | 0.708 | 15.9% [14.5, 17.3] | 33.3% |

Across all 13 regression baselines, the mean reference nRMSE was 0.918, the mean raw shift slope was -0.172, and the mean localizer-like fraction was 4.6%. Thus, the segmentation objectives are not only more accurate at the reference crop, but also move the output in a more event-like direction under temporal crop shifts.

## Camera-ready paragraph

To test whether the models learned an event-localizing representation or a stimulus-locked scalar shortcut, we introduced a shifted-crop evaluation. For each R11 trial, the model was evaluated on 2 s crops extracted from the same 5 s EEG window with crop starts from 0.2 s to 0.8 s, while keeping the behavioral RT label fixed. A model that localizes the response-relevant event within the crop should shift its within-crop prediction by the negative crop displacement, yielding a raw shift slope of -1. In contrast, a stimulus-aware scalar regressor that predicts trial-wise response tendency should be approximately crop-invariant, yielding a slope near 0. We evaluated this on the common-inside subset, where RT was observable in all shifted crops.

The segmentation models preserved or improved reference-crop RT accuracy relative to the best scalar regression baseline (nRMSE 0.856/0.859 for CE/EventNLL vs. 0.874 for SneddyRTNet), while producing a more event-like response to temporal shifts. EventNLL achieved a raw shift slope of -0.292 [95% CI: -0.311, -0.272], compared with -0.253 [-0.270, -0.235] for the best regression baseline and -0.172 on average across regression baselines. The fraction of localizer-like trials also increased from 10.1% for the best scalar regressor to 14.8% for CE and 15.9% for EventNLL. These results indicate that distributional temporal supervision changes the semantics of the output: the model is not merely fitting a scalar RT target, but partially aligns its predictions with the latent timing of response-relevant EEG evidence.

## Interpretation

The shifted-crop result should be presented as evidence for a representational difference, not as a perfect equivariance result. The segmentation models are closer to the event-localizer ideal than scalar regression baselines, but their slopes remain far from -1. This means the current model still mixes event localization with trial-wise response tendency. That limitation is useful to state directly: it motivates the event-time formulation and opens a concrete methodological direction for stronger shift-equivariant training objectives.

## Artifact paths

- Regression/segmentation summary table: `benchmarks/experiments/shifted_eval_camera_ready_summary.csv`
- CE shifted eval: `benchmarks/experiments/02_segmentation_ablations/unet_deeper_ce_only__20260703_193119/shifted_eval/`
- EventNLL shifted eval: `benchmarks/experiments/02_segmentation_ablations/unet_deeper_event_nll__20260703_191318/shifted_eval/`
