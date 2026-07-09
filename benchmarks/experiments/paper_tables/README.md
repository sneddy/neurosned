# Paper Tables

This directory stores paper-facing Markdown tables only. The source of truth for
the numbers is the experiment tree under `benchmarks/experiments/` and the
experiment journal `benchmarks/EXPERIMENTS.md`.

## Main Tables

| file | intended placement | role |
| --- | --- | --- |
| `main_01_regression_baselines.md` | `Regression Controls` | Scalar RT and external EEG backbone baselines. |
| `main_02_event_time_objectives.md` | `Event-Time Posterior Formulation` | ETS-U-Net objective comparison under the fixed event-time backbone. |
| `main_03_shift_jitter_summary.md` | `Posterior Readout and Diagnostics / Shift-Jitter Summary` | Fixed-vs-jitter comparison of holdout accuracy, shifted-crop robustness, and localization behavior. |
| `main_04_posterior_geometry.md` | `Posterior Geometry Diagnostics` | Distributional posterior geometry, concentration, and coverage metrics. |
| `main_05_effect_size.md` | `Event-Time Posterior Formulation` | Practical effect-size summary for the strongest scalar baseline versus the strongest event-time objectives. |
| `main_06_experiments_map.md` | `Dataset and Evaluation Protocol / Shared Training, Readout Tuning, and Inference Protocol` | Design map linking comparison blocks to backbone capacity, temporal readout, and distributional supervision axes. |

## Appendix Tables

| file | intended placement | role |
| --- | --- | --- |
| `appendix_01_regression_shifted_crop.md` | Appendix shifted-crop details | Shifted-crop diagnostic for all scalar regression baselines. |
| `appendix_02_fixed_shifted_details.md` | Appendix shifted-crop details | Detailed shifted-crop diagnostic for fixed-window ETS-U-Net objectives. |
| `appendix_03_jitter_shifted_details.md` | Appendix shifted-crop details | Detailed shifted-crop diagnostic after shift-jitter training. |
| `appendix_04_observation_noise_calibration.md` | Appendix posterior calibration details | Post-hoc RT observation-noise calibration for EventNLL-family predictive intervals. |

## Shifted-Crop Metric Convention

Accuracy metrics (`rel nRMSE`, RMSE, MAE) are computed only on crop examples
where the behavioral response remains observable inside the evaluated 2 s
window. Shift-tracking metrics (`Sensitivity`, `Direction`, and shift error in
appendix tables) are computed on the matched common-inside trial subset, so the
same trials are present for every crop start.

`rel nRMSE` is normalized by `std(target_abs - crop_start)` on the pooled
crop-relative target set. It is a shifted-crop pseudo-validation score and is
not numerically interchangeable with ordinary holdout nRMSE.
