# Paper Tables

This directory stores paper-facing Markdown tables only. The source of truth for
the numbers is the experiment tree under `benchmarks/experiments/` and the
experiment journal `benchmarks/EXPERIMENTS.md`.

## Main Tables

| file | intended placement | role |
| --- | --- | --- |
| `main_01_regression_baselines.md` | `Regression Controls` | Scalar RT and external EEG backbone baselines. |
| `main_02_event_time_objectives.md` | `Event-Time Posterior Formulation` | ETS-U-Net objective comparison under the fixed event-time backbone. |
| `main_03_shifted_crop_core.md` | `Posterior Readout and Diagnostics / Shifted-Crop Localization Diagnostic` | Compact shortcut-vs-localization diagnostic table. |

## Appendix Tables

| file | intended placement | role |
| --- | --- | --- |
| `appendix_01_regression_shifted_crop.md` | Appendix shifted-crop details | Shifted-crop diagnostic for all scalar regression baselines. |
| `appendix_02_fixed_window_segmentation_shifted_crop.md` | Appendix shifted-crop details | Post-hoc shifted-crop diagnostic for fixed-window ETS-U-Net objectives. |
| `appendix_03_shift_jitter_training.md` | Appendix or optional main extension | Shifted-crop diagnostic after shift-jitter training. |

## Shifted-Crop Metric Convention

Accuracy metrics (`rel nRMSE`, RMSE, MAE) use `mask=inside_crop`,
`start_group=all_starts`: every evaluated crop example contains the response.
Shift-tracking metrics (`Sensitivity`, `Direction`, and shift error in appendix
tables) use matched trial-level crop pairs with `mask=common_inside`,
`start_group=all_starts`, so the same trials are present for every crop start.

`rel nRMSE` is normalized by `std(target_abs - crop_start)` on the pooled
crop-relative target set. It is a shifted-crop pseudo-validation score and is
not numerically interchangeable with ordinary holdout nRMSE.
