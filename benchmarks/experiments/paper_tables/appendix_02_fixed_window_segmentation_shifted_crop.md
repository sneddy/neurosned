# Segmentation Shifted-Crop Diagnostic

Evaluated on the 5 s holdout dataset with 2 s crops starting at
`0.2, 0.3, ..., 0.8` seconds. For comparability with the standard fixed-window
evaluation, accuracy metrics are computed only over crop examples in which the
behavioral response remains observable within the evaluated 2 s window.
Shift-tracking metrics use the common trial subset for which the response
remains inside every evaluated crop, so prediction changes can be attributed to
the imposed temporal displacement rather than to a changing trial set.

`rel nRMSE` is normalized by `std(target_abs - crop_start)` over the
pooled valid crop-relative target set. It is a shifted-crop
pseudo-validation score and should not be compared numerically to
ordinary holdout nRMSE.

| Model | Seeds | Acc trials | Acc rows | rel nRMSE | RMSE, s | MAE, s | Shift error, s | Sensitivity | Direction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ETS-U-Net mixture EventNLL | 5/5 | 15472 | 105218 | 0.866 +/- 0.004 | 0.332 +/- 0.001 | 0.234 +/- 0.003 | 0.130 +/- 0.004 | 0.602 +/- 0.048 | 0.773 +/- 0.006 |
| ETS-U-Net CE | 5/5 | 15472 | 105218 | 0.868 +/- 0.002 | 0.333 +/- 0.001 | 0.237 +/- 0.002 | 0.133 +/- 0.003 | 0.581 +/- 0.035 | 0.778 +/- 0.004 |
| ETS-U-Net EventNLL | 5/5 | 15472 | 105218 | 0.869 +/- 0.002 | 0.333 +/- 0.001 | 0.237 +/- 0.002 | 0.132 +/- 0.002 | 0.584 +/- 0.029 | 0.774 +/- 0.006 |
| ETS-U-Net hazard EventNLL | 5/5 | 15472 | 105218 | 0.869 +/- 0.003 | 0.334 +/- 0.001 | 0.238 +/- 0.003 | 0.135 +/- 0.004 | 0.562 +/- 0.038 | 0.769 +/- 0.009 |
| ETS-U-Net soft-argmax RT loss | 5/5 | 15472 | 105218 | 0.886 +/- 0.007 | 0.340 +/- 0.003 | 0.247 +/- 0.005 | 0.140 +/- 0.008 | 0.538 +/- 0.059 | 0.759 +/- 0.018 |
| ETS-U-Net Wasserstein | 5/5 | 15472 | 105218 | 0.893 +/- 0.003 | 0.343 +/- 0.001 | 0.239 +/- 0.004 | 0.133 +/- 0.008 | 0.668 +/- 0.071 | 0.783 +/- 0.020 |

Interpretation: original fixed-crop event-time segmentation does not solve crop-relative localization. It is slightly more sensitive to crop shifts than many scalar baselines, but the sensitivity remains far from the ideal value of `1.0`.
