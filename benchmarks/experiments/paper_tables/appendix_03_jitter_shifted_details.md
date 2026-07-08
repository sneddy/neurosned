# Appendix Table 3: Shift-Jitter Segmentation Shifted-Crop Details

Intended placement: appendix shifted-crop details.

Caption draft: Shifted-crop diagnostic for event-time models retrained with
random crop-start jitter. Holdout tau nRMSE is re-evaluated on the canonical
fixed-window support (`0.5 <= RT <= 2.5`). Shifted-crop metrics are evaluated on
the same 5 s holdout crops used in the main shifted-crop diagnostic. For
comparability with the standard fixed-window evaluation, accuracy metrics are
computed only over crop examples in which the behavioral response remains
observable within the evaluated 2 s window. To isolate crop-induced prediction
changes, shift-tracking metrics are computed on the common trial subset for
which the response remains inside every evaluated crop. Relative nRMSE is
normalized by the standard deviation of the pooled crop-relative target.

| Model | Seeds | Holdout tau nRMSE | rel nRMSE | RMSE, s | MAE, s | Shift error, s | Sensitivity | Direction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ETS-U-Net CE shift-jitter | 5/5 | 0.8749 +/- 0.0060 | 0.857 +/- 0.006 | 0.329 +/- 0.002 | 0.235 +/- 0.004 | 0.126 +/- 0.004 | 0.583 +/- 0.041 | 0.792 +/- 0.006 |
| ETS-U-Net mixture EventNLL shift-jitter | 5/5 | 0.8734 +/- 0.0033 | 0.858 +/- 0.004 | 0.329 +/- 0.002 | 0.231 +/- 0.003 | 0.122 +/- 0.004 | 0.625 +/- 0.031 | 0.792 +/- 0.008 |
| ETS-U-Net EventNLL shift-jitter | 5/5 | 0.8771 +/- 0.0040 | 0.859 +/- 0.003 | 0.330 +/- 0.001 | 0.232 +/- 0.002 | 0.123 +/- 0.003 | 0.617 +/- 0.022 | 0.794 +/- 0.007 |
| ETS-U-Net hazard EventNLL shift-jitter | 5/5 | 0.8806 +/- 0.0034 | 0.861 +/- 0.005 | 0.330 +/- 0.002 | 0.235 +/- 0.003 | 0.126 +/- 0.004 | 0.589 +/- 0.026 | 0.792 +/- 0.007 |
| ETS-U-Net soft-argmax RT loss shift-jitter | 5/5 | 0.8836 +/- 0.0035 | 0.861 +/- 0.003 | 0.330 +/- 0.001 | 0.239 +/- 0.003 | 0.126 +/- 0.003 | 0.578 +/- 0.036 | 0.795 +/- 0.005 |
| ETS-U-Net Wasserstein shift-jitter | 5/5 | 0.8922 +/- 0.0066 | 0.877 +/- 0.010 | 0.337 +/- 0.004 | 0.233 +/- 0.004 | 0.123 +/- 0.005 | 0.685 +/- 0.042 | 0.803 +/- 0.012 |

Paper note: shift-jitter improves shifted-crop accuracy relative to the
original fixed-window segmentation rows, but does not solve crop-relative
localization. CE gives the best shifted accuracy; Wasserstein gives the
strongest sensitivity and direction.
