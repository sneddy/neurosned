# Appendix Table 3: Shift-Jitter Training

Intended placement: appendix, or optional main-text extension if shift-jitter is
made part of the central story.

Caption draft: Shifted-crop diagnostic for shift-jitter-trained event-time
models on the 5 s holdout dataset. Accuracy metrics use valid crop examples
(`mask=inside_crop`, `start_group=all_starts`), while sensitivity and direction
use matched common-inside crop pairs (`mask=common_inside`,
`start_group=all_starts`). Relative nRMSE is normalized by
`std(RT - crop_start)` on the pooled valid crop-relative target set.

| Model | rel nRMSE | RMSE, s | MAE, s | Shift error, s | Sensitivity | Direction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ETS-U-Net CE shift-jitter | 0.857 +/- 0.006 | 0.329 +/- 0.002 | 0.235 +/- 0.004 | 0.126 +/- 0.004 | 0.583 +/- 0.041 | 0.792 +/- 0.006 |
| ETS-U-Net mixture EventNLL shift-jitter | 0.858 +/- 0.004 | 0.329 +/- 0.002 | 0.231 +/- 0.003 | 0.122 +/- 0.004 | 0.625 +/- 0.031 | 0.792 +/- 0.008 |
| ETS-U-Net EventNLL shift-jitter | 0.859 +/- 0.003 | 0.330 +/- 0.001 | 0.232 +/- 0.002 | 0.123 +/- 0.003 | 0.617 +/- 0.022 | 0.794 +/- 0.007 |
| ETS-U-Net hazard EventNLL shift-jitter | 0.861 +/- 0.005 | 0.330 +/- 0.002 | 0.235 +/- 0.003 | 0.126 +/- 0.004 | 0.589 +/- 0.026 | 0.792 +/- 0.007 |
| ETS-U-Net soft-argmax RT loss shift-jitter | 0.861 +/- 0.003 | 0.330 +/- 0.001 | 0.239 +/- 0.003 | 0.126 +/- 0.003 | 0.578 +/- 0.036 | 0.795 +/- 0.005 |
| ETS-U-Net Wasserstein shift-jitter | 0.877 +/- 0.010 | 0.337 +/- 0.004 | 0.233 +/- 0.004 | 0.123 +/- 0.005 | 0.685 +/- 0.042 | 0.803 +/- 0.012 |

Paper note: shift-jitter improves shifted-crop accuracy relative to the
original fixed-window segmentation rows, but does not solve crop-relative
localization. CE gives the best shifted accuracy; Wasserstein gives the
strongest sensitivity and direction.
