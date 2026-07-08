# Main Table 3: Shifted-Crop Core Diagnostic

Intended placement: `Posterior Readout and Diagnostics / Shifted-Crop
Localization Diagnostic`.

Caption draft: Shifted-crop diagnostic on the 5 s holdout dataset. Each trained
model is evaluated on multiple 2 s crops from the same trial, with crop starts
ranging from 0.2 to 0.8 s after stimulus onset. For comparability with the
standard fixed-window evaluation, accuracy metrics are computed only over crop
examples in which the behavioral response remains observable within the
evaluated 2 s window. To isolate crop-induced prediction changes,
shift-tracking metrics are computed on the common trial subset for which the
response remains inside every evaluated crop. Sensitivity quantifies the
fraction of the imposed crop shift reflected in the prediction, with 1
indicating ideal crop-relative localization and 0 indicating crop-invariant
behavior; direction is the fraction of shifted examples whose prediction moves
in the expected localizer direction.

| Model | Family | rel nRMSE | RMSE, s | MAE, s | Sensitivity | Direction |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ETR-CNN large | scalar regression | 0.851 +/- 0.008 | 0.326 +/- 0.003 | 0.245 +/- 0.002 | 0.533 +/- 0.057 | 0.773 +/- 0.013 |
| ETS-U-Net mixture EventNLL | event-time likelihood | 0.866 +/- 0.004 | 0.332 +/- 0.001 | 0.234 +/- 0.003 | 0.602 +/- 0.048 | 0.773 +/- 0.006 |
| ETS-U-Net CE | event-time soft target | 0.868 +/- 0.002 | 0.333 +/- 0.001 | 0.237 +/- 0.002 | 0.581 +/- 0.035 | 0.778 +/- 0.004 |
| ETS-U-Net EventNLL | event-time likelihood | 0.869 +/- 0.002 | 0.333 +/- 0.001 | 0.237 +/- 0.002 | 0.584 +/- 0.029 | 0.774 +/- 0.006 |
| ETS-U-Net soft-argmax RT loss | scalar readout control | 0.886 +/- 0.007 | 0.340 +/- 0.003 | 0.247 +/- 0.005 | 0.538 +/- 0.059 | 0.759 +/- 0.018 |
| ETS-U-Net Wasserstein | geometry control | 0.893 +/- 0.003 | 0.343 +/- 0.001 | 0.239 +/- 0.004 | 0.668 +/- 0.071 | 0.783 +/- 0.020 |

Paper note: fixed-window models show partial crop sensitivity rather than
solved crop-relative localization. Event-time models provide explicit posterior
outputs for this diagnostic, and Wasserstein gives the strongest sensitivity,
but stronger sensitivity does not automatically imply the best scalar shifted
accuracy.
