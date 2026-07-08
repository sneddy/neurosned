# Appendix Table 1: Full Regression Shifted-Crop Diagnostic

Caption draft: Shifted-crop diagnostic for all scalar regression baselines on
the 5 s holdout dataset. Each model is evaluated on 2 s crops starting from 0.2 to
0.8 s after stimulus onset. Prediction error is computed on valid crop examples
where the behavioral response falls inside the evaluated crop. Shift-tracking
metrics are computed on matched trials where the response remains visible under
every crop start, so the measured prediction shift is not confounded by a
changing trial set. Values are mean +/- sample standard deviation over five
training seeds.

Table note: `rel nRMSE` is normalized by the standard deviation of the
crop-relative target over the pooled valid crop examples. The accuracy cohort
contains 15,472 holdout trials and 105,218 valid crop examples across starts; the
matched shift-tracking cohort contains 14,183 trials and 99,281 crop examples.
These counts are shared by all rows and are therefore not repeated as columns.

| Model | rel nRMSE | RMSE, s | MAE, s | Shift error, s | Sensitivity | Direction |
|---|---:|---:|---:|---:|---:|---:|
| ETR-CNN large | 0.851 +/- 0.008 | 0.326 +/- 0.003 | 0.245 +/- 0.002 | 0.142 +/- 0.005 | 0.533 +/- 0.057 | 0.773 +/- 0.013 |
| ETR-CNN | 0.853 +/- 0.008 | 0.327 +/- 0.003 | 0.246 +/- 0.002 | 0.145 +/- 0.002 | 0.517 +/- 0.036 | 0.761 +/- 0.004 |
| MSP-CNN | 0.857 +/- 0.014 | 0.329 +/- 0.005 | 0.247 +/- 0.003 | 0.143 +/- 0.004 | 0.553 +/- 0.076 | 0.761 +/- 0.010 |
| TIDNet wrapped | 0.858 +/- 0.002 | 0.329 +/- 0.001 | 0.252 +/- 0.001 | 0.168 +/- 0.001 | 0.516 +/- 0.047 | 0.640 +/- 0.004 |
| EEGNet wrapped | 0.863 +/- 0.003 | 0.331 +/- 0.001 | 0.255 +/- 0.001 | 0.167 +/- 0.001 | 0.365 +/- 0.010 | 0.664 +/- 0.006 |
| ShallowFBCSPNet wrapped | 0.864 +/- 0.003 | 0.331 +/- 0.001 | 0.254 +/- 0.001 | 0.167 +/- 0.002 | 0.348 +/- 0.031 | 0.672 +/- 0.005 |
| Deep4Net wrapped | 0.865 +/- 0.004 | 0.332 +/- 0.002 | 0.253 +/- 0.001 | 0.166 +/- 0.003 | 0.481 +/- 0.016 | 0.652 +/- 0.010 |
| LaBraM wrapped | 0.870 +/- 0.004 | 0.334 +/- 0.002 | 0.253 +/- 0.001 | 0.170 +/- 0.002 | 0.549 +/- 0.067 | 0.660 +/- 0.007 |
| EEGConformer wrapped | 0.871 +/- 0.010 | 0.334 +/- 0.004 | 0.252 +/- 0.002 | 0.157 +/- 0.006 | 0.436 +/- 0.054 | 0.759 +/- 0.007 |
| Medformer wrapped | 0.873 +/- 0.003 | 0.335 +/- 0.001 | 0.258 +/- 0.001 | 0.182 +/- 0.004 | 0.326 +/- 0.060 | 0.623 +/- 0.008 |
| EEGPT wrapped | 0.874 +/- 0.008 | 0.335 +/- 0.003 | 0.259 +/- 0.004 | 0.179 +/- 0.011 | 0.333 +/- 0.095 | 0.610 +/- 0.041 |
| ATCNet wrapped | 0.875 +/- 0.006 | 0.336 +/- 0.002 | 0.261 +/- 0.003 | 0.185 +/- 0.006 | 0.313 +/- 0.110 | 0.589 +/- 0.022 |

Interpretation: scalar regression is not fully crop-invariant, but it is also
not a solved crop-relative localizer. The strongest scalar models reach
sensitivity around 0.5, far from the ideal localizer value of 1.0.
