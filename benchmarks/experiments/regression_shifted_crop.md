# Regression Shifted-Crop Diagnostic

Evaluated on the R11 5 s dataset with 2 s crops starting at
`0.2, 0.3, ..., 0.8` seconds. Metrics are aggregated over 5 seeds per model
using the common-inside subset `0.8 <= RT <= 2.2`, where the response remains
inside every shifted crop.

Interpretation: an ideal crop-relative localizer has raw shift slope near `-1`;
a crop-invariant shortcut has raw shift slope near `0`.

| Model | Seeds | Ref nRMSE @0.5 | Mean shifted nRMSE | Worst shifted nRMSE | Raw shift slope | Localizer-like | Invariant-like |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ETR-CNN large | 5/5 | 0.8734 +/- 0.0046 | 0.9503 +/- 0.0131 | 1.0574 +/- 0.0263 | -0.294 +/- 0.029 | 0.139 +/- 0.048 | 0.316 +/- 0.054 |
| ETR-CNN | 5/5 | 0.8799 +/- 0.0095 | 0.9536 +/- 0.0120 | 1.0475 +/- 0.0163 | -0.277 +/- 0.011 | 0.118 +/- 0.025 | 0.330 +/- 0.027 |
| MSP-CNN | 5/5 | 0.8891 +/- 0.0129 | 0.9636 +/- 0.0218 | 1.0670 +/- 0.0386 | -0.299 +/- 0.029 | 0.149 +/- 0.063 | 0.315 +/- 0.043 |
| TIDNet wrapped | 5/5 | 0.9106 +/- 0.0042 | 0.9654 +/- 0.0028 | 1.0438 +/- 0.0098 | -0.210 +/- 0.014 | 0.075 +/- 0.014 | 0.466 +/- 0.023 |
| EEGNet wrapped | 5/5 | 0.9159 +/- 0.0034 | 0.9638 +/- 0.0045 | 1.0370 +/- 0.0126 | -0.173 +/- 0.009 | 0.043 +/- 0.011 | 0.511 +/- 0.009 |
| Deep4Net wrapped | 5/5 | 0.9177 +/- 0.0058 | 0.9714 +/- 0.0056 | 1.0420 +/- 0.0094 | -0.208 +/- 0.018 | 0.083 +/- 0.016 | 0.462 +/- 0.019 |
| EEGConformer wrapped | 5/5 | 0.9242 +/- 0.0107 | 0.9799 +/- 0.0166 | 1.0621 +/- 0.0201 | -0.218 +/- 0.030 | 0.041 +/- 0.025 | 0.365 +/- 0.053 |
| ShallowFBCSPNet wrapped | 5/5 | 0.9243 +/- 0.0018 | 0.9677 +/- 0.0044 | 1.0249 +/- 0.0116 | -0.168 +/- 0.017 | 0.038 +/- 0.020 | 0.509 +/- 0.033 |
| LaBraM wrapped | 5/5 | 0.9287 +/- 0.0124 | 0.9800 +/- 0.0066 | 1.0529 +/- 0.0117 | -0.187 +/- 0.017 | 0.048 +/- 0.016 | 0.439 +/- 0.031 |
| EEGPT wrapped | 5/5 | 0.9503 +/- 0.0245 | 0.9804 +/- 0.0125 | 1.0272 +/- 0.0255 | -0.111 +/- 0.058 | 0.023 +/- 0.026 | 0.654 +/- 0.199 |
| Medformer wrapped | 5/5 | 0.9532 +/- 0.0079 | 0.9810 +/- 0.0041 | 1.0199 +/- 0.0114 | -0.095 +/- 0.021 | 0.005 +/- 0.007 | 0.634 +/- 0.095 |
| ATCNet wrapped | 5/5 | 0.9633 +/- 0.0169 | 0.9837 +/- 0.0075 | 1.0266 +/- 0.0209 | -0.087 +/- 0.043 | 0.008 +/- 0.015 | 0.771 +/- 0.144 |

## Per-Start Average

Aggregated over all 60 regression seed-runs.

| Crop start | Raw nRMSE | Raw MAE | Corrected nRMSE |
| ---: | ---: | ---: | ---: |
| 0.2 | 1.0290 +/- 0.0277 | 0.2447 +/- 0.0075 | 1.2978 +/- 0.0548 |
| 0.3 | 0.9746 +/- 0.0210 | 0.2279 +/- 0.0064 | 1.0935 +/- 0.0441 |
| 0.4 | 0.9351 +/- 0.0259 | 0.2145 +/- 0.0095 | 0.9598 +/- 0.0317 |
| 0.5 | 0.9192 +/- 0.0292 | 0.2081 +/- 0.0115 | 0.9192 +/- 0.0292 |
| 0.6 | 0.9324 +/- 0.0252 | 0.2122 +/- 0.0098 | 0.9799 +/- 0.0403 |
| 0.7 | 0.9711 +/- 0.0180 | 0.2254 +/- 0.0064 | 1.1233 +/- 0.0570 |
| 0.8 | 1.0291 +/- 0.0244 | 0.2448 +/- 0.0082 | 1.3147 +/- 0.0782 |

## Interpretation

The regression baselines are strongest on the canonical crop start `0.5` and
degrade for earlier/later crops. Their raw shift slopes mostly lie between
`-0.1` and `-0.3`, far from the crop-relative localizer ideal of `-1`. This
supports the diagnostic claim that fixed-window scalar RT prediction permits
stimulus-locked shortcut behavior.
