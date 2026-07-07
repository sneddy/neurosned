# Segmentation Shifted-Crop Diagnostic

Evaluated on the R11 5 s dataset with 2 s crops starting at
`0.2, 0.3, ..., 0.8` seconds. Metrics are aggregated over the common-inside
subset `0.8 <= RT <= 2.2`, where the response remains inside every shifted
crop.

Interpretation: an ideal crop-relative localizer has raw shift slope near `-1`;
a crop-invariant shortcut has raw shift slope near `0`.

| Model | Seeds | Ref nRMSE @0.5 | Mean shifted nRMSE | Worst shifted nRMSE | Raw shift slope | Localizer-like | Invariant-like | Wrong-direction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ETS-U-Net CE | 5/5 | 0.8593 +/- 0.0071 | 0.9470 +/- 0.0087 | 1.0655 +/- 0.0135 | -0.339 +/- 0.017 | 0.209 +/- 0.031 | 0.286 +/- 0.016 | 0.222 +/- 0.006 |
| ETS-U-Net EventNLL | 5/5 | 0.8656 +/- 0.0049 | 0.9514 +/- 0.0063 | 1.0616 +/- 0.0129 | -0.344 +/- 0.014 | 0.221 +/- 0.026 | 0.292 +/- 0.016 | 0.221 +/- 0.003 |
| ETS-U-Net time-only | 5/5 | 0.8802 +/- 0.0034 | 0.9563 +/- 0.0113 | 1.0531 +/- 0.0224 | -0.301 +/- 0.043 | 0.160 +/- 0.066 | 0.323 +/- 0.046 | 0.238 +/- 0.013 |
| ETS-U-Net Wasserstein | 5/5 | 0.8851 +/- 0.0169 | 0.9810 +/- 0.0217 | 1.1083 +/- 0.0327 | -0.337 +/- 0.044 | 0.271 +/- 0.065 | 0.270 +/- 0.045 | 0.248 +/- 0.018 |
| ETS-U-Net hazard EventNLL | 3/5 partial | 0.8639 +/- 0.0064 | 0.9445 +/- 0.0119 | 1.0422 +/- 0.0197 | -0.329 +/- 0.028 | 0.187 +/- 0.056 | 0.301 +/- 0.033 | 0.222 +/- 0.003 |

## Per-Start Average

Aggregated over the currently available 23 segmentation seed-runs.

| Crop start | Raw nRMSE | Raw MAE | Corrected nRMSE |
| ---: | ---: | ---: | ---: |
| 0.2 | 1.0474 +/- 0.0261 | 0.2526 +/- 0.0075 | 1.2120 +/- 0.0343 |
| 0.3 | 0.9580 +/- 0.0184 | 0.2226 +/- 0.0046 | 1.0199 +/- 0.0251 |
| 0.4 | 0.8936 +/- 0.0131 | 0.1967 +/- 0.0036 | 0.9023 +/- 0.0150 |
| 0.5 | 0.8714 +/- 0.0134 | 0.1853 +/- 0.0038 | 0.8714 +/- 0.0134 |
| 0.6 | 0.8985 +/- 0.0178 | 0.1968 +/- 0.0039 | 0.9287 +/- 0.0198 |
| 0.7 | 0.9670 +/- 0.0250 | 0.2241 +/- 0.0067 | 1.0558 +/- 0.0282 |
| 0.8 | 1.0632 +/- 0.0335 | 0.2582 +/- 0.0098 | 1.2245 +/- 0.0352 |

## Interpretation

The completed event-time rows are better scalar predictors than the regression
baselines at the canonical crop, but shifted-crop behavior is still not close to
ideal crop-relative localization. CE and EventNLL have raw shift slopes around
`-0.34`; time-only is weaker at `-0.30`. Wasserstein has the highest
localizer-like fraction so far, but its scalar nRMSE and shifted nRMSE are worse
than CE/EventNLL.

This supports a cautious narrative: event-time supervision makes the output
more localization-like than ordinary scalar regression, but the fixed-window
protocol still permits shortcut behavior. Shift-jitter training remains the
natural next intervention if the paper wants to claim improved localization,
not only a diagnostic.
