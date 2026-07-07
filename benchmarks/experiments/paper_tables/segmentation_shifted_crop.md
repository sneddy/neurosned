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
| ETS-U-Net mixture EventNLL | 5/5 | 0.8623 +/- 0.0070 | 0.9520 +/- 0.0077 | 1.0678 +/- 0.0151 | -0.355 +/- 0.024 | 0.246 +/- 0.042 | 0.290 +/- 0.026 | 0.221 +/- 0.004 |
| ETS-U-Net hazard EventNLL | 5/5 | 0.8605 +/- 0.0065 | 0.9434 +/- 0.0086 | 1.0466 +/- 0.0152 | -0.328 +/- 0.020 | 0.196 +/- 0.042 | 0.305 +/- 0.024 | 0.228 +/- 0.009 |
| ETS-U-Net time-only | 5/5 | 0.8802 +/- 0.0034 | 0.9563 +/- 0.0113 | 1.0531 +/- 0.0224 | -0.301 +/- 0.043 | 0.160 +/- 0.066 | 0.323 +/- 0.046 | 0.238 +/- 0.013 |
| ETS-U-Net Wasserstein | 5/5 | 0.8851 +/- 0.0169 | 0.9810 +/- 0.0217 | 1.1083 +/- 0.0327 | -0.337 +/- 0.044 | 0.271 +/- 0.065 | 0.270 +/- 0.045 | 0.248 +/- 0.018 |

## Per-Start Average

Aggregated over the currently available 30 segmentation seed-runs.

| Crop start | Raw nRMSE | Raw MAE | Corrected nRMSE |
| ---: | ---: | ---: | ---: |
| 0.2 | 1.0465 +/- 0.0238 | 0.2525 +/- 0.0068 | 1.2106 +/- 0.0313 |
| 0.3 | 0.9562 +/- 0.0171 | 0.2221 +/- 0.0043 | 1.0178 +/- 0.0228 |
| 0.4 | 0.8910 +/- 0.0126 | 0.1959 +/- 0.0036 | 0.8998 +/- 0.0139 |
| 0.5 | 0.8688 +/- 0.0130 | 0.1844 +/- 0.0037 | 0.8688 +/- 0.0130 |
| 0.6 | 0.8959 +/- 0.0167 | 0.1963 +/- 0.0037 | 0.9251 +/- 0.0190 |
| 0.7 | 0.9651 +/- 0.0228 | 0.2241 +/- 0.0061 | 1.0511 +/- 0.0268 |
| 0.8 | 1.0626 +/- 0.0303 | 0.2585 +/- 0.0089 | 1.2179 +/- 0.0339 |

## Interpretation

The completed event-time rows are better scalar predictors than the regression
baselines at the canonical crop, but shifted-crop behavior is still not close to
ideal crop-relative localization. Mixture EventNLL has the most localizer-like
shift slope among the likelihood-style objectives (`-0.355`), while
Wasserstein has the highest localizer-like fraction but worse scalar and shifted
nRMSE. Time-only remains weaker than the distributional objectives.

This supports a cautious narrative: event-time supervision makes the output
more localization-like than ordinary scalar regression, but the fixed-window
protocol still permits shortcut behavior. Shift-jitter training remains the
natural next intervention if the paper wants to claim improved localization,
not only a diagnostic.
