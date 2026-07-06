# quantitative_posterior_geometry_table

## Draft Caption

Quantitative posterior geometry on R11 for matched event-time segmentation losses. Scalar nRMSE and MAE summarize point-readout accuracy; CRPS and fixed-kernel EventNLL are proper distributional scores computed from the full posterior; width, near-target mass, mode-mean gap, and empirical coverage quantify posterior concentration and calibration. CRPS is reported in milliseconds. Fixed-kernel EventNLL uses the same Gaussian observation kernel for all models (`sigma=0.15 s`), so lower values indicate that the observed RT has higher likelihood under the predicted event-time mixture.

| Model | nRMSE | MAE ms | CRPS ms | Fixed-kernel EventNLL | Width80 ms | Mass +/-150 ms | Mode-mean gap ms | Coverage80 | Coverage MAE |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CE | 0.937 | 216 | 155 | 0.115 | 820 | 0.328 | 57 | 0.895 | 0.113 |
| CE+time | 0.936 | 219 | 157 | 0.144 | 860 | 0.311 | 57 | 0.899 | 0.116 |
| EventNLL | 0.941 | 216 | 162 | -0.027 | 450 | 0.487 | 72 | 0.480 | 0.290 |
| Time-only | 0.945 | 228 | 167 | 0.203 | 1070 | 0.311 | 129 | 0.913 | 0.113 |
| Wasserstein | 0.950 | 222 | 165 | 0.219 | 740 | 0.296 | 50 | 0.812 | 0.053 |

## Camera-Ready Summary

EventNLL produces the sharpest and most target-concentrated event-time posteriors, but these posteriors are under-calibrated as uncertainty estimates. Thus, EventNLL is better interpreted as a localization objective, whereas coverage-based metrics quantify whether posterior concentration corresponds to calibrated uncertainty.
