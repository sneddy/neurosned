# quantitative_posterior_geometry_table

## Draft Caption

Quantitative posterior geometry on R11 for matched event-time segmentation losses. Scalar nRMSE and MAE summarize point-readout accuracy; CRPS and fixed-kernel EventNLL are proper distributional scores computed from the full posterior; width, near-target mass, mode-mean gap, and empirical coverage quantify posterior concentration and calibration. CRPS is reported in milliseconds. Fixed-kernel EventNLL uses the same Gaussian observation kernel for all models (`sigma=0.15 s`), so lower values indicate that the observed RT has higher likelihood under the predicted event-time mixture.

| Model | nRMSE | MAE ms | CRPS ms | Fixed-kernel EventNLL | Width80 ms | Mass +/-150 ms | Mode-mean gap ms | Coverage80 | Coverage MAE |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CE | 0.939 | 215 | 154 | 0.102 | 770 | 0.337 | 53 | 0.883 | 0.100 |
| CE+time | 0.940 | 215 | 155 | 0.106 | 730 | 0.336 | 47 | 0.868 | 0.081 |
| EventNLL | 0.943 | 214 | 163 | -0.024 | 410 | 0.495 | 65 | 0.454 | 0.312 |
| Time-only | 0.952 | 225 | 166 | 0.159 | 860 | 0.348 | 108 | 0.846 | 0.039 |
| Wasserstein | 0.961 | 218 | 172 | 0.428 | 370 | 0.412 | 41 | 0.588 | 0.173 |

## Camera-Ready Summary

EventNLL produces the sharpest and most target-concentrated event-time posteriors, but these posteriors are under-calibrated as uncertainty estimates. Thus, EventNLL is better interpreted as a localization objective, whereas coverage-based metrics quantify whether posterior concentration corresponds to calibrated uncertainty.
