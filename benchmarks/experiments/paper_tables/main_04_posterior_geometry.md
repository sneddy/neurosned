# Main Table 4: Posterior Geometry Diagnostics

Intended placement: `Posterior Geometry Diagnostics`.

Source artifact: `benchmarks/experiments/paper_figures/csv/posterior_geometry_group_summary.csv`.

Caption draft: Quantitative posterior geometry on R11 for matched event-time segmentation losses. Values are averages across five seeds after R9-R10 readout-temperature selection. nRMSE measures posterior-mean scalar prediction; CRPS and fixed-kernel EventNLL score the full posterior distribution; Width80, near-target mass, and interval coverage summarize posterior concentration and empirical reliability. CRPS is reported in milliseconds. Fixed-kernel EventNLL uses a common Gaussian observation kernel for all models (`sigma = 0.12 s`). For Coverage80, values closer to 0.80 indicate better nominal 80% interval coverage; for Coverage MAE, lower is better.

| Objective | nRMSE | CRPS ms | Fixed NLL | Width80 ms | Mass +/-150 ms | Coverage80 | Coverage MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CE | 0.875 | 153 | 0.08 | 766 | 0.334 | 0.883 | 0.101 |
| EventNLL | 0.877 | 160 | -0.06 | 502 | 0.490 | 0.500 | 0.273 |
| Mixture EventNLL | 0.874 | 155 | -0.08 | 528 | 0.471 | 0.573 | 0.207 |
| Hazard EventNLL | 0.878 | 161 | -0.06 | 445 | 0.502 | 0.468 | 0.302 |
| Soft-argmax RT loss | 0.892 | 161 | 0.11 | 844 | 0.357 | 0.843 | 0.040 |
| Wasserstein | 0.890 | 161 | 0.20 | 632 | 0.334 | 0.774 | 0.059 |

Paper note: CE and mixture EventNLL provide the strongest scalar readouts. EventNLL-family objectives are sharper and more target-concentrated, but they under-cover nominal intervals. The soft-argmax RT-loss control has the lowest Coverage MAE because its posterior is broad, but it is weaker as a scalar predictor and lacks distributional event-time supervision. Wasserstein is closest to nominal 80% coverage and later behaves most localizer-like under shifted-crop diagnostics, but it has weaker scalar accuracy and lower near-target mass. This supports the main claim that scalar nRMSE hides differences in posterior concentration, distributional scoring, and interval behavior.
