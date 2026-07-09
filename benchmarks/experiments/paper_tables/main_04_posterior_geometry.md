# Main Table 4: Posterior Geometry Diagnostics

Intended placement: `Posterior Geometry Diagnostics`.

Source artifact: `benchmarks/experiments/paper_figures/csv/posterior_geometry_group_summary.csv`.

Caption draft: Quantitative posterior geometry on R11 for matched event-time segmentation losses. Values are mean +/- sample standard deviation across seeds after R9-R10 readout-temperature selection. nRMSE measures posterior-mean scalar prediction; fixed-kernel EventNLL scores the full posterior distribution under a common Gaussian observation kernel for all models (`sigma = 0.12 s`); Width80, near-target mass, and interval coverage summarize posterior concentration and empirical reliability. For Coverage80, values closer to 0.80 indicate better nominal 80% interval coverage; for Coverage MAE, lower is better.

| Objective | nRMSE | Fixed-kernel EventNLL | Width80 ms | Mass +/-150 ms | Coverage80 | Coverage MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CE | 0.875 +/- 0.004 | 0.08 +/- 0.01 | 766 +/- 36 | 0.334 +/- 0.008 | 0.883 +/- 0.013 | 0.101 +/- 0.014 |
| EventNLL | 0.877 +/- 0.002 | -0.06 +/- 0.01 | 502 +/- 15 | 0.490 +/- 0.008 | 0.500 +/- 0.025 | 0.273 +/- 0.021 |
| Mixture EventNLL | 0.874 +/- 0.005 | -0.08 +/- 0.01 | 528 +/- 22 | 0.471 +/- 0.008 | 0.573 +/- 0.020 | 0.207 +/- 0.017 |
| Hazard EventNLL | 0.878 +/- 0.004 | -0.06 +/- 0.01 | 445 +/- 14 | 0.502 +/- 0.005 | 0.468 +/- 0.007 | 0.302 +/- 0.006 |
| Soft-argmax RT loss | 0.892 +/- 0.005 | 0.11 +/- 0.03 | 844 +/- 96 | 0.357 +/- 0.015 | 0.843 +/- 0.029 | 0.040 +/- 0.024 |
| Wasserstein | 0.890 +/- 0.003 | 0.20 +/- 0.04 | 632 +/- 101 | 0.334 +/- 0.028 | 0.774 +/- 0.061 | 0.059 +/- 0.021 |

Paper note: CE and mixture EventNLL provide the strongest scalar readouts. EventNLL-family objectives are sharper and more target-concentrated, but they under-cover nominal intervals. The soft-argmax RT-loss control has the lowest Coverage MAE because its posterior is broad, but it is weaker as a scalar predictor and lacks distributional event-time supervision. Wasserstein is closest to nominal 80% coverage and later behaves most localizer-like under shifted-crop diagnostics, but it has weaker scalar accuracy and lower near-target mass. This supports the main claim that scalar nRMSE hides differences in posterior concentration, distributional scoring, and interval behavior.
