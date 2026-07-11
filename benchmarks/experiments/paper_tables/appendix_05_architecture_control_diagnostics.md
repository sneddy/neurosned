# Appendix Table 5: Architecture-Control Diagnostic Robustness

Intended placement: appendix architecture-control details.

Manuscript label: `tab:architecture_control_diagnostics`

**Caption:** Shifted-crop and posterior-geometry diagnostics across dense
temporal backbones. Shifted-crop metrics follow the evaluation convention in
Section~\ref{sec:shifted_crop_diagnostic}; posterior metrics are evaluated on
the fixed holdout window after development-set readout-temperature selection.
Values are mean ± standard deviation across five seeds.

| Objective | Shifted rel. nRMSE ↓ | Sensitivity ↑ | Direction ↑ | Shared-kernel RT NLL ↓ | Mass ±150 ms ↑ | Coverage MAE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **ETS-U-Net** |  |  |  |  |  |  |
| RT-only soft-argmax | 0.886 ± 0.007 | 0.538 ± 0.059 | 0.759 ± 0.018 | 0.1070 ± 0.0303 | 0.357 ± 0.015 | 0.040 ± 0.024 |
| CE | 0.868 ± 0.002 | 0.581 ± 0.035 | 0.778 ± 0.004 | 0.0770 ± 0.0144 | 0.334 ± 0.008 | 0.101 ± 0.014 |
| Mixture EventNLL | 0.866 ± 0.004 | 0.602 ± 0.048 | 0.773 ± 0.006 | -0.0820 ± 0.0122 | 0.471 ± 0.008 | 0.207 ± 0.017 |
| **ETS-TCN** |  |  |  |  |  |  |
| RT-only soft-argmax | 0.881 ± 0.006 | 0.519 ± 0.042 | 0.758 ± 0.012 | 0.0268 ± 0.0114 | 0.410 ± 0.012 | 0.080 ± 0.031 |
| CE | 0.870 ± 0.008 | 0.551 ± 0.025 | 0.774 ± 0.006 | 0.0710 ± 0.0243 | 0.336 ± 0.013 | 0.099 ± 0.016 |
| Mixture EventNLL | 0.867 ± 0.005 | 0.544 ± 0.036 | 0.761 ± 0.010 | -0.0835 ± 0.0039 | 0.471 ± 0.007 | 0.187 ± 0.017 |
| **ETS-InceptionPyramid** |  |  |  |  |  |  |
| RT-only soft-argmax | 0.882 ± 0.005 | 0.506 ± 0.035 | 0.758 ± 0.007 | 0.1634 ± 0.0195 | 0.318 ± 0.007 | 0.103 ± 0.020 |
| CE | 0.863 ± 0.004 | 0.533 ± 0.032 | 0.781 ± 0.012 | 0.0605 ± 0.0173 | 0.340 ± 0.012 | 0.097 ± 0.017 |
| Mixture EventNLL | 0.866 ± 0.002 | 0.589 ± 0.037 | 0.774 ± 0.005 | -0.0925 ± 0.0049 | 0.473 ± 0.008 | 0.185 ± 0.019 |
| **ETS-AttnSeg** |  |  |  |  |  |  |
| RT-only soft-argmax | 0.899 ± 0.033 | 0.479 ± 0.156 | 0.749 ± 0.028 | 0.2977 ± 0.1669 | 0.268 ± 0.048 | 0.156 ± 0.046 |
| CE | 0.868 ± 0.009 | 0.566 ± 0.057 | 0.771 ± 0.005 | 0.0771 ± 0.0276 | 0.334 ± 0.016 | 0.103 ± 0.019 |
| Mixture EventNLL | 0.867 ± 0.008 | 0.639 ± 0.034 | 0.767 ± 0.004 | -0.0937 ± 0.0083 | 0.477 ± 0.007 | 0.173 ± 0.023 |
