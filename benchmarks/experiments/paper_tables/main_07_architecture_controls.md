# Architecture Controls for Dense Event-Time Prediction

This document mirrors the main-text architecture-control table currently
reported in the manuscript.

## Main-Text Scalar Accuracy Comparison

Manuscript label: `tab:architecture_robustness`

**Caption:** Scalar RT accuracy across event-time backbones and objectives.

| Objective | Valid nRMSE | Holdout nRMSE | Holdout τ-nRMSE |
| --- | ---: | ---: | ---: |
| **ETS-U-Net** |  |  |  |
| RT-only soft-argmax | 0.8944 ± 0.0048 | 0.8943 ± 0.0025 | 0.8917 ± 0.0046 |
| CE | 0.8763 ± 0.0044 | 0.8774 ± 0.0044 | 0.8753 ± 0.0039 |
| Mixture EventNLL | 0.8744 ± 0.0018 | 0.8785 ± 0.0047 | **0.8745 ± 0.0053** |
| **ETS-TCN** |  |  |  |
| RT-only soft-argmax | 0.8865 ± 0.0029 | 0.8853 ± 0.0044 | 0.8842 ± 0.0045 |
| CE | 0.8717 ± 0.0025 | 0.8751 ± 0.0085 | 0.8730 ± 0.0038 |
| Mixture EventNLL | 0.8718 ± 0.0046 | 0.8729 ± 0.0020 | **0.8722 ± 0.0021** |
| **ETS-InceptionPyramid** |  |  |  |
| RT-only soft-argmax | 0.8970 ± 0.0043 | 0.8894 ± 0.0054 | 0.8886 ± 0.0044 |
| CE | 0.8710 ± 0.0037 | 0.8717 ± 0.0017 | **0.8717 ± 0.0036** |
| Mixture EventNLL | 0.8703 ± 0.0018 | 0.8780 ± 0.0026 | 0.8746 ± 0.0028 |
| **ETS-AttnSeg** |  |  |  |
| RT-only soft-argmax | 0.9114 ± 0.0350 | 0.9105 ± 0.0294 | 0.9052 ± 0.0300 |
| CE | 0.8742 ± 0.0036 | 0.8811 ± 0.0097 | 0.8780 ± 0.0082 |
| Mixture EventNLL | 0.8719 ± 0.0082 | 0.8823 ± 0.0079 | **0.8752 ± 0.0050** |
