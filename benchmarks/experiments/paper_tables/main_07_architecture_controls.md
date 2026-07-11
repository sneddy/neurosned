# Architecture Controls for Dense Event-Time Prediction

This document contains manuscript-ready text and compact appendix tables for
the completed architecture-control experiments. The proposed integration keeps
the main text focused on the output formulation and places the numerical
architecture comparison in the appendix.

## Main-Text Insertion

**Placement:** at the end of *Event-Time Segmentation Architecture*, after the
ETS-U-Net description. Replace the appendix label only if a different label is
used during manuscript integration.

```latex
ETS-U-Net serves as the primary fixed backbone for the objective comparisons,
but the supervision effect is not specific to its encoder--decoder design. We
repeated the RT-only soft-argmax, CE, and mixture EventNLL comparisons with
three additional dense temporal backbones: a full-resolution dilated temporal
convolutional network (ETS-TCN), an explicit multi-scale temporal model
(ETS-InceptionPyramid), and an attention-based temporal segmenter
(ETS-AttnSeg). Across all four backbones, mixture EventNLL improved holdout
\(\tau\)-nRMSE over RT-only posterior-mean supervision and remained competitive
with CE. These controls therefore support the conclusion that the gain arises
from distributional event-time supervision rather than from the ETS-U-Net
architecture or differentiable expectation readout alone; full results are
reported in Appendix~\ref{app:architecture_controls}.
```

## Appendix-Ready Subsection

### Architecture Controls for Dense Event-Time Prediction

ETS-U-Net is the primary backbone used to isolate the effect of the event-time
objective. To test whether the resulting supervision effect depends on its
encoder-decoder structure, we repeated the RT-only soft-argmax, CE, and mixture
EventNLL comparisons with three alternative dense temporal backbones. ETS-TCN
uses full-resolution residual dilated convolutions to obtain long temporal
context without temporal pooling or decoder skip connections.
ETS-InceptionPyramid applies parallel temporal filters at multiple receptive
field widths and fuses them at the original temporal resolution. ETS-AttnSeg
combines temporal self-attention, local depthwise temporal convolution, and
feed-forward blocks. Thus, the controls span distinct temporal inductive biases
while preserving the same 200-bin event-time output grid and posterior-mean
readout. They are trained from scratch under the shared preprocessing,
optimization, augmentation, checkpoint-selection, temperature-selection, and
holdout-evaluation protocol. Model capacity is also kept within a narrow range
(3.05--3.25 million trainable parameters), but parameter counts are omitted
from the results tables because capacity is not the experimental variable of
interest.

Table~\ref{tab:architecture_control_accuracy} reports scalar accuracy. Mixture
EventNLL improves holdout \(\tau\)-nRMSE over RT-only posterior-mean supervision
for every backbone, with absolute reductions of 0.0172 for ETS-U-Net, 0.0120
for ETS-TCN, 0.0140 for ETS-InceptionPyramid, and 0.0300 for ETS-AttnSeg. CE and
mixture EventNLL form the strongest objective group: mixture EventNLL has the
lower mean for ETS-U-Net, ETS-TCN, and ETS-AttnSeg, whereas CE has the lower
mean for ETS-InceptionPyramid. The purpose of these controls is therefore not
to rank architectures, but to show that the advantage of distributional
event-time supervision over an RT-only temporal expectation readout persists
across alternative dense temporal models. The larger variability of the
ETS-AttnSeg RT-only result reflects one markedly weaker run; all five seeds are
retained in the summary.

**Table caption:** Architecture-control comparison for dense temporal
event-time models. All models use the same 200-bin output grid, posterior-mean
readout, readout-temperature selection on R9--R10, and fixed R11 holdout
evaluation. Values are mean ± standard deviation across five seeds; lower is
better. Bold marks the lowest mean holdout \(\tau\)-nRMSE within each
architecture.

| Architecture | Objective | Valid nRMSE | Holdout nRMSE | Holdout τ-nRMSE |
| --- | --- | ---: | ---: | ---: |
| ETS-U-Net | RT-only soft-argmax | 0.8944 ± 0.0048 | 0.8943 ± 0.0025 | 0.8917 ± 0.0046 |
| ETS-U-Net | CE | 0.8763 ± 0.0044 | 0.8774 ± 0.0044 | 0.8753 ± 0.0039 |
| ETS-U-Net | Mixture EventNLL | 0.8744 ± 0.0018 | 0.8785 ± 0.0047 | **0.8745 ± 0.0053** |
| ETS-TCN | RT-only soft-argmax | 0.8865 ± 0.0029 | 0.8853 ± 0.0044 | 0.8842 ± 0.0045 |
| ETS-TCN | CE | 0.8717 ± 0.0025 | 0.8751 ± 0.0085 | 0.8730 ± 0.0038 |
| ETS-TCN | Mixture EventNLL | 0.8718 ± 0.0046 | 0.8729 ± 0.0020 | **0.8722 ± 0.0021** |
| ETS-InceptionPyramid | RT-only soft-argmax | 0.8970 ± 0.0043 | 0.8894 ± 0.0054 | 0.8886 ± 0.0044 |
| ETS-InceptionPyramid | CE | 0.8710 ± 0.0037 | 0.8717 ± 0.0017 | **0.8717 ± 0.0036** |
| ETS-InceptionPyramid | Mixture EventNLL | 0.8703 ± 0.0018 | 0.8780 ± 0.0026 | 0.8746 ± 0.0028 |
| ETS-AttnSeg | RT-only soft-argmax | 0.9114 ± 0.0350 | 0.9105 ± 0.0294 | 0.9052 ± 0.0300 |
| ETS-AttnSeg | CE | 0.8742 ± 0.0036 | 0.8811 ± 0.0097 | 0.8780 ± 0.0082 |
| ETS-AttnSeg | Mixture EventNLL | 0.8719 ± 0.0082 | 0.8823 ± 0.0079 | **0.8752 ± 0.0050** |

### Shifted-Crop Diagnostics

The architecture controls preserve the distinction between fixed-window scalar
accuracy and crop-relative localization behavior. Table~\ref{tab:architecture_control_shifted_crop}
uses the same shifted-crop convention as the main analysis: shifted relative
nRMSE is evaluated on crop examples in which the behavioral response remains
inside the evaluated crop, whereas sensitivity and direction use the common
trial subset for which the response remains inside every crop. Architecture
and objective choice affect crop-relative behavior, but sensitivity remains
well below the ideal value of 1 for all models.

**Table caption:** Shifted-crop diagnostics for the architecture controls.
Values are mean ± standard deviation across five seeds. Lower shifted relative
nRMSE is better; higher sensitivity and direction are more localizer-like.

| Architecture | Objective | Shifted rel. nRMSE | Sensitivity | Direction |
| --- | --- | ---: | ---: | ---: |
| ETS-U-Net | RT-only soft-argmax | 0.886 ± 0.007 | 0.538 ± 0.059 | 0.759 ± 0.018 |
| ETS-U-Net | CE | 0.868 ± 0.002 | 0.581 ± 0.035 | 0.778 ± 0.004 |
| ETS-U-Net | Mixture EventNLL | 0.866 ± 0.004 | 0.602 ± 0.048 | 0.773 ± 0.006 |
| ETS-TCN | RT-only soft-argmax | 0.881 ± 0.006 | 0.519 ± 0.042 | 0.758 ± 0.012 |
| ETS-TCN | CE | 0.870 ± 0.008 | 0.551 ± 0.025 | 0.774 ± 0.006 |
| ETS-TCN | Mixture EventNLL | 0.867 ± 0.005 | 0.544 ± 0.036 | 0.761 ± 0.010 |
| ETS-InceptionPyramid | RT-only soft-argmax | 0.882 ± 0.005 | 0.506 ± 0.035 | 0.758 ± 0.007 |
| ETS-InceptionPyramid | CE | 0.863 ± 0.004 | 0.533 ± 0.032 | 0.781 ± 0.012 |
| ETS-InceptionPyramid | Mixture EventNLL | 0.866 ± 0.002 | 0.589 ± 0.037 | 0.774 ± 0.005 |
| ETS-AttnSeg | RT-only soft-argmax | 0.899 ± 0.033 | 0.479 ± 0.156 | 0.749 ± 0.028 |
| ETS-AttnSeg | CE | 0.868 ± 0.009 | 0.566 ± 0.057 | 0.771 ± 0.005 |
| ETS-AttnSeg | Mixture EventNLL | 0.867 ± 0.008 | 0.639 ± 0.034 | 0.767 ± 0.004 |

### Posterior Geometry

Posterior diagnostics show that the distinction between scalar accuracy and
posterior structure also transfers across backbones. As shown in
Table~\ref{tab:architecture_control_posterior_geometry}, mixture EventNLL
consistently yields the best fixed-kernel EventNLL and greater target-aligned
mass than CE or RT-only supervision. Its posterior intervals are substantially
narrower than those produced by CE, but their empirical latent-event coverage
is lower. Thus, the posterior-geometry trade-off is a property of the
supervision objective rather than an ETS-U-Net-specific artifact.

**Table caption:** Posterior geometry for the architecture controls on the
holdout split after development-set readout-temperature selection. Values are
mean ± standard deviation across five seeds. Lower is better for fixed-kernel
EventNLL, Width80, and Coverage MAE; higher is better for Mass ±150 ms.
Coverage80 is reported against its nominal level of 0.8.

| Architecture | Objective | Fixed-kernel EventNLL | Width80, ms | Mass ±150 ms | Coverage80 | Coverage MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ETS-U-Net | RT-only soft-argmax | 0.1070 ± 0.0303 | 844.0 ± 95.6 | 0.357 ± 0.015 | 0.843 ± 0.029 | 0.040 ± 0.024 |
| ETS-U-Net | CE | 0.0770 ± 0.0144 | 766.0 ± 35.8 | 0.334 ± 0.008 | 0.883 ± 0.013 | 0.101 ± 0.014 |
| ETS-U-Net | Mixture EventNLL | -0.0820 ± 0.0122 | 528.0 ± 21.7 | 0.471 ± 0.008 | 0.573 ± 0.020 | 0.207 ± 0.017 |
| ETS-TCN | RT-only soft-argmax | 0.0268 ± 0.0114 | 526.0 ± 40.4 | 0.410 ± 0.012 | 0.717 ± 0.034 | 0.080 ± 0.031 |
| ETS-TCN | CE | 0.0710 ± 0.0243 | 758.0 ± 47.6 | 0.336 ± 0.013 | 0.883 ± 0.014 | 0.099 ± 0.016 |
| ETS-TCN | Mixture EventNLL | -0.0835 ± 0.0039 | 534.0 ± 30.5 | 0.471 ± 0.007 | 0.596 ± 0.020 | 0.187 ± 0.017 |
| ETS-InceptionPyramid | RT-only soft-argmax | 0.1634 ± 0.0195 | 986.0 ± 67.7 | 0.318 ± 0.007 | 0.896 ± 0.018 | 0.103 ± 0.020 |
| ETS-InceptionPyramid | CE | 0.0605 ± 0.0173 | 748.0 ± 44.4 | 0.340 ± 0.012 | 0.879 ± 0.015 | 0.097 ± 0.017 |
| ETS-InceptionPyramid | Mixture EventNLL | -0.0925 ± 0.0049 | 540.0 ± 38.1 | 0.473 ± 0.008 | 0.600 ± 0.023 | 0.185 ± 0.019 |
| ETS-AttnSeg | RT-only soft-argmax | 0.2977 ± 0.1669 | 1154.0 ± 206.6 | 0.268 ± 0.048 | 0.926 ± 0.029 | 0.156 ± 0.046 |
| ETS-AttnSeg | CE | 0.0771 ± 0.0276 | 776.0 ± 57.7 | 0.334 ± 0.016 | 0.885 ± 0.015 | 0.103 ± 0.019 |
| ETS-AttnSeg | Mixture EventNLL | -0.0937 ± 0.0083 | 568.0 ± 23.9 | 0.477 ± 0.007 | 0.608 ± 0.027 | 0.173 ± 0.023 |

### Interpretation Boundary

These controls establish robustness across the tested dense temporal
architectures; they are not a general benchmark of EEG backbones and do not
establish that ETS-U-Net is the best architecture. The supported conclusion is
that distributional event-time supervision, including mixture EventNLL,
outperforms RT-only posterior-mean supervision across the tested backbones while
preserving objective-dependent posterior-geometry trade-offs.
