# Main Table 1: Regression Baselines

Intended placement: `Regression Controls`.

Caption draft: Direct-regression and wrapped external backbone baselines. Values
are mean +/- sample standard deviation over seeds 2025-2029. Lower nRMSE is
better.

| Model | Role | Valid nRMSE | Holdout nRMSE |
| --- | --- | ---: | ---: |
| MSP-CNN | segment-pooling scalar baseline | 0.9006 +/- 0.0051 | 0.8998 +/- 0.0080 |
| ETR-CNN | temporal-readout direct regression | 0.9008 +/- 0.0060 | 0.8977 +/- 0.0068 |
| ETR-CNN large | capacity ablation | 0.8972 +/- 0.0040 | 0.8928 +/- 0.0042 |
| TIDNet wrapped | thinker-invariant CNN baseline | 0.9235 +/- 0.0024 | 0.9192 +/- 0.0027 |
| EEGConformer wrapped | conv-transformer baseline | 0.9188 +/- 0.0026 | 0.9287 +/- 0.0057 |
| EEGNet wrapped | compact EEG CNN baseline | 0.9350 +/- 0.0054 | 0.9335 +/- 0.0028 |
| LaBraM wrapped | foundation-style architecture from scratch | 0.9304 +/- 0.0061 | 0.9327 +/- 0.0086 |
| Deep4Net wrapped | classical deep ConvNet baseline | 0.9269 +/- 0.0045 | 0.9260 +/- 0.0044 |
| ShallowFBCSPNet wrapped | shallow FBCSP-style CNN baseline | 0.9324 +/- 0.0016 | 0.9343 +/- 0.0024 |
| ATCNet wrapped | conv/attention/TCN baseline | 0.9686 +/- 0.0183 | 0.9666 +/- 0.0151 |
| EEGPT wrapped | foundation-style architecture from scratch | 0.9616 +/- 0.0201 | 0.9584 +/- 0.0185 |
| Medformer wrapped | larger transformer/time-series baseline | 0.9623 +/- 0.0046 | 0.9585 +/- 0.0051 |

Paper note: ETR-CNN large is the strongest scalar baseline, while MSP-CNN and
base ETR-CNN remain close. The wrapped external EEG architectures train
meaningfully under the same protocol but do not close the gap to the compact
task-specific temporal controls.
