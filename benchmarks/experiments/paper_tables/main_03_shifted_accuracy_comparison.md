# Main Table 3: Shift-Jitter Training and Shifted-Crop Accuracy

Intended placement: `Posterior Readout and Diagnostics / Shifted-Crop
Localization Diagnostic`.

Caption draft: Accuracy comparison between fixed-window-trained and
shift-jitter-trained event-time models. Holdout tau nRMSE reports the
temperature-calibrated fixed-window score. Shifted rel nRMSE reports
crop-relative prediction accuracy when the same holdout trials are evaluated
with multiple 2 s crops starting from 0.2 to 0.8 s after stimulus onset. The two
nRMSE scores use different target distributions and should not be subtracted
from each other; the shifted delta compares only shifted-crop rel nRMSE after
shift-jitter training against the corresponding fixed-window-trained model.
The holdout delta analogously compares jitter-trained and fixed-window-trained
models on the fixed-window holdout score. Negative deltas indicate lower nRMSE
after jitter training.

| Objective | Holdout tau nRMSE, fixed train | Holdout tau nRMSE, jitter train | Holdout delta | Shifted rel nRMSE, fixed train | Shifted rel nRMSE, jitter train | Shifted delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CE | 0.8753 +/- 0.0039 | 0.8453 +/- 0.0075 | -0.0301 | 0.8679 +/- 0.0024 | 0.8569 +/- 0.0060 | -0.0110 |
| Mixture EventNLL | 0.8745 +/- 0.0053 | 0.8505 +/- 0.0065 | -0.0240 | 0.8663 +/- 0.0037 | 0.8576 +/- 0.0041 | -0.0087 |
| EventNLL | 0.8772 +/- 0.0018 | 0.8503 +/- 0.0049 | -0.0269 | 0.8685 +/- 0.0023 | 0.8593 +/- 0.0028 | -0.0092 |
| Hazard EventNLL | 0.8778 +/- 0.0041 | 0.8542 +/- 0.0063 | -0.0236 | 0.8692 +/- 0.0028 | 0.8609 +/- 0.0045 | -0.0083 |
| Soft-argmax RT loss | 0.8917 +/- 0.0046 | 0.8618 +/- 0.0030 | -0.0299 | 0.8857 +/- 0.0069 | 0.8613 +/- 0.0032 | -0.0244 |
| Wasserstein | 0.8896 +/- 0.0033 | 0.8658 +/- 0.0045 | -0.0238 | 0.8932 +/- 0.0033 | 0.8774 +/- 0.0099 | -0.0158 |

Paper note: shift-jitter training consistently improves shifted-crop rel nRMSE,
indicating better robustness to crop placement. Holdout tau nRMSE is reported
as fixed-window context, while the shifted delta is the direct crop-robustness
comparison.
