# Main Table 3: Shift-Jitter Training Summary

Intended placement: `Posterior Readout and Diagnostics / Shifted-Crop
Diagnostic`.

Caption draft: Summary of the shift-jitter intervention under the canonical
holdout protocol. Holdout tau nRMSE is evaluated on the standard fixed-window
holdout support (`0.5 <= RT <= 2.5`) after temperature calibration. Shifted rel
nRMSE measures crop-relative prediction accuracy on the 5 s holdout shifted-crop
diagnostic. Sensitivity quantifies the fraction of the imposed crop shift
reflected in the prediction, with 1 indicating ideal crop-relative localization
and 0 indicating crop-invariant behavior; direction is the fraction of shifted
examples whose prediction moves in the expected localizer direction. Shift-jitter
preserves ordinary fixed-window accuracy, improves shifted-crop robustness, and
modestly improves localization behavior.

| Objective | Holdout tau nRMSE fixed | Holdout tau nRMSE jitter | Shifted rel nRMSE fixed | Shifted rel nRMSE jitter | Sensitivity fixed | Sensitivity jitter | Direction fixed | Direction jitter |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CE | 0.8753 +/- 0.0039 | 0.8749 +/- 0.0060 | 0.8679 +/- 0.0024 | 0.8569 +/- 0.0060 | 0.5810 +/- 0.0351 | 0.5831 +/- 0.0406 | 0.7780 +/- 0.0041 | 0.7924 +/- 0.0056 |
| Mixture EventNLL | 0.8745 +/- 0.0053 | 0.8734 +/- 0.0033 | 0.8663 +/- 0.0037 | 0.8576 +/- 0.0041 | 0.6021 +/- 0.0484 | 0.6249 +/- 0.0313 | 0.7730 +/- 0.0062 | 0.7925 +/- 0.0076 |
| EventNLL | 0.8772 +/- 0.0018 | 0.8771 +/- 0.0040 | 0.8685 +/- 0.0023 | 0.8593 +/- 0.0028 | 0.5842 +/- 0.0287 | 0.6174 +/- 0.0222 | 0.7739 +/- 0.0059 | 0.7937 +/- 0.0068 |
| Hazard EventNLL | 0.8778 +/- 0.0041 | 0.8806 +/- 0.0034 | 0.8692 +/- 0.0028 | 0.8609 +/- 0.0045 | 0.5618 +/- 0.0376 | 0.5891 +/- 0.0259 | 0.7694 +/- 0.0090 | 0.7925 +/- 0.0074 |
| Soft-argmax RT loss | 0.8917 +/- 0.0046 | 0.8836 +/- 0.0035 | 0.8857 +/- 0.0069 | 0.8613 +/- 0.0032 | 0.5381 +/- 0.0586 | 0.5775 +/- 0.0362 | 0.7592 +/- 0.0183 | 0.7951 +/- 0.0052 |
| Wasserstein | 0.8896 +/- 0.0033 | 0.8922 +/- 0.0066 | 0.8932 +/- 0.0033 | 0.8774 +/- 0.0099 | 0.6684 +/- 0.0711 | 0.6852 +/- 0.0416 | 0.7830 +/- 0.0197 | 0.8026 +/- 0.0118 |

Paper note: the intervention does not produce a broad ordinary-holdout gain
under the canonical support. Its main effect is improved shifted-crop robustness,
with only partial improvement in localization behavior.
