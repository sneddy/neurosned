# Main Table 4: Shift-Jitter Training and Localization Behavior

Intended placement: `Posterior Readout and Diagnostics / Shift-Jitter Training
Intervention`.

Caption draft: Localization-behavior comparison between fixed-window-trained
and shift-jitter-trained event-time models under the shifted-crop diagnostic.
Sensitivity quantifies the fraction of the imposed crop shift reflected in the
prediction, with 1 indicating ideal crop-relative localization and 0 indicating
crop-invariant behavior. Direction is the fraction of shifted examples whose
prediction moves in the expected localizer direction. Deltas are computed as
shift-jitter minus fixed-window training; positive values indicate stronger
localizer-like behavior after jitter training.

| Objective | Sensitivity fixed | Sensitivity jitter | Sensitivity delta | Direction fixed | Direction jitter | Direction delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CE | 0.5810 +/- 0.0351 | 0.5831 +/- 0.0406 | +0.0021 | 0.7780 +/- 0.0041 | 0.7924 +/- 0.0056 | +0.0144 |
| Mixture EventNLL | 0.6021 +/- 0.0484 | 0.6249 +/- 0.0313 | +0.0228 | 0.7730 +/- 0.0062 | 0.7925 +/- 0.0076 | +0.0195 |
| EventNLL | 0.5842 +/- 0.0287 | 0.6174 +/- 0.0222 | +0.0332 | 0.7739 +/- 0.0059 | 0.7937 +/- 0.0068 | +0.0199 |
| Hazard EventNLL | 0.5618 +/- 0.0376 | 0.5891 +/- 0.0259 | +0.0272 | 0.7694 +/- 0.0090 | 0.7925 +/- 0.0074 | +0.0231 |
| Soft-argmax RT loss | 0.5381 +/- 0.0586 | 0.5775 +/- 0.0362 | +0.0395 | 0.7592 +/- 0.0183 | 0.7951 +/- 0.0052 | +0.0359 |
| Wasserstein | 0.6684 +/- 0.0711 | 0.6852 +/- 0.0416 | +0.0167 | 0.7830 +/- 0.0197 | 0.8026 +/- 0.0118 | +0.0196 |

Paper note: shift-jitter training modestly improves localization behavior, but
the best sensitivity remains well below the ideal value of 1.0. This supports a
partial-localization interpretation rather than a claim of solved
crop-relative event localization.
