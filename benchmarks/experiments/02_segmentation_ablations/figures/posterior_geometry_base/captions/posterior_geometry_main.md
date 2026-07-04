# posterior_geometry_main

## Draft Caption

Output geometry of event-time posteriors learned by matched segmentation losses. All panels use the `base` readout from saved temporal logits. (A) R11 scalar NRMSE anchors the comparison and shows that several losses have similar scalar RT error. (B) Predicted posterior distributions are aligned to each trial's observed RT and averaged, so zero on the x-axis denotes the true RT. (C-E) Subject-level distributions of posterior width, near-target mass, and mode-mean gap summarize whether the scalar prediction is supported by a localized temporal event distribution. (F) Empirical coverage of central posterior intervals evaluates whether posterior concentration should be interpreted as calibrated uncertainty. In this run set, EventNLL produces a more concentrated posterior (80% width 410 ms; mass +/-150 ms 0.495) whereas time-only training has the largest mode-mean gap (108 ms).

## Analysis Notes

Posterior-geometry summaries use 15,164/15,751 representable trials with targets in [0.50, 2.49] s; 587 trials outside this event-time window are excluded from these summaries.

Key summary values:

- CE: NRMSE=0.939, 80% width=770 ms, mass +/-150 ms=0.337, mode-mean gap=53 ms.
- CE+time: NRMSE=0.940, 80% width=730 ms, mass +/-150 ms=0.336, mode-mean gap=47 ms.
- EventNLL: NRMSE=0.943, 80% width=410 ms, mass +/-150 ms=0.495, mode-mean gap=65 ms.
- Time-only: NRMSE=0.952, 80% width=860 ms, mass +/-150 ms=0.348, mode-mean gap=108 ms.
- Wasserstein: NRMSE=0.961, 80% width=370 ms, mass +/-150 ms=0.412, mode-mean gap=41 ms.

Interpretation: scalar RT error alone hides output geometry. EventNLL does not primarily improve scalar NRMSE here; it changes the learned posterior by concentrating probability mass near the observed RT. Time-only training can recover the posterior mean while leaving a broader or less coherent temporal event map.
