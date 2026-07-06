# posterior_geometry_main

## Draft Caption

Output geometry of event-time posteriors learned by matched segmentation losses. All panels use the `calibrated` readout from saved temporal logits. (A) R11 scalar NRMSE anchors the comparison and shows that several losses have similar scalar RT error. (B) Predicted posterior distributions are aligned to each trial's observed RT and averaged, so zero on the x-axis denotes the true RT. (C-E) Subject-level distributions of posterior width, near-target mass, and mode-mean gap summarize whether the scalar prediction is supported by a localized temporal event distribution. In panel E, triangles at the upper axis boundary mark subject-level mode-mean gaps above 300 ms, clipped for readability. (F) Empirical coverage of central posterior intervals evaluates whether posterior concentration should be interpreted as calibrated uncertainty. In this run set, EventNLL produces a more concentrated posterior (80% width 450 ms; mass +/-150 ms 0.487) whereas time-only training has the largest mode-mean gap (129 ms).

## Analysis Notes

Posterior-geometry summaries use 15,164/15,751 representable trials with targets in [0.50, 2.49] s; 587 trials outside this event-time window are excluded from these summaries.

Key summary values:

- CE: NRMSE=0.937, 80% width=820 ms, mass +/-150 ms=0.328, mode-mean gap=57 ms.
- CE+time: NRMSE=0.936, 80% width=860 ms, mass +/-150 ms=0.311, mode-mean gap=57 ms.
- EventNLL: NRMSE=0.941, 80% width=450 ms, mass +/-150 ms=0.487, mode-mean gap=72 ms.
- Time-only: NRMSE=0.945, 80% width=1070 ms, mass +/-150 ms=0.311, mode-mean gap=129 ms.
- Wasserstein: NRMSE=0.950, 80% width=740 ms, mass +/-150 ms=0.296, mode-mean gap=50 ms.

Interpretation: scalar RT error alone hides output geometry. EventNLL does not primarily improve scalar NRMSE here; it changes the learned posterior by concentrating probability mass near the observed RT. Time-only training can recover the posterior mean while leaving a broader or less coherent temporal event map.
