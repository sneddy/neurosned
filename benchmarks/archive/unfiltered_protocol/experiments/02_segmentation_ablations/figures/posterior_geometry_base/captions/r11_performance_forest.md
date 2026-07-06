# r11_performance_forest

## Draft Caption

R11 scalar readout performance with subject-bootstrap confidence intervals. The direct-regression row is automatically selected as the best available R11 direct-regression baseline from `benchmarks/experiments/01_regression_baselines`; event-time rows use the calibrated segmentation readout from saved logits. Lower nRMSE is better.

## Values

- Best direct regression SneddyRTNet: nRMSE=0.946, CI=[0.930, 0.964].
- CE: nRMSE=0.939, CI=[0.922, 0.958].
- CE+time: nRMSE=0.940, CI=[0.921, 0.960].
- EventNLL: nRMSE=0.943, CI=[0.925, 0.963].
- Time-only: nRMSE=0.952, CI=[0.937, 0.968].
- Wasserstein: nRMSE=0.961, CI=[0.944, 0.979].

## Interpretation

This plot anchors the output-geometry analysis: several event-time losses are close in scalar R11 nRMSE, motivating posterior-level visualization rather than relying only on point-prediction error.
