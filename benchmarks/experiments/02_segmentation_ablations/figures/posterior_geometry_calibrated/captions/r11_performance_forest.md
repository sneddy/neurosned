# r11_performance_forest

## Draft Caption

R11 scalar readout performance with subject-bootstrap confidence intervals. The direct-regression row is automatically selected as the best available R11 direct-regression baseline from `benchmarks/experiments/01_regression_baselines`; event-time rows use the calibrated segmentation readout from saved logits. Lower nRMSE is better.

## Values

- Best direct regression SneddyRTNet: nRMSE=0.946, CI=[0.930, 0.964].
- CE: nRMSE=0.937, CI=[0.920, 0.956].
- CE+time: nRMSE=0.936, CI=[0.918, 0.954].
- EventNLL: nRMSE=0.941, CI=[0.923, 0.960].
- Time-only: nRMSE=0.945, CI=[0.931, 0.960].
- Wasserstein: nRMSE=0.950, CI=[0.935, 0.966].

## Interpretation

This plot anchors the output-geometry analysis: several event-time losses are close in scalar R11 nRMSE, motivating posterior-level visualization rather than relying only on point-prediction error.
