# 01 Regression Baselines

Five-seed, support-filtered direct-regression baselines for the paper protocol.

All configs use the release-separated split:

- train: `R1-R8`
- validation/development: `R9-R10`
- holdout: `R11`

The primary protocol filters train, validation, and holdout rows to the
observable 2 s inference window:

```yaml
data:
  target_min: 0.5
  target_max: 2.5
```

Each config enables repeated runs with seeds `2025, 2026, 2027, 2028, 2029`.
Each config also enables the shifted-crop diagnostic on
`data/new_validation/r11_test_5sec.pkl` without an extra target-range filter;
the saved `shifted_summary.csv` contains `all`, `inside_crop`, and
`common_inside` rows for paper-facing metric selection.

Run with:

```bash
python benchmarks/scripts/run_repeated.py benchmarks/configs/01_regression_baselines/<config>.yaml
```

Paper-facing in-house model names:

- `msp_cnn`: MSP-CNN, the compact multiscale segment-pooling scalar baseline.
- `etr_cnn`: ETR-CNN, the event-time-readout direct-regression baseline.
- `etr_cnn_large`: ETR-CNN capacity ablation.

Wrapped external baselines use the same per-window standardization protocol as
the in-house models.
