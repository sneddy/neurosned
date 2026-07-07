# 03 Crop Shift Jitter

Paper-facing configs for the shift-jitter training experiment under the
support-filtered protocol.

The goal is to test whether event-time supervision becomes more localizer-like
when the fixed-window shortcut is weakened during training. Models still
receive 2 s inputs, but training crops are sampled from 5 s R1-R8 windows with
crop starts drawn from `0.2` to `0.8` s after stimulus onset.

The train/evaluation support is restricted to `0.8 <= RT <= 2.2`. This is the
common-inside target range for all 2 s crops starting in `0.2, 0.3, ..., 0.8`,
so the response remains inside every shifted crop used by the diagnostic.

Training uses:

- `data/new_validation/r1_r8_train_5sec.pkl`
- `TrainCroppingDataset`
- `crop_start_min: 0.2`
- `crop_start_max: 0.8`
- `crop_sec: 2.0`

Validation and standard R11 holdout still use the canonical fixed 2 s windows,
but filtered to the same `0.8--2.2` support. Shifted-crop evaluation runs
automatically on `data/new_validation/r11_test_5sec.pkl`.

Run all configs:

```bash
sh /home/sneddy/sneddy_projects/neurosned/benchmarks/runners/run_crop_shift_jitter.sh
```

Primary comparison:

- `ets_unet_ce_shift_jitter`: soft-label event-time CE.
- `ets_unet_event_nll_shift_jitter`: latent EventNLL objective.
- `ets_unet_event_nll_mixture_shift_jitter`: two-scale EventNLL observation
  kernel.
- `ets_unet_hazard_event_nll_shift_jitter`: hazard/survival event-time
  parameterization.
- `ets_unet_time_only_shift_jitter`: soft-argmax scalar control.
- `ets_unet_wasserstein_shift_jitter`: CDF-distance geometry control.
