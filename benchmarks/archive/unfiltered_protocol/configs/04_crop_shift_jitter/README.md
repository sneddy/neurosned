# 04 Crop Shift Jitter

Paper-facing configs for the shift-jitter training experiment.

The purpose of this group is to test whether event-time supervision becomes
more localizer-like when the fixed-window shortcut is weakened during training.
Each config trains on 2 s crops sampled from the 5 s training windows with
crop starts sampled from `0.2` to `0.8` s after stimulus onset. The target is
converted to crop-relative time by `TrainCroppingDataset`.

Validation and standard holdout evaluation still use the original 2 s
`0.5--2.5` s windows. The shifted-crop diagnostic is then run automatically on
the R11 5 s split through `evaluation.shifted_crop`. Shifted-crop evaluation is
restricted to `0.8 <= RT <= 2.2`, the common-inside target range for crop starts
`0.2--0.8` with 2 s crops.

Run examples:

```bash
python benchmarks/scripts/run.py benchmarks/configs/04_crop_shift_jitter/unet_deeper_ce_shift_jitter.yaml
python benchmarks/scripts/run.py benchmarks/configs/04_crop_shift_jitter/unet_deeper_comboloss_shift_jitter.yaml
python benchmarks/scripts/run.py benchmarks/configs/04_crop_shift_jitter/unet_deeper_event_nll_shift_jitter.yaml
python benchmarks/scripts/run.py benchmarks/configs/04_crop_shift_jitter/unet_deeper_time_only_shift_jitter.yaml
```

Primary comparison:

- `unet_deeper_ce_shift_jitter`: distributional soft-label event-time CE.
- `unet_deeper_comboloss_shift_jitter`: current strong CE+time scalar row under
  shift-jitter training.
- `unet_deeper_event_nll_shift_jitter`: principled latent EventNLL objective.
- `unet_deeper_time_only_shift_jitter`: control for soft-argmax readout plus
  shift-jitter training without distributional supervision.

Expected paper question:

> Does crop-relative shift-jitter training move event-time models closer to
> true within-crop localization than fixed-window training?
