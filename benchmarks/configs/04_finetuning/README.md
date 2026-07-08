# 04 Finetuning

Optional final-stage configs after the main baseline and shift-jitter stages are
frozen.

The intent is not to expand the ablation matrix. These configs start from a
completed checkpoint and test whether a short, lower-LR finetuning stage with a
more aggressive legacy augmentation profile improves the final ETS-U-Net
recipe.

Current first probe:

| config | start checkpoint | role |
| --- | --- | --- |
| `ets_unet_nll_mixture.yaml` | best validation mixture EventNLL shift-jitter seed2026 checkpoint | Single-seed finetune probe using mixture EventNLL, the shift-jitter crop profile, no global target filter, SGD, large batches, mixup, and stronger legacy-style augmentations. |

Important constraint:

- `run_repeated.py` does not yet substitute seed-specific input checkpoints.
  This first config is therefore a single-checkpoint probe, not a full
  five-seed finetuning result.
- This probe intentionally sets `target_min: null` and `target_max: null`.
  Training crops still use `TrainCroppingDataset`, which clips sampled crop
  starts to keep the RT inside the 2 s crop whenever possible.

Run:

```bash
python benchmarks/scripts/run.py benchmarks/configs/04_finetuning/ets_unet_nll_mixture.yaml --device cuda
```

If this single probe is useful, the next clean step is either to add
seed-specific finetune configs or extend the repeated runner to map each seed
to its matching base checkpoint.
