# Benchmark Runs

This file is the manual history of benchmark runs. Keep the command and output
directory here so each result can be reproduced from the exact YAML config.

## Running Commands

Run the current demo config from the project root:

```bash
python benchmarks/run.py benchmarks/configs/0_demo/unet_deeper_demo.yaml
```

Run the two-stage protocol-calibration demo:

```bash
python benchmarks/run.py benchmarks/configs/0_demo/unet_deeper_two_stage_demo.yaml
```

The artefacts will be written under:

```text
experiments/00_protocol_calibration/<run_name>/
```

## Run Designs

| name | date | config | run dir | notes |
| --- | --- | --- | --- | --- |
| unet_deeper_default | 2026-06-29 12:29 UTC | `configs/00_protocol_calibration/`<br>`unet_deeper_default.yaml` | `experiments/00_protocol_calibration/`<br>`unet_deeper_default__20260629_122932` | Simple protocol baseline: no warm-start, Adam lr=1e-3, no plateau reload, sigma=0.15. This run was written before the nrmse scale fix, so the stored raw score is divided by 1000 here for comparison. |
| unet_deeper_lr2e4 | 2026-06-29 13:22 UTC | `configs/00_protocol_calibration/`<br>`unet_deeper_lr2e4.yaml` | `experiments/00_protocol_calibration/`<br>`unet_deeper_lr2e4__20260629_132219` | Same as default, only Adam lr reduced to 2e-4. This run was written before the nrmse scale fix, so the stored raw score is divided by 1000 here for comparison. |
| unet_deeper_sgd | 2026-06-29 14:18 UTC | `configs/00_protocol_calibration/`<br>`unet_deeper_sgd.yaml` | `experiments/00_protocol_calibration/`<br>`unet_deeper_sgd__20260629_141821` | Same as default, only optimizer changed to SGD lr=1e-3. This run was written before the nrmse scale fix and was stopped by the old best-metric scale mismatch. |
| unet_deeper_adamw | 2026-06-29 14:50 UTC | `configs/00_protocol_calibration/`<br>`unet_deeper_adamw.yaml` | `experiments/00_protocol_calibration/`<br>`unet_deeper_adamw__20260629_145009` | Same as default, only optimizer changed to AdamW lr=1e-3 with weight_decay=0. Rerun after the nrmse scale fix; best checkpoint and predictions are now saved correctly. Because weight decay is zero, this is expected to be nearly identical to Adam. |

## Run Results

| name | epoch | valid_score | result note |
| --- | --- | --- | --- |
| unet_deeper_default | 21 | 0.930441 | Current best simple-protocol baseline by a tiny margin; validation worsened after epoch 21 while train kept improving. |
| unet_deeper_lr2e4 | 12 | 0.949753 | Worse than default; lr=2e-4 looks too conservative for this simple protocol. |
| unet_deeper_sgd | 10 | 1.048704 | Clearly worse early trajectory than Adam; not worth continuing as the next simple-protocol baseline. |
| unet_deeper_adamw | 21 | 0.930456 | Essentially tied with Adam baseline. With weight_decay=0, AdamW is not a meaningful decoupled-weight-decay ablation; run AdamW with nonzero weight decay if we want to test that idea. |

## Table Template

| name | date | config | run dir | notes |
| --- | --- | --- | --- | --- |
| run_name | YYYY-MM-DD HH:MM UTC | `configs/...`<br>`run.yaml` | `experiments/<experiment>/`<br>`<run_name>` | One controlled change relative to the baseline. |

| name | epoch | valid_score | result note |
| --- | --- | --- | --- |
| run_name | best epoch | best validation score | Short interpretation for the next experiment choice. |
