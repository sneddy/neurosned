# 03 Seed Robustness

Five-seed robustness configs for the final paper-facing claims. These YAML files
reuse the finished single-seed recipes and change only:

- `experiment: 03_seed_robustness`
- `evaluation.repeated_runs.enabled: true`

The seed set is fixed across configs: `2025, 2026, 2027, 2028, 2029`.

## Core Runs

Run these first. They cover the claims that need seed-level support.

| config | reason |
| --- | --- |
| `core/sneddy_rt_net.yaml` | Best direct-regression scalar baseline. |
| `core/unet_deeper_ce_only.yaml` | Simple event-time segmentation baseline and strong scalar readout. |
| `core/unet_deeper_comboloss.yaml` | Best current calibrated scalar segmentation readout. |
| `core/unet_deeper_event_nll.yaml` | Main principled latent event-time likelihood objective. |
| `core/unet_deeper_time_only.yaml` | Negative control: scalar soft-argmax supervision without distributional event-time learning. |
| `core/unet_deeper_wass_only.yaml` | Negative control: not every distributional loss recovers the segmentation gain. |

## Extensions

Run if the core package finishes cleanly and compute remains.

| config | reason |
| --- | --- |
| `extensions/unet_deeper_hazard_event_nll.yaml` | Alternative hazard/survival parameterization of the event-time posterior. |
| `extensions/unet_deeper_gaussian_mixture_event_nll.yaml` | Two-scale observation-kernel extension of EventNLL. |

## External Optional

Run only if the final paper needs seed robustness for the strongest external EEG
backbone.

| config | reason |
| --- | --- |
| `external_optional/tidnet_wrapped.yaml` | Strongest completed external wrapped baseline on R11. |

## Commands

Run one config:

```bash
python benchmarks/scripts/run_repeated.py benchmarks/configs/03_seed_robustness/core/sneddy_rt_net.yaml --device auto
```

Run the full core set:

```bash
for cfg in benchmarks/configs/03_seed_robustness/core/*.yaml; do
  python benchmarks/scripts/run_repeated.py "$cfg" --device auto
done
```

The outputs go under one repeated-run container per config:

```text
benchmarks/experiments/03_seed_robustness/<config>_repeated__<timestamp>/
  repeated_summary.csv
  repeated_summary.json
  summary.csv
  summary.md
  seed2025/
  seed2026/
  seed2027/
  seed2028/
  seed2029/
```

Each seed directory is a normal benchmark run directory with its own checkpoint,
predictions, logits where enabled, temperature calibration, metrics, and logs.

## Rationale

Do not run every regression model for five seeds by default. The paper needs
seed robustness for the models that support the main claims: the best scalar
baseline, the strongest event-time segmentation readouts, the principled
EventNLL objective, and the key negative controls. Broad external-backbone seed
sweeps are expensive and do not materially strengthen the event-time
formulation claim.
