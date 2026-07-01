# 01 Regression Baselines

This folder is reserved for clean paper-facing regression baseline configs.
Exploratory recipe search lives in `../00_protocol_calibration/`; configs here
should be copied in only after the protocol is frozen.

## Configs

| config | model | role |
| --- | --- | --- |
| `sneddy_net.yaml` | SneddyNet | Main direct-regression baseline from our model family. |
| `eegnet.yaml` | Braindecode EEGNet | Canonical compact EEG baseline. |
| `eegnet_wrapped.yaml` | Braindecode EEGNet + per-sample standardization | Same EEGNet architecture with the input normalization used by our models. |
| `labram.yaml` | Braindecode LaBraM from scratch | Larger modern baseline without pretrained weights. |

## Running

Single seed:

```bash
python benchmarks/run.py benchmarks/configs/01_regression_baselines/sneddy_net.yaml
```

Repeated seeds are listed in the YAML files but disabled by default for the
first production pass. Enable `evaluation.repeated_runs.enabled` before running
the repeated launcher:

```bash
python benchmarks/run_repeated.py benchmarks/configs/01_regression_baselines/sneddy_net.yaml
```

`run_repeated.py` reuses the normal runner for each seed and then writes a
compact `repeated_summary.csv` and `repeated_summary.json` under the experiment
artefact directory.

## Goal

The regression baselines answer a narrow question: how far can direct scalar
reaction-time regression go under the same data split and training protocol,
before comparing it with the event-time segmentation formulation.

These runs are not meant to be a model-specific hyperparameter search. Each
config should change the model, or one explicitly documented ablation, while
keeping the protocol fixed.

## Frozen Protocol

Use the same split roles as the rest of the benchmark:

| split | role |
| --- | --- |
| R1-R8 | train base models |
| R9-R10 | development split for checkpointing and diagnostics |
| R11 | untouched final held-out release |

Default regression recipe:

| component | value |
| --- | --- |
| input | fixed 2 s EEG windows |
| target | scalar reaction time |
| train batch size | 128 |
| validation batch size | 256 |
| optimizer | Adam |
| learning rate | 1e-3 |
| weight decay | 0 |
| early stopping | patience 20 |
| monitor | validation NRMSE |
| mixup | disabled |
| augmentation | mild v2 recipe |
| holdout evaluation | enabled on `data.test` after training |
| repeated-run seeds | 2025, 2026, 2027, 2028, 2029 |
| confidence interval | subject bootstrap, 1000 resamples |

The mild v2 augmentation recipe is:

| parameter | value |
| --- | ---: |
| channel_dropout_proba | 0.25 |
| channel_dropout_max_ratio | 0.3 |
| cutout_proba | 0.25 |
| cutout_min_len | 10 |
| cutout_max_len | 50 |
| noise_proba | 0.3 |
| noise_base_std | 0.01 |
| noise_random_std | 0.01 |

Current protocol-calibration evidence supports this choice: no mixup improved
direct regression, batch size 128 improved convergence, mild v2 augmentation
beat no augmentation, and Adam weight decay 1e-6 did not improve over
weight_decay 0.

NRMSE is normalized within each split by that split's target standard
deviation: train by R1-R8, validation by R9-R10, and holdout by R11.

## Paper Defense

In the paper, this block should be presented as a controlled baseline family:

- the protocol is selected before the final baseline comparison;
- R11 is not used for checkpointing, calibration, stacking, or hyperparameter
  selection;
- all main claims are reported on R11;
- R9-R10 results are development diagnostics;
- model-specific deviations are allowed only for shape compatibility or memory
  constraints, and must be documented in the config notes;
- pretrained models should not be mixed into the main regression table unless
  clearly labeled as a separate foundation-model appendix.

This makes the comparison defensible: SneddyNet, EEGNet, EEGConformer, TIDNet
or Deep4Net, and LaBraM-from-scratch can be compared as direct-regression
baselines under one protocol, while the main method claim remains the comparison
between the best scalar-regression baseline and event-time segmentation.
