# 07 Lagged Dynamics Regression

Exploratory scalar-regression experiments for the two-scale lagged-dynamics
architecture. These configs keep the data protocol, augmentation, optimizer,
trainer, five seeds, holdout evaluation, and shifted-crop evaluation identical
to `01_regression_baselines`. Only the model is changed.

## Representation

For each `(B, 128, 200)` EEG window, the model:

1. learns a normalized spatial projection from 128 channels to 24 latent
   components;
2. extracts seven overlapping 0.5 s segments with a 0.25 s stride;
3. computes a shrinkage covariance and lagged operators at 50, 100, and 200 ms
   inside each segment;
4. maps the matrices to a `(B, 7, 384)` token sequence;
5. models the evolution of those tokens with a small dilated TCN and returns a
   standard `(B, 1)` RT prediction.

The covariance branch uses a log-Cholesky SPD representation. The lagged
branch contains normalized cross-correlation and differentiable
ridge-transition operators. This explicitly separates within-segment delayed
dependence from across-segment temporal evolution.

The 24-dimensional projection is not only a compression choice: a raw
128-channel covariance estimated from 50 samples has rank at most 49. Projection
plus learned shrinkage makes the local SPD estimate well-conditioned. The
matrix encoders use all entries rather than a local 2D convolution because
adjacent row or column indices do not define a meaningful translation axis for
EEG electrodes (or learned latent components).

## Configs

| config | enabled matrix information | question |
| --- | --- | --- |
| `lagged_dynamics_full.yaml` | covariance + lagged correlation + transition | Does the complete two-scale representation compete with existing regressors? |
| `lagged_dynamics_covariance_only.yaml` | covariance | Do delayed operators add information beyond local zero-lag second-order structure? |
| `lagged_dynamics_lagged_only.yaml` | lagged correlation + transition | Does the zero-lag SPD branch add information beyond delayed dynamics? |

Configured trainable parameter counts are approximately 2.85M, 2.34M, and
2.61M, respectively. The full model is close in scale to the existing 3.01M
MSP-CNN baseline and remains substantially smaller than ETR-CNN.

## Run

Before a long repeated run, the real-data optimizer smoke test can be used to
check the current device and numerical path:

```bash
python benchmarks/scripts/smoke_lagged_dynamics.py --device cuda --steps 12
```

Then run the prespecified five-seed experiment matrix:

```bash
DEVICE=cuda sh benchmarks/runners/run_lagged_dynamics.sh
```

The primary comparison is the five-seed holdout nRMSE against the completed
rows in `benchmarks/experiments/paper_tables/main_01_regression_baselines.md`.
The key architectural contrast is full versus covariance-only; full versus
lagged-only tests whether static and delayed second-order structure are
complementary. Shifted-crop metrics are secondary diagnostics and should not be
used to retune the architecture on R11.

After all runs finish, the runner writes a sorted comparison and paired-seed
deltas to `benchmarks/experiments/07_lagged_dynamics/comparison.md`. It can also
be regenerated independently:

```bash
python benchmarks/scripts/compare_lagged_dynamics.py
```
