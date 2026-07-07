# Regression Leaderboard

Support-filtered regression baseline leaderboard for the current paper-facing
protocol.

Last updated: 2026-07-07.

Protocol:

| component | value |
| --- | --- |
| train split | R1-R8 |
| development split | R9-R10 |
| final holdout | R11 |
| target support | `0.5 <= RT <= 2.5` |
| seeds | 2025, 2026, 2027, 2028, 2029 |
| metric | nRMSE, lower is better |

## Current Results

Completed 5/5 rows only:

| rank | model | seeds | valid nRMSE mean +/- std | R11 nRMSE mean +/- std | R11 range | run directory |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `etr_cnn_large` | 5/5 | 0.8972 +/- 0.0040 | 0.8928 +/- 0.0042 | 0.8873-0.8977 | `benchmarks/experiments/01_regression_baselines/etr_cnn_large_repeated__20260706_200136` |
| 2 | `etr_cnn` | 5/5 | 0.9008 +/- 0.0060 | 0.8977 +/- 0.0068 | 0.8922-0.9085 | `benchmarks/experiments/01_regression_baselines/etr_cnn_repeated__20260706_183507` |
| 3 | `msp_cnn` | 5/5 | 0.9006 +/- 0.0051 | 0.8998 +/- 0.0080 | 0.8927-0.9112 | `benchmarks/experiments/01_regression_baselines/msp_cnn_repeated__20260706_174609` |
| 4 | `tidnet_wrapped` | 5/5 | 0.9235 +/- 0.0024 | 0.9192 +/- 0.0027 | 0.9146-0.9219 | `benchmarks/experiments/01_regression_baselines/tidnet_wrapped_repeated__20260706_215011` |
| 5 | `deep4net_wrapped` | 5/5 | 0.9269 +/- 0.0045 | 0.9260 +/- 0.0044 | 0.9210-0.9309 | `benchmarks/experiments/01_regression_baselines/deep4net_wrapped_repeated__20260707_032916` |
| 6 | `eegconformer_wrapped` | 5/5 | 0.9188 +/- 0.0026 | 0.9287 +/- 0.0057 | 0.9225-0.9342 | `benchmarks/experiments/01_regression_baselines/eegconformer_wrapped_repeated__20260706_230013` |
| 7 | `eegnet_wrapped` | 5/5 | 0.9350 +/- 0.0054 | 0.9335 +/- 0.0028 | 0.9292-0.9370 | `benchmarks/experiments/01_regression_baselines/eegnet_wrapped_repeated__20260707_021052` |
| 8 | `shallowfbcspnet_wrapped` | 5/5 | 0.9324 +/- 0.0016 | 0.9343 +/- 0.0024 | 0.9316-0.9372 | `benchmarks/experiments/01_regression_baselines/shallowfbcspnet_wrapped_repeated__20260707_045531` |
| 9 | `atcnet_wrapped` | 5/5 | 0.9686 +/- 0.0183 | 0.9666 +/- 0.0151 | 0.9533-0.9920 | `benchmarks/experiments/01_regression_baselines/atcnet_wrapped_repeated__20260707_055533` |

## Partial / Pending

| model | state | partial valid nRMSE mean +/- std | partial R11 nRMSE mean +/- std | R11 range | run directory |
| --- | --- | ---: | ---: | ---: | --- |
| `labram_wrapped` | 3/5 finished, seed 2028 running | 0.9313 +/- 0.0083 | 0.9357 +/- 0.0100 | 0.9262-0.9462 | `benchmarks/experiments/01_regression_baselines/labram_wrapped_repeated__20260707_065438` |
| `eegpt_wrapped` | pending | - | - | - | - |
| `medformer_wrapped` | pending | - | - | - | - |

Partial rows are not ranked until all five seeds finish.

## Interpretation

- `etr_cnn_large` is the current strongest completed scalar baseline.
- `etr_cnn` improves over `msp_cnn`, but the gap is modest relative to seed
  variation.
- Wrapped external architectures train meaningfully under the fixed
  normalization protocol but remain weaker than the compact scalar models in
  the completed set.
- Partial `labram_wrapped` results currently sit near the weaker wrapped
  baselines, but the row should stay out of the ranked table until 5/5 seeds
  finish.
