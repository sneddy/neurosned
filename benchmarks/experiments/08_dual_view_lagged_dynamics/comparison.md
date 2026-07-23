# Dual-view lagged-dynamics comparison

Lower nRMSE is better. Each row is the latest repeated summary for that config.

| family | config | seeds | valid nRMSE | holdout nRMSE |
| --- | --- | ---: | ---: | ---: |
| baseline | `etr_cnn_large` | 5 | 0.8972 +/- 0.0040 | 0.8928 +/- 0.0042 |
| baseline | `etr_cnn` | 5 | 0.9008 +/- 0.0060 | 0.8977 +/- 0.0068 |
| baseline | `msp_cnn` | 5 | 0.9006 +/- 0.0051 | 0.8998 +/- 0.0080 |
| dual-view group | `dual_view_covariance_only` | 5 | 0.9075 +/- 0.0047 | 0.9057 +/- 0.0040 |
| dual-view group | `raw_view_only` | 5 | 0.9074 +/- 0.0026 | 0.9092 +/- 0.0043 |
| dual-view group | `dual_view_full` | 5 | 0.9117 +/- 0.0045 | 0.9162 +/- 0.0075 |
| baseline | `tidnet_wrapped` | 5 | 0.9235 +/- 0.0024 | 0.9192 +/- 0.0027 |
| baseline | `deep4net_wrapped` | 5 | 0.9269 +/- 0.0045 | 0.9260 +/- 0.0044 |
| baseline | `eegconformer_wrapped` | 5 | 0.9188 +/- 0.0026 | 0.9287 +/- 0.0057 |
| baseline | `labram_wrapped` | 5 | 0.9304 +/- 0.0061 | 0.9327 +/- 0.0086 |
| baseline | `eegnet_wrapped` | 5 | 0.9350 +/- 0.0054 | 0.9335 +/- 0.0028 |
| baseline | `shallowfbcspnet_wrapped` | 5 | 0.9324 +/- 0.0016 | 0.9343 +/- 0.0024 |
| baseline | `eegpt_wrapped` | 5 | 0.9616 +/- 0.0201 | 0.9584 +/- 0.0185 |
| baseline | `medformer_wrapped` | 5 | 0.9623 +/- 0.0046 | 0.9585 +/- 0.0051 |
| baseline | `atcnet_wrapped` | 5 | 0.9686 +/- 0.0183 | 0.9666 +/- 0.0151 |
| matrix only | `lagged_dynamics_full` | 5 | 0.9686 +/- 0.0065 | 0.9736 +/- 0.0077 |

## Paired-seed contrasts

A negative delta favors the first model.

| contrast | paired seeds | delta holdout nRMSE |
| --- | ---: | ---: |
| `dual_view_full` - `raw_view_only` | 5 | +0.0070 +/- 0.0107 |
| `dual_view_full` - `lagged_dynamics_full` | 5 | -0.0574 +/- 0.0086 |
| `dual_view_full` - `dual_view_covariance_only` | 5 | +0.0105 +/- 0.0056 |
| `dual_view_covariance_only` - `raw_view_only` | 5 | -0.0035 +/- 0.0067 |
| `dual_view_full` - `etr_cnn_large` | 5 | +0.0234 +/- 0.0081 |
