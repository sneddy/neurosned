| Method | Seeds | R11 nRMSE | R11 MAE | Delta vs best single | Delta vs RT-only stacker |
| --- | ---: | ---: | ---: | ---: | ---: |
| Equal-weight scalar RT blend | 5 | 0.8575 +/- 0.0022 | 0.2057 +/- 0.0008 | -0.0207 +/- 0.0033 |  |
| Equal-weight logits soft-argmax blend | 5 | 0.8763 +/- 0.0025 | 0.2119 +/- 0.0008 | -0.0019 +/- 0.0028 |  |
| Ridge stacking, RT only | 5 | 0.8580 +/- 0.0025 | 0.2054 +/- 0.0007 | -0.0202 +/- 0.0034 |  |
| Boosting stacking, RT only | 5 | 0.8592 +/- 0.0025 | 0.2058 +/- 0.0007 | -0.0189 +/- 0.0035 |  |
| Ridge stacking, posterior meta-features | 5 | 0.8609 +/- 0.0031 | 0.2054 +/- 0.0007 | -0.0173 +/- 0.0039 | +0.0029 +/- 0.0009 |
| Boosting stacking, posterior meta-features | 5 | 0.8598 +/- 0.0028 | 0.2053 +/- 0.0008 | -0.0184 +/- 0.0035 | +0.0006 +/- 0.0009 |
