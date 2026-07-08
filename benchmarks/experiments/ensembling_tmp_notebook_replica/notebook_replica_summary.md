# Notebook Replica Stacking Diagnostic

This table uses the old `neurosned.wrappers.challenge_1` feature extractor and meta-regressors directly.

| Method | Seeds | R11 nRMSE | R11 MAE | Delta vs best single |
| --- | ---: | ---: | ---: | ---: |
| Best single model | 5 | 0.8782 +/- 0.0050 | 0.2100 +/- 0.0023 | 0.0000 +/- 0.0000 |
| Equal scalar blend | 5 | 0.8575 +/- 0.0022 | 0.2057 +/- 0.0008 | -0.0207 +/- 0.0033 |
| Equal logits soft-argmax blend | 5 | 0.8763 +/- 0.0025 | 0.2119 +/- 0.0008 | -0.0019 +/- 0.0028 |
| Notebook Ridge posterior features | 5 | 0.8609 +/- 0.0030 | 0.2054 +/- 0.0007 | -0.0172 +/- 0.0039 |
| Notebook HGB posterior features | 5 | 0.8598 +/- 0.0026 | 0.2052 +/- 0.0008 | -0.0184 +/- 0.0036 |
