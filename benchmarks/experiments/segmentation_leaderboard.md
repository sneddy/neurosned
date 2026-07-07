| Model | Role | Seeds | Valid nRMSE (mean +/- std) | R11 nRMSE (mean +/- std) | R11 tau nRMSE (mean +/- std) | Shift slope (mean +/- std) | Localizer-like (mean +/- std) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ETS-U-Net CE | soft-label event-time CE baseline | 5/5 | 0.8763 +/- 0.0044 | 0.8774 +/- 0.0044 | 0.8753 +/- 0.0039 | -0.339 +/- 0.017 | 0.209 +/- 0.031 |
| ETS-U-Net EventNLL | latent event-time likelihood | 5/5 | 0.8769 +/- 0.0030 | 0.8805 +/- 0.0021 | 0.8772 +/- 0.0018 | -0.344 +/- 0.014 | 0.221 +/- 0.026 |
| ETS-U-Net time-only | soft-argmax scalar control | 5/5 | 0.8944 +/- 0.0048 | 0.8943 +/- 0.0025 | 0.8917 +/- 0.0046 | -0.301 +/- 0.043 | 0.160 +/- 0.066 |
| ETS-U-Net Wasserstein | CDF-distance event-time control | 5/5 | 0.8997 +/- 0.0035 | 0.8995 +/- 0.0078 | 0.8896 +/- 0.0033 | -0.337 +/- 0.044 | 0.271 +/- 0.065 |
| ETS-U-Net hazard EventNLL | hazard/survival posterior parameterization | 3/5 partial | 0.8743 +/- 0.0024 | 0.8783 +/- 0.0042 | 0.8786 +/- 0.0055 | -0.329 +/- 0.028 | 0.187 +/- 0.056 |
| ETS-U-Net CE+time | hybrid CE plus time-readout loss | pending | - | - | - | - | - |
| ETS-U-Net mixture EventNLL | two-scale Gaussian observation kernel | pending | - | - | - | - | - |
