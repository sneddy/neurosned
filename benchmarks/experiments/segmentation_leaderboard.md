| Model | Role | Seeds | Valid nRMSE (mean +/- std) | R11 nRMSE (mean +/- std) | R11 tau nRMSE (mean +/- std) | Shift slope (mean +/- std) | Localizer-like (mean +/- std) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ETS-U-Net CE | soft-label event-time CE baseline | 5/5 | 0.8763 +/- 0.0044 | 0.8774 +/- 0.0044 | 0.8753 +/- 0.0039 | -0.339 +/- 0.017 | 0.209 +/- 0.031 |
| ETS-U-Net EventNLL | latent event-time likelihood | 5/5 | 0.8769 +/- 0.0030 | 0.8805 +/- 0.0021 | 0.8772 +/- 0.0018 | -0.344 +/- 0.014 | 0.221 +/- 0.026 |
| ETS-U-Net time-only | soft-argmax scalar control | 0/5 (+1 marked running) | - | - | - | - | - |
| ETS-U-Net CE+time | hybrid CE plus time-readout loss | pending | - | - | - | - | - |
| ETS-U-Net Wasserstein | CDF-distance event-time control | pending | - | - | - | - | - |
| ETS-U-Net mixture EventNLL | two-scale Gaussian observation kernel | pending | - | - | - | - | - |
| ETS-U-Net hazard EventNLL | hazard/survival posterior parameterization | pending | - | - | - | - | - |
