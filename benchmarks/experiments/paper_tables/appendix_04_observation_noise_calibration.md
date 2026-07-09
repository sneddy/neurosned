# Observation-Noise Calibration for EventNLL-Family Posteriors

Intended placement: appendix or a short supporting table for `Posterior Geometry Diagnostics`.

Caption draft: Post-hoc RT observation-noise calibration for EventNLL-family models. The trained EEG model, event-time posterior, and posterior-mean scalar readout are fixed. A single multiplicative scale on the RT observation kernel is selected on R9-R10 by `coverage_mae` and applied unchanged to R11. Each model block separates the latent event-time posterior from the behavioral-RT predictive distribution obtained by convolving that posterior with the observation kernel. Values are mean +/- sample standard deviation across seeds.

### EventNLL

| Distribution | Scale c | Coverage MAE | Coverage80 | Width80 ms | Predictive NLL |
| --- | ---: | ---: | ---: | ---: | ---: |
| Latent event-time posterior | - | 0.273 +/- 0.021 | 0.500 +/- 0.025 | 502 +/- 15 | - |
| Base predictive RT | 1.00 | 0.059 +/- 0.004 | 0.850 +/- 0.004 | 699 +/- 13 | -0.044 +/- 0.010 |
| Calibrated predictive RT | 0.70 +/- 0.00 | 0.006 +/- 0.003 | 0.793 +/- 0.004 | 632 +/- 14 | -0.054 +/- 0.016 |

### Mixture EventNLL

| Distribution | Scale c | Coverage MAE | Coverage80 | Width80 ms | Predictive NLL |
| --- | ---: | ---: | ---: | ---: | ---: |
| Latent event-time posterior | - | 0.207 +/- 0.017 | 0.573 +/- 0.020 | 528 +/- 22 | - |
| Base predictive RT | 1.00 | 0.040 +/- 0.010 | 0.835 +/- 0.009 | 671 +/- 17 | -0.098 +/- 0.012 |
| Calibrated predictive RT | 0.73 +/- 0.04 | 0.005 +/- 0.004 | 0.795 +/- 0.003 | 622 +/- 12 | -0.111 +/- 0.013 |

### Hazard EventNLL

| Distribution | Scale c | Coverage MAE | Coverage80 | Width80 ms | Predictive NLL |
| --- | ---: | ---: | ---: | ---: | ---: |
| Latent event-time posterior | - | 0.303 +/- 0.006 | 0.467 +/- 0.007 | 440 +/- 17 | - |
| Base predictive RT | 1.00 | 0.044 +/- 0.006 | 0.834 +/- 0.006 | 639 +/- 15 | -0.044 +/- 0.006 |
| Calibrated predictive RT | 0.78 +/- 0.04 | 0.008 +/- 0.002 | 0.790 +/- 0.005 | 587 +/- 17 | -0.059 +/- 0.005 |

Coverage MAE is averaged over central interval levels `0.50, 0.60, 0.70, 0.80, 0.90`.
Interpretation note: this calibration changes only the RT observation-noise layer used for probabilistic prediction. It does not change trained weights, posterior-mean RT predictions, or tau-nRMSE.
