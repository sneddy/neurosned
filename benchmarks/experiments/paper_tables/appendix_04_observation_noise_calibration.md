# Observation-Noise Calibration for EventNLL-Family Posteriors

Intended placement: appendix or a short supporting table for `Posterior Geometry Diagnostics`.

Caption draft: Post-hoc RT observation-noise calibration for EventNLL-family models. The trained EEG model, event-time posterior, and posterior-mean scalar readout are fixed. A single multiplicative scale on the RT observation kernel is selected on R9-R10 by `coverage_mae` and applied unchanged to R11. Latent columns summarize central intervals of the event-time posterior itself. Predictive columns summarize the behavioral-RT predictive distribution obtained by convolving the latent posterior with the observation kernel. Values are mean +/- sample standard deviation across seeds.

| Model | Scale c | Latent Cov80 | Base Pred Cov80 | Cal Pred Cov80 | Base Pred CovMAE | Cal Pred CovMAE | Cal Pred Width80 ms | Base Pred NLL | Cal Pred NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EventNLL | 0.70 +/- 0.00 | 0.500 +/- 0.025 | 0.850 +/- 0.004 | 0.793 +/- 0.004 | 0.059 +/- 0.004 | 0.006 +/- 0.003 | 632 +/- 14 | -0.044 +/- 0.010 | -0.054 +/- 0.016 |
| Mixture EventNLL | 0.73 +/- 0.04 | 0.573 +/- 0.020 | 0.835 +/- 0.009 | 0.795 +/- 0.003 | 0.040 +/- 0.010 | 0.005 +/- 0.004 | 622 +/- 12 | -0.098 +/- 0.012 | -0.111 +/- 0.013 |
| Hazard EventNLL | 0.78 +/- 0.04 | 0.467 +/- 0.007 | 0.834 +/- 0.006 | 0.790 +/- 0.005 | 0.044 +/- 0.006 | 0.008 +/- 0.002 | 587 +/- 17 | -0.044 +/- 0.006 | -0.059 +/- 0.005 |

Coverage MAE is averaged over central interval levels `0.50, 0.60, 0.70, 0.80, 0.90`.
Interpretation note: this calibration changes only the RT observation-noise layer used for probabilistic prediction. It does not change trained weights, posterior-mean RT predictions, or tau-nRMSE.
