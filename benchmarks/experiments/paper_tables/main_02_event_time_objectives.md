# Main Table 2: Event-Time Objective Comparison

Intended placement: `Event-Time Posterior Formulation / Formulation Comparison
and Robustness`.

Caption draft: Event-time objective variants with a fixed ETS-U-Net backbone.
Temperature is selected on R9-R10 for posterior-mean readout and then applied
unchanged to the holdout split. Values are mean +/- sample standard deviation
over seeds 2025-2029. Lower nRMSE is better.

| Objective | Supervision signal | Valid nRMSE | Holdout nRMSE | Holdout tau nRMSE |
| --- | --- | ---: | ---: | ---: |
| **Soft-target family** |  |  |  |  |
| CE | Gaussian soft event-time label | 0.8763 +/- 0.0044 | 0.8774 +/- 0.0044 | 0.8753 +/- 0.0039 |
| **Likelihood family** |  |  |  |  |
| EventNLL | latent event time + Gaussian RT likelihood | 0.8769 +/- 0.0030 | 0.8805 +/- 0.0021 | 0.8772 +/- 0.0018 |
| Mixture EventNLL | Two-component core-tail RT likelihood | 0.8744 +/- 0.0018 | 0.8785 +/- 0.0047 | 0.8745 +/- 0.0053 |
| Hazard EventNLL | hazard posterior + Gaussian RT likelihood | 0.8755 +/- 0.0027 | 0.8776 +/- 0.0031 | 0.8778 +/- 0.0041 |
| **Scalar-readout and distance controls** |  |  |  |  |
| Soft-argmax RT loss | posterior-mean RT error only; no distribution target | 0.8944 +/- 0.0048 | 0.8943 +/- 0.0025 | 0.8917 +/- 0.0046 |
| Wasserstein | CDF match to Gaussian soft event-time label | 0.8997 +/- 0.0035 | 0.8995 +/- 0.0078 | 0.8896 +/- 0.0033 |

Paper note: CE and the EventNLL-family objectives form the strongest scalar
readout group after temperature tuning. The soft-argmax RT-loss control keeps
the temporal event-time head, but collapses supervision to scalar posterior-mean
RT error; its gap to CE/EventNLL supports the claim that the gain is not just a
differentiable expectation readout. Wasserstein is useful as a geometry control,
but it is not the best scalar predictor.
