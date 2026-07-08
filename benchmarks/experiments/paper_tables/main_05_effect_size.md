# Effect Size Summary: Practical Magnitude of Event-Time Supervision

Intended placement: supporting paragraph or compact table near `Event-Time Posterior Formulation / Formulation Comparison and Robustness`.

Caption draft: Practical magnitude of the gain over the strongest scalar baseline. Scalar and event-time values are mean +/- sample standard deviation over matched seeds 2025-2029. Delta columns report scalar-minus-event improvement with subject-bootstrap 95% confidence intervals over R11 subjects, using seed-averaged per-trial errors. Positive deltas indicate better event-time performance.

| Comparison | Scalar tau nRMSE | Event tau nRMSE | Delta tau nRMSE | Relative gain | Delta RMSE ms | Delta MAE ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ETR-CNN large -> CE | 0.8928 +/- 0.0042 | 0.8753 +/- 0.0039 | 0.0175 [0.0133, 0.0218] | 2.0% | 5.94 [4.54, 7.39] | 7.66 [6.47, 8.86] |
| ETR-CNN large -> Mixture EventNLL | 0.8928 +/- 0.0042 | 0.8745 +/- 0.0053 | 0.0184 [0.0132, 0.0236] | 2.1% | 6.24 [4.50, 7.97] | 8.94 [7.43, 10.49] |

Paper note: This table is intended to calibrate the practical size of the main scalar accuracy gain. The absolute improvement is moderate in milliseconds but consistent across matched seeds and R11 subjects.
