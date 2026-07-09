# Main Table 6: Controlled Comparison and Diagnostic Map

Intended placement: `Dataset and Evaluation Protocol / Shared Training,
Readout Tuning, and Inference Protocol`.

Caption draft: Controlled comparison and diagnostic blocks under the shared
protocol. The first two blocks define the main scalar-performance comparisons;
the final block evaluates posterior readout, posterior geometry, and
shortcut-vs-localization behavior after event-time models have been trained.

| Comparison block / model family | Model capacity | Temporal readout | Supervision | Role |
| --- | --- | --- | --- | --- |
| **Scalar regression controls** |  |  |  |  |
| External EEG backbones | Standard EEG architectures trained from scratch | Scalar RT head | Scalar RT | Assesses whether generic EEG backbone capacity explains performance |
| MSP-CNN family | Compact scalar CNN | Segment-pooling scalar readout | Scalar RT | Scalar baseline with coarse temporal pooling |
| ETR-CNN family | Compact scalar CNN | Temporal expectation readout | Scalar RT | Isolates temporal expectation readout under scalar RT supervision |
| **Event-time objective comparison** |  |  |  |  |
| Soft-argmax RT-loss control | Shared segmentation backbone | Posterior-mean readout | Scalar RT only | Isolates posterior-mean readout without distributional supervision |
| CE soft-target objective | Shared segmentation backbone | Posterior-mean readout | Soft event-time target | Direct soft-label event-time supervision |
| Likelihood-based objectives | Shared segmentation backbone | Posterior-mean readout | Latent event-time likelihood | Probabilistic latent-event supervision |
| Wasserstein control | Shared segmentation backbone | Posterior-mean readout | Distributional geometry loss | Alternative geometry-based distributional supervision |
| **Posterior and localization diagnostics** |  |  |  |  |
| Readout-temperature tuning | Trained event-time models | Temperature-adjusted posterior mean | Learned posterior | Standardizes scalar posterior-mean readout |
| Posterior geometry diagnostics | Trained event-time models | Posterior summaries | Learned posterior | Exposes posterior structure beyond scalar nRMSE |
| Shifted-crop diagnostic | Trained fixed-window models | Crop-relative posterior mean | No new training | Probes shortcut-vs-localization behavior |
| Shift-jitter intervention | Matched event-time models | Crop-relative posterior mean | Shifted crop-relative targets | Intervenes on crop-start variation to improve robustness and localizer-like movement |

Paper note: This table is a design map rather than a result table. It links
the shared protocol to the paper's three main experimental axes: EEG backbone
capacity, temporal readout parameterization, and distributional event-time
supervision. Posterior geometry and shifted-crop analyses are diagnostic
extensions, not an additional scalar benchmark family.
