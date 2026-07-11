# Main Table 2: Controlled Comparison and Diagnostic Map

Intended placement: `Dataset and Evaluation Protocol / Shared Evaluation
Protocol`.

Caption: Controlled comparison and diagnostic blocks under the shared protocol.

| Control or analysis | Readout | Supervision | Question addressed |
| --- | --- | --- | --- |
| **Scalar regression controls** |  |  |  |
| External EEG backbones | Scalar RT head | Scalar RT | Does generic EEG backbone capacity explain performance? |
| MSP-CNN family | Segment-pooling scalar readout | Scalar RT | How far does coarse temporal pooling support scalar regression? |
| ETR-CNN family | Temporal expectation readout | Scalar RT | Does a learned temporal readout improve scalar regression? |
| **Event-time objective comparison with fixed ETS-U-Net** |  |  |  |
| Soft-argmax RT-loss control | Posterior mean | Scalar RT only | Is posterior-mean readout sufficient without distributional supervision? |
| CE soft-target objective | Posterior mean | Soft event-time target | Does direct distributional supervision improve RT prediction? |
| Likelihood-based objectives | Posterior mean | Latent event-time likelihood | Does probabilistic latent-event supervision provide the same benefit? |
| Wasserstein control | Posterior mean | Distributional geometry loss | Is an alternative geometry-based distributional objective sufficient? |
| **Architecture robustness controls** |  |  |  |
| ETS-TCN | Posterior mean | RT-only, CE, mixture EventNLL | Does the supervision effect persist with dilated temporal convolution? |
| ETS-InceptionPyramid | Posterior mean | RT-only, CE, mixture EventNLL | Does the supervision effect persist with multi-scale temporal filtering? |
| ETS-AttnSeg | Posterior mean | RT-only, CE, mixture EventNLL | Does the supervision effect persist with attention and local convolution? |
| **Posterior and localization diagnostics** |  |  |  |
| Readout-temperature tuning | Temperature-adjusted posterior mean | Learned posterior | How should the scalar posterior readout be standardized across objectives? |
| Posterior geometry diagnostics | Posterior summaries | Learned posterior | What posterior behavior is hidden by scalar nRMSE? |
| Shifted-crop diagnostic | Crop-relative posterior mean | No new training | Do predictions behave as crop-relative temporal localizers? |
| Shift-jitter intervention | Crop-relative posterior mean | Shifted crop-relative targets | Does crop-start augmentation reduce shortcut behavior? |

Paper note: This table is a design map rather than a result table. Detailed
architecture capacity and hyperparameters are reported in the reproducibility
table. Posterior geometry and shifted-crop analyses are diagnostic extensions,
not an additional scalar benchmark family.
