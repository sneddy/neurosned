# Main Table 5: Distribution-Aware Stacking

Intended placement: `Secondary ensemble use of posterior features` or appendix
if the main text must stay focused on single-model evidence.

Caption draft: Distribution-aware stacking from frozen fixed-window ETS-U-Net
outputs. Each row reports mean +/- sample standard deviation across five
seed-matched stacking replications; within each seed, six segmentation
objectives are combined using their saved fixed-window R9-R10 and R11 outputs.
The best-solo row selects the single base model with the best R9-R10 scalar
nRMSE within each seed. Stackers are trained only on R9-R10 with
subject-disjoint folds and evaluated once on R11. `Delta vs
matched RT-only` compares each posterior-based row with the corresponding
scalar-RT-only reference: SubmitWrapper-style logit soft-argmax blending
versus scalar equal blending, Ridge posterior features versus Ridge RT-only,
and boosting posterior features versus boosting RT-only. The equal-logit row
averages raw segmentation logits first and then applies softmax/soft-argmax
with temperature 0.92. Lower nRMSE and MAE are better.

| Method | Input to combiner | Learner | R11 nRMSE | R11 MAE | Delta vs matched RT-only |
| --- | --- | --- | ---: | ---: | ---: |
| Best single model | dev-selected solo model | model selection | 0.8782 +/- 0.0050 | 0.2100 +/- 0.0023 | baseline |
| **Equal scalar blend** | scalar RT predictions | equal weights | **0.8575 +/- 0.0022** | 0.2057 +/- 0.0008 | reference |
| Equal logits soft-argmax blend | raw segmentation logits | equal weights | 0.8763 +/- 0.0025 | 0.2119 +/- 0.0008 | +0.0187 |
| Ridge RT-only stacking | scalar RT predictions | Ridge | 0.8580 +/- 0.0025 | 0.2054 +/- 0.0007 | reference |
| Boosting RT-only stacking | scalar RT predictions | HGBR | 0.8592 +/- 0.0025 | 0.2058 +/- 0.0007 | reference |
| Ridge posterior-feature stacking | scalar RT + posterior summaries | Ridge | 0.8609 +/- 0.0031 | 0.2054 +/- 0.0007 | +0.0029 |
| Boosting posterior-feature stacking | scalar RT + posterior summaries | HGBR | 0.8598 +/- 0.0028 | **0.2053 +/- 0.0008** | +0.0006 |

Paper note: on the main fixed-window protocol, simple scalar ensembling already
captures most of the stacking gain over the dev-selected best single model.
SubmitWrapper-style logit blending is substantially better than averaging hard
posterior modes, but scalar blending remains the strongest simple ensemble in
this fixed-window setting. The ported challenge-style posterior meta-features
remove the previous unstable Ridge failure mode, but they still do not improve
R11 nRMSE over RT-only stacking in this run. This supports treating
posterior-feature stacking as a secondary utility analysis rather than as core
evidence for the event-time formulation.

Best-solo selections by seed: seed2025 `ets_unet_hazard_event_nll`,
seed2026 `ets_unet_event_nll`, seed2027 `ets_unet_event_nll_mixture`,
seed2028 `ets_unet_ce`, and seed2029 `ets_unet_event_nll_mixture`.
