# Architecture-Control Extension: Dense Temporal Backbones

Working status: architecture-control runs complete. This file is the maintained
working chapter for the architecture-control extension. U-Net, TCN,
InceptionPyramid, and AttnSeg all have RT-only, CE, and Mixture EventNLL
five-seed runs.

Intended placement: either a compact main-results table after the ETS-U-Net
objective comparison, or an appendix table with a short main-text synthesis.

## Recommended Integration

Recommended integration: add a compact main-text subsection after the ETS-U-Net
objective comparison, with a fuller appendix table for shifted-crop behavior and
posterior geometry. The main text should keep the focus on output formulation:
the architecture controls test whether the RT-only versus distributional
supervision pattern survives changes in dense temporal backbone while preserving
the same 200-bin event-time output space. AttnSeg remains in the integration
plan as the attention-based dense temporal control.

## Reviewer-Facing Rationale

Reviewer-facing rationale: this experiment addresses whether the gain is
specific to the ETS-U-Net encoder-decoder backbone. Because RT-only
posterior-mean supervision remains weaker than CE/EventNLL across full-resolution
TCN and explicit multi-scale InceptionPyramid controls, and because AttnSeg CE
and Mixture EventNLL also land in the same performance range as the
convolutional dense segmenters, the result supports the output-supervision claim
rather than a U-Net-only architecture claim.

## Compact Main-Text Candidate

This is the compact table shape for the main text. It intentionally omits
diagnostic columns so the main paper does not become an architecture benchmark.
AttnSeg is included as a dense attention-based control.

| Architecture | Objective | Params | Holdout tau nRMSE |
| --- | --- | ---: | ---: |
| ETS-U-Net | RT-only softargmax | 3.10M | 0.8917 +/- 0.0046 |
| ETS-U-Net | CE | 3.10M | 0.8753 +/- 0.0039 |
| ETS-U-Net | Mixture EventNLL | 3.10M | 0.8745 +/- 0.0053 |
| ETS-TCN | RT-only softargmax | 3.13M | 0.8842 +/- 0.0045 |
| ETS-TCN | CE | 3.13M | 0.8730 +/- 0.0038 |
| ETS-TCN | Mixture EventNLL | 3.13M | 0.8722 +/- 0.0021 |
| ETS-InceptionPyramid | RT-only softargmax | 3.25M | 0.8886 +/- 0.0044 |
| ETS-InceptionPyramid | CE | 3.25M | 0.8717 +/- 0.0036 |
| ETS-InceptionPyramid | Mixture EventNLL | 3.25M | 0.8746 +/- 0.0028 |
| ETS-AttnSeg | RT-only softargmax | 3.05M | 0.9052 +/- 0.0300 |
| ETS-AttnSeg | CE | 3.05M | 0.8780 +/- 0.0082 |
| ETS-AttnSeg | Mixture EventNLL | 3.05M | 0.8752 +/- 0.0050 |

## Guardrails

Do not claim that ETS-U-Net is the best backbone. The completed controls show
that competitive dense temporal backbones can match or slightly improve scalar
accuracy. The main claim is that distributional event-time supervision improves
over RT-only posterior-mean supervision across temporal backbones.

Do not present this as a general architecture benchmark or as evidence about
pretrained foundation models. All architecture controls are trained from scratch
and keep the same event-time output contract.

## Model Capacity

Parameter counts are computed from the current CE config for each architecture.
The controls are intentionally kept in the same approximate capacity range, so
the comparison emphasizes temporal inductive bias and output supervision rather
than a large parameter-count difference.

| Architecture | Class | Trainable parameters | Approx. size |
| --- | --- | ---: | ---: |
| ETS-U-Net | `EventTimeUNet1D` | 3,098,401 | 3.10M |
| ETS-TCN | `ETSTCN1D` | 3,129,985 | 3.13M |
| ETS-InceptionPyramid | `ETSInceptionPyramid1D` | 3,254,689 | 3.25M |
| ETS-AttnSeg | `ETSAttnSeg1D` | 3,046,529 | 3.05M |

## Current Scalar Accuracy Table

Caption draft: Architecture-control comparison for dense temporal event-time
models. All models preserve the 200-bin time grid and use the same posterior
readout, temperature selection on R9-R10, fixed R11 holdout evaluation, and
five-seed protocol. Parameter counts are matched within a narrow range of
roughly 3.05M to 3.25M trainable parameters. Lower nRMSE is better.

| Architecture | Objective | Temporal inductive bias | Params | Seeds | Valid nRMSE | Holdout nRMSE | Holdout tau nRMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ETS-U-Net | RT-only softargmax | U-Net encoder-decoder + skips | 3.10M | 5/5 | 0.8944 +/- 0.0048 | 0.8943 +/- 0.0025 | 0.8917 +/- 0.0046 |
| ETS-U-Net | CE | U-Net encoder-decoder + skips | 3.10M | 5/5 | 0.8763 +/- 0.0044 | 0.8774 +/- 0.0044 | 0.8753 +/- 0.0039 |
| ETS-U-Net | Mixture EventNLL | U-Net encoder-decoder + skips | 3.10M | 5/5 | 0.8744 +/- 0.0018 | 0.8785 +/- 0.0047 | 0.8745 +/- 0.0053 |
| ETS-TCN | RT-only softargmax | full-resolution dilated temporal conv | 3.13M | 5/5 | 0.8865 +/- 0.0029 | 0.8853 +/- 0.0044 | 0.8842 +/- 0.0045 |
| ETS-TCN | CE | full-resolution dilated temporal conv | 3.13M | 5/5 | 0.8717 +/- 0.0025 | 0.8751 +/- 0.0085 | 0.8730 +/- 0.0038 |
| ETS-TCN | Mixture EventNLL | full-resolution dilated temporal conv | 3.13M | 5/5 | 0.8718 +/- 0.0046 | 0.8729 +/- 0.0020 | 0.8722 +/- 0.0021 |
| ETS-InceptionPyramid | RT-only softargmax | explicit multi-scale temporal filters | 3.25M | 5/5 | 0.8970 +/- 0.0043 | 0.8894 +/- 0.0054 | 0.8886 +/- 0.0044 |
| ETS-InceptionPyramid | CE | explicit multi-scale temporal filters | 3.25M | 5/5 | 0.8710 +/- 0.0037 | 0.8717 +/- 0.0017 | 0.8717 +/- 0.0036 |
| ETS-InceptionPyramid | Mixture EventNLL | explicit multi-scale temporal filters | 3.25M | 5/5 | 0.8703 +/- 0.0018 | 0.8780 +/- 0.0026 | 0.8746 +/- 0.0028 |
| ETS-AttnSeg | RT-only softargmax | attention + local depthwise temporal conv | 3.05M | 5/5 | 0.9114 +/- 0.0350 | 0.9105 +/- 0.0294 | 0.9052 +/- 0.0300 |
| ETS-AttnSeg | CE | attention + local depthwise temporal conv | 3.05M | 5/5 | 0.8742 +/- 0.0036 | 0.8811 +/- 0.0097 | 0.8780 +/- 0.0082 |
| ETS-AttnSeg | Mixture EventNLL | attention + local depthwise temporal conv | 3.05M | 5/5 | 0.8719 +/- 0.0082 | 0.8823 +/- 0.0079 | 0.8752 +/- 0.0050 |

Working interpretation: in the completed U-Net, TCN, and InceptionPyramid runs,
distributional event-time supervision remains better than RT-only posterior-mean
supervision. The completed AttnSeg CE and Mixture EventNLL runs are also in the
same accuracy range as the convolutional dense segmenters and improve clearly
over AttnSeg RT-only. The AttnSeg RT-only result is noisier because one seed is
much weaker, but the completed AttnSeg comparison still supports the same
RT-only versus distributional-supervision pattern.

*AttnSeg RT-only showed higher seed variability, driven by one failed seed,
whereas CE recovered performance in the same range as the convolutional dense
segmenters.*

## Current Shifted-Crop Snapshot

This table is optional for the main text. It is useful if we want to show that
architecture controls preserve the shortcut-vs-localization distinction rather
than only scalar accuracy. This follows the appendix shifted-crop convention:
accuracy metrics are computed on crop examples where the response remains
inside the evaluated crop, while shift-tracking metrics use the common-inside
trial subset.

| Model | Seeds | Holdout tau nRMSE | Acc trials | Acc rows | rel nRMSE | RMSE, s | MAE, s | Shift error, s | Sensitivity | Direction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ETS-U-Net RT-only softargmax | 5/5 | 0.8917 +/- 0.0046 | 15472 | 105218 | 0.886 +/- 0.007 | 0.340 +/- 0.003 | 0.247 +/- 0.005 | 0.140 +/- 0.008 | 0.538 +/- 0.059 | 0.759 +/- 0.018 |
| ETS-U-Net CE | 5/5 | 0.8753 +/- 0.0039 | 15472 | 105218 | 0.868 +/- 0.002 | 0.333 +/- 0.001 | 0.237 +/- 0.002 | 0.133 +/- 0.003 | 0.581 +/- 0.035 | 0.778 +/- 0.004 |
| ETS-U-Net Mixture EventNLL | 5/5 | 0.8745 +/- 0.0053 | 15472 | 105218 | 0.866 +/- 0.004 | 0.332 +/- 0.001 | 0.234 +/- 0.003 | 0.130 +/- 0.004 | 0.602 +/- 0.048 | 0.773 +/- 0.006 |
| ETS-TCN RT-only softargmax | 5/5 | 0.8842 +/- 0.0045 | 15472 | 105218 | 0.881 +/- 0.006 | 0.338 +/- 0.002 | 0.245 +/- 0.003 | 0.143 +/- 0.004 | 0.519 +/- 0.042 | 0.758 +/- 0.012 |
| ETS-TCN CE | 5/5 | 0.8730 +/- 0.0038 | 15472 | 105218 | 0.870 +/- 0.008 | 0.334 +/- 0.003 | 0.239 +/- 0.001 | 0.135 +/- 0.002 | 0.551 +/- 0.025 | 0.774 +/- 0.006 |
| ETS-TCN Mixture EventNLL | 5/5 | 0.8722 +/- 0.0021 | 15472 | 105218 | 0.867 +/- 0.005 | 0.333 +/- 0.002 | 0.238 +/- 0.002 | 0.137 +/- 0.004 | 0.544 +/- 0.036 | 0.761 +/- 0.010 |
| ETS-InceptionPyramid RT-only softargmax | 5/5 | 0.8886 +/- 0.0044 | 15472 | 105218 | 0.882 +/- 0.005 | 0.338 +/- 0.002 | 0.249 +/- 0.002 | 0.143 +/- 0.003 | 0.506 +/- 0.035 | 0.758 +/- 0.007 |
| ETS-InceptionPyramid CE | 5/5 | 0.8717 +/- 0.0036 | 15472 | 105218 | 0.863 +/- 0.004 | 0.331 +/- 0.001 | 0.238 +/- 0.003 | 0.134 +/- 0.004 | 0.533 +/- 0.032 | 0.781 +/- 0.012 |
| ETS-InceptionPyramid Mixture EventNLL | 5/5 | 0.8746 +/- 0.0028 | 15472 | 105218 | 0.866 +/- 0.002 | 0.332 +/- 0.001 | 0.234 +/- 0.002 | 0.131 +/- 0.003 | 0.589 +/- 0.037 | 0.774 +/- 0.005 |
| ETS-AttnSeg RT-only softargmax | 5/5 | 0.9052 +/- 0.0300 | 15472 | 105218 | 0.899 +/- 0.033 | 0.345 +/- 0.013 | 0.257 +/- 0.019 | 0.149 +/- 0.019 | 0.479 +/- 0.156 | 0.749 +/- 0.028 |
| ETS-AttnSeg CE | 5/5 | 0.8780 +/- 0.0082 | 15472 | 105218 | 0.868 +/- 0.009 | 0.333 +/- 0.003 | 0.238 +/- 0.004 | 0.135 +/- 0.004 | 0.566 +/- 0.057 | 0.771 +/- 0.005 |
| ETS-AttnSeg Mixture EventNLL | 5/5 | 0.8752 +/- 0.0050 | 15472 | 105218 | 0.867 +/- 0.008 | 0.333 +/- 0.003 | 0.232 +/- 0.001 | 0.130 +/- 0.002 | 0.639 +/- 0.034 | 0.767 +/- 0.004 |

Working interpretation: the architecture controls can match or improve shifted
relative nRMSE while still differing in crop-relative sensitivity. This is
useful because it separates point accuracy from localizer-like behavior.
AttnSeg Mixture EventNLL is especially strong on crop-relative sensitivity among
the architecture-control rows.

## Posterior Geometry on Architecture Controls

The architecture-control runs should not only be compared by scalar tau-nRMSE.
The same posterior-geometry diagnostics used in the main U-Net analysis can be
applied to completed architecture-control runs. The table below uses calibrated
posterior readout and the same representable-target filter, fixed-kernel
EventNLL score, 80% central interval width, +/-150 ms target-aligned mass, and
coverage metrics as the main posterior-geometry table.

| Architecture | Objective | nRMSE | Fixed-kernel EventNLL | Width80 ms | Mass +/-150 ms | Coverage80 | Coverage MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ETS-U-Net | RT-only softargmax | 0.8917 +/- 0.0046 | 0.1070 +/- 0.0303 | 844.0 +/- 95.6 | 0.357 +/- 0.015 | 0.843 +/- 0.029 | 0.040 +/- 0.024 |
| ETS-U-Net | CE | 0.8753 +/- 0.0039 | 0.0770 +/- 0.0144 | 766.0 +/- 35.8 | 0.334 +/- 0.008 | 0.883 +/- 0.013 | 0.101 +/- 0.014 |
| ETS-U-Net | Mixture EventNLL | 0.8745 +/- 0.0053 | -0.0820 +/- 0.0122 | 528.0 +/- 21.7 | 0.471 +/- 0.008 | 0.573 +/- 0.020 | 0.207 +/- 0.017 |
| ETS-TCN | RT-only softargmax | 0.8842 +/- 0.0045 | 0.0268 +/- 0.0114 | 526.0 +/- 40.4 | 0.410 +/- 0.012 | 0.717 +/- 0.034 | 0.080 +/- 0.031 |
| ETS-TCN | CE | 0.8730 +/- 0.0038 | 0.0710 +/- 0.0243 | 758.0 +/- 47.6 | 0.336 +/- 0.013 | 0.883 +/- 0.014 | 0.099 +/- 0.016 |
| ETS-TCN | Mixture EventNLL | 0.8722 +/- 0.0021 | -0.0835 +/- 0.0039 | 534.0 +/- 30.5 | 0.471 +/- 0.007 | 0.596 +/- 0.020 | 0.187 +/- 0.017 |
| ETS-InceptionPyramid | RT-only softargmax | 0.8886 +/- 0.0044 | 0.1634 +/- 0.0195 | 986.0 +/- 67.7 | 0.318 +/- 0.007 | 0.896 +/- 0.018 | 0.103 +/- 0.020 |
| ETS-InceptionPyramid | CE | 0.8717 +/- 0.0036 | 0.0605 +/- 0.0173 | 748.0 +/- 44.4 | 0.340 +/- 0.012 | 0.879 +/- 0.015 | 0.097 +/- 0.017 |
| ETS-InceptionPyramid | Mixture EventNLL | 0.8746 +/- 0.0028 | -0.0925 +/- 0.0049 | 540.0 +/- 38.1 | 0.473 +/- 0.008 | 0.600 +/- 0.023 | 0.185 +/- 0.019 |
| ETS-AttnSeg | RT-only softargmax | 0.9052 +/- 0.0300 | 0.2977 +/- 0.1669 | 1154.0 +/- 206.6 | 0.268 +/- 0.048 | 0.926 +/- 0.029 | 0.156 +/- 0.046 |
| ETS-AttnSeg | CE | 0.8780 +/- 0.0082 | 0.0771 +/- 0.0276 | 776.0 +/- 57.7 | 0.334 +/- 0.016 | 0.885 +/- 0.015 | 0.103 +/- 0.019 |
| ETS-AttnSeg | Mixture EventNLL | 0.8752 +/- 0.0050 | -0.0937 +/- 0.0083 | 568.0 +/- 23.9 | 0.477 +/- 0.007 | 0.608 +/- 0.027 | 0.173 +/- 0.023 |

Candidate interpretation: posterior diagnostics preserve the distinction
between scalar accuracy and posterior structure across architecture controls.
CE gives strong posterior-mean accuracy and empirical latent-event coverage,
whereas Mixture EventNLL gives the strongest fixed-kernel distributional score
and more concentrated target-aligned posterior mass. This supports the
posterior-geometry contribution as a property of event-time supervision rather
than a U-Net-specific artifact.

## Takeaways

1. The supervision effect is not U-Net-specific. RT-only softargmax remains
weaker than CE/Mixture EventNLL across U-Net, TCN, InceptionPyramid, and
AttnSeg controls. This supports the claim that the gain comes from
distributional event-time supervision rather than from the ETS-U-Net backbone
or differentiable temporal expectation readout alone.

2. U-Net should not be presented as the best backbone. TCN Mixture,
InceptionPyramid CE, U-Net Mixture, and AttnSeg Mixture all occupy a similar
scalar-accuracy range, with TCN/Inception slightly ahead on mean tau-nRMSE in
these runs. The architecture-control result is therefore best framed as
robustness of the output-formulation effect, not as an architecture ranking.

3. RT-only posterior-mean supervision is less reliable. This is clearest for
AttnSeg, where RT-only has high seed variability driven by one failed seed,
while CE and Mixture EventNLL recover performance in the same range as the
convolutional dense segmenters.

4. Posterior-geometry trade-offs reproduce across backbones. Mixture EventNLL
gives the strongest fixed-kernel EventNLL and more concentrated target-aligned
posterior mass, while CE tends to give strong scalar accuracy and empirical
latent-event coverage. This keeps the posterior-diagnostics story from being a
U-Net-only artifact.

5. Shifted-crop behavior remains a separate diagnostic axis. AttnSeg Mixture
shows the strongest sensitivity among the architecture-control rows, but all
models remain well below ideal crop-relative sensitivity. Architecture and
objective choice can improve localizer-like behavior without solving full
shift-equivariant response-timing localization.

Concise paper-facing synthesis:

> Across matched dense temporal backbones, distributional event-time
> supervision consistently improves over RT-only posterior-mean supervision.
> The effect is therefore not explained by the ETS-U-Net backbone alone.
> Architecture choice changes the scalar optimum and shifted-crop behavior, but
> the main supervision pattern and posterior-geometry trade-offs persist across
> U-Net, TCN, InceptionPyramid, and AttnSeg controls.

## How to Add This to the Paper

### Option A: Compact Main Table, Details in Appendix

Use one compact main table with RT-only, CE, and Mixture EventNLL across
ETS-U-Net, ETS-TCN, ETS-InceptionPyramid, and ETS-AttnSeg. Keep shifted-crop
architecture details and full posterior geometry diagnostics in the appendix.

Best when: we want the strongest journal-style synthesis without making the
main Results feel like an architecture zoo.

Main-text claim:

> Distributional event-time supervision improves posterior-mean RT prediction
> across dense temporal backbones with different temporal inductive biases.

### Option B: Appendix-First Architecture Robustness

Keep the current main Results centered on ETS-U-Net objective comparisons. Add
one short paragraph in the main text noting that architecture-control runs are
reported in the appendix. Put scalar accuracy, shifted-crop behavior, and
posterior diagnostics for TCN/InceptionPyramid/AttnSeg in appendix tables.

Best when: page budget is tight or the journal version should stay focused on
the output-formulation contribution rather than architecture benchmarking.

Main-text claim:

> The same RT-only versus distributional-supervision pattern is reproduced in
> additional dense temporal backbones, reported in Appendix Table X.

### Option C: Dedicated Results Subsection

Add a short Results subsection after the U-Net objective comparison:
`Architecture Controls for Dense Event-Time Prediction`. Include the compact
scalar table in main text. Place shifted-crop and posterior-geometry extensions
in appendix.

Best when: we want to actively preempt the concern that the contribution is
U-Net-specific.

Subsection flow:

1. State that ETS-U-Net is the primary fixed backbone for objective ablations.
2. Introduce dense temporal controls with different inductive biases.
3. Present the compact table.
4. Interpret the replicated gap between RT-only softargmax and CE/EventNLL.
5. Point to appendix posterior/shifted-crop diagnostics.

## Candidate Wording

### Why These Architectures

Compact version:

> We chose architecture controls that are native to one-dimensional temporal
> prediction rather than image-classification backbones. ETS-TCN provides a
> full-resolution residual dilated-convolution control, testing whether long
> temporal context without encoder-decoder skips is sufficient.
> ETS-InceptionPyramid provides an explicit multi-scale temporal filtering
> control, testing whether parallel receptive-field scales can replace
> U-Net-style temporal compression and reconstruction. ETS-AttnSeg provides an
> attention-based dense temporal control, testing whether global temporal
> context changes the effect of event-time supervision. All controls preserve
> the 200-bin time grid and use the same per-time event posterior readout.

Shorter version:

> The architecture controls target temporal inductive biases rather than generic
> image-backbone capacity: dilated full-resolution convolution (ETS-TCN),
> explicit multi-scale temporal filtering (ETS-InceptionPyramid), and
> attention-based global temporal context (ETS-AttnSeg). Each model maps the EEG
> window to per-time logits on the same 200-bin grid.

### Why ETS-U-Net Is the Primary Backbone

Less defensive version:

> ETS-U-Net serves as the primary backbone for objective comparisons because it
> is a compact dense temporal segmenter that maps naturally onto the event-time
> output space. It combines local and broader temporal context while preserving
> a per-time logit readout. We therefore fix this backbone when comparing output
> objectives, and use full-resolution TCN, InceptionPyramid, and AttnSeg
> controls to assess whether the same supervision effects transfer to
> alternative temporal inductive biases.

More explicit version:

> We use ETS-U-Net as the primary backbone for the objective comparisons because
> it is a simple dense temporal segmentation architecture that naturally
> matches the event-time output space: it maps each EEG window to per-time
> logits while combining local and broader temporal context through an
> encoder-decoder path with skip connections. The goal of this comparison is to
> isolate the effect of output supervision under a fixed, strong segmentation
> model. Additional full-resolution TCN, InceptionPyramid, and AttnSeg controls
> then test whether the conclusions transfer beyond the U-Net inductive bias.

### What This Experiment Shows

Main result framing:

> Across dense temporal backbones, RT-only posterior-mean supervision remains
> weaker than distributional event-time supervision. This shows that the gain is
> not simply a consequence of replacing a scalar head with a differentiable
> temporal readout; it depends on supervising the output distribution over
> event time.

Architecture robustness framing:

> The architecture-control runs shift the interpretation from "ETS-U-Net works"
> to "the event-time supervision effect transfers across dense temporal
> predictors." U-Net, TCN, and InceptionPyramid differ in how they obtain
> temporal context, but all preserve the same per-time posterior output space.

Careful limitation:

> These controls are not intended to rank all possible EEG backbones. They test
> whether the main supervision effect survives changes in temporal inductive
> bias while keeping the event-time output contract fixed.

### How to Introduce InceptionPyramid

> ETS-InceptionPyramid exposes multiple temporal scales explicitly: each block
> applies parallel temporal filters at several receptive-field widths, fuses the
> resulting features at the original 200-bin resolution, and applies residual
> temporal refinement before the per-time output head. Unlike ETS-U-Net, this
> model has no temporal pooling, decoder, or skip reconstruction; it tests
> whether explicit multi-scale filtering is sufficient for event-time posterior
> modeling.

### How to Introduce AttnSeg

> ETS-AttnSeg is an attention-based dense temporal segmenter. It preserves the
> 200-bin output grid while combining temporal self-attention, local depthwise
> temporal convolution, and feed-forward branches. This tests whether access to
> global temporal context changes the effect of event-time supervision without
> introducing scalar pooling or a pretrained representation-learning objective.

### Posterior Geometry Extension

> Scalar accuracy is not the only test of architecture transfer. The same
> posterior diagnostics used for ETS-U-Net can be applied to the architecture
> controls to ask whether distributional supervision changes posterior geometry
> in the same way across backbones. In the repeated-run summaries, CE provides
> strong posterior-mean accuracy and low CRPS, whereas Mixture EventNLL improves
> fixed-kernel distributional scoring. Full geometry diagnostics should further
> separate posterior concentration, target-aligned mass, and empirical interval
> coverage.

## Refresh Checklist

- [x] Wait for `06_inception_pyramid/ets_inception_pyramid_event_nll_mixture` to finish 5/5 seeds.
- [x] Run `04_attnseg` for RT-only, CE, and Mixture EventNLL.
- [x] Refresh the scalar accuracy table with all completed seeds.
- [x] Refresh shifted-crop architecture-control rows with all completed seeds.
- [x] Extend posterior-geometry summaries to TCN/InceptionPyramid and completed AttnSeg RT-only/CE/Mixture.
- [ ] Decide paper placement: compact main table plus appendix details, or appendix-only robustness.
