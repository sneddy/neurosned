# trial_sorted_posterior_raster

## Draft Caption

Trial-sorted event-time posterior maps on R11 for matched segmentation losses (CE, CE+time, EventNLL, Time-only, Wasserstein). Rows are quantile bins of representable R11 trials sorted by observed RT; the x-axis is time from stimulus onset; color shows log-transformed posterior density, `log(1 + p(t|x)/dt)`, averaged within each bin. The black curve marks the mean observed RT in each bin. A coherent event-time localization model should form posterior mass near this curve. The raster displays 260 RT-sorted bins formed from 15,164 representable R11 trials.

## Interpretation

This figure is qualitative support for the aggregate posterior-geometry panels: scalar RT error can be similar even when the learned temporal evidence map is sharp, diffuse, shifted, or multimodal.
