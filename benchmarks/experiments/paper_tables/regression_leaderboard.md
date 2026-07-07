| Model | Role | Valid nRMSE $\downarrow$ (mean +/- std) | R11 nRMSE $\downarrow$ (mean +/- std) |
| --- | --- | ---: | ---: |
| MSP-CNN | compact scalar regression baseline | 0.900595 +/- 0.005096 | 0.899795 +/- 0.008045 |
| ETR-CNN | temporal-readout direct regression | 0.900822 +/- 0.006043 | 0.897679 +/- 0.006773 |
| ETR-CNN large | capacity ablation | 0.897233 +/- 0.004022 | 0.892783 +/- 0.004161 |
| TIDNet wrapped | strongest external supervised baseline in this comparison | 0.923506 +/- 0.002357 | 0.919204 +/- 0.002739 |
| EEGConformer wrapped | modern supervised transformer-style baseline | 0.918845 +/- 0.002634 | 0.928656 +/- 0.005658 |
| EEGNet wrapped | canonical compact EEG baseline | 0.935037 +/- 0.005384 | 0.933504 +/- 0.002832 |
| LaBraM wrapped | foundation-style architecture from scratch | 0.930388 +/- 0.006065 | 0.932722 +/- 0.008568 |
| Deep4Net wrapped | classical convolutional baseline | 0.926883 +/- 0.004521 | 0.925972 +/- 0.004354 |
| ShallowFBCSPNet wrapped | classical filter-bank spatial convolution baseline | 0.932353 +/- 0.001632 | 0.934285 +/- 0.002423 |
| ATCNet wrapped | supervised conv/attention/TCN baseline | 0.968640 +/- 0.018283 | 0.966578 +/- 0.015093 |
| EEGPT wrapped | foundation-style architecture from scratch | 0.961615 +/- 0.020110 | 0.958448 +/- 0.018547 |
| Medformer wrapped | larger transformer/time-series baseline | 0.962349 +/- 0.004576 | 0.958536 +/- 0.005125 |
