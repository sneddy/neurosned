# Release-Separated CCD Protocol Summary

Created: 2026-07-07T08:28:03.849290+00:00

Prepared split directory: `/home/sneddy/sneddy_projects/neurosned/data/new_validation`

Main analyzed-trial support: `0.50 <= RT <= 2.50` seconds.

| Partition | Releases | Role | Prepared 2 s trials | Analyzed trials | Subjects |
| --- | --- | --- | ---: | ---: | ---: |
| Train | R1--R8 | model fitting | 74,576 | 73,030 | 1,221 |
| Development | R9--R10 | early stopping, checkpoint selection, temperature tuning | 17,867 | 17,348 | 308 |
| Test | R11 | one-shot final holdout evaluation | 15,751 | 15,164 | 292 |

Prepared trials are CCD windows with a stimulus and response annotation. Analyzed trials additionally satisfy the main RT-support filter, matching the fixed 2 s inference window.

## Subject Overlap After Filtering

| Split pair | Overlap subjects |
| --- | ---: |
| r1_r8_train vs r9_r10_val | 0 |
| r1_r8_train vs r11_test | 0 |
| r9_r10_val vs r11_test | 0 |

The train, development, and R11 test partitions are subject-disjoint after the main support filter.

## Shifted-Crop Common-Inside Subset

The shifted-crop diagnostic uses the common-inside subset `0.80 <= RT <= 2.20` seconds.

| Partition | 2 s shifted-subset trials | 2 s shifted-subset subjects | 5 s shifted-subset trials |
| --- | ---: | ---: | ---: |
| Train | 67,938 | 1,219 | 67,935 |
| Development | 16,042 | 308 | 16,042 |
| Test | 14,209 | 292 | 14,208 |

## Package Versions

| Package | Version |
| --- | --- |
| braindecode | 1.5.2 |
| eegdash | 0.8.3 |
| mne | 1.12.1 |
| numpy | 2.3.5 |
| pandas | 2.3.2 |
| torch | 2.9.0 |

