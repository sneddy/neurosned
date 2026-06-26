# Benchmark Preparation

Reproducible entrypoints for turning EEG Challenge releases into benchmark
pickle datasets. Run commands from the project root with the local Python
environment already activated.

## Running commands

```bash
python benchmarks/preparation/scripts/download_releases.py
python benchmarks/preparation/scripts/check_releases.py
python benchmarks/preparation/scripts/prepare_splitted_datasets.py
```

Small smoke runs:

```bash
python benchmarks/preparation/scripts/download_releases.py --releases R11 --output-dir /tmp/neurosned-release-data
python benchmarks/preparation/scripts/check_releases.py --releases R11 --input-dir /tmp/neurosned-release-data
```

## Pipeline

High level:

1. `download_releases.py` materializes `EEGChallengeDataset` recordings into
   `release_data/` and writes `download_manifest.json`.
2. `check_releases.py` reopens cached recordings and writes
   `check_manifest.json` with readable/failed counts.
3. `prepare_splitted_datasets.py` builds release-based `BaseConcatDataset`
   pickle files in `data/new_validation/` and writes `prepare_manifest.json`.

Low-level preprocessing:

1. Load selected releases through `EEGChallengeDataset`.
2. Annotate trials with `rt_from_stimulus` targets and add stimulus anchors.
3. Drop clearly broken cached BDF headers and skip unreadable recordings.
4. Remove stimulus anchors that cannot fit inside the requested window.
5. Create event windows at 100 Hz:
   - `2sec`: 0.5 s after stimulus, 2.0 s window.
   - `5sec`: 0.0 s after stimulus, 5.0 s window.
6. Add trial metadata columns and save the resulting datasets as pickle files.

Outputs:

- `data/new_validation/r1_r8_train.pkl`
- `data/new_validation/r1_r8_train_5sec.pkl`
- `data/new_validation/r9_r10_val.pkl`
- `data/new_validation/r9_r10_val_5sec.pkl`
- `data/new_validation/r11_test.pkl`

## Config defaults

Defaults live in `benchmarks/preparation/config.py`.

- Raw release cache: `release_data/`
- Prepared split output: `data/new_validation/`
- Task: `contrastChangeDetection`
- Releases: `R1` through `R11`
- Split policy: `R1-R8` train, `R9-R10` validation, `R11` holdout
- Metadata fields: `subject`, `session`, `run`, `task`, `age`, `gender`,
  `sex`, `p_factor`
- Window metadata: target, RTs, stimulus/response onset, correctness, response
  type
