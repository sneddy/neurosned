## Empirical sanity check (HBN, pseudo-devices via channel subsets)

**Goal:** show that masked reconstruction tends to retain *acquisition/view* information `u` (here: pseudo-device), and that adding an explicit invariance loss reduces (i) `u` decodability from embeddings and (ii) the ID→OOD drop for a downstream target (RT).

### Setup
- **Dataset:** HBN EEG, pick **one active task** (e.g., CCD) with trial-level **reaction time (RT)**.
- **Target:** `RT' = log(RT)` or z-score per subject.
- **Windows:** fixed-length EEG window per trial (same alignment/duration for all samples).
- **Splits:** subject-wise train/val/test.

### Pseudo-devices `u`
Create 2–3 channel subsets of equal size: `S1, S2 (, S3)`.
- Build regions (e.g., `F, C, P, O, TL, TR`) using electrode geometry.
- Sample the same number of channels from each region into each subset.
- **Important:** keep a fixed channel axis (e.g., union of channels) and use a **channel-mask** for missing channels so input dimensionality is constant across pseudo-devices.

### Pretraining (two variants)
1) **MAE-recon (baseline)**
   - For each sample, choose a pseudo-device `Sk` and keep only channels in `Sk` visible.
   - Apply *time/patch masking* **within visible channels only**.
   - Optimize reconstruction loss (MSE/Huber) on masked time points/patches.

2) **MAE-recon + invariance**
   - Same reconstruction objective.
   - Additionally, for the *same trial*, form two views `x_Si` and `x_Sj` (different pseudo-devices).
   - Compute pooled embeddings `h_i = f(x_Si)`, `h_j = f(x_Sj)`.
   - Add an invariance penalty, e.g. `L_inv = ||norm(h_i) - norm(h_j)||^2` (or InfoNCE).
   - Total: `L = L_recon + λ * L_inv` (fixed small `λ`, no heavy tuning).

### Evaluation metrics (test subjects)
1) **Reconstruction quality**
   - `ReconMSE(S1)`, `ReconMSE(S2)` (and `ReconMSE(S3)` if used).

2) **Nuisance predictability (does embedding encode `u`?)**
   - Train a **linear probe** to predict pseudo-device `u ∈ {S1, S2 (, S3)}` from embeddings `h`.
   - Report accuracy/AUROC: `Probe(u)`.

3) **Shift gap on RT (does `u` hurt transfer?)**
   - Freeze encoder `f`; train a linear RT head.
   - **ID:** train+test on the same pseudo-device `Sk`.
   - **OOD:** train on `Si`, test on `Sj` (average over pairs).
   - Report Pearson `r` and/or RMSE, plus `gap = r_ID - r_OOD` (or `gap = RMSE_OOD - RMSE_ID`).

### Expected pattern
- **MAE-recon:** high `Probe(u)` and a noticeable ID→OOD RT gap.
- **MAE + invariance:** lower `Probe(u)` and smaller gap (better OOD), with comparable reconstruction/ID performance.
