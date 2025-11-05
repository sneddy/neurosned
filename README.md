# NeuroSned – EEG Reaction-Time Modeling (Competition Track)

This repository contains the code and notebooks we used to build a winning-level solution for the **EEG Foundation Model** competition task on **reaction-time prediction**.

## 🔑 Core ideas

- **Stimulus-locked preprocessing.** We convert run-level EEG into fixed **2‑second windows** starting **0.5 s after stimulus onset**, using anchor annotations. Data are already normalized by the challenge preprocessing (100 Hz, 0.5–50 Hz band).
- **Targets as time distributions.** Instead of a single scalar, we supervise a **soft 1‑D distribution** over time steps (`q`), and train the model to predict logits over time. We then compute the **expected time** `t_hat` from the distribution for RMSE.
- **Loss cocktail.** Weighted combination of **CE** on soft labels, **time RMSE**, optional **KL**, **Wasserstein‑1 (CDF)** and **entropy** regularization. An optional **hazard** head converts logits to a discrete-time hazard distribution.
- **Architectures.** 1D segmentation‑style models (UNet family, attention UNet, factorized UNet, Inception‑style, recurrent UNet) with a unified output `(B,1,T)`.
- **Robust training.** Time‑aware data augmentation (time scaling, channel dropout, cutout, Gaussian noise), **mixup** over time‑bins, early stopping with **LR halving & restart from best checkpoint**.
- **Ensembling.** Average distributions / expected times across multiple architectures and seeds for best validation scores.

> **Disclaimer**  
> You can choose **any model** below for testing — all the scores mentioned in comments are **achievable using the same training code**, given enough runs and proper tuning. The provided setups represent **the most stable base configurations** for both initial training and fine‑tuning modes. Exact leaderboard-level performance may require **multiple restarts and hyperparameter tweaks** for each model individually.

---

## ⚙️ Environment

We keep base dependencies in `requirements.txt` (without hard‑pinning PyTorch to a specific CUDA). Install **PyTorch** for your platform first, then the rest:

```bash
# 1) Install PyTorch for your platform (choose ONE)
# CUDA 12.4
pip install --extra-index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
# CUDA 12.1
# pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
# CPU-only
# pip install --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio

# 2) Install project requirements
pip install -r requirements.txt
```

---

## 📁 Data

We use the challenge‑ready HBN data via `EEGChallengeDataset` (preprocessed to 100 Hz, 0.5–50 Hz).  
Set `DATA_DIR` to your data cache location (the notebooks will populate it on first run).

```
artefacts/
  data/      # pickled datasets / cached windows
  models/    # checkpoints
  logs/      # optional logs & figures
```

---

## ▶️ Reproduction by notebooks (run top‑to‑bottom)

> All notebooks assume **Python ≥ 3.10** and **PyTorch ≥ 2.3**.  
> Open in JupyterLab and run sequentially; edit paths at the top where marked.

### 1. `1_data_preparation.ipynb` — annotate & window runs
- Load releases `R1..R11` with `EEGChallengeDataset` (competition preprocessing).  
- **Annotate trials** and **add stimulus anchors**.  
- **Filter late anchors** and **create 2‑s windows** (start at +0.5 s).  
- Save train/valid datasets to `artefacts/data/*.pkl`.

**Edit before running:** `DATA_DIR`, `release_list`, window params (`EPOCH_LEN_S=2.0`, `SFREQ=100`).

---

### 2. `2_regression.ipynb` — direct reaction-time regression

**Chronologically, we started with direct RT regression:** the model predicts the reaction time as a scalar from a stimulus-locked 2-s EEG crop. We designed several **lightweight custom models** (see `models/regression`) and trained them with strong augmentations and a **two-stage schedule**: base training → fine-tuning.

- **Augmentations:** time cutout, channel dropout, Gaussian noise, time-scaling, mixup.
- **Outcome:** this setup reliably beat the baseline and briefly held **#1** on the leaderboard.
- **Limitation:** the model did not fully capture the **event-detection** nature of the task; it tended to over-focus on micro-structure rather than “*when the event happens*”.

> *Notes:* The exact leaderboard scores are attainable with these pipelines, but typically require **multiple restarts and per-model hyperparameter tuning** (LR, sigma, augmentations, schedulers, seeds).

---

### 3. `3_segmentation.ipynb` — turning RT into a segmentation problem

To address detection explicitly, we introduced an **artificial segmentation target** over time and reframed RT prediction as **temporal segmentation** (the model outputs a distribution over time; we then take its expectation).

- **Effect:** large performance boost and faster, more stable convergence.
- **Practical benefit:** segmentation made **crop augmentations** safer/effective; the model learned *where* the event occurs.

**Losses & tricks.**
- **Dataset trick:** pool of **5-second crops** → sample 2-s training windows.
- **Cross-Entropy (CE)** on soft time labels.
- **Wasserstein-1 (CDF)** to align distribution shapes.
- **Entropy** regularization for sharpness control.
- **Focal** weighting for hard time bins.
- **KL divergence** (q ‖ p) for distribution matching.
- **Hazard head:** logits → discrete-time hazard → normalized event-time distribution.

#### Inference: soft-argmax with temperature

Let logits be \(z_t\) for \(t=0,\dots,T-1\), sampling step \(\Delta t=1/\mathrm{sfreq}\),
grid \(g_t=t\,\Delta t\), and window offset \(t_0=0.5\,\mathrm{s}\).

$$
p_t(\tau)=\frac{e^{z_t/\tau}}{\sum_{k=0}^{T-1} e^{z_k/\tau}},\qquad
\hat t_{\mathrm{rel}}(\tau)=\sum_{t=0}^{T-1} p_t(\tau)\,g_t,\qquad
\hat t_{\mathrm{abs}}(\tau)=t_0+\hat t_{\mathrm{rel}}(\tau).
$$

**Temperature selection**

$$
\tau^\star = \arg\min_{\tau\in\mathcal{T}}
\ \mathrm{RMSE}_{\text{val}}\big(\hat t_{\mathrm{abs}}(\tau)\big).
$$

#### Target: segmentation label with \(\sigma\)

For true relative time \(y_{\mathrm{rel}}\in[0,\mathrm{crop\_sec}]\) we build a Gaussian label on \(g_t\):

$$
\tilde q_t=\exp\!\left(-\frac{(g_t-y_{\mathrm{rel}})^2}{2\sigma^2}\right),\qquad
q_t=\frac{\tilde q_t}{\sum_{k=0}^{T-1}\tilde q_k\,\Delta t},\qquad
\sum_{t=0}^{T-1} q_t\,\Delta t=1.
$$


This smooth label stabilizes training while preserving temporal localization.

### 4. `4_ensembling.ipynb` — stacking & calibration over time-distributions

**Observation.** After moving to segmentation-style outputs, each model predicts a **time distribution** (logits over T). This made ensembling more powerful (we aggregate *richer* signals than scalars), but also trickier: every model has its own optimal **temperature** (confidence scale). Naïve averaging over-weights overconfident models and under-weights the best but **underconfident** ones.

**Solution: out-of-fold stacking with meta-features.**
We split validation into **5 folds** with a matched target (RT) distribution, extract features from each model’s predictions on its **OOF** split, and train a meta-model (Ridge, Gradient Boosting) to **calibrate confidence** and combine models.

**Pipeline.**
1. **5-fold OOF:** for each fold, train base models on 4/5 and predict the held-out 1/5 (store logits `Z ∈ ℝ^{N×T}` or probabilities).
2. **Meta-features per model** (see `neurosned/wrappers/challenge_1/meta_features.py`):
   - **Hard peak:**  
     `t_hard = argmax(Z)·dt + win_offset`, peak score `z_max`, top-2 **margin** (`z_max - z_second`).
   - For multiple temperatures `t ∈ temps`:
     - `p = softmax(Z / t)`; **softargmax time** `t_abs = E_p[t] + win_offset`.
     - **Entropy** `H(p)`, peak prob `pmax`, **probability margin**, **time variance** `Var_p[t]`.
     - **Quantiles** of event time (`q10`, `q50`, `q90`, …) from the CDF of `p`.
3. **Stacking model:** concatenate features across models; train **Ridge** (L2) and **GBDT** on OOF to predict RT. Pick the best (on our data it was GBDT).
4. **Inference:** compute the same features on validation/test predictions; apply the trained meta-model to get the final RT.

**Why it helps.** Stacking learns **per-model calibration** and trusts models differently across regimes (sharp vs. diffuse distributions), turning heterogeneous confidences into a coherent final estimate.



## 🧪 Practical tips

- **Seeds & restarts.** Fix seeds for comparability; do several runs to hit peak scores.  
- **Batch size & workers.** Increase gradually until near GPU RAM limit; set `num_workers` to CPU cores.  
- **Suppress benign logs.** We locally silence verbose data notices and out‑of‑range annotations (see notebook cells).  
- **CUDA mismatches.** If you see “no kernel image available” or `.so` load errors, install PyTorch matching your CUDA (cu121/cu124) or use CPU build.
- **Checkpoints.** Best weights are auto‑saved; restarts reload best and halve LR on plateaus.

---

## 📜 Citation

If you use this code or models, please cite this repository. A `CITATION.cff` is provided for automatic BibTeX/APA.

```bibtex
@software{neurosned2025,
  title   = {NeuroSned: Stimulus-locked EEG modeling for reaction-time prediction},
  year    = {2025},
  author  = {Your Team},
  url     = {https://github.com/<user>/<repo>}
}
```

---

## 📝 License

- Code: Apache-2.0 (recommended).  
- Model weights / docs: CC BY 4.0 (or CC BY‑NC 4.0 if you restrict commercial use).
