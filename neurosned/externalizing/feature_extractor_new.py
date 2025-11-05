import numpy as np

class ExternalizingFeaturesExtractor:
    """
    Build features from lagged cross-correlation and a ridge-regularized
    transition matrix A across one or more lags.

    Usage:
        det = ExternalizingFeaturesExtractor(lags=(10, 25, 100), lam=1e-2, downsample=1)
        features, mats = det.extract_features(eeg)  # eeg shape: (n_channels, n_time)
    """

    def __init__(self, lags=(10,), lam: float = 1e-2, downsample: int = 1):
        self.lags = tuple(lags)
        self.lam = float(lam)
        self.downsample = int(downsample)

    @staticmethod
    def safe_norm(arr: np.ndarray) -> np.ndarray:
        """
        Normalize by channel: zero-mean, unit-std along axis=1.
        Std==0 rows left unchanged (std set to 1).
        """
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        mean = arr.mean(axis=1, keepdims=True)
        std = arr.std(axis=1, keepdims=True)
        std_safe = np.where(std == 0, 1, std)
        return (arr - mean) / std_safe

    @staticmethod
    def get_lag_slices(eeg_norm: np.ndarray, lag: int):
        Xt   = eeg_norm[:, :-lag]
        Xlag = eeg_norm[:,  lag:]
        T = Xt.shape[1]
        return Xt, Xlag, T

    @staticmethod
    def offdiag_mask(n: int) -> np.ndarray:
        """Boolean mask for off-diagonal entries in an n×n matrix."""
        m = np.ones((n, n), dtype=bool)
        np.fill_diagonal(m, False)
        return m

    @staticmethod
    def safe_entropy(x: np.ndarray, eps: float = 1e-12) -> float:
        """Shannon entropy of |x| normalized to a probability vector (natural log)."""
        x = np.abs(x).ravel()
        s = x.sum()
        if s <= eps:
            return 0.0
        p = x / s
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    # -------- core computations --------
    @staticmethod
    def compute_lagged_corr_mat(eeg_norm: np.ndarray, lag: int = 10) -> np.ndarray:
        """
        Channel-by-channel correlation between x(t) and x(t+lag).
        Mirrors np.corrcoef(..., ddof=0) semantics used in original code.
        """
        current_eeg, next_eeg, _ = ExternalizingFeaturesExtractor.get_lag_slices(eeg_norm, lag=lag)
        with np.errstate(invalid='ignore', divide='ignore'):
            corr = np.corrcoef(current_eeg, next_eeg)[:current_eeg.shape[0], next_eeg.shape[0]:]
        corr = np.nan_to_num(corr, nan=0.0)
        return np.asarray(corr)

    @staticmethod
    def estimate_transition_matrix(eeg_norm: np.ndarray, lag: int = 10, lam: float = 1e-2) -> np.ndarray:
        """
        Ridge-regularized linear mapping A such that X_{t+lag} ≈ A X_t.
        Uses A = Cxy @ (Cxx + lam*alpha*I)^{-1}, alpha = trace(Cxx)/n.
        """
        X_t, X_lag, T = ExternalizingFeaturesExtractor.get_lag_slices(eeg_norm, lag=lag)
        Cxx = (X_t @ X_t.T) / T
        Cxy = (X_lag @ X_t.T) / T
        n = Cxx.shape[0]
        alpha = float(np.trace(Cxx)) / n
        C_reg = Cxx + (lam * alpha) * np.eye(n, dtype=Cxx.dtype)
        try:
            A = Cxy @ np.linalg.solve(C_reg, np.eye(n, dtype=Cxx.dtype))
        except np.linalg.LinAlgError:
            # fallback: pseudoinverse with stronger regularization
            A = Cxy @ np.linalg.pinv(C_reg + 1e-1 * np.eye(n, dtype=Cxx.dtype))

        resid = X_lag - (A @ X_t)
        num = float(np.linalg.norm(resid, 'fro') ** 2)
        den = float(np.linalg.norm(X_lag, 'fro') ** 2 + 1e-12)
        R2 = float(1.0 - num / den)
        return A, R2

    @staticmethod
    def _spectral_radius(A):
        try:
            eig = np.linalg.eigvals(A)
        except np.linalg.LinAlgError:
            eig = np.linalg.eigvals((A + A.T) / 2.0)
        return float(np.max(np.abs(eig)))

    # @staticmethod
    # def get_global_feats(feats: dict) -> dict:
    #     """
    #     Compute a compact set of global (lag-averaged) features
    #     across the most informative per-lag metrics.
    #     Returns:
    #         dict[str, float]: keys like 'corr_mean_abs_avg', 'A_R2_avg', etc.
    #     """
    #     def collect(prefix):
    #         vals = [v for k, v in feats.items() if k.startswith(prefix)]
    #         return np.array(vals, dtype=float) if vals else None

    #     global_feats = {}

    #     # --- correlation-based ---
    #     if (arr := collect("lag") ) is not None:
    #         corr_mean_abs = collect("_corr_mean_abs")
    #         corr_entropy  = collect("_corr_entropy")
    #         if corr_mean_abs is not None and corr_mean_abs.size:
    #             global_feats["corr_mean_abs_avg"] = float(np.mean(corr_mean_abs))
    #             global_feats["corr_mean_abs_std"] = float(np.std(corr_mean_abs))
    #         if corr_entropy is not None and corr_entropy.size:
    #             global_feats["corr_entropy_avg"] = float(np.mean(corr_entropy))
    #             global_feats["corr_entropy_std"] = float(np.std(corr_entropy))

    #     # --- A-based dynamic summaries ---
    #     for name in ["A_fro", "A_asym_mean_abs", "A_spectral_radius", "A_R2"]:
    #         arr = collect(f"lag")
    #         arr = [v for k, v in feats.items() if k.endswith(name)]
    #         if len(arr):
    #             arr = np.array(arr, dtype=float)
    #             global_feats[f"{name}_avg"] = float(np.mean(arr))
    #             global_feats[f"{name}_std"] = float(np.std(arr))

    #     return global_feats

    # -------- public API --------
    def extract_features(self, eeg: np.ndarray):
        """
        Compute features and matrices for each lag in self.lags.
        Returns:
            feats: dict[str, float]
            mats:  dict[lag, {"corr": np.ndarray, "A": np.ndarray}]
        """
        X = np.asarray(eeg, dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if self.downsample and self.downsample > 1:
            X = X[:, ::self.downsample]
        eeg_norm = self.safe_norm(X)

        n_ch, n_t = eeg_norm.shape
        off_mask = self.offdiag_mask(n_ch)

        feats = {}
        mats = {}

        for lag in self.lags:
            # --- lagged correlation block (n×n)
            C = self.compute_lagged_corr_mat(eeg_norm, lag=lag)
            C_abs = np.abs(C)

            feats[f"lag{lag}_corr_mean_abs"]  = float(C_abs.mean())
            feats[f"lag{lag}_corr_diag_mean"] = float(np.diag(C).mean())
            feats[f"lag{lag}_corr_off_mean"]  = float(C[off_mask].mean())
            feats[f"lag{lag}_corr_entropy"]   = self.safe_entropy(C)

            # --- transition matrix A (n×n)
            A, r2 = self.estimate_transition_matrix(eeg_norm, lag=lag, lam=self.lam)
            A = np.asarray(A)
            A_abs = np.abs(A)

            feats[f"lag{lag}_A_mean_abs"]      = float(A_abs.mean())
            feats[f"lag{lag}_A_diag_mean"]     = float(np.diag(A).mean())
            feats[f"lag{lag}_A_off_mean"]      = float(A[off_mask].mean())
            feats[f"lag{lag}_A_fro"]           = float(np.linalg.norm(A, 'fro'))
            feats[f"lag{lag}_A_asym_mean_abs"] = float(np.mean(np.abs(A - A.T)))
            feats[f"lag{lag}_A_sparsity@0.05"] = float(np.mean(A_abs < 0.05))
            # new
            node_strength_c = C_abs.sum(axis=1)
            node_strength_a = A_abs.sum(axis=1)
            feats[f"lag{lag}_C_strength_cv"]        = float(node_strength_c.std() / (node_strength_c.mean() + 1e-12))
            feats[f"lag{lag}_C_diag_energy_ratio"]  = float((np.diag(C) ** 2).sum() / ((C ** 2).sum() + 1e-12))
            feats[f"lag{lag}_A_strength_cv"]        = float(node_strength_a.std() / (node_strength_a.mean() + 1e-12))
            feats[f"lag{lag}_A_spectral_radius"]    = self._spectral_radius(A)
            feats[f"lag{lag}_A_R2"]                 = r2
            mats[lag] = {"corr": C, "A": A}

        return feats, mats