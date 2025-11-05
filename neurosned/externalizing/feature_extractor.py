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
        return A

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
            A = self.estimate_transition_matrix(eeg_norm, lag=lag, lam=self.lam)
            A = np.asarray(A)
            A_abs = np.abs(A)

            feats[f"lag{lag}_A_mean_abs"]      = float(A_abs.mean())
            feats[f"lag{lag}_A_diag_mean"]     = float(np.diag(A).mean())
            feats[f"lag{lag}_A_off_mean"]      = float(A[off_mask].mean())
            feats[f"lag{lag}_A_fro"]           = float(np.linalg.norm(A, 'fro'))
            feats[f"lag{lag}_A_asym_mean_abs"] = float(np.mean(np.abs(A - A.T)))
            feats[f"lag{lag}_A_sparsity@0.05"] = float(np.mean(A_abs < 0.05))
            mats[lag] = {"corr": C, "A": A}

        return feats, mats