import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

class MedianBootstrapRidgeModel(nn.Module):
    """
    Torch-модель с медианным баггингом Ridge:
    - нормализуем фичи (по всему train)
    - делаем B бутстрапов (по строкам или кластерам subject_id)
    - обучаем Ridge на каждом бутстрапе
    - берём медиану коэфов и смещения, сохраняем как state_dict
    Интерфейс и методы совместимы с RidgeModel.
    """
    def __init__(self, usecols: list[str]):
        super().__init__()
        self.usecols = usecols
        self.register_buffer("coef_", torch.empty(len(usecols), dtype=torch.float32))
        self.register_buffer("bias_", torch.tensor(0.0, dtype=torch.float32))  # scalar
        self.register_buffer("feat_means", torch.empty(len(usecols), dtype=torch.float32))
        self.register_buffer("feat_stds", torch.empty(len(usecols), dtype=torch.float32))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        feats = (feats - self.feat_means) / self.feat_stds
        feats = feats[:, :len(self.usecols)]
        pred = feats @ self.coef_ + self.bias_
        return pred.unsqueeze(1)

    @classmethod
    def make_checkpoint(cls,
                        X_train: pd.DataFrame,
                        y_train: pd.Series | np.ndarray,
                        usecols: list[str],
                        C: float = 1.0,
                        n_estimators: int = 200,
                        subject_col: str | None = None,
                        cluster_bootstrap: bool = True,
                        random_state: int = 42,
                        force_bias = None,
                        save_path: str = "median_bootstrap_ridge.pt"):
        """
        Обучает ансамбль Ridge на бутстрап-выборках и сохраняет медианные параметры.

        Args:
            X_train: DataFrame с фичами (+ опц. колонка subject_col).
            y_train: целевая переменная.
            usecols: имена признаков (важен порядок).
            C: обратная регуляризация Ridge (alpha = 1 / C).
            n_estimators: число бутстрапов.
            subject_col: если задан, бутстрап идёт по субъектам.
            cluster_bootstrap: True = кластерный бутстрап (выбираем субъекты с возвращением и берём ВСЕ их строки).
                               False = обычный бутстрап по строкам.
            random_state: сид генератора.
            force_bias: если нужно зафиксировать bias (иначе медиана intercept’ов).
            save_path: путь для torch.save.
        """
        rng = np.random.default_rng(random_state)

        # 1) Подготовка данных и нормировок
        X_sub = X_train[usecols].to_numpy(dtype=np.float32)
        y_arr = np.asarray(y_train, dtype=np.float32).reshape(-1)
        assert X_sub.shape[0] == y_arr.shape[0], "X_train and y_train length mismatch"

        feat_means = X_sub.mean(axis=0)
        feat_stds  = X_sub.std(axis=0)
        feat_stds[feat_stds == 0] = 1.0

        Xn_full = (X_sub - feat_means) / feat_stds

        # 2) Подготовка индексов для бутстрапа
        if subject_col is not None and cluster_bootstrap:
            subjects = X_train[subject_col].to_numpy()
            uniq_subj = np.unique(subjects)
            # маппинг: subj -> индексы строк
            subj_to_idx = {s: np.flatnonzero(subjects == s) for s in uniq_subj}

        # 3) Бутстрап-обучение
        al = 1.0 / C
        coefs = []
        biases = []
        for _ in range(n_estimators):
            if subject_col is not None and cluster_bootstrap:
                # кластерный бутстрап: субъекты с возвращением, берём ВСЕ их строки
                picked = rng.choice(uniq_subj, size=len(uniq_subj), replace=True)
                idxs = np.concatenate([subj_to_idx[s] for s in picked])
            else:
                # обычный бутстрап по строкам
                n = Xn_full.shape[0]
                idxs = rng.integers(0, n, size=n, endpoint=False)

            Xb = Xn_full[idxs]
            yb = y_arr[idxs]

            model = Ridge(alpha=al, fit_intercept=True)
            model.fit(Xb, yb)

            coefs.append(model.coef_.ravel())
            biases.append(model.intercept_)

        coefs = np.asarray(coefs, dtype=np.float32)         # (B, F)
        biases = np.asarray(biases, dtype=np.float32)       # (B,)

        coef_med = np.median(coefs, axis=0)
        bias_med = float(np.median(biases)) if force_bias is None else float(force_bias)

        # 4) Сохранение чекпойнта
        state_dict = {
            "coef_": torch.tensor(coef_med, dtype=torch.float32),
            "bias_" : torch.tensor(bias_med, dtype=torch.float32),
            "feat_means": torch.tensor(feat_means, dtype=torch.float32),
            "feat_stds" : torch.tensor(feat_stds, dtype=torch.float32),
        }
        meta = {
            "usecols": usecols,
            "C": C,
            "n_estimators": n_estimators,
            "subject_col": subject_col,
            "cluster_bootstrap": cluster_bootstrap,
            "random_state": random_state,
        }
        torch.save({"state_dict": state_dict, "meta": meta}, save_path)
        print(f"✅ Median-Bootstrap Ridge checkpoint saved to {save_path}")

        # 5) Готовая к использованию модель
        model = cls(usecols)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model

    def forward_df(self, X: pd.DataFrame, clip_low=None, clip_high=None):
        X_full = X[self.usecols].to_numpy(dtype=np.float32)
        feats = torch.from_numpy(X_full)
        preds_t = self.forward(feats).squeeze(1)
        preds = preds_t.detach().cpu().numpy()
        if clip_low is not None or clip_high is not None:
            preds = np.clip(preds, a_min=clip_low, a_max=clip_high)
        return preds

    def eval_df(self, X: pd.DataFrame, y, clip_low=None, clip_high=None) -> float:
        preds = self.forward_df(X, clip_low=clip_low, clip_high=clip_high)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        mse = float(np.mean((y - preds) ** 2))
        nrmse = float(np.sqrt(mse) / (np.std(y) if np.std(y) > 0 else 1.0))
        return nrmse
