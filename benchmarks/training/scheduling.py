"""Training schedule helpers."""

from __future__ import annotations

from pathlib import Path

import torch


class ReloadBestOnPlateau:
    """Reload the best checkpoint and decay learning rate on plateau.

    This mirrors the notebook pattern used for segmentation experiments: when
    the trainer hits early-stopping patience, it reloads the best available
    checkpoint, recreates the optimizer on current model parameters, multiplies
    LR by `factor`, and lets training continue. If the current run has not saved
    a checkpoint yet, `fallback_checkpoint_path` can point to the starting
    weights.
    """

    def __init__(
        self,
        *,
        optimizer_factory,
        lr: float,
        factor: float = 0.5,
        optimizer_kwargs: dict | None = None,
        fallback_checkpoint_path: str | Path | None = None,
    ):
        self.optimizer_factory = optimizer_factory
        self.current_lr = float(lr)
        self.factor = float(factor)
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.fallback_checkpoint_path = Path(fallback_checkpoint_path) if fallback_checkpoint_path is not None else None

    def step(self, trainer) -> bool:
        """Apply reload-best plus learning-rate decay and continue training."""
        reload_path = self._reload_path(trainer)
        print(f"Restart from checkpoint {reload_path}. Best Val NRMSE: {trainer.best_metric:.6f} (epoch {trainer.best_epoch})")
        trainer.model.load_state_dict(torch.load(reload_path, map_location=trainer.device))

        next_lr = self.current_lr * self.factor
        print(f"Updating learning rate {self.current_lr} -> {next_lr}")
        self.current_lr = next_lr
        trainer.optimizer = self.optimizer_factory(
            trainer.model.parameters(),
            lr=self.current_lr,
            **self.optimizer_kwargs,
        )
        return True

    def _reload_path(self, trainer) -> Path:
        checkpoint_path = trainer.checkpoint_path
        if checkpoint_path is not None and checkpoint_path.exists():
            return checkpoint_path
        if self.fallback_checkpoint_path is not None:
            return self.fallback_checkpoint_path
        raise FileNotFoundError("No checkpoint is available for plateau reload.")
