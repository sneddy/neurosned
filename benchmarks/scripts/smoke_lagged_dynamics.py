"""Run a few real-data optimizer steps for a lagged-dynamics config."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

os.environ["MNE_DONTWRITE_HOME"] = "true"
os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["MPLCONFIGDIR"] = "/tmp/neurosned-matplotlib"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from benchmarks.pkg.config import load_experiment_config
from benchmarks.pkg.evaluation.factory import build_dataset_wrapper
from benchmarks.pkg.runtime import choose_device
from benchmarks.pkg.utils import set_seed


DEFAULT_CONFIG = PROJECT_ROOT / "benchmarks" / "configs" / "07_lagged_dynamics" / "lagged_dynamics_full.yaml"


def build_parser() -> argparse.ArgumentParser:
    """Build the smoke-test CLI."""
    parser = argparse.ArgumentParser(description="Run finite-value checks on real EEG optimizer steps.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser


def tensors_are_finite(parameters, *, gradients: bool) -> bool:
    """Return whether all present parameters or gradients are finite."""
    for parameter in parameters:
        tensor = parameter.grad if gradients else parameter
        if tensor is not None and not torch.isfinite(tensor).all():
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    """Run real-data forward/backward checks without writing run artifacts."""
    args = build_parser().parse_args(argv)
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")

    config = load_experiment_config(args.config.resolve())
    set_seed(config.seed)
    device = choose_device(args.device)
    base_dataset = config.build_dataset("train", PROJECT_ROOT)
    model = config.model.build().to(device)
    channels = np.arange(model.n_chans) if hasattr(model, "n_chans") else None
    dataset = build_dataset_wrapper(config.data.train_dataset, base_dataset, channels)
    batch_size = args.batch_size or config.loaders.train.batch_size
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = config.optimizer.build(model.parameters())

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    model.train()
    for step, (inputs, targets) in enumerate(loader, start=1):
        inputs = inputs.to(device).float().contiguous()
        targets = targets.to(device).float()
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = torch.nn.functional.mse_loss(predictions, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}: {loss.detach().item()}")
        loss.backward()
        if not tensors_are_finite(model.parameters(), gradients=True):
            raise FloatingPointError(f"Non-finite gradient at step {step}.")
        optimizer.step()
        if not tensors_are_finite(model.parameters(), gradients=False):
            raise FloatingPointError(f"Non-finite parameter at step {step}.")
        print(
            f"step={step:02d} loss={loss.detach().item():.6f} "
            f"pred_mean={predictions.detach().mean().item():.6f} "
            f"pred_std={predictions.detach().std().item():.6f}"
        )
        if step >= args.steps:
            break

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"cuda_peak_memory_mb={peak_mb:.1f}")
    print(f"elapsed_seconds={time.perf_counter() - started:.2f}")
    print("status=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
