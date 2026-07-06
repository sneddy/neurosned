"""Evaluate trained RT models on temporally shifted crops."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["MNE_DONTWRITE_HOME"] = "true"
os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["MPLCONFIGDIR"] = "/tmp/neurosned-matplotlib"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.pkg.evaluation.shifted import DEFAULT_DATASET, DEFAULT_STARTS, run_shifted_eval_from_run_dir


def build_parser() -> argparse.ArgumentParser:
    """Build the shifted-crop evaluation CLI."""
    parser = argparse.ArgumentParser(description="Evaluate a trained regression or segmentation run on shifted 2 s crops.")
    parser.add_argument("run_dir", type=Path, help="Existing run directory with config.yaml and best_model.pth.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Long-window dataset pickle used for shifted crops. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--starts",
        type=float,
        nargs="+",
        default=list(DEFAULT_STARTS),
        help="Crop starts in seconds relative to stimulus onset. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--reference-start",
        type=float,
        default=0.5,
        help="Canonical crop start used by the trained protocol. Defaults to %(default)s.",
    )
    parser.add_argument("--target-min", type=float, default=None, help="Optional minimum absolute RT kept for evaluation.")
    parser.add_argument("--target-max", type=float, default=None, help="Optional maximum absolute RT kept for evaluation.")
    parser.add_argument("--crop-sec", type=float, default=None, help="Crop length in seconds. Defaults to model n_times/sfreq.")
    parser.add_argument("--sfreq", type=float, default=None, help="Sampling rate. Defaults to model/config sfreq, then 100.")
    parser.add_argument("--batch-size", type=int, default=None, help="Evaluation batch size. Defaults to config valid batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers. Defaults to config valid workers.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto", help="Torch device selection.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Defaults to <run_dir>/shifted_eval.")
    parser.add_argument(
        "--segmentation-temperature",
        type=float,
        default=None,
        help="Softmax temperature for segmentation soft-argmax. Defaults to eval_temperature/temperature from config.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000, help="Subject-bootstrap samples for CI tables.")
    parser.add_argument("--bootstrap-seed", type=int, default=2025, help="Subject-bootstrap random seed.")
    parser.add_argument(
        "--save-predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save full shifted_predictions.csv. Defaults to true for CLI backward compatibility.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    run_shifted_eval_from_run_dir(
        run_dir=args.run_dir,
        dataset_path=args.dataset,
        target_min=args.target_min,
        target_max=args.target_max,
        starts=args.starts,
        reference_start=args.reference_start,
        crop_sec=args.crop_sec,
        sfreq=args.sfreq,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        segmentation_temperature_override=args.segmentation_temperature,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        output_dir=args.output_dir,
        save_predictions=args.save_predictions,
        device_name=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
