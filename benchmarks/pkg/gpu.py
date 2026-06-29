"""Lightweight GPU utilization monitoring for benchmark runs."""

from __future__ import annotations

import csv
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class GpuSample:
    """One GPU utilization sample."""

    elapsed_sec: float
    gpu_util_pct: float
    memory_used_mb: float
    memory_total_mb: float


class GpuMonitor:
    """Sample `nvidia-smi` during training and save CSV plus plot."""

    def __init__(self, *, gpu_index: int, csv_path: str | Path, plot_path: str | Path, interval_sec: float = 10.0):
        self.gpu_index = int(gpu_index)
        self.csv_path = Path(csv_path)
        self.plot_path = Path(plot_path)
        self.interval_sec = float(interval_sec)
        self.samples: list[GpuSample] = []
        self.gpu_name: str | None = None
        self._started_at: float | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: str | None = None

    def start(self) -> None:
        """Start background sampling."""
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.plot_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["elapsed_sec", "gpu_util_pct", "memory_used_mb", "memory_total_mb"])
        self._started_at = time.perf_counter()
        try:
            self.gpu_name = self._read_gpu_name()
        except Exception as exc:
            self._error = repr(exc)
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        """Stop sampling, save the plot and return aggregate metrics."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.interval_sec + 1.0, 2.0))
        self._plot()
        return self.summary()

    def summary(self) -> dict[str, Any]:
        """Return aggregate GPU monitoring metrics."""
        if not self.samples:
            return {
                "gpu_name": self.gpu_name,
                "gpu_samples": 0,
                "gpu_monitoring_error": self._error,
                "gpu_csv": self.csv_path,
                "gpu_plot": None,
            }

        util = np.array([sample.gpu_util_pct for sample in self.samples], dtype=float)
        memory = np.array([sample.memory_used_mb for sample in self.samples], dtype=float)
        total = float(self.samples[-1].memory_total_mb)
        return {
            "gpu_name": self.gpu_name,
            "gpu_memory_total_mb": total,
            "gpu_memory_peak_mb": float(memory.max()),
            "gpu_memory_mean_mb": float(memory.mean()),
            "gpu_util_mean_pct": float(util.mean()),
            "gpu_util_max_pct": float(util.max()),
            "gpu_samples": len(self.samples),
            "gpu_monitoring_error": self._error,
            "gpu_csv": self.csv_path,
            "gpu_plot": self.plot_path if self.plot_path.exists() else None,
        }

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._sample_once()
            self._stop_event.wait(self.interval_sec)

    def _sample_once(self) -> None:
        if self._started_at is None:
            return
        try:
            util, memory_used, memory_total = self._read_gpu_stats()
        except Exception as exc:
            self._error = repr(exc)
            self._stop_event.set()
            return
        sample = GpuSample(
            elapsed_sec=time.perf_counter() - self._started_at,
            gpu_util_pct=util,
            memory_used_mb=memory_used,
            memory_total_mb=memory_total,
        )
        self.samples.append(sample)
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    f"{sample.elapsed_sec:.3f}",
                    f"{sample.gpu_util_pct:.3f}",
                    f"{sample.memory_used_mb:.3f}",
                    f"{sample.memory_total_mb:.3f}",
                ]
            )

    def _read_gpu_name(self) -> str:
        output = self._run_nvidia_smi(["--query-gpu=name", "--format=csv,noheader"])
        return output.strip().splitlines()[0].strip()

    def _read_gpu_stats(self) -> tuple[float, float, float]:
        output = self._run_nvidia_smi(
            [
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ]
        )
        parts = [part.strip() for part in output.strip().splitlines()[0].split(",")]
        return float(parts[0]), float(parts[1]), float(parts[2])

    def _run_nvidia_smi(self, args: list[str]) -> str:
        command = ["nvidia-smi", f"--id={self.gpu_index}", *args]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return result.stdout

    def _plot(self) -> None:
        if not self.samples:
            return
        self.plot_path.parent.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/neurosned-matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        elapsed = np.array([sample.elapsed_sec for sample in self.samples], dtype=float)
        util = np.array([sample.gpu_util_pct for sample in self.samples], dtype=float)
        memory_gb = np.array([sample.memory_used_mb for sample in self.samples], dtype=float) / 1024.0

        fig, ax_util = plt.subplots(figsize=(9, 4.5))
        ax_memory = ax_util.twinx()

        util_line = ax_util.plot(elapsed, util, color="#2f6fbd", label="GPU util (%)", linewidth=2)
        memory_line = ax_memory.plot(elapsed, memory_gb, color="#c44e52", label="Memory used (GB)", linewidth=2)

        ax_util.set_xlabel("Training time (s)")
        ax_util.set_ylabel("GPU utilization (%)")
        ax_memory.set_ylabel("Memory used (GB)")
        ax_util.set_ylim(0, 100)
        ax_util.grid(True, alpha=0.25)

        lines = util_line + memory_line
        labels = [line.get_label() for line in lines]
        ax_util.legend(lines, labels, loc="upper right")

        title = "GPU usage"
        if self.gpu_name:
            title = f"{title}: {self.gpu_name}"
        ax_util.set_title(title)
        fig.tight_layout()
        fig.savefig(self.plot_path, dpi=150)
        plt.close(fig)
