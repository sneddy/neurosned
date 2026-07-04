"""Runtime helpers shared by benchmark command-line tools."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def choose_device(name: str) -> torch.device:
    """Select the torch device for a benchmark command."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return torch.device(name)


def path_text(path: Path | str | None, *, project_root: Path | None = None) -> str:
    """Return a readable path relative to the project root when possible."""
    if path is None:
        return "None"
    root = PROJECT_ROOT if project_root is None else Path(project_root)
    path = Path(path)
    try:
        return str(path.relative_to(root))
    except ValueError:
        try:
            return str(path.resolve().relative_to(root))
        except ValueError:
            return str(path)


class TeeStream:
    """Write text to the terminal and a log file."""

    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, text: str) -> int:
        self.terminal.write(text)
        self.log_file.write(text)
        return len(text)

    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return self.terminal.isatty()


@contextmanager
def tee_output(log_path: Path):
    """Mirror stdout and stderr to a run log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
        sys.stdout = TeeStream(old_stdout, log_file)
        sys.stderr = TeeStream(old_stderr, log_file)
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
