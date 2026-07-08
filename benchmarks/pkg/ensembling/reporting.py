"""CSV and Markdown reporting for stacking results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


METRIC_COLUMNS = [
    "dev_oof_nrmse",
    "dev_oof_mae",
    "test_nrmse",
    "test_mae",
    "delta_vs_best_single_nrmse",
    "delta_vs_rt_only_nrmse",
]


def aggregate_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-seed stacking rows into mean/std paper rows."""
    records = []
    for method, group in rows.groupby("method", sort=False, observed=False):
        record = {
            "method": method,
            "seeds": int(group["seed"].nunique()),
            "n_models_mean": float(group["n_models"].mean()),
        }
        for column in METRIC_COLUMNS:
            values = pd.to_numeric(group[column], errors="coerce")
            record[f"{column}_mean"] = float(values.mean()) if values.notna().any() else np.nan
            record[f"{column}_std"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else 0.0
        records.append(record)
    return pd.DataFrame(records)


def write_markdown_table(aggregate: pd.DataFrame, path: Path) -> None:
    """Write a compact camera-ready Markdown table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "Method",
        "Seeds",
        "R11 nRMSE",
        "R11 MAE",
        "Delta vs best single",
        "Delta vs RT-only stacker",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---", "---:", "---:", "---:", "---:", "---:"]) + " |",
    ]
    for _, row in aggregate.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    str(int(row["seeds"])),
                    _fmt_pm(row, "test_nrmse"),
                    _fmt_pm(row, "test_mae"),
                    _fmt_pm(row, "delta_vs_best_single_nrmse", signed=True),
                    _fmt_pm(row, "delta_vs_rt_only_nrmse", signed=True),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt_pm(row: pd.Series, name: str, *, signed: bool = False) -> str:
    mean = row.get(f"{name}_mean", np.nan)
    std = row.get(f"{name}_std", np.nan)
    if pd.isna(mean):
        return ""
    prefix = "+" if signed and float(mean) > 0 else ""
    return f"{prefix}{float(mean):.4f} +/- {float(std):.4f}"
