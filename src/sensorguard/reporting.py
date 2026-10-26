"""Reproducible artifact exports."""

import json
from pathlib import Path

import pandas as pd

from .profiling import quality_summary


def export_report(frame: pd.DataFrame, output: str | Path) -> Path:
    """Write deterministic JSON summary and return its path."""
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(quality_summary(frame), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return target


def export_cleaned(frame: pd.DataFrame, output: str | Path) -> Path:
    """Export all parsed rows plus issue evidence as CSV; no suspicious row is dropped."""
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False, date_format="%Y-%m-%dT%H:%M:%S.%f")
    return target
