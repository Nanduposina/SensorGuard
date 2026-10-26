"""Non-destructive issue annotation and duplicate detection."""

from __future__ import annotations

import pandas as pd

from .config import SensorGuardConfig


def validate_frame(frame: pd.DataFrame, config: SensorGuardConfig | None = None) -> pd.DataFrame:
    """Return a copy with quality flags; never removes or repairs observations."""
    cfg = config or SensorGuardConfig()
    result = frame.copy()
    result["issues"] = result["issues"].map(lambda v: list(v) if isinstance(v, list) else [])
    for idx, row in result.iterrows():
        issues = row["issues"]
        mote = row.get("moteid")
        if pd.notna(mote) and int(mote) not in cfg.sensor_ids:
            issues.append("invalid_sensor_id")
        for feature, (low, high) in cfg.physical_limits.items():
            value = row.get(feature)
            if pd.notna(value) and not low <= float(value) <= high:
                issues.append(f"suspicious_{feature}")
    result["issues"] = result["issues"].map(lambda x: list(dict.fromkeys(x)))
    result["duplicate"] = result.duplicated(subset=["timestamp", "moteid"], keep=False)
    result["out_of_order"] = False
    result["sampling_gap_seconds"] = float("nan")
    valid = result.dropna(subset=["timestamp", "moteid"]).sort_values(["moteid", "timestamp"])
    gaps = valid.groupby("moteid")["timestamp"].diff().dt.total_seconds()
    result.loc[valid.index, "sampling_gap_seconds"] = gaps
    result.loc[valid.index[gaps > cfg.sampling_gap_seconds], "issues"] = result.loc[
        valid.index[gaps > cfg.sampling_gap_seconds], "issues"
    ].map(lambda x: x + ["sampling_gap"])
    original_order = result.dropna(subset=["timestamp", "moteid"])
    backwards = original_order.groupby("moteid")["timestamp"].diff() < pd.Timedelta(0)
    result.loc[original_order.index[backwards], "out_of_order"] = True
    return result
