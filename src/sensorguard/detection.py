"""Shared result schema and non-destructive alert merging."""

from __future__ import annotations

import pandas as pd

RESULT_COLUMNS = [
    "timestamp",
    "sensor_id",
    "feature",
    "observed_value",
    "anomaly_score",
    "is_anomaly",
    "detector",
    "reason",
    "severity",
]


def normalize_input(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize legacy moteid column while preserving source metadata."""
    data = frame.copy()
    if "sensor_id" not in data and "moteid" in data:
        data["sensor_id"] = data["moteid"]
    if "timestamp" not in data:
        raise ValueError("frame must contain timestamp")
    if "sensor_id" not in data:
        raise ValueError("frame must contain sensor_id or moteid")
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    return data


def empty_results() -> pd.DataFrame:
    return pd.DataFrame(columns=RESULT_COLUMNS)


def merge_alerts(*results: pd.DataFrame) -> pd.DataFrame:
    """Combine detector evidence for a reading-feature without discarding reasons."""
    valid = [r.copy() for r in results if r is not None and not r.empty]
    if not valid:
        return empty_results()
    combined = pd.concat(valid, ignore_index=True)
    return combined.groupby(
        ["timestamp", "sensor_id", "feature"], dropna=False, as_index=False
    ).agg(
        observed_value=("observed_value", "first"),
        anomaly_score=("anomaly_score", "max"),
        is_anomaly=("is_anomaly", "any"),
        detector=("detector", lambda s: ",".join(sorted(set(s)))),
        reason=("reason", lambda s: "; ".join(dict.fromkeys(str(v) for v in s if v))),
        severity=(
            "severity",
            lambda s: (
                "critical"
                if "critical" in set(s)
                else "high"
                if "high" in set(s)
                else "medium"
                if "medium" in set(s)
                else "low"
            ),
        ),
    )
