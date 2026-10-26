"""Data quality summaries."""

import pandas as pd


def quality_summary(frame: pd.DataFrame) -> dict:
    """Build per-feature and per-sensor summaries from validated rows."""
    features = [c for c in ("temperature", "humidity", "light", "voltage") if c in frame]
    feature = {
        c: {
            "missing": int(frame[c].isna().sum()),
            "minimum": _num(frame[c].min()),
            "maximum": _num(frame[c].max()),
            "mean": _num(frame[c].mean()),
        }
        for c in features
    }
    sensor = {}
    if "moteid" in frame:
        for sid, group in frame.groupby("moteid", dropna=False):
            sensor[str(sid)] = {
                "rows": len(group),
                "flagged_rows": int(group["issues"].map(bool).sum()),
                "first_timestamp": str(group["timestamp"].min()),
                "last_timestamp": str(group["timestamp"].max()),
            }
    return {
        "rows": len(frame),
        "feature_summary": feature,
        "sensor_summary": sensor,
        "issue_counts": {str(k): int(v) for k, v in _issue_counts(frame).items()},
    }


def _num(value):
    return None if pd.isna(value) else float(value)


def _issue_counts(frame):
    return frame["issues"].explode().dropna().value_counts() if "issues" in frame else {}
