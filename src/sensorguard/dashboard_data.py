"""UI-independent data loading and transformations for the Streamlit dashboard."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .ingestion import iter_records

FEATURES = ("temperature", "humidity", "light", "voltage")


def configured_paths() -> dict[str, Path]:
    """Resolve files from environment overrides or repository defaults."""
    root = Path(__file__).resolve().parents[2]
    return {
        "root": root,
        "data": Path(os.getenv("SENSORGUARD_DATA_PATH", root / "data.txt" / "data.txt")),
        "experiment": Path(
            os.getenv("SENSORGUARD_EXPERIMENT_DIR", root / "artifacts" / "experiment")
        ),
        "profile": Path(
            os.getenv("SENSORGUARD_PROFILE_PATH", root / "artifacts" / "data_profile.md")
        ),
        "models": Path(os.getenv("SENSORGUARD_MODEL_DIR", root / "artifacts" / "models")),
        "reviews": Path(
            os.getenv("SENSORGUARD_REVIEW_STORE", root / ".sensorguard" / "reviews.sqlite3")
        ),
    }


def read_csv_optional(path: str | Path) -> pd.DataFrame:
    """Read an optional CSV; missing or empty artifacts become an empty frame."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(p, low_memory=False)
    except (OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame()


def read_json_optional(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def profile_row_count(path: str | Path) -> int | None:
    """Extract prior measured profile count, clearly separate from live scope."""
    p = Path(path)
    if not p.exists():
        return None
    match = re.search(
        r"^- Rows(?: parsed)?: ([\d,]+)$",
        p.read_text(encoding="utf-8", errors="replace"),
        re.MULTILINE,
    )
    return int(match.group(1).replace(",", "")) if match else None


def load_sensor_records(path: str | Path, sensor_id: int, max_records=5000) -> pd.DataFrame:
    """Read bounded records for one sensor from the streaming source."""
    rows = []
    for row in iter_records(path):
        if row.get("moteid") == sensor_id and row.get("timestamp") is not None:
            rows.append(row)
            if len(rows) >= max_records:
                break
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["sensor_id"] = frame["moteid"]
    return frame.sort_values("timestamp", kind="stable").reset_index(drop=True)


def normalize_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize mixed current experiment prediction artifacts and types."""
    data = frame.copy()
    if data.empty:
        return data
    if "timestamp" in data:
        data["timestamp"] = pd.to_datetime(data.timestamp, errors="coerce")
    if "sensor_id" in data:
        data["sensor_id"] = pd.to_numeric(data.sensor_id, errors="coerce")
    if "experiment_detector" in data and "detector" in data:
        data["detector_group"] = data["experiment_detector"].fillna(data["detector"])
    elif "detector" in data:
        data["detector_group"] = data["detector"]
    if "predicted" not in data and "is_anomaly" in data:
        data["predicted"] = data["is_anomaly"]
    if "score" not in data and "anomaly_score" in data:
        data["score"] = data["anomaly_score"]
    for col in ("ground_truth", "predicted", "is_anomaly"):
        if col in data:
            values = (
                data[col]
                .astype("string")
                .str.lower()
                .map(
                    {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}
                )
            )
            data[col] = (
                values.astype("boolean")
                if col == "ground_truth"
                else values.fillna(False).astype(bool)
            )
    for col in ("score", "anomaly_score", "observed_value"):
        if col in data:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    return data


def filter_alerts(
    alerts: pd.DataFrame,
    *,
    sensor_ids=None,
    detectors=None,
    features=None,
    search="",
    start=None,
    end=None,
    include_false=False,
) -> pd.DataFrame:
    """Filter precomputed prediction rows with a searchable time/sensor scope."""
    data = normalize_predictions(alerts)
    if data.empty:
        return data
    if not include_false and "predicted" in data:
        data = data[data.predicted]
    if sensor_ids is not None:
        data = data[data.sensor_id.isin(sensor_ids)]
    if detectors is not None:
        column = "detector_group" if "detector_group" in data else "detector"
        data = data[data[column].isin(detectors)]
    if features is not None:
        data = data[data.feature.isin(features)]
    if start is not None:
        data = data[data.timestamp >= pd.Timestamp(start)]
    if end is not None:
        data = data[data.timestamp <= pd.Timestamp(end)]
    if search:
        searchable = [
            c
            for c in (
                "reason",
                "fault_type",
                "fault_window_type",
                "detector",
                "feature",
                "sensor_id",
            )
            if c in data
        ]
        mask = pd.Series(False, index=data.index)
        for col in searchable:
            mask |= data[col].astype(str).str.contains(search, re.IGNORECASE, na=False)
        data = data[mask]
    return data.sort_values("timestamp", ascending=False, kind="stable").reset_index(drop=True)


def add_observed_values(predictions: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    """Attach selected raw values to alerts using timestamp, sensor and feature."""
    pred = normalize_predictions(predictions)
    if pred.empty or records.empty:
        return pred
    raw = records.copy()
    if "sensor_id" not in raw and "moteid" in raw:
        raw["sensor_id"] = raw.moteid
    raw["timestamp"] = pd.to_datetime(raw.timestamp, errors="coerce")
    values = raw.melt(
        id_vars=["timestamp", "sensor_id"],
        value_vars=[f for f in FEATURES if f in raw],
        var_name="feature",
        value_name="observed_value",
    )
    join = [c for c in ("timestamp", "sensor_id", "feature") if c in pred]
    if "observed_value" in pred:
        pred = pred.drop(columns="observed_value")
    return pred.merge(values, on=join, how="left", validate="many_to_one")


def sensor_summary(records: pd.DataFrame, gaps_seconds=300.0) -> pd.DataFrame:
    """Summarize completeness and observed timestamp gaps per sensor sample."""
    if records.empty:
        return pd.DataFrame()
    data = records.copy()
    if "sensor_id" not in data and "moteid" in data:
        data["sensor_id"] = data.moteid
    data["timestamp"] = pd.to_datetime(data.timestamp, errors="coerce")
    rows = []
    for sid, g in data.sort_values("timestamp").groupby("sensor_id"):
        total = len(g)
        missing = sum(int(g[f].isna().sum()) for f in FEATURES if f in g)
        cells = total * sum(f in g for f in FEATURES)
        deltas = g.timestamp.diff().dt.total_seconds().dropna()
        rows.append(
            {
                "sensor_id": sid,
                "records_in_sample": total,
                "measurement_completeness": 1 - missing / cells if cells else None,
                "missing_measurement_cells": missing,
                "gaps_over_threshold": int((deltas > gaps_seconds).sum()),
                "largest_gap_seconds": float(deltas.max()) if not deltas.empty else None,
                "first_timestamp": g.timestamp.min(),
                "last_timestamp": g.timestamp.max(),
                "voltage_start": pd.to_numeric(g.voltage, errors="coerce").dropna().iloc[0]
                if "voltage" in g and g.voltage.notna().any()
                else None,
                "voltage_end": pd.to_numeric(g.voltage, errors="coerce").dropna().iloc[-1]
                if "voltage" in g and g.voltage.notna().any()
                else None,
            }
        )
    return pd.DataFrame(rows)


def align_context(
    sensor: pd.DataFrame, peers: pd.DataFrame, feature: str, tolerance="2min"
) -> pd.DataFrame:
    """Causally align peer sensor readings using backward as-of joins."""
    if sensor.empty or peers.empty:
        return pd.DataFrame(
            columns=["timestamp", "sensor_value", "reference_value", "peer_sensor_id"]
        )
    left = (
        sensor[["timestamp", feature]]
        .dropna()
        .copy()
        .sort_values("timestamp")
        .rename(columns={feature: "sensor_value"})
    )
    right = (
        peers[["timestamp", "sensor_id", feature]]
        .dropna()
        .copy()
        .sort_values("timestamp")
        .rename(columns={feature: "reference_value", "sensor_id": "peer_sensor_id"})
    )
    return pd.merge_asof(
        left, right, on="timestamp", direction="backward", tolerance=pd.Timedelta(tolerance)
    )


def confusion_counts(rows: pd.DataFrame, detector: str) -> dict[str, int]:
    """Return counts for explicitly labeled pairs only for one detector."""
    data = normalize_predictions(rows)
    if data.empty:
        return {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    column = "detector_group" if "detector_group" in data else "detector"
    data = data[data[column] == detector]
    if not {"ground_truth", "predicted"}.issubset(data):
        return {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    data = data[data.ground_truth.notna()]
    truth = data.ground_truth.fillna(False)
    pred = data.predicted
    return {
        "tn": int((~truth & ~pred).sum()),
        "fp": int((~truth & pred).sum()),
        "fn": int((truth & ~pred).sum()),
        "tp": int((truth & pred).sum()),
    }
