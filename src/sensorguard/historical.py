"""Exploratory unlabeled historical voltage case studies, never fault labels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .detection import normalize_input


def voltage_case_studies(frame: pd.DataFrame, *, minimum_rows=30) -> pd.DataFrame:
    """Rank voltage patterns after linear temperature adjustment per sensor.

    The regression is exploratory, not causal. Candidates remain unverified and
    can reflect load, calibration, placement, truncation, or temperature model error.
    """
    data = (
        normalize_input(frame)
        .dropna(subset=["timestamp", "sensor_id", "voltage", "temperature"])
        .copy()
    )
    results = []
    for sid, g in data.sort_values("timestamp").groupby("sensor_id"):
        if len(g) < minimum_rows:
            continue
        t = pd.to_numeric(g.temperature, errors="coerce").to_numpy(float)
        v = pd.to_numeric(g.voltage, errors="coerce").to_numpy(float)
        seconds = (g.timestamp - g.timestamp.min()).dt.total_seconds().to_numpy(float)
        if np.std(t) == 0 or np.std(v) == 0 or np.ptp(seconds) == 0:
            continue
        temp_slope, temp_intercept = np.polyfit(t, v, 1)
        adjusted = v - (temp_intercept + temp_slope * t)
        day = seconds / 86400
        residual_slope = float(np.polyfit(day, adjusted, 1)[0])
        corr = float(np.corrcoef(t, v)[0, 1])
        diffs = np.diff(adjusted)
        robust_scale = max(float(np.median(np.abs(diffs - np.median(diffs))) * 1.4826), 1e-9)
        abrupt = float(np.max(np.abs(diffs)) / robust_scale) if len(diffs) else 0.0
        results.append(
            {
                "sensor_id": sid,
                "observations": len(g),
                "start_timestamp": g.timestamp.min(),
                "end_timestamp": g.timestamp.max(),
                "temperature_voltage_correlation": corr,
                "voltage_per_celsius_slope": float(temp_slope),
                "temperature_adjusted_voltage_slope_per_day": residual_slope,
                "voltage_min": float(np.min(v)),
                "voltage_max": float(np.max(v)),
                "largest_adjusted_step_robust_z": abrupt,
                "candidate_decline": bool(residual_slope < -0.005),
                "candidate_abrupt_change": bool(abrupt >= 8.0),
                "interpretation": "Unlabeled candidate only; temperature adjustment is linear and cannot establish battery failure.",
            }
        )
    return (
        pd.DataFrame(results).sort_values(
            [
                "candidate_decline",
                "candidate_abrupt_change",
                "temperature_adjusted_voltage_slope_per_day",
            ],
            ascending=[False, False, True],
        )
        if results
        else pd.DataFrame()
    )
