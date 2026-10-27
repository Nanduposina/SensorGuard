"""Exploratory unlabeled historical voltage case studies, never fault labels."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime

import numpy as np
import pandas as pd

from .detection import normalize_input


def voltage_window_candidates(
    records: Iterable[dict],
    *,
    window_days: int = 7,
    minimum_rows: int = 100,
    decline_threshold_v_per_day: float = -0.005,
) -> pd.DataFrame:
    """Stream records into fixed calendar windows and fit voltage ~ temp + time.

    The fitted time coefficient is a linear temperature-adjusted voltage trend.
    Windows are exploratory candidates only; there are no verified battery-failure
    labels in the source. Memory use is proportional to sensor-window groups,
    not readings. The input may be the streaming records yielded by ``iter_records``.
    """
    if window_days < 1 or minimum_rows < 3:
        raise ValueError("window_days must be positive and minimum_rows at least three")
    stats: dict[tuple[int, date], dict[str, float | int | pd.Timestamp]] = {}
    for row in records:
        timestamp = row.get("timestamp")
        if not isinstance(timestamp, (datetime, pd.Timestamp)):
            timestamp = pd.to_datetime(timestamp, errors="coerce")
        sensor_id = row.get("sensor_id", row.get("moteid"))
        temperature = row.get("temperature")
        voltage = row.get("voltage")
        if not isinstance(temperature, (int, float, np.number)):
            temperature = pd.to_numeric(temperature, errors="coerce")
        if not isinstance(voltage, (int, float, np.number)):
            voltage = pd.to_numeric(voltage, errors="coerce")
        if pd.isna(timestamp) or pd.isna(sensor_id) or pd.isna(temperature) or pd.isna(voltage):
            continue
        sensor_id = int(sensor_id)
        start_ordinal = ((timestamp.date().toordinal() - 1) // window_days) * window_days + 1
        period_start = date.fromordinal(start_ordinal)
        key = (sensor_id, period_start)
        group = stats.setdefault(
            key,
            {
                "n": 0,
                "sum_temp": 0.0,
                "sum_time": 0.0,
                "sum_voltage": 0.0,
                "sum_temp2": 0.0,
                "sum_time2": 0.0,
                "sum_temp_time": 0.0,
                "sum_temp_voltage": 0.0,
                "sum_time_voltage": 0.0,
                "start_timestamp": timestamp,
                "end_timestamp": timestamp,
                "voltage_min": float(voltage),
                "voltage_max": float(voltage),
            },
        )
        period_datetime = datetime.combine(period_start, datetime.min.time())
        time_days = (timestamp - period_datetime).total_seconds() / 86400.0
        t = float(temperature)
        v = float(voltage)
        group["n"] += 1
        group["sum_temp"] += t
        group["sum_time"] += time_days
        group["sum_voltage"] += v
        group["sum_temp2"] += t * t
        group["sum_time2"] += time_days * time_days
        group["sum_temp_time"] += t * time_days
        group["sum_temp_voltage"] += t * v
        group["sum_time_voltage"] += time_days * v
        group["start_timestamp"] = min(group["start_timestamp"], timestamp)
        group["end_timestamp"] = max(group["end_timestamp"], timestamp)
        group["voltage_min"] = min(group["voltage_min"], v)
        group["voltage_max"] = max(group["voltage_max"], v)

    results = []
    for (sensor_id, period_start), values in sorted(stats.items()):
        n = int(values["n"])
        if n < minimum_rows:
            continue
        matrix = np.array(
            [
                [n, values["sum_temp"], values["sum_time"]],
                [values["sum_temp"], values["sum_temp2"], values["sum_temp_time"]],
                [values["sum_time"], values["sum_temp_time"], values["sum_time2"]],
            ],
            dtype=float,
        )
        target = np.array(
            [values["sum_voltage"], values["sum_temp_voltage"], values["sum_time_voltage"]],
            dtype=float,
        )
        try:
            _intercept, temperature_slope, time_slope = np.linalg.solve(matrix, target)
        except np.linalg.LinAlgError:
            continue
        results.append(
            {
                "sensor_id": sensor_id,
                "period_start": period_start.isoformat(),
                "start_timestamp": values["start_timestamp"],
                "end_timestamp": values["end_timestamp"],
                "observations": n,
                "temperature_adjusted_voltage_slope_v_per_day": float(time_slope),
                "voltage_per_celsius_slope": float(temperature_slope),
                "voltage_min": float(values["voltage_min"]),
                "voltage_max": float(values["voltage_max"]),
                "candidate_decline": bool(time_slope < decline_threshold_v_per_day),
                "candidate_threshold_v_per_day": decline_threshold_v_per_day,
                "interpretation": "Unverified voltage-decline candidate; temperature-adjusted linear trend is not causal and is not a battery-failure label.",
            }
        )
    return pd.DataFrame(results)


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
