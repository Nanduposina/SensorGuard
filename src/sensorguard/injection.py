"""Seeded synthetic fault injection with immutable source and audit mapping."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .detection import normalize_input
from .isolation import FEATURES

FAULT_TYPES = (
    "spike",
    "drop",
    "level_shift",
    "gradual_drift",
    "stuck_at",
    "missing_block",
    "sensor_outage",
    "random_noise",
    "increased_variance",
    "physical_value",
    "voltage_decline",
    "timestamp_delay",
    "irregular_sampling",
)


@dataclass(frozen=True)
class FaultSpec:
    """One fault request. start_index indexes chronological readings of that sensor."""

    fault_type: str
    sensor_id: int | None = None
    feature: str | None = None
    start_index: int | None = None
    duration: int = 1
    magnitude: float | None = None
    severity: str = "moderate"
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class InjectionResult:
    frame: pd.DataFrame
    labels: pd.DataFrame
    fault_manifest: pd.DataFrame
    observations: pd.DataFrame


def inject_faults(
    frame: pd.DataFrame, specs: list[FaultSpec], *, seed: int = 42
) -> InjectionResult:
    """Return a modified copy, feature-reading labels, manifest and original-value audit.

    Call only on the held-out test partition. `frame` is copied before changes.
    Labels mark directly modified feature-reading pairs; timestamp faults use feature
    `record`. Neighboring unaffected observations remain explicitly negative.
    """
    source = normalize_input(frame).reset_index(drop=True)
    derived = source.copy(deep=True)
    rng = np.random.default_rng(seed)
    label_rows = []
    manifest = []
    observations = []
    for ordinal, spec in enumerate(specs):
        if spec.fault_type not in FAULT_TYPES:
            raise ValueError(f"unsupported fault type: {spec.fault_type}")
        if spec.duration < 1:
            raise ValueError("duration must be positive")
        sensors = sorted(derived.sensor_id.dropna().unique().tolist())
        sid = spec.sensor_id if spec.sensor_id is not None else int(rng.choice(sensors))
        if sid not in sensors:
            raise ValueError(f"sensor {sid} not present")
        group = derived.loc[derived.sensor_id == sid].sort_values("timestamp", kind="stable")
        start = (
            spec.start_index
            if spec.start_index is not None
            else int(rng.integers(0, max(1, len(group) - spec.duration + 1)))
        )
        if start < 0 or start + spec.duration > len(group):
            raise ValueError("fault span is outside sensor readings")
        affected = group.iloc[start : start + spec.duration]
        fault_id = f"F{ordinal + 1:04d}-{uuid.uuid5(uuid.NAMESPACE_OID, f'{seed}:{ordinal}:{sid}:{start}:{spec.fault_type}').hex[:8]}"
        f = spec.feature or ("voltage" if spec.fault_type == "voltage_decline" else "temperature")
        if (
            spec.fault_type not in {"sensor_outage", "timestamp_delay", "irregular_sampling"}
            and f not in FEATURES
        ):
            raise ValueError(f"invalid feature: {f}")
        vals = (
            pd.to_numeric(affected[f], errors="coerce").to_numpy(float)
            if f in FEATURES
            else np.full(len(affected), np.nan)
        )
        magnitude = (
            spec.magnitude
            if spec.magnitude is not None
            else float(spec.parameters.get("magnitude", 10.0))
        )
        modifications = {}
        if spec.fault_type in {"spike", "drop"}:
            amount = abs(magnitude)
            changes = np.full(len(vals), amount if spec.fault_type == "spike" else -amount)
            modifications[f] = vals + changes
        elif spec.fault_type == "level_shift":
            modifications[f] = vals + magnitude
        elif spec.fault_type in {"gradual_drift", "voltage_decline"}:
            delta = abs(magnitude) if spec.fault_type == "gradual_drift" else -abs(magnitude)
            # Every labeled point must be modified, including a one-point drift.
            modifications[f] = vals + np.linspace(
                delta / spec.duration, delta, spec.duration, endpoint=True
            )
        elif spec.fault_type == "stuck_at":
            fallback = vals[0] + (magnitude if magnitude else 1.0) if np.isfinite(vals[0]) else 1.0
            target = float(spec.parameters.get("value", fallback))
            modifications[f] = np.full(len(vals), target)
        elif spec.fault_type == "missing_block":
            modifications[f] = np.full(len(vals), np.nan)
        elif spec.fault_type == "sensor_outage":
            modifications = {col: np.full(len(affected), np.nan) for col in FEATURES}
        elif spec.fault_type in {"random_noise", "increased_variance"}:
            sigma = abs(magnitude)
            if spec.fault_type == "increased_variance":
                sigma *= float(spec.parameters.get("multiplier", 5.0))
            modifications[f] = vals + rng.normal(0, sigma, size=len(vals))
        elif spec.fault_type == "physical_value":
            default = (
                200.0
                if f == "temperature"
                else 150.0
                if f == "humidity"
                else -10.0
                if f in {"light", "voltage"}
                else magnitude
            )
            modifications[f] = np.full(len(vals), float(spec.parameters.get("value", default)))
        elif spec.fault_type in {"timestamp_delay", "irregular_sampling"}:
            seconds = float(
                spec.parameters.get("seconds", magnitude if spec.magnitude is not None else 60.0)
            )
            offsets = (
                np.full(len(affected), seconds)
                if spec.fault_type == "timestamp_delay"
                else np.arange(1, len(affected) + 1) * seconds
            )
            modifications["timestamp"] = (
                pd.to_datetime(affected.timestamp).to_numpy()
                + pd.to_timedelta(offsets, unit="s").to_numpy()
            )
        start_ts = affected.timestamp.iloc[0]
        end_ts = affected.timestamp.iloc[-1]
        modified_features = list(modifications)
        affected_start = len(observations)
        affected_label_start = len(label_rows)
        for col, newvalues in modifications.items():
            for pos, (idx, newval) in enumerate(zip(affected.index, newvalues)):
                old = derived.at[idx, col]
                if (pd.isna(old) and pd.isna(newval)) or old == newval:
                    continue
                derived.at[idx, col] = newval
                label_rows.append(
                    {
                        "timestamp": derived.at[idx, "timestamp"],
                        "sensor_id": sid,
                        "feature": col if col in FEATURES else "record",
                        "is_fault": True,
                        "fault_id": fault_id,
                        "fault_type": spec.fault_type,
                    }
                )
                observations.append(
                    {
                        "fault_id": fault_id,
                        "fault_type": spec.fault_type,
                        "sensor_id": sid,
                        "timestamp_original": source.at[idx, "timestamp"],
                        "timestamp_modified": derived.at[idx, "timestamp"],
                        "feature": col,
                        "original_value": old,
                        "modified_value": newval,
                        "row_index": int(idx),
                        "affected": True,
                    }
                )
        if len(label_rows) == affected_label_start:
            raise ValueError(f"fault specification {spec.fault_type!r} made no data changes")
        manifest.append(
            {
                "fault_id": fault_id,
                "fault_type": spec.fault_type,
                "sensor_id": sid,
                "features": ";".join(modified_features),
                "start_index": start,
                "end_index": start + spec.duration - 1,
                "start_timestamp": start_ts,
                "end_timestamp": end_ts,
                "severity": spec.severity,
                "magnitude": magnitude,
                "parameters": dict(spec.parameters),
                "random_seed": seed,
                "affected_observations": len(observations) - affected_start,
            }
        )
    labels = pd.DataFrame(
        label_rows,
        columns=["timestamp", "sensor_id", "feature", "is_fault", "fault_id", "fault_type"],
    )
    if not labels.empty:
        labels = labels.groupby(["timestamp", "sensor_id", "feature"], as_index=False).agg(
            is_fault=("is_fault", "any"),
            fault_id=("fault_id", lambda s: ";".join(dict.fromkeys(s))),
            fault_type=("fault_type", lambda s: ";".join(dict.fromkeys(s))),
        )
    return InjectionResult(derived, labels, pd.DataFrame(manifest), pd.DataFrame(observations))
