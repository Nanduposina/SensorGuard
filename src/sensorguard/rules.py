"""Transparent non-destructive checks and causal cross-sensor context."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SensorGuardConfig
from .detection import RESULT_COLUMNS, empty_results, normalize_input

FEATURES = ("temperature", "humidity", "light", "voltage")


class RuleDetector:
    """Create structured findings; thresholds are screening assumptions."""

    detector_name = "rules"

    def __init__(self, config: SensorGuardConfig | None = None):
        self.config = config or SensorGuardConfig()
        self.sensor_baselines = {}
        self.baseline_scale_floors = {}
        self.sensor_deviation_thresholds = {}

    def fit(self, training: pd.DataFrame, validation: pd.DataFrame | None = None):
        """Fit sensor baselines on training and calibrate deviations on validation.

        Nonzero sensor MAD scales provide training-derived floors for near-constant
        sensors. This avoids assigning effectively infinite robust z-scores when a
        sensor's short training sample happens to be constant. Validation readings
        set one upper-tail score cutoff per feature; they are not treated as labeled
        normal observations and no injected points are used for calibration.
        """
        data = normalize_input(training)
        self.sensor_baselines = {}
        raw_baselines = {}
        for sid, group in data.groupby("sensor_id"):
            for feature in FEATURES:
                if feature not in group:
                    continue
                values = pd.to_numeric(group[feature], errors="coerce").dropna()
                if len(values) >= self.config.sensor_baseline_minimum_rows:
                    median = float(values.median())
                    scale = float(1.4826 * (values - median).abs().median())
                    raw_baselines[(sid, feature)] = (median, scale)
        self.baseline_scale_floors = {}
        for feature in FEATURES:
            observed_scales = [
                scale
                for (sid, name), (_, scale) in raw_baselines.items()
                if name == feature and scale > self.config.sensor_baseline_minimum_scale[feature]
            ]
            self.baseline_scale_floors[feature] = max(
                self.config.sensor_baseline_minimum_scale[feature],
                float(np.median(observed_scales)) if observed_scales else 0.0,
            )
        self.sensor_baselines = {
            key: (median, max(scale, self.baseline_scale_floors[key[1]]))
            for key, (median, scale) in raw_baselines.items()
        }
        self.sensor_deviation_thresholds = {}
        if validation is not None and len(validation):
            if not 0 < self.config.sensor_deviation_validation_quantile < 1:
                raise ValueError("sensor deviation validation quantile must be between 0 and 1")
            val = normalize_input(validation)
            scores = {feature: [] for feature in FEATURES}
            for row in val.itertuples(index=False):
                sid = getattr(row, "sensor_id")
                for feature in FEATURES:
                    baseline = self.sensor_baselines.get((sid, feature))
                    value = getattr(row, feature, np.nan)
                    if baseline is not None and pd.notna(value):
                        center, scale = baseline
                        scores[feature].append(abs(float(value) - center) / scale)
            self.sensor_deviation_thresholds = {
                feature: float(
                    np.nextafter(
                        np.quantile(values, self.config.sensor_deviation_validation_quantile),
                        np.inf,
                    )
                )
                for feature, values in scores.items()
                if values
            }
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        data = normalize_input(frame).reset_index(drop=True)
        found = []

        def add(ts, sid, feature, value, rule, explanation, severity="medium", score=1.0):
            found.append(
                {
                    "timestamp": ts,
                    "sensor_id": sid,
                    "feature": feature,
                    "observed_value": value,
                    "anomaly_score": float(score),
                    "is_anomaly": True,
                    "detector": "rules",
                    "reason": f"{rule}: {explanation}",
                    "severity": severity,
                    "rule_id": rule,
                    "explanation": explanation,
                }
            )

        for i, r in data.iterrows():
            sid, ts = r.sensor_id, r.timestamp
            issues = r.get("issues", [])
            issue_codes = set()
            if isinstance(issues, (list, tuple)):
                for issue in issues:
                    issue_codes.add(str(issue))
                    feature = next((f for f in FEATURES if f in str(issue)), "record")
                    add(
                        ts,
                        sid,
                        feature,
                        r.get(feature, np.nan),
                        str(issue),
                        f"Input validation flagged {issue}.",
                        "high"
                        if str(issue).startswith(("invalid", "malformed", "suspicious"))
                        else "medium",
                    )
            if (
                pd.notna(sid)
                and int(sid) not in self.config.sensor_ids
                and "invalid_sensor_id" not in issue_codes
            ):
                add(
                    ts,
                    sid,
                    "record",
                    np.nan,
                    "invalid_sensor_id",
                    "Sensor ID is outside the configured valid set.",
                    "high",
                )
            for f, bounds in self.config.physical_limits.items():
                value = r.get(f, np.nan)
                if pd.isna(value):
                    has_specific_issue = any(
                        f"{prefix}_{f}" in issue_codes for prefix in ("missing", "invalid")
                    )
                    if not has_specific_issue:
                        add(
                            ts,
                            sid,
                            f,
                            np.nan,
                            "missing_measurement",
                            f"{f} measurement is missing or unparsable.",
                        )
                elif not bounds[0] <= float(value) <= bounds[1]:
                    add(
                        ts,
                        sid,
                        f,
                        value,
                        "physical_limit",
                        f"{f} is outside configured screening bounds {bounds}.",
                        "high",
                    )
                baseline = self.sensor_baselines.get((sid, f))
                if pd.notna(value) and baseline:
                    center, scale = baseline
                    z = abs(float(value) - center) / scale
                    threshold = self.sensor_deviation_thresholds.get(
                        f, self.config.sensor_deviation_z_threshold
                    )
                    if z >= threshold:
                        add(
                            ts,
                            sid,
                            f,
                            value,
                            "sensor_baseline_deviation",
                            f"Deviation from training sensor median {center:.5g} has robust z {z:.5g}, above {threshold:.5g} (validation quantile {self.config.sensor_deviation_validation_quantile:.3g} when available).",
                            "medium",
                            z,
                        )
            if bool(r.get("duplicate", False)):
                add(
                    ts,
                    sid,
                    "record",
                    np.nan,
                    "duplicate_record",
                    "Timestamp and sensor ID duplicate another row.",
                )
            if bool(r.get("out_of_order", False)):
                add(
                    ts,
                    sid,
                    "record",
                    np.nan,
                    "out_of_order",
                    "Timestamp precedes the prior input record for this sensor.",
                )
        duplicates = data.duplicated(["timestamp", "sensor_id"], keep=False)
        for i in np.flatnonzero(duplicates):
            r = data.iloc[i]
            if not any(
                x["rule_id"] == "duplicate_record"
                and x["sensor_id"] == r.sensor_id
                and x["timestamp"] == r.timestamp
                for x in found
            ):
                add(
                    r.timestamp,
                    r.sensor_id,
                    "record",
                    np.nan,
                    "duplicate_record",
                    "Timestamp and sensor ID duplicate another row.",
                )
        valid = data.dropna(subset=["timestamp", "sensor_id"])
        back = valid.groupby("sensor_id", sort=False).timestamp.diff() < pd.Timedelta(0)
        for i in valid.index[back.fillna(False)]:
            r = data.loc[i]
            add(
                r.timestamp,
                r.sensor_id,
                "record",
                np.nan,
                "out_of_order",
                "Timestamp precedes the prior input record for this sensor.",
            )
        chronological = valid.sort_values(["sensor_id", "timestamp"])
        gaps = chronological.groupby("sensor_id").timestamp.diff().dt.total_seconds()
        for i in chronological.index[gaps > self.config.sampling_gap_seconds]:
            r = data.loc[i]
            add(
                r.timestamp,
                r.sensor_id,
                "record",
                gaps.loc[i],
                "sampling_gap",
                "Time since prior observed reading exceeds configured maximum.",
            )
        for sid, group in chronological.groupby("sensor_id", sort=False):
            for f in FEATURES:
                if f not in group:
                    continue
                vals = pd.to_numeric(group[f], errors="coerce")
                changes = vals.diff().abs()
                lim = self.config.abrupt_change_limits[f]
                for idx in group.index[changes > lim]:
                    add(
                        data.loc[idx, "timestamp"],
                        sid,
                        f,
                        data.loc[idx, f],
                        "abrupt_change",
                        f"Absolute change exceeds configured {lim} threshold.",
                        score=changes.loc[idx] / lim,
                    )
                min_obs = self.config.constant_minimum_observations
                tol = self.config.constant_tolerance
                run_start = None
                previous = None
                for pos, value in enumerate(vals.to_numpy(dtype=float)):
                    if not np.isfinite(value):
                        run_start = None
                        previous = None
                        continue
                    if previous is None or abs(value - previous) > tol:
                        run_start = pos
                    previous = value
                    if run_start is not None and pos - run_start + 1 == min_obs:
                        idx = group.index[pos]
                        add(
                            data.loc[idx, "timestamp"],
                            sid,
                            f,
                            data.loc[idx, f],
                            "constant_reading",
                            f"At least {min_obs} consecutive observations are constant within tolerance; emitted once at the run threshold to reduce duplicate alerts.",
                            "low",
                        )
        if not found:
            return empty_results()
        findings = pd.DataFrame(found).drop_duplicates(
            subset=["timestamp", "sensor_id", "feature", "rule_id"]
        )
        # A reading-feature has one alert; all firing rules remain attached as evidence.
        return (
            findings.groupby(["timestamp", "sensor_id", "feature"], dropna=False, as_index=False)
            .agg(
                observed_value=("observed_value", "first"),
                anomaly_score=("anomaly_score", "max"),
                is_anomaly=("is_anomaly", "any"),
                detector=("detector", "first"),
                reason=("reason", lambda s: "; ".join(dict.fromkeys(s))),
                severity=(
                    "severity",
                    lambda s: (
                        "high" if "high" in set(s) else "medium" if "medium" in set(s) else "low"
                    ),
                ),
                rule_id=("rule_id", lambda s: "; ".join(dict.fromkeys(s))),
                explanation=("explanation", lambda s: "; ".join(dict.fromkeys(s))),
            )
            .reset_index(drop=True)
        )


def contextual_deviations(
    frame: pd.DataFrame,
    *,
    tolerance: str = "2min",
    min_peers: int = 3,
    training_baselines: dict | None = None,
    z_threshold: float = 4.0,
) -> pd.DataFrame:
    """Compare baseline-adjusted readings with prior peer readings within tolerance.

    Backward-only time matching prevents future leakage. Baselines are optional
    training-only sensor/feature medians. Context findings are weak evidence.
    """
    data = normalize_input(frame).dropna(subset=["timestamp", "sensor_id"])
    tol = pd.Timedelta(tolerance)
    baselines = training_baselines or {}
    findings = []
    for f in FEATURES:
        if f not in data:
            continue
        work = data[["timestamp", "sensor_id", f]].dropna().copy()
        work["adjusted"] = [
            float(v) - float(baselines.get((sid, f), 0.0))
            for sid, v in zip(work.sensor_id, work[f])
        ]
        for _, row in work.iterrows():
            peers = work[
                (work.sensor_id != row.sensor_id)
                & (work.timestamp <= row.timestamp)
                & (work.timestamp >= row.timestamp - tol)
            ]
            peers = peers.sort_values("timestamp").groupby("sensor_id").tail(1)
            if len(peers) < min_peers:
                continue
            vals = peers.adjusted.to_numpy(float)
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med)))
            scale = max(1.4826 * mad, float(np.std(vals)) * 0.05, 1e-6)
            score = abs(float(row.adjusted) - med) / scale
            if score >= z_threshold:
                findings.append(
                    {
                        "timestamp": row.timestamp,
                        "sensor_id": row.sensor_id,
                        "feature": f,
                        "observed_value": row[f],
                        "anomaly_score": score,
                        "is_anomaly": True,
                        "detector": "context",
                        "reason": f"Deviation from {len(peers)} prior peer sensors (robust z={score:.2f}); local environmental changes may be legitimate.",
                        "severity": "low",
                    }
                )
    return pd.DataFrame(findings, columns=RESULT_COLUMNS) if findings else empty_results()


def fit_contextual_baselines(training: pd.DataFrame, minimum_rows: int = 20) -> dict:
    """Return per-sensor/feature medians fitted on training rows only."""
    data = normalize_input(training)
    baselines = {}
    for sid, group in data.groupby("sensor_id"):
        for feature in FEATURES:
            if feature in group:
                values = pd.to_numeric(group[feature], errors="coerce").dropna()
                if len(values) >= minimum_rows:
                    baselines[(sid, feature)] = float(values.median())
    return baselines
