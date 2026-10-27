"""Causal statistical peer features for asynchronous sensor observations."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .coordinates import physical_neighborhoods
from .detection import normalize_input

FEATURES = ("temperature", "humidity", "light", "voltage")

CONTEXT_FEATURES = tuple(
    name
    for feature in FEATURES
    for name in (
        f"{feature}_peer_delta",
        f"{feature}_peer_robust_z",
        f"{feature}_peer_count",
        f"{feature}_peer_age_fraction",
    )
)


def peer_feature_columns() -> tuple[str, ...]:
    """Return the ordered feature columns added by :func:`add_peer_features`."""
    return CONTEXT_FEATURES


def fit_peer_baselines(training: pd.DataFrame, minimum_rows: int = 20) -> dict:
    """Fit sensor/feature medians on training rows only for level adjustment."""
    if minimum_rows < 1:
        raise ValueError("minimum_rows must be positive")
    data = normalize_input(training)
    baselines = {}
    for sensor_id, group in data.groupby("sensor_id", dropna=True):
        for feature in FEATURES:
            if feature not in group:
                continue
            values = pd.to_numeric(group[feature], errors="coerce").dropna()
            if len(values) >= minimum_rows:
                baselines[(sensor_id, feature)] = float(values.median())
    return baselines


def add_peer_features(
    targets: pd.DataFrame,
    references: pd.DataFrame,
    *,
    baselines: Mapping[tuple[object, str], float],
    tolerance: str | pd.Timedelta = "2min",
    minimum_peers: int = 2,
    neighborhoods: Mapping[int, tuple[int, ...] | list[int]] | None = None,
    radius_m: float = 8.0,
) -> pd.DataFrame:
    """Add backward-aligned peer context features without target self-reference.

    The default peer graph uses the official mote coordinates and an 8-meter
    Euclidean radius. An explicit ``neighborhoods`` mapping can override it.
    IDs without coordinates have no default neighbors. For each target row, each
    allowed physical peer contributes its latest finite reading
    at or before the target timestamp and within ``tolerance``. Per-sensor medians
    must be fitted on the training partition and are subtracted before the peer
    reference is formed, so sensors with different operating levels are not
    compared as raw values. Same-time peer values are allowed; future readings,
    the target sensor, duplicate timestamps within one peer, and stale readings
    are excluded. Context values remain missing when the target baseline or the
    minimum peer count is unavailable; companion count/age features make coverage
    explicit for models.
    """
    if minimum_peers < 1:
        raise ValueError("minimum_peers must be positive")
    tolerance_delta = pd.Timedelta(tolerance)
    if tolerance_delta <= pd.Timedelta(0):
        raise ValueError("tolerance must be positive")
    allowed_neighbors = (
        neighborhoods if neighborhoods is not None else physical_neighborhoods(radius_m=radius_m)
    )
    target = normalize_input(targets).reset_index(drop=True)
    refs = normalize_input(references).dropna(subset=["timestamp", "sensor_id"]).copy()
    refs = refs.sort_values(["sensor_id", "timestamp"], kind="stable")
    arrays: dict[str, dict[object, tuple[np.ndarray, np.ndarray]]] = {}
    for feature in FEATURES:
        arrays[feature] = {}
        if feature not in refs:
            continue
        valid = refs.dropna(subset=[feature])
        for sensor_id, group in valid.groupby("sensor_id", sort=False):
            group = group.drop_duplicates("timestamp", keep="last")
            timestamps = group.timestamp.to_numpy(dtype="datetime64[ns]").astype("int64")
            values = pd.to_numeric(group[feature], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(values)
            arrays[feature][sensor_id] = (timestamps[finite], values[finite])

    tol_ns = int(tolerance_delta.value)
    outputs = {name: np.full(len(target), np.nan, dtype=float) for name in CONTEXT_FEATURES}
    for position, row in target.iterrows():
        timestamp = row.timestamp
        sensor_id = row.sensor_id
        if pd.isna(timestamp) or pd.isna(sensor_id):
            continue
        timestamp_ns = int(pd.Timestamp(timestamp).value)
        lower_ns = timestamp_ns - tol_ns
        allowed = set(allowed_neighbors.get(int(sensor_id), ()))
        for feature in FEATURES:
            if feature not in target or feature not in arrays:
                continue
            candidates = []
            ages = []
            for peer_id, (peer_times, peer_values) in arrays[feature].items():
                if peer_id == sensor_id or int(peer_id) not in allowed or not len(peer_times):
                    continue
                latest = int(np.searchsorted(peer_times, timestamp_ns, side="right") - 1)
                if latest < 0 or int(peer_times[latest]) < lower_ns:
                    continue
                baseline = baselines.get((peer_id, feature))
                if baseline is None or not np.isfinite(float(baseline)):
                    continue
                candidates.append(float(peer_values[latest]) - float(baseline))
                ages.append((timestamp_ns - int(peer_times[latest])) / tol_ns)

            count_name = f"{feature}_peer_count"
            age_name = f"{feature}_peer_age_fraction"
            delta_name = f"{feature}_peer_delta"
            z_name = f"{feature}_peer_robust_z"
            outputs[count_name][position] = len(candidates)
            if ages:
                outputs[age_name][position] = float(np.median(ages))
            target_baseline = baselines.get((sensor_id, feature))
            observed = pd.to_numeric(pd.Series([row.get(feature)]), errors="coerce").iloc[0]
            if len(candidates) < minimum_peers or target_baseline is None or pd.isna(observed):
                continue
            peer_median = float(np.median(candidates))
            target_residual = float(observed) - float(target_baseline)
            peer_delta = target_residual - peer_median
            mad = float(np.median(np.abs(np.asarray(candidates) - peer_median)))
            spread = max(1.4826 * mad, float(np.std(candidates)) * 0.05, 1e-6)
            outputs[delta_name][position] = peer_delta
            outputs[z_name][position] = abs(peer_delta) / spread

    result = target.copy()
    for name, values in outputs.items():
        result[name] = values
    return result
