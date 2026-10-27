"""Reproducible Isolation Forest with training-only preprocessing."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .context import CONTEXT_FEATURES
from .detection import RESULT_COLUMNS, empty_results, normalize_input

FEATURES = ("temperature", "humidity", "light", "voltage")


class IsolationForestDetector:
    """Robust numeric imputation, log-light, sensor baseline and causal deltas."""

    def __init__(
        self,
        *,
        contamination="auto",
        n_estimators=200,
        random_state=42,
        threshold: float | None = None,
        minimum_baseline_rows: int = 20,
        use_contextual_features: bool = False,
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.threshold = threshold
        self.minimum_baseline_rows = minimum_baseline_rows
        self.use_contextual_features = use_contextual_features
        self.model = None
        self.imputer = None
        self.scaler = None
        self.baselines = {}
        self.feature_names = []

    def _features(self, frame, fitting=False):
        data = (
            normalize_input(frame)
            .reset_index(drop=True)
            .sort_values(["timestamp", "sensor_id"], kind="stable")
        )
        cols = [f for f in FEATURES if f in data]
        numeric = (
            data[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        )
        if self.use_contextual_features:
            context_cols = [f for f in CONTEXT_FEATURES if f in data]
            if not context_cols:
                raise ValueError("contextual features were enabled but not supplied to the frame")
            contextual = data[context_cols].apply(pd.to_numeric, errors="coerce")
            numeric = pd.concat([numeric, contextual], axis=1)
        if "light" in numeric:
            numeric["light"] = np.log1p(numeric["light"].clip(lower=0))
        if fitting:
            self.baselines = {}
            for sid, g in data.groupby("sensor_id"):
                if len(g) >= self.minimum_baseline_rows:
                    for f in cols:
                        self.baselines[(sid, f)] = float(
                            pd.to_numeric(g[f], errors="coerce").median()
                        )
        for f in cols:
            baseline = data.sensor_id.map(lambda sid: self.baselines.get((sid, f), np.nan))
            numeric[f + "_baseline_residual"] = (
                pd.to_numeric(data[f], errors="coerce").to_numpy() - baseline.to_numpy()
            )
            # A missing previous observation stays missing and is imputed by training median.
            numeric[f + "_delta"] = pd.to_numeric(
                data.groupby("sensor_id", sort=False)[f].diff(), errors="coerce"
            ).to_numpy()
        ts = data.timestamp
        hour = ts.dt.hour + ts.dt.minute / 60
        numeric["time_sin"] = np.sin(2 * np.pi * hour / 24)
        numeric["time_cos"] = np.cos(2 * np.pi * hour / 24)
        numeric["sensor_id_numeric"] = pd.to_numeric(data.sensor_id, errors="coerce").to_numpy()
        if fitting:
            self.feature_names = list(numeric.columns)
        return data, numeric.reindex(columns=self.feature_names) if not fitting else numeric

    def fit(self, frame: pd.DataFrame):
        data, x = self._features(frame, True)
        if len(x) < 2:
            raise ValueError("Isolation Forest requires at least two training rows")
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(x)
        transformed = self.imputer.transform(x)
        self.scaler = StandardScaler().fit(transformed)
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=1,
        ).fit(self.scaler.transform(transformed))
        return self

    def score_samples(self, frame):
        if self.model is None:
            raise RuntimeError("detector must be fitted")
        data, x = self._features(frame)
        z = self.scaler.transform(self.imputer.transform(x))
        return data, -self.model.score_samples(z)

    def predict(self, frame, threshold=None):
        data, scores = self.score_samples(frame)
        cutoff = threshold if threshold is not None else self.threshold
        if cutoff is None:
            cutoff = float(-self.model.offset_)
        rows = []
        for i, score in enumerate(scores):
            r = data.iloc[i]
            for f in FEATURES:
                if f in data and pd.notna(r[f]):
                    rows.append(
                        {
                            "timestamp": r.timestamp,
                            "sensor_id": r.sensor_id,
                            "feature": f,
                            "observed_value": r[f],
                            "anomaly_score": float(score),
                            "is_anomaly": bool(score >= cutoff),
                            "detector": "isolation_forest",
                            "reason": f"Aggregate multivariate Isolation Forest score {score:.5g} vs threshold {cutoff:.5g}.",
                            "severity": "high" if score >= cutoff else "low",
                        }
                    )
        return pd.DataFrame(rows, columns=RESULT_COLUMNS) if rows else empty_results()

    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        return joblib.load(path)
