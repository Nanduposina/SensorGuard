"""Local, human-reviewed alert feedback with stable observation identifiers."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REVIEW_SCHEMA = "sensorguard.alert_reviews"
REVIEW_SCHEMA_VERSION = 1
LABELS = ("confirmed_anomaly", "normal_behavior", "uncertain")
LABEL_TITLES = {
    "confirmed_anomaly": "Confirmed anomaly",
    "normal_behavior": "Normal behavior",
    "uncertain": "Uncertain / needs investigation",
}
IDENTITY_FIELDS = ("timestamp", "sensor_id", "feature", "observed_value", "detector")
REVIEW_FIELDS = (
    "alert_id",
    *IDENTITY_FIELDS,
    "anomaly_score",
    "severity",
    "explanation",
    "label",
    "notes",
    "reviewed_at",
)


class ReviewImportError(ValueError):
    """Raised when a review export does not match the supported schema."""


class ReviewConflictError(ReviewImportError):
    """Raised when an import would replace an existing review without consent."""


def _json_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        stamp = pd.Timestamp(value)
        if pd.isna(stamp):
            return None
        return stamp.isoformat()
    if isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    return str(value)


def alert_identity(alert: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize identity fields used to match the same reviewed observation.

    The score, severity, and explanation are deliberately excluded because they can
    change when detector versions or thresholds change. Exact duplicate source rows
    with identical identity fields cannot be distinguished by current artifacts.
    """
    row = dict(alert)
    timestamp = _json_value(row.get("timestamp"))
    sensor = _json_value(row.get("sensor_id", row.get("moteid")))
    feature = _json_value(row.get("feature"))
    detector_value = row.get("detector_group")
    if detector_value is None or pd.isna(detector_value):
        detector_value = row.get("detector")
    detector = _json_value(detector_value)
    observed = _json_value(row.get("observed_value"))
    if not timestamp or sensor is None or feature is None or detector is None:
        raise ValueError("Alert identity requires timestamp, sensor_id, feature, and detector")
    try:
        parsed_timestamp = pd.Timestamp(timestamp)
        if pd.isna(parsed_timestamp):
            raise ValueError
        timestamp = parsed_timestamp.isoformat()
        sensor_num = float(sensor)
        if not math.isfinite(sensor_num) or not sensor_num.is_integer():
            raise ValueError
        sensor = int(sensor_num)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Alert timestamp and sensor_id must be valid") from exc
    feature = str(feature).strip().lower()
    detector = str(detector).strip().lower()
    if not feature or not detector:
        raise ValueError("Alert feature and detector cannot be blank")
    return {
        "timestamp": timestamp,
        "sensor_id": sensor,
        "feature": feature,
        "observed_value": float(observed) if observed is not None else None,
        "detector": detector,
    }


def stable_alert_id(alert: Mapping[str, Any]) -> str:
    """Return a stable ID for a source reading and detector, independent of score."""
    identity = alert_identity(alert)
    raw = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sg1_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def prepare_review_alerts(alerts: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Keep predicted alert rows, add stable IDs, and remove duplicate identities."""
    if alerts.empty:
        return alerts.copy(), 0
    data = alerts.copy()
    if "predicted" in data:
        predicted = data.predicted.astype("string").str.lower().isin(["true", "1", "yes"])
        data = data[predicted].copy()
    identities = []
    invalid = 0
    for row in data.to_dict(orient="records"):
        try:
            identities.append(stable_alert_id(row))
        except ValueError:
            identities.append(None)
            invalid += 1
    data["alert_id"] = identities
    data = data[data.alert_id.notna()].drop_duplicates("alert_id", keep="first")
    return data.reset_index(drop=True), invalid


def reviewed_alert_summary(alerts: pd.DataFrame, reviews: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize only reviews that exactly match current alert identities."""
    current, invalid = prepare_review_alerts(alerts)
    current_ids = set(current.alert_id) if not current.empty else set()
    matched = [review for review in reviews if review["alert_id"] in current_ids]
    counts = {label: sum(review["label"] == label for review in matched) for label in LABELS}
    binary = counts["confirmed_anomaly"] + counts["normal_behavior"]
    return {
        "current_alerts": len(current),
        "matched_reviews": len(matched),
        "confirmed_anomaly": counts["confirmed_anomaly"],
        "normal_behavior": counts["normal_behavior"],
        "uncertain": counts["uncertain"],
        "binary_reviewed": binary,
        "reviewed_alert_confirmation_rate": (
            counts["confirmed_anomaly"] / binary if binary else None
        ),
        "unmatched_stored_reviews": sum(
            review["alert_id"] not in current_ids for review in reviews
        ),
        "invalid_current_alerts": invalid,
    }


def export_reviews(reviews: list[dict[str, Any]]) -> str:
    """Serialize reviews to a versioned, portable JSON export."""
    payload = {
        "schema": REVIEW_SCHEMA,
        "schema_version": REVIEW_SCHEMA_VERSION,
        "reviews": [{field: review.get(field) for field in REVIEW_FIELDS} for review in reviews],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def validate_review_export(payload: str | bytes) -> list[dict[str, Any]]:
    """Parse and strictly validate a review JSON export before any writes."""
    try:
        decoded = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
        raise ReviewImportError("Review file is not valid UTF-8 JSON") from exc
    if (
        not isinstance(decoded, dict)
        or decoded.get("schema") != REVIEW_SCHEMA
        or decoded.get("schema_version") != REVIEW_SCHEMA_VERSION
        or not isinstance(decoded.get("reviews"), list)
    ):
        raise ReviewImportError("Unsupported review export schema or version")
    reviews: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(decoded["reviews"]):
        if not isinstance(item, dict) or set(item) != set(REVIEW_FIELDS):
            raise ReviewImportError(f"Review record {index + 1} has an invalid field set")
        if item["label"] not in LABELS:
            raise ReviewImportError(f"Review record {index + 1} has an invalid label")
        try:
            identity = alert_identity(item)
            expected = stable_alert_id(item)
        except (TypeError, ValueError) as exc:
            raise ReviewImportError(
                f"Review record {index + 1} has invalid alert identity"
            ) from exc
        if item["alert_id"] != expected:
            raise ReviewImportError(
                f"Review record {index + 1} alert_id does not match its identity"
            )
        if expected in seen:
            raise ReviewImportError(f"Duplicate alert_id in review file: {expected}")
        seen.add(expected)
        notes = item["notes"]
        if notes is not None and not isinstance(notes, str):
            raise ReviewImportError(f"Review record {index + 1} notes must be text or null")
        if not isinstance(item["reviewed_at"], str) or not item["reviewed_at"]:
            raise ReviewImportError(f"Review record {index + 1} needs reviewed_at")
        try:
            reviewed_at = pd.Timestamp(item["reviewed_at"])
        except (TypeError, ValueError) as exc:
            raise ReviewImportError(f"Review record {index + 1} has invalid reviewed_at") from exc
        if pd.isna(reviewed_at):
            raise ReviewImportError(f"Review record {index + 1} has invalid reviewed_at")
        normalized = {field: item.get(field) for field in REVIEW_FIELDS}
        normalized.update(identity)
        for field in ("severity", "explanation"):
            if normalized[field] is not None and not isinstance(normalized[field], str):
                raise ReviewImportError(f"Review record {index + 1} {field} must be text or null")
        if normalized["anomaly_score"] is not None:
            try:
                score = float(normalized["anomaly_score"])
            except (TypeError, ValueError) as exc:
                raise ReviewImportError(
                    f"Review record {index + 1} anomaly_score must be numeric"
                ) from exc
            if not math.isfinite(score):
                raise ReviewImportError(f"Review record {index + 1} anomaly_score must be finite")
            normalized["anomaly_score"] = score
        reviews.append(normalized)
    return reviews


class ReviewStore:
    """SQLite-backed local review store with transactional updates and imports."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS reviews (
                    alert_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    sensor_id INTEGER NOT NULL,
                    feature TEXT NOT NULL,
                    observed_value REAL,
                    detector TEXT NOT NULL,
                    anomaly_score REAL,
                    severity TEXT,
                    explanation TEXT,
                    label TEXT NOT NULL CHECK(label IN ('confirmed_anomaly', 'normal_behavior', 'uncertain')),
                    notes TEXT,
                    reviewed_at TEXT NOT NULL
                )"""
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            connection.execute(
                "INSERT OR IGNORE INTO metadata(key, value) VALUES ('schema_version', ?)",
                (str(REVIEW_SCHEMA_VERSION),),
            )
            version = connection.execute(
                "SELECT value FROM metadata WHERE key='schema_version'"
            ).fetchone()[0]
            if version != str(REVIEW_SCHEMA_VERSION):
                raise ReviewImportError(
                    f"Review store schema version {version} is not supported; export a JSON backup before migration"
                )

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _make_review(alert: Mapping[str, Any], label: str, notes: str = "") -> dict[str, Any]:
        if label not in LABELS:
            raise ValueError(f"label must be one of {LABELS}")
        identity = alert_identity(alert)
        score_value = alert.get("anomaly_score")
        if score_value is None or pd.isna(score_value):
            score_value = alert.get("score")
        score = _json_value(score_value)
        if score is not None:
            try:
                score = float(score)
                if not math.isfinite(score):
                    score = None
            except (TypeError, ValueError):
                score = None
        explanation = alert.get("explanation")
        if explanation is None or pd.isna(explanation):
            explanation = alert.get("reason")
        return {
            "alert_id": stable_alert_id(alert),
            **identity,
            "anomaly_score": score,
            "severity": _json_value(alert.get("severity")),
            "explanation": _json_value(explanation),
            "label": label,
            "notes": str(notes).strip() or None,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _upsert(connection: sqlite3.Connection, review: Mapping[str, Any]) -> None:
        columns = ", ".join(REVIEW_FIELDS)
        placeholders = ", ".join("?" for _ in REVIEW_FIELDS)
        updates = ", ".join(
            f"{field}=excluded.{field}" for field in REVIEW_FIELDS if field != "alert_id"
        )
        values = [review.get(field) for field in REVIEW_FIELDS]
        connection.execute(
            f"INSERT INTO reviews ({columns}) VALUES ({placeholders}) "
            f"ON CONFLICT(alert_id) DO UPDATE SET {updates}",
            values,
        )

    def save_review(self, alert: Mapping[str, Any], label: str, notes: str = "") -> dict[str, Any]:
        """Save or update one explicit reviewer decision."""
        review = self._make_review(alert, label, notes)
        with self._connect() as connection:
            self._upsert(connection, review)
        return review

    def list_reviews(self) -> list[dict[str, Any]]:
        """Return stored reviews ordered by latest review time."""
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT {', '.join(REVIEW_FIELDS)} FROM reviews ORDER BY reviewed_at DESC, alert_id"
            ).fetchall()
        return [dict(row) for row in rows]

    def import_reviews(self, payload: str | bytes, *, overwrite: bool = False) -> dict[str, int]:
        """Import validated records atomically; collisions require explicit overwrite."""
        reviews = validate_review_export(payload)
        with self._connect() as connection:
            existing = {
                row[0] for row in connection.execute("SELECT alert_id FROM reviews").fetchall()
            }
            conflicts = existing.intersection(review["alert_id"] for review in reviews)
            if conflicts and not overwrite:
                raise ReviewConflictError(
                    f"Import contains {len(conflicts)} existing review(s); enable overwrite to replace them"
                )
            for review in reviews:
                self._upsert(connection, review)
        return {"imported": len(reviews) - len(conflicts), "overwritten": len(conflicts)}
