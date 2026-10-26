import json

import pandas as pd
import pytest

from sensorguard.review_store import (
    LABELS,
    REVIEW_FIELDS,
    REVIEW_SCHEMA,
    REVIEW_SCHEMA_VERSION,
    ReviewConflictError,
    ReviewImportError,
    ReviewStore,
    export_reviews,
    prepare_review_alerts,
    reviewed_alert_summary,
    stable_alert_id,
)


def alert(**updates):
    row = {
        "timestamp": pd.Timestamp("2004-02-28 01:02:03"),
        "sensor_id": 12,
        "feature": "temperature",
        "observed_value": 21.25,
        "detector": "rules",
        "anomaly_score": 8.2,
        "severity": "medium",
        "reason": "sensor_baseline_deviation",
        "predicted": True,
    }
    row.update(updates)
    return row


def test_stable_alert_id_tracks_observation_but_not_volatile_score_or_reason():
    original = alert()
    changed_detection = alert(anomaly_score=19.0, reason="new explanation")
    assert stable_alert_id(original) == stable_alert_id(changed_detection)
    assert stable_alert_id(original) != stable_alert_id(alert(observed_value=21.5))
    assert stable_alert_id(original) != stable_alert_id(alert(detector="isolation_forest"))


def test_review_persistence_and_update(tmp_path):
    store = ReviewStore(tmp_path / "local" / "reviews.sqlite3")
    assert store.list_reviews() == []
    first = store.save_review(alert(), "confirmed_anomaly", "Verified against maintenance note.")
    assert first["alert_id"] == stable_alert_id(alert())
    assert first["notes"] == "Verified against maintenance note."
    assert first["observed_value"] == pytest.approx(21.25)
    assert first["explanation"] == "sensor_baseline_deviation"

    updated = store.save_review(alert(anomaly_score=10.0), "uncertain", "Need a second reviewer.")
    stored = store.list_reviews()
    assert len(stored) == 1
    assert stored[0]["label"] == "uncertain"
    assert stored[0]["notes"] == "Need a second reviewer."
    assert updated["alert_id"] == first["alert_id"]


def test_import_export_round_trip_and_conflict_requires_explicit_overwrite(tmp_path):
    source = ReviewStore(tmp_path / "source.sqlite3")
    review = source.save_review(alert(), "normal_behavior", "Expected warm room reading.")
    exported = export_reviews(source.list_reviews())
    target = ReviewStore(tmp_path / "target.sqlite3")
    assert target.import_reviews(exported) == {"imported": 1, "overwritten": 0}
    assert target.list_reviews()[0]["alert_id"] == review["alert_id"]

    replacement = source.save_review(alert(), "confirmed_anomaly", "Independent evidence found.")
    new_review = source.save_review(
        alert(timestamp="2004-02-28 01:03:03", feature="voltage"), "normal_behavior"
    )
    with pytest.raises(ReviewConflictError):
        target.import_reviews(export_reviews([replacement, new_review]))
    assert len(target.list_reviews()) == 1
    assert target.list_reviews()[0]["label"] == "normal_behavior"
    assert target.import_reviews(export_reviews([replacement, new_review]), overwrite=True) == {
        "imported": 1,
        "overwritten": 1,
    }
    assert {r["label"] for r in target.list_reviews()} == {"confirmed_anomaly", "normal_behavior"}


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        json.dumps({"schema": "other", "schema_version": 1, "reviews": []}),
        json.dumps({"schema": REVIEW_SCHEMA, "schema_version": 2, "reviews": []}),
        json.dumps({"schema": REVIEW_SCHEMA, "schema_version": 1, "reviews": [{"label": "bad"}]}),
    ],
)
def test_invalid_import_schema_is_rejected_without_writes(tmp_path, payload):
    store = ReviewStore(tmp_path / "reviews.sqlite3")
    with pytest.raises(ReviewImportError):
        store.import_reviews(payload)
    assert store.list_reviews() == []


def test_import_rejects_tampered_identity_and_duplicate_records(tmp_path):
    store = ReviewStore(tmp_path / "reviews.sqlite3")
    review = store._make_review(alert(), "confirmed_anomaly")
    forged = dict(review, observed_value=99.0)
    payload = {
        "schema": REVIEW_SCHEMA,
        "schema_version": REVIEW_SCHEMA_VERSION,
        "reviews": [forged],
    }
    with pytest.raises(ReviewImportError, match="alert_id"):
        store.import_reviews(json.dumps(payload))

    duplicate_payload = {
        "schema": REVIEW_SCHEMA,
        "schema_version": REVIEW_SCHEMA_VERSION,
        "reviews": [review, review],
    }
    with pytest.raises(ReviewImportError, match="Duplicate"):
        store.import_reviews(json.dumps(duplicate_payload))
    assert store.list_reviews() == []


def test_import_rejects_missing_identity_fields_as_schema_error(tmp_path):
    store = ReviewStore(tmp_path / "reviews.sqlite3")
    review = store._make_review(alert(), "confirmed_anomaly")
    del review["sensor_id"]
    payload = {
        "schema": REVIEW_SCHEMA,
        "schema_version": REVIEW_SCHEMA_VERSION,
        "reviews": [review],
    }
    with pytest.raises(ReviewImportError, match="field set"):
        store.import_reviews(json.dumps(payload))


def test_duplicate_alert_rows_are_deduplicated_and_bad_rows_reported():
    rows = pd.DataFrame([alert(), alert(anomaly_score=99), {"predicted": True}])
    prepared, invalid = prepare_review_alerts(rows)
    assert len(prepared) == 1
    assert invalid == 1


def test_review_summary_matches_ids_excludes_uncertain_from_binary_rate(tmp_path):
    rows = pd.DataFrame([alert(), alert(detector="isolation_forest")])
    prepared, _ = prepare_review_alerts(rows)
    store = ReviewStore(tmp_path / "reviews.sqlite3")
    store.save_review(prepared.iloc[0].to_dict(), "confirmed_anomaly")
    store.save_review(prepared.iloc[1].to_dict(), "uncertain")
    store.save_review(alert(timestamp="2004-03-01 00:00:00"), "normal_behavior")

    summary = reviewed_alert_summary(rows, store.list_reviews())
    assert summary["current_alerts"] == 2
    assert summary["matched_reviews"] == 2
    assert summary["confirmed_anomaly"] == 1
    assert summary["normal_behavior"] == 0
    assert summary["uncertain"] == 1
    assert summary["binary_reviewed"] == 1
    assert summary["reviewed_alert_confirmation_rate"] == 1.0
    assert summary["unmatched_stored_reviews"] == 1
    assert set(LABELS) == {"confirmed_anomaly", "normal_behavior", "uncertain"}
    assert len(REVIEW_FIELDS) == len(set(REVIEW_FIELDS))
