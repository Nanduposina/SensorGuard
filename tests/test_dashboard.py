from pathlib import Path

import pandas as pd

from sensorguard.dashboard_data import (
    align_context,
    confusion_counts,
    filter_alerts,
    normalize_predictions,
    profile_row_count,
    sensor_summary,
)


def test_filter_and_normalize_predictions():
    rows = pd.DataFrame(
        {
            "timestamp": ["2004-02-28T00:00:00", "2004-02-28T00:01:00"],
            "sensor_id": [1, 2],
            "feature": ["temperature", "voltage"],
            "detector": ["rules", "if"],
            "predicted": ["True", "False"],
            "score": ["2.5", "0.1"],
            "reason": ["spike", "normal"],
        }
    )
    result = filter_alerts(rows, detectors=["rules"], search="spike")
    assert len(result) == 1 and result.iloc[0].sensor_id == 1
    assert result.iloc[0].predicted and result.iloc[0].score == 2.5


def test_sensor_summary_missing_and_gap_transforms():
    rows = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2004-01-01 00:00:00", "2004-01-01 00:10:00"]),
            "moteid": [1, 1],
            "temperature": [20, None],
            "humidity": [40, 41],
            "light": [5, 6],
            "voltage": [2.7, 2.6],
        }
    )
    summary = sensor_summary(rows, gaps_seconds=300).iloc[0]
    assert summary.records_in_sample == 2 and summary.missing_measurement_cells == 1
    assert summary.gaps_over_threshold == 1 and summary.largest_gap_seconds == 600


def test_context_alignment_is_backward_only():
    sensor = pd.DataFrame(
        {"timestamp": pd.to_datetime(["2004-01-01 00:01:00"]), "temperature": [25.0]}
    )
    peers = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2004-01-01 00:00:30", "2004-01-01 00:01:30"]),
            "sensor_id": [2, 2],
            "temperature": [20.0, 99.0],
        }
    )
    aligned = align_context(sensor, peers, "temperature", "2min")
    assert aligned.reference_value.iloc[0] == 20


def test_confusion_counts():
    rows = pd.DataFrame(
        {
            "detector": ["rules"] * 4,
            "ground_truth": [False, False, True, True],
            "predicted": [False, True, False, True],
        }
    )
    assert confusion_counts(rows, "rules") == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}


def test_dashboard_preserves_unknown_ground_truth_and_excludes_it_from_counts():
    rows = normalize_predictions(
        pd.DataFrame(
            {
                "detector": ["rules"] * 3,
                "ground_truth": [True, None, False],
                "predicted": [True, True, False],
            }
        )
    )
    assert rows.ground_truth.isna().sum() == 1
    assert rows.ground_truth.dtype.name == "boolean"
    assert confusion_counts(rows, "rules") == {"tn": 1, "fp": 0, "fn": 0, "tp": 1}


def test_profile_row_count_accepts_current_and_legacy_report_labels(tmp_path):
    path = tmp_path / "profile.md"
    path.write_text("- Rows parsed: 2,313,682\n", encoding="utf-8")
    assert profile_row_count(path) == 2_313_682
    path.write_text("- Rows: 12,345\n", encoding="utf-8")
    assert profile_row_count(path) == 12_345


def test_streamlit_smoke(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    experiment_dir = tmp_path / "experiment"
    experiment_dir.mkdir()
    pd.DataFrame(
        {
            "timestamp": ["2004-02-28 00:00:00"],
            "sensor_id": [1],
            "feature": ["temperature"],
            "observed_value": [21.0],
            "score": [0.9],
            "predicted": [True],
            "detector": ["rules"],
            "reason": ["fixture alert"],
            "severity": ["low"],
        }
    ).to_csv(experiment_dir / "clean_test_prediction_details.csv", index=False)
    monkeypatch.setenv("SENSORGUARD_EXPERIMENT_DIR", str(experiment_dir))
    monkeypatch.setenv("SENSORGUARD_DATA_PATH", str(tmp_path / "missing.txt"))
    monkeypatch.setenv("SENSORGUARD_PROFILE_PATH", str(tmp_path / "missing-profile.md"))
    monkeypatch.setenv("SENSORGUARD_MODEL_DIR", str(tmp_path / "missing-models"))
    monkeypatch.setenv("SENSORGUARD_REVIEW_STORE", str(tmp_path / "reviews.sqlite3"))

    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "app.py", default_timeout=20
    ).run()
    assert not app.exception
    assert any("SensorGuard" in title.value for title in app.title)
    for page in [
        "Overview",
        "Sensor explorer",
        "Alert investigation",
        "Alert review",
        "Model evaluation",
        "Sensor health",
    ]:
        app.radio(key="workspace_navigation").set_value(page).run()
        assert not app.exception, f"Dashboard page {page!r} raised an exception"
        if page == "Alert investigation":
            download = app.get("download_button")[0]
            assert download.label == "Export filtered alerts as CSV"
            assert download.proto.url.endswith(".csv")
        if page == "Alert review":
            assert app.get("download_button")[-1].label == "Download review backup (JSON)"


def test_streamlit_empty_artifact_smoke(tmp_path, monkeypatch):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("SENSORGUARD_DATA_PATH", str(tmp_path / "missing.txt"))
    monkeypatch.setenv("SENSORGUARD_EXPERIMENT_DIR", str(tmp_path / "no-results"))
    monkeypatch.setenv("SENSORGUARD_PROFILE_PATH", str(tmp_path / "no-profile.md"))
    monkeypatch.setenv("SENSORGUARD_MODEL_DIR", str(tmp_path / "no-models"))
    monkeypatch.setenv("SENSORGUARD_REVIEW_STORE", str(tmp_path / "no-reviews.sqlite3"))
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
    assert not app.exception
    for page in [
        "Overview",
        "Sensor explorer",
        "Alert investigation",
        "Alert review",
        "Model evaluation",
        "Sensor health",
    ]:
        app.radio(key="workspace_navigation").set_value(page).run()
        assert not app.exception, f"Empty-state page {page!r} raised an exception"


def test_alert_review_streamlit_saves_and_updates_decision(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    experiment_dir = tmp_path / "experiment"
    experiment_dir.mkdir()
    pd.DataFrame(
        {
            "timestamp": ["2004-02-28 00:00:00", "2004-02-28 00:00:00"],
            "sensor_id": [1, 1],
            "feature": ["temperature", "temperature"],
            "observed_value": [21.0, 21.0],
            "score": [0.9, 0.9],
            "predicted": [True, True],
            "detector": ["rules", "rules"],
            "reason": ["fixture alert", "fixture alert"],
            "severity": ["medium", "medium"],
        }
    ).to_csv(experiment_dir / "clean_test_prediction_details.csv", index=False)
    review_path = tmp_path / "feedback" / "reviews.sqlite3"
    monkeypatch.setenv("SENSORGUARD_EXPERIMENT_DIR", str(experiment_dir))
    monkeypatch.setenv("SENSORGUARD_DATA_PATH", str(tmp_path / "missing.txt"))
    monkeypatch.setenv("SENSORGUARD_PROFILE_PATH", str(tmp_path / "missing-profile.md"))
    monkeypatch.setenv("SENSORGUARD_MODEL_DIR", str(tmp_path / "missing-models"))
    monkeypatch.setenv("SENSORGUARD_REVIEW_STORE", str(review_path))

    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
    app.radio(key="workspace_navigation").set_value("Alert review").run()
    assert not app.exception
    assert "1 matching reviewable alert" in " ".join(item.value for item in app.caption)
    app.radio(key="review_decision").set_value("confirmed_anomaly").run()
    app.text_area(key="review_notes").set_value("Checked against lab notes.")
    app.button(key="save_review").click().run()
    assert not app.exception

    from sensorguard.review_store import ReviewStore

    stored = ReviewStore(review_path).list_reviews()
    assert len(stored) == 1
    assert stored[0]["label"] == "confirmed_anomaly"
    assert stored[0]["notes"] == "Checked against lab notes."

    app.radio(key="review_decision").set_value("normal_behavior").run()
    app.button(key="save_review").click().run()
    assert not app.exception
    stored = ReviewStore(review_path).list_reviews()
    assert len(stored) == 1
    assert stored[0]["label"] == "normal_behavior"


def test_model_evaluation_shows_matched_human_review_summary(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from sensorguard.review_store import ReviewStore

    experiment_dir = tmp_path / "experiment"
    experiment_dir.mkdir()
    alert = {
        "timestamp": "2004-02-28 00:00:00",
        "sensor_id": 1,
        "feature": "temperature",
        "observed_value": 21.0,
        "score": 0.9,
        "anomaly_score": 0.9,
        "predicted": True,
        "detector": "rules",
        "experiment_detector": "rules",
        "reason": "fixture alert",
        "severity": "medium",
    }
    pd.DataFrame([alert]).to_csv(experiment_dir / "clean_test_prediction_details.csv", index=False)
    pd.DataFrame(
        {
            "detector": ["rules"],
            "recall": [None],
            "true_positives": [0],
            "positive_pairs": [0],
            "alerts_on_unlabeled_pairs": [1],
            "unlabeled_alert_rate": [0.1],
        }
    ).to_csv(experiment_dir / "metrics.csv", index=False)
    store_path = tmp_path / "reviews.sqlite3"
    ReviewStore(store_path).save_review(alert, "confirmed_anomaly", "Independent review")
    monkeypatch.setenv("SENSORGUARD_EXPERIMENT_DIR", str(experiment_dir))
    monkeypatch.setenv("SENSORGUARD_DATA_PATH", str(tmp_path / "missing.txt"))
    monkeypatch.setenv("SENSORGUARD_PROFILE_PATH", str(tmp_path / "missing-profile.md"))
    monkeypatch.setenv("SENSORGUARD_MODEL_DIR", str(tmp_path / "missing-models"))
    monkeypatch.setenv("SENSORGUARD_REVIEW_STORE", str(store_path))

    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
    app.radio(key="workspace_navigation").set_value("Model evaluation").run()
    assert not app.exception
    assert any(metric.label == "Confirmed anomaly" and metric.value == "1" for metric in app.metric)
