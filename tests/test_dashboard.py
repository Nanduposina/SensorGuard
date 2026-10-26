from pathlib import Path

import pandas as pd

from sensorguard.dashboard_data import (
    align_context,
    confusion_counts,
    filter_alerts,
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


def test_streamlit_smoke():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "app.py", default_timeout=20
    ).run()
    assert not app.exception
    assert any("SensorGuard" in title.value for title in app.title)
    for page in [
        "Overview",
        "Sensor explorer",
        "Alert investigation",
        "Model evaluation",
        "Sensor health",
    ]:
        app.radio[0].set_value(page).run()
        assert not app.exception, f"Dashboard page {page!r} raised an exception"


def test_streamlit_empty_artifact_smoke(tmp_path, monkeypatch):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("SENSORGUARD_DATA_PATH", str(tmp_path / "missing.txt"))
    monkeypatch.setenv("SENSORGUARD_EXPERIMENT_DIR", str(tmp_path / "no-results"))
    monkeypatch.setenv("SENSORGUARD_PROFILE_PATH", str(tmp_path / "no-profile.md"))
    monkeypatch.setenv("SENSORGUARD_MODEL_DIR", str(tmp_path / "no-models"))
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
    assert not app.exception
    for page in [
        "Overview",
        "Sensor explorer",
        "Alert investigation",
        "Model evaluation",
        "Sensor health",
    ]:
        app.radio[0].set_value(page).run()
        assert not app.exception, f"Empty-state page {page!r} raised an exception"
