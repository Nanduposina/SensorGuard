import json

import numpy as np
import pandas as pd
import pytest

from sensorguard.cleaning import (
    clean_frame,
    cleaning_summary,
    fit_imputation_statistics,
)


def sample_rows():
    return pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2004-01-01 00:00:00"),
                "epoch": 1,
                "moteid": 1,
                "temperature": 20.0,
                "humidity": 40.0,
                "light": 10.0,
                "voltage": 2.5,
                "raw": "line-1",
                "line_number": 1,
                "issues": [],
            },
            {
                "timestamp": pd.Timestamp("2004-01-01 00:01:00"),
                "epoch": 2,
                "moteid": 1,
                "temperature": np.nan,
                "humidity": 40.0,
                "light": 10.0,
                "voltage": 2.5,
                "raw": "line-2",
                "line_number": 2,
                "issues": ["missing_temperature"],
            },
            {
                "timestamp": pd.Timestamp("2004-01-01 00:00:30"),
                "epoch": 3,
                "moteid": 1,
                "temperature": 120.0,
                "humidity": 40.0,
                "light": 10.0,
                "voltage": 2.5,
                "raw": "line-3",
                "line_number": 3,
                "issues": [],
            },
            {
                "timestamp": pd.NaT,
                "epoch": None,
                "moteid": 70,
                "temperature": 20.0,
                "humidity": 40.0,
                "light": 10.0,
                "voltage": 2.5,
                "raw": "malformed timestamp",
                "line_number": 4,
                "issues": ["invalid_timestamp", "malformed_field_count"],
            },
        ]
    )


def test_cleaner_flags_quality_classes_without_modifying_source():
    source = sample_rows()
    before = source.copy(deep=True)
    cleaned = clean_frame(source)
    assert source.equals(before)
    assert cleaned.retained.all()
    assert "missing_temperature" in cleaned.loc[1, "quality_issues"]
    assert "out_of_order" in cleaned.loc[2, "quality_issues"]
    assert "implausible_temperature" in cleaned.loc[2, "quality_issues"]
    assert "invalid_timestamp" in cleaned.loc[3, "quality_issues"]
    assert "unexpected_sensor_id" in cleaned.loc[3, "quality_issues"]
    assert json.loads(cleaned.loc[2, "cleaning_provenance"])["line_number"] == 3


def test_duplicate_exact_and_key_are_distinguished_and_drop_is_explicit():
    source = sample_rows().iloc[[0]].copy()
    exact = source.copy()
    key_duplicate = source.copy()
    key_duplicate["raw"] = "different line with same key"
    key_duplicate["temperature"] = 21.0
    data = pd.concat([source, exact, key_duplicate], ignore_index=True)
    result = clean_frame(data, policies={"duplicate_key": "drop"})
    assert result.retained.tolist() == [True, False, False]
    repeated_actions = json.loads(result.loc[1, "cleaning_provenance"])["actions"]
    assert {action["issue"] for action in repeated_actions} >= {"duplicate_key", "duplicate_record"}
    assert "duplicate_record" not in result.loc[2, "quality_issues"]
    assert cleaning_summary(result)["excluded_rows"] == 2


def test_missing_measurement_imputation_uses_only_explicit_training_rows():
    source = sample_rows()
    mask = np.array([True, False, False, False])
    statistics = fit_imputation_statistics(source, mask)
    result = clean_frame(
        source,
        policies={"missing_measurement": "impute"},
        imputation_statistics=statistics,
    )
    assert result.loc[1, "temperature"] == 20.0
    assert result.loc[1, "original_temperature"] != result.loc[1, "temperature"]
    action = json.loads(result.loc[1, "cleaning_provenance"])["actions"]
    assert any(a["policy"] == "impute" and a["result"] == 20.0 for a in action)


def test_imputation_requires_training_statistics_and_supported_category():
    source = sample_rows()
    with pytest.raises(ValueError, match="training-fitted"):
        clean_frame(source, policies={"missing_measurement": "impute"})
    with pytest.raises(ValueError, match="only supported"):
        clean_frame(
            source,
            policies={"timestamp": "impute"},
            imputation_statistics={"temperature": {1: 20.0}},
        )


def test_gap_policy_drops_only_observation_after_gap_and_preserves_audit():
    source = sample_rows().iloc[[0, 1]].copy()
    source.loc[source.index[1], "timestamp"] = pd.Timestamp("2004-01-01 00:20:00")
    result = clean_frame(source, policies={"sampling_gap": "drop"})
    assert result.retained.tolist() == [True, False]
    assert "sampling_gap" in result.loc[1, "quality_issues"]
    assert json.loads(result.loc[1, "cleaning_provenance"])["retained"] is False


def test_empty_malformed_inputs_and_invalid_policy_are_safe():
    empty = pd.DataFrame(columns=["timestamp", "moteid", "issues"])
    output = clean_frame(empty)
    assert output.empty and cleaning_summary(output)["input_rows"] == 0
    malformed = pd.DataFrame([{"raw": "not a reading", "issues": ["malformed_field_count"]}])
    output = clean_frame(malformed)
    assert len(output) == 1 and "malformed_field_count" in output.loc[0, "quality_issues"]
    with pytest.raises(ValueError, match="unknown cleaning policy"):
        clean_frame(sample_rows(), policies={"sampling_gap": "silently_fix"})


def test_streaming_cli_tracks_cross_chunk_duplicates_gaps_and_provenance(tmp_path):
    from scripts.clean_dataset import run_cleaning

    source = tmp_path / "synthetic.txt"
    source.write_text(
        "\n".join(
            [
                "2004-01-01 00:00:00 1 1 20 40 10 2.5",
                "2004-01-01 00:10:00 2 1 21 40 10 2.5",
                "2004-01-01 00:00:00 3 1 22 40 10 2.5",
                "2004-01-01 00:00:00 3 1 22 40 10 2.5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    first_dir = tmp_path / "run1"
    second_dir = tmp_path / "run2"
    first = run_cleaning(source, first_dir, chunk_size=1)
    run_cleaning(source, second_dir, chunk_size=2)
    assert first["input_rows"] == first["retained_rows"] == 4
    assert first["issue_counts"]["sampling_gap"] >= 1
    assert first["issue_counts"]["out_of_order"] >= 1
    assert first["issue_counts"]["duplicate_key"] >= 1
    assert first["issue_counts"]["duplicate_record"] >= 1
    assert (first_dir / "provenance.csv").read_bytes() == (
        second_dir / "provenance.csv"
    ).read_bytes()
    assert (first_dir / "cleaned.csv").read_bytes() == (second_dir / "cleaned.csv").read_bytes()
    assert json.loads((first_dir / "cleaning_report.json").read_text()) == json.loads(
        (second_dir / "cleaning_report.json").read_text()
    )
