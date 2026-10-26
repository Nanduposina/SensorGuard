import numpy as np
import pandas as pd

from scripts.run_experiment import alert_breakdowns
from sensorguard.experiment import evaluate_track
from sensorguard.historical import voltage_case_studies
from sensorguard.injection import FAULT_TYPES, FaultSpec, inject_faults


def frame(n=80):
    rows = []
    for sid, offset in [(1, 0), (2, 1), (3, -1), (4, 2)]:
        for i, t in enumerate(pd.date_range("2004-03-01", periods=n, freq="min")):
            rows.append(
                {
                    "timestamp": t,
                    "moteid": sid,
                    "temperature": 20 + offset + np.sin(i / 5),
                    "humidity": 40 + np.cos(i / 6),
                    "light": 100 + i % 5,
                    "voltage": 2.7 - 0.0001 * i,
                }
            )
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def test_each_injection_type_is_seeded_audited_and_non_mutating():
    source = frame(50)
    before = source.copy(deep=True)
    specs = [
        FaultSpec("spike", 1, "temperature", 10, 1, 15),
        FaultSpec("drop", 1, "humidity", 12, 1, 10),
        FaultSpec("level_shift", 1, "temperature", 15, 3, 5),
        FaultSpec("gradual_drift", 1, "humidity", 20, 4, 2),
        FaultSpec("stuck_at", 1, "light", 26, 4),
        FaultSpec("missing_block", 1, "temperature", 32, 3),
        FaultSpec("sensor_outage", 1, None, 36, 2),
        FaultSpec("random_noise", 1, "temperature", 39, 3, 2),
        FaultSpec("increased_variance", 1, "humidity", 43, 3, 1),
        FaultSpec("physical_value", 1, "temperature", 46, 1),
        FaultSpec("voltage_decline", 1, "voltage", 5, 3, 0.1),
        FaultSpec("timestamp_delay", 1, None, 8, 2, parameters={"seconds": 90}),
        FaultSpec("irregular_sampling", 1, None, 2, 3, parameters={"seconds": 45}),
    ]
    result = inject_faults(source, specs, seed=99)
    assert set(result.fault_manifest.fault_type) == set(FAULT_TYPES)
    assert source.equals(before)
    assert len(result.observations) > len(specs) and result.labels.is_fault.all()
    assert result.fault_manifest.random_seed.eq(99).all()
    second = inject_faults(source, specs, seed=99)
    pd.testing.assert_frame_equal(result.frame, second.frame)
    assert (result.observations.original_value != result.observations.modified_value).all()
    drift_rows = result.observations[result.observations.fault_type == "gradual_drift"]
    assert (drift_rows.original_value != drift_rows.modified_value).all()
    timestamp_rows = result.observations[
        result.observations.fault_type.isin(["timestamp_delay", "irregular_sampling"])
    ]
    assert (timestamp_rows.timestamp_original != timestamp_rows.timestamp_modified).all()
    labeled_pairs = {
        (row.timestamp, int(row.sensor_id), row.feature)
        for row in result.labels.itertuples()
        if row.is_fault
    }
    for row in result.observations.itertuples():
        feature = "record" if row.feature == "timestamp" else row.feature
        assert (row.timestamp_modified, int(row.sensor_id), feature) in labeled_pairs


def test_complete_small_experiment():
    result = evaluate_track(
        frame(), seed=5, autoencoder_epochs=2, n_estimators=20, threshold_quantile=0.9
    )
    assert set(result["metrics"].detector) == {
        "rules",
        "isolation_forest",
        "autoencoder",
        "quality_rule_baseline",
        "statistical_rule_baseline",
        "combined_rules_if",
        "combined_ml_if_ae",
        "combined_rules_if_ae",
    }
    assert set(
        [
            "precision",
            "recall",
            "f1",
            "false_positive_rate",
            "false_negatives",
            "false_positives",
            "true_negatives",
            "pr_auc",
            "alerts_on_unlabeled_pairs",
            "unlabeled_pairs",
        ]
    ).issubset(result["metrics"].columns)
    assert (
        result["split_sizes"]["train"]
        and result["split_sizes"]["validation"]
        and result["split_sizes"]["test"]
    )
    assert not result["fault_manifest"].empty
    assert result["metrics"].evaluated_pairs.nunique() == 1
    assert (result["metrics"].evaluated_pairs == result["metrics"].evaluated_records * 5).all()
    assert result["metrics"].fault_events.eq(len(result["fault_manifest"])).all()
    assert result["metrics"].precision.isna().all()
    assert result["metrics"].false_positives.isna().all()
    assert result["metrics"].true_negatives.isna().all()
    assert result["metrics"].pr_auc.isna().all()
    assert result["metrics"].labeled_pairs.eq(result["metrics"].positive_pairs).all()
    assert (
        result["metrics"]
        .unlabeled_pairs.eq(result["metrics"].evaluated_pairs - result["metrics"].positive_pairs)
        .all()
    )
    assert result["prediction_details"].ground_truth.isna().any()
    assert result["prediction_details"].ground_truth.notna().sum() == (
        result["metrics"].positive_pairs.iloc[0] * len(result["predictions"])
    )
    assert result["per_fault"].injected_pairs.sum() == (
        result["metrics"].positive_pairs.iloc[0] * len(result["predictions"])
    )
    for output in result["predictions"].values():
        assert not output.duplicated(["timestamp", "sensor_id", "feature"]).any()
    assert not result["pr_curves"]
    assert result["clean_baseline"].unlabeled_alert_rate.dropna().between(0, 1).all()
    assert not result["clean_prediction_details"].empty


def test_historical_voltage_adjusts_temperature_and_caveats():
    df = frame(50)
    cases = voltage_case_studies(df, minimum_rows=20)
    assert len(cases) == 4
    assert "candidate_decline" in cases
    assert cases.interpretation.str.contains("Unlabeled candidate").all()


def test_constant_rule_emits_once_for_long_plateau():
    from sensorguard.rules import RuleDetector

    df = frame(30)
    df["temperature"] = 20.0
    out = RuleDetector().predict(df[df.moteid == 1])
    constant = out[out.reason.str.contains("constant_reading")]
    assert len(constant) == 1


def test_alert_breakdown_deduplicates_pairs_and_counts_readings():
    predictions = pd.DataFrame(
        [
            {
                "detector": "isolation_forest",
                "timestamp": pd.Timestamp("2004-03-01"),
                "sensor_id": 1,
                "feature": feature,
                "predicted": True,
                "rule_id": None,
            }
            for feature in ("temperature", "humidity", "temperature")
        ]
    )
    pairs, _, readings = alert_breakdowns(predictions)
    assert pairs.alert_pairs.sum() == 2
    assert readings.feature_alert_pairs.iloc[0] == 2
    assert readings.unique_alerted_readings.iloc[0] == 1
