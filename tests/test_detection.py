import numpy as np
import pandas as pd
import pytest

from sensorguard.autoencoder import AutoencoderDetector
from sensorguard.cli import calibrate_isolation_threshold
from sensorguard.detection import RESULT_COLUMNS, merge_alerts
from sensorguard.evaluation import binary_metrics, chronological_split
from sensorguard.isolation import IsolationForestDetector
from sensorguard.rules import RuleDetector, contextual_deviations


def sample(n=50):
    rng = np.random.default_rng(4)
    ts = pd.date_range("2004-01-01", periods=n, freq="min")
    rows = []
    for sid, offset in [(1, 0), (2, 2), (3, -2), (4, 1)]:
        for i, t in enumerate(ts):
            rows.append(
                {
                    "timestamp": t,
                    "moteid": sid,
                    "temperature": 20 + offset + rng.normal(0, 0.1),
                    "humidity": 40 + rng.normal(0, 0.2),
                    "light": 100 + rng.normal(0, 2),
                    "voltage": 2.7,
                }
            )
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def test_isolation_schema_missing_reproducibility_and_save(tmp_path):
    df = sample()
    df.loc[0, "temperature"] = np.nan
    a = IsolationForestDetector(n_estimators=40, random_state=7).fit(df)
    b = IsolationForestDetector(n_estimators=40, random_state=7).fit(df)
    pred = a.predict(df)
    assert set(RESULT_COLUMNS).issubset(pred.columns)
    assert np.allclose(a.score_samples(df)[1], b.score_samples(df)[1])
    path = tmp_path / "if.joblib"
    a.save(path)
    loaded = IsolationForestDetector.load(path)
    assert np.allclose(a.score_samples(df)[1], loaded.score_samples(df)[1])
    assert pred.is_anomaly.sum() > 0


def test_rules_and_context_causal_overlap():
    df = sample(8)
    df.loc[0, "temperature"] = -100
    out = RuleDetector().predict(df)
    assert "physical_limit" in " ".join(out.reason)
    ctx = contextual_deviations(df, tolerance="2min", min_peers=3)
    assert set(RESULT_COLUMNS).issubset(ctx.columns)


def test_merge_and_chronological_split():
    df = sample()
    tr, va, te = chronological_split(df)
    assert tr.timestamp.max() <= va.timestamp.min() <= te.timestamp.min()
    repeated = pd.concat([df.head(2), df.head(1)], ignore_index=True)
    r = RuleDetector().predict(repeated)
    merged = merge_alerts(r, r)
    assert merged.detector.iloc[0] == "rules"
    assert len(merged) == len(r)


def test_binary_metrics_counts_and_undefined_false_positive_rate():
    got = binary_metrics([1, 1, 0, 0], [1, 0, 1, 0])
    assert (got["true_positives"], got["false_positives"]) == (1, 1)
    assert (got["false_negatives"], got["true_negatives"]) == (1, 1)
    assert got["evaluated_pairs"] == 4 and got["positive_pairs"] == 2
    assert got["precision"] == got["recall"] == got["f1"] == 0.5
    assert got["false_positive_rate"] == 0.5
    no_negative_class = binary_metrics([1, 1], [0, 1])
    assert no_negative_class["false_positive_rate"] is None
    assert no_negative_class["recall"] == 0.5
    with pytest.raises(ValueError, match="equally sized"):
        binary_metrics([1], [0, 1])


def test_autoencoder_model_fit_predict_serialize():
    pytest.importorskip("torch")
    df = sample(30)
    train = df[df.timestamp < df.timestamp.quantile(0.7)]
    val = df[df.timestamp >= df.timestamp.quantile(0.7)]
    model = AutoencoderDetector(
        hidden_dims=(8, 3), epochs=4, patience=2, threshold_quantile=0.9, seed=2
    ).fit(train, val)
    pred = model.predict(val)
    assert set(RESULT_COLUMNS).issubset(pred.columns)
    assert model.loss_history and np.isfinite(model.threshold)
    path = __import__("pathlib").Path("ae-test.pt")
    model.save(path)
    loaded = AutoencoderDetector.load(path)
    assert np.allclose(model.predict(val).anomaly_score, loaded.predict(val).anomaly_score)
    assert not model.predict(val, threshold=1e20).is_anomaly.any()
    path.unlink()


def test_constant_feature_safe():
    df = sample(8)
    df["voltage"] = 2.5
    model = IsolationForestDetector(n_estimators=10).fit(df)
    assert np.isfinite(model.score_samples(df)[1]).all()


def test_rule_training_baseline_and_context_baselines():
    from sensorguard.rules import fit_contextual_baselines

    df = sample(30)
    detector = RuleDetector().fit(df)
    altered = df.iloc[[0]].copy()
    altered["temperature"] = 80.0
    assert "sensor_baseline_deviation" in " ".join(detector.predict(altered).reason)
    baselines = fit_contextual_baselines(df)
    assert (1, "temperature") in baselines


def test_cli_isolation_cutoff_uses_validation_scores():
    data = sample(40)
    train, validation, _ = chronological_split(data, 0.6, 0.2)
    detector = IsolationForestDetector(random_state=13, n_estimators=20).fit(train)
    _, validation_scores = detector.score_samples(validation)
    threshold = calibrate_isolation_threshold(detector, validation, 0.9)
    assert threshold == pytest.approx(np.quantile(validation_scores, 0.9))
    assert detector.threshold == threshold
    with pytest.raises(ValueError, match="between 0 and 1"):
        calibrate_isolation_threshold(detector, validation, 1.0)
