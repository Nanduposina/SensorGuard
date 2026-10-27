import numpy as np
import pandas as pd

from sensorguard.context import add_peer_features, fit_peer_baselines
from sensorguard.coordinates import MOTE_COORDINATES, physical_neighborhoods
from sensorguard.experiment import evaluate_track


def test_peer_features_adjust_sensor_levels_exclude_self_and_use_past_only():
    train_times = pd.date_range("2004-01-01", periods=5, freq="min")
    training = pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "sensor_id": sensor_id,
                "temperature": level,
                "humidity": 40.0,
                "light": 10.0,
                "voltage": 2.5,
            }
            for timestamp in train_times
            for sensor_id, level in [(1, 20.0), (2, 30.0), (3, 40.0)]
        ]
    )
    baselines = fit_peer_baselines(training, minimum_rows=3)
    target_time = pd.Timestamp("2004-01-01 00:05:00")
    target = pd.DataFrame([{"timestamp": target_time, "sensor_id": 1, "temperature": 25.0}])
    references = pd.concat(
        [
            training,
            pd.DataFrame(
                [
                    {
                        "timestamp": target_time - pd.Timedelta(seconds=30),
                        "sensor_id": 2,
                        "temperature": 30.0,
                    },
                    {
                        "timestamp": target_time - pd.Timedelta(seconds=30),
                        "sensor_id": 3,
                        "temperature": 40.0,
                    },
                    {
                        "timestamp": target_time + pd.Timedelta(seconds=30),
                        "sensor_id": 2,
                        "temperature": 1000.0,
                    },
                    {"timestamp": target_time, "sensor_id": 1, "temperature": 999.0},
                ]
            ),
        ],
        ignore_index=True,
    )
    result = add_peer_features(
        target,
        references,
        baselines=baselines,
        tolerance="2min",
        minimum_peers=2,
    )
    assert result.temperature_peer_count.iloc[0] == 2
    assert result.temperature_peer_delta.iloc[0] == 5.0
    assert result.temperature_peer_age_fraction.iloc[0] == 0.25


def test_sparse_peer_data_is_reported_but_not_used_as_a_reference():
    ts = pd.Timestamp("2004-01-01")
    targets = pd.DataFrame([{"timestamp": ts, "sensor_id": 1, "temperature": 30.0}])
    refs = pd.DataFrame(
        [
            {"timestamp": ts, "sensor_id": 1, "temperature": 30.0},
            {"timestamp": ts, "sensor_id": 2, "temperature": 20.0},
        ]
    )
    result = add_peer_features(
        targets,
        refs,
        baselines={(1, "temperature"): 20.0, (2, "temperature"): 10.0},
        minimum_peers=2,
    )
    assert result.temperature_peer_count.iloc[0] == 1
    assert pd.isna(result.temperature_peer_delta.iloc[0])
    assert pd.isna(result.temperature_peer_robust_z.iloc[0])


def test_official_physical_neighborhoods_do_not_assume_id_adjacency():
    neighbors = physical_neighborhoods(radius_m=5.0)
    assert set(MOTE_COORDINATES) == set(range(1, 55))
    assert 2 in neighbors[1]  # 4.24 m away in the official coordinate file.
    assert 54 not in neighbors[1]
    assert 1 not in neighbors[1]
    assert 999 not in neighbors


def test_contextual_experiment_consumes_features_and_is_reproducible():
    rows = []
    for sensor_id, offset in [(1, 0.0), (2, 1.0), (3, -1.0), (4, 2.0)]:
        for i, timestamp in enumerate(pd.date_range("2004-02-01", periods=48, freq="min")):
            rows.append(
                {
                    "timestamp": timestamp,
                    "sensor_id": sensor_id,
                    "temperature": 20 + offset + np.sin(i / 6),
                    "humidity": 40 + np.cos(i / 7),
                    "light": 100 + i % 5,
                    "voltage": 2.7 - 0.0001 * i,
                }
            )
    frame = pd.DataFrame(rows).sort_values("timestamp", kind="stable").reset_index(drop=True)
    first = evaluate_track(
        frame,
        seed=17,
        threshold_quantile=0.9,
        autoencoder_epochs=1,
        n_estimators=10,
        use_contextual_features=True,
    )
    second = evaluate_track(
        frame,
        seed=17,
        threshold_quantile=0.9,
        autoencoder_epochs=1,
        n_estimators=10,
        use_contextual_features=True,
    )
    assert "temperature_peer_robust_z" in first["test_frame"]
    assert set(first["contextual_features"]) <= set(first["test_frame"])
    assert set(first["contextual_coverage"].feature) == {
        "temperature",
        "humidity",
        "light",
        "voltage",
    }
    assert {"isolation_forest_contextual", "autoencoder_contextual"} <= set(
        first["metrics"].detector
    )
    assert first["thresholds"]["contextual_enabled"] is True
    pd.testing.assert_frame_equal(first["metrics"], second["metrics"])
