import numpy as np
import pandas as pd
import pytest

from sensorguard.battery_evaluation import (
    BATTERY_FEATURES,
    _top_fraction_vehicle_score,
    evaluate_labeled_battery_vehicles,
    load_vehicle_labeled_segments,
)


def _battery_frame(normal=12, faulty=2, segments=6):
    rows = []
    for car in range(normal + faulty):
        label = int(car >= normal)
        for order in range(segments):
            row = {
                "vehicle_id": car + 1,
                "vehicle_label": label,
                "mileage": 1000.0 + order * 50,
                "charge_segment": str(order),
                "source_file": f"{car}_{order}.pkl",
                "timestamp": pd.Timestamp("2000-01-01") + pd.Timedelta(seconds=order),
            }
            for j, feature in enumerate(BATTERY_FEATURES):
                row[feature] = (
                    (3.5 if feature.startswith("voltage") else 0.0)
                    + car * 0.01
                    + order * 0.001
                    + j * 0.1
                )
            rows.append(row)
    return pd.DataFrame(rows)


def test_public_dataset_adapter_preserves_vehicle_and_segment_provenance(tmp_path):
    import torch

    brand = tmp_path / "battery_brand3"
    (brand / "label").mkdir(parents=True)
    (brand / "data").mkdir()
    pd.DataFrame({"car": [401, 402, 403], "label": [0, 1, 0]}).to_csv(
        brand / "label" / "all_label.csv", index=False
    )
    columns = [
        "volt",
        "current",
        "soc",
        "max_single_volt",
        "min_single_volt",
        "max_temp",
        "min_temp",
        "timestamp",
    ]
    torch.save(columns, brand / "column.pkl")
    for idx, car in enumerate([401, 402, 403]):
        values = np.column_stack(
            [
                np.linspace(3.2, 3.6, 8),
                np.linspace(-1.0, -0.2, 8),
                np.linspace(20, 90, 8),
                np.full(8, 3.7),
                np.full(8, 3.1),
                np.linspace(20, 25, 8),
                np.linspace(15, 20, 8),
                np.arange(8) * 10,
            ]
        )
        torch.save(
            (values, {"car": car, "charge_segment": str(idx), "mileage": 100 + idx}),
            brand / "data" / f"{idx}.pkl",
        )
    got = load_vehicle_labeled_segments(brand)
    assert len(got) == 3
    assert got.vehicle_id.tolist() == [401, 402, 403]
    assert got.vehicle_label.tolist() == [0, 1, 0]
    assert got.source_file.tolist() == ["0.pkl", "1.pkl", "2.pkl"]
    assert got.source_label.eq("paper_vehicle_level_label").all()
    assert got.voltage_mean.notna().all()
    assert got.relative_segment_order.eq(0).all()


def test_real_battery_evaluation_has_disjoint_vehicle_splits_and_normal_denominator(tmp_path):
    frame = _battery_frame()
    result = evaluate_labeled_battery_vehicles(
        frame, seed=4, epochs=2, n_estimators=15, threshold_quantile=0.8
    )
    groups = result["vehicle_groups"]
    assert not set(groups["train"]) & set(groups["validation"])
    assert not set(groups["train"]) & set(groups["test"])
    assert not set(groups["validation"]) & set(groups["test"])
    assert all(car <= 12 for car in groups["train"] + groups["validation"])
    assert {13, 14}.issubset(set(groups["test"]))
    assert result["metrics"].evaluation_unit.eq("held-out vehicle").all()
    assert (
        result["metrics"]
        .evaluated_vehicles.eq(
            result["metrics"].test_labeled_normal_vehicles
            + result["metrics"].positive_fault_vehicles
        )
        .all()
    )
    assert result["metrics"].positive_fault_vehicles.eq(2).all()
    for row in result["metrics"].itertuples():
        expected_normal = row.test_labeled_normal_vehicles
        assert row.false_positive_normal_vehicles <= expected_normal
        assert row.false_positive_rate == pytest.approx(
            row.false_positive_normal_vehicles / expected_normal
        )

    iso_path = tmp_path / "battery-if.joblib"
    ae_path = tmp_path / "battery-ae.pt"
    result["detectors"]["isolation_forest"].save(iso_path)
    result["detectors"]["autoencoder"].save(ae_path)
    from sensorguard.autoencoder import AutoencoderDetector
    from sensorguard.isolation import IsolationForestDetector

    iso = IsolationForestDetector.load(iso_path)
    ae = AutoencoderDetector.load(ae_path)
    assert iso.measurement_features == BATTERY_FEATURES
    assert not iso.include_time_features and not iso.include_sensor_id_feature
    assert ae.configured_measurement_features == BATTERY_FEATURES
    model_frame = frame.assign(sensor_id=frame.vehicle_id)
    assert np.allclose(
        result["detectors"]["isolation_forest"].score_samples(model_frame)[1],
        iso.score_samples(model_frame)[1],
    )
    assert np.allclose(
        result["detectors"]["autoencoder"].score_samples(model_frame)[1],
        ae.score_samples(model_frame)[1],
    )


def test_vehicle_scores_are_reassociated_by_source_segment_identity():
    frame = _battery_frame(normal=12, faulty=2, segments=10)
    scored_order = frame.sample(frac=1, random_state=11).reset_index(drop=True)
    got = _top_fraction_vehicle_score(
        frame,
        scored_order.vehicle_id.to_numpy(dtype=float),
        score_frame=scored_order,
    )
    assert got.set_index("vehicle_id").score.to_dict() == {car: float(car) for car in range(1, 15)}


def test_unknown_label_values_are_not_accepted_as_normal(tmp_path):
    brand = tmp_path / "battery_brand3"
    (brand / "label").mkdir(parents=True)
    (brand / "data").mkdir()
    pd.DataFrame({"car": [1, 2], "label": [0, "unknown"]}).to_csv(
        brand / "label" / "all_label.csv", index=False
    )
    with pytest.raises(ValueError, match="unique binary values"):
        load_vehicle_labeled_segments(brand)
