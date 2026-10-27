"""Vehicle-level evaluation on independently labelled EV battery telemetry.

The optional dataset is the brand 3 release accompanying Zhang et al. (2023).
Its anomaly labels are at vehicle level; this module never promotes them to
timestamp-level failure labels.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .autoencoder import AutoencoderDetector
from .evaluation import binary_metrics
from .isolation import IsolationForestDetector

BATTERY_FEATURES = (
    "voltage_mean",
    "voltage_std",
    "voltage_slope",
    "current_mean",
    "current_std",
    "current_slope",
    "soc_mean",
    "soc_std",
    "soc_slope",
    "max_temp_mean",
    "max_temp_std",
    "max_temp_slope",
)
EXPECTED_COLUMNS = (
    "volt",
    "current",
    "soc",
    "max_single_volt",
    "min_single_volt",
    "max_temp",
    "min_temp",
    "timestamp",
)


def _load_research_pickle(path: Path):
    """Load the study's PyTorch-serialized NumPy tuple from a user-supplied folder."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency is part of the project
        raise RuntimeError("Install SensorGuard dependencies to read this dataset") from exc
    # This archive uses pickle-backed torch serialization. Only load files obtained
    # from the cited Figshare release; never accept uploads through the dashboard.
    return torch.load(path, map_location="cpu", weights_only=False)


def _summarize_segment(values: np.ndarray, columns: list[str]) -> dict[str, float]:
    if values.ndim != 2 or values.shape[1] != len(columns) or values.shape[0] < 2:
        raise ValueError("each source segment must be a 2D array with at least two samples")
    result: dict[str, float] = {}
    time_index = columns.index("timestamp")
    x = np.asarray(values[:, time_index], dtype=float)
    if not np.isfinite(x).all() or np.ptp(x) == 0:
        x = np.arange(len(values), dtype=float)
    for source, prefix in (
        ("volt", "voltage"),
        ("current", "current"),
        ("soc", "soc"),
        ("max_temp", "max_temp"),
    ):
        y = np.asarray(values[:, columns.index(source)], dtype=float)
        finite = np.isfinite(y)
        if not finite.any():
            result[f"{prefix}_mean"] = math.nan
            result[f"{prefix}_std"] = math.nan
            result[f"{prefix}_slope"] = math.nan
            continue
        xf = x[finite]
        yf = y[finite]
        result[f"{prefix}_mean"] = float(np.mean(yf))
        result[f"{prefix}_std"] = float(np.std(yf))
        result[f"{prefix}_slope"] = float(np.polyfit(xf, yf, 1)[0]) if len(yf) > 1 else 0.0
    return result


def load_vehicle_labeled_segments(dataset_dir: str | Path) -> pd.DataFrame:
    """Load snippet summaries and retain source paths, vehicle IDs, and mileage.

    `timestamp` in the returned table is an internal chronological ordering key
    derived from each vehicle's mileage-ordered segment rank, not a wall-clock
    timestamp. The source's 128-point elapsed-time vector remains in each pickle;
    failure dates are not provided by this release.
    """
    root = Path(dataset_dir)
    labels_path = root / "label" / "all_label.csv"
    data_dir = root / "data"
    if not labels_path.is_file() or not data_dir.is_dir():
        raise FileNotFoundError("Expected brand directory with label/all_label.csv and data/*.pkl")
    labels_frame = pd.read_csv(labels_path)
    if not {"car", "label"}.issubset(labels_frame.columns):
        raise ValueError("vehicle label table must contain car and label columns")
    if labels_frame.car.duplicated().any() or not labels_frame.label.isin([0, 1]).all():
        raise ValueError("vehicle labels must be unique binary values")
    labels = dict(zip(labels_frame.car.astype(int), labels_frame.label.astype(int), strict=True))

    column_payload = _load_research_pickle(root / "column.pkl")
    if tuple(column_payload) != EXPECTED_COLUMNS:
        raise ValueError(f"unexpected source columns: {column_payload!r}")
    paths = sorted(data_dir.glob("*.pkl"), key=lambda p: (int(p.stem), p.name))
    if not paths:
        raise ValueError("no segment pickle files found")

    rows: list[dict[str, object]] = []
    for path in paths:
        payload = _load_research_pickle(path)
        if not isinstance(payload, tuple) or len(payload) != 2:
            raise ValueError(f"invalid segment payload: {path.name}")
        values, metadata = payload
        values = np.asarray(values, dtype=float)
        car = int(metadata["car"])
        if car not in labels:
            raise ValueError(f"segment {path.name} has no matching vehicle label")
        row: dict[str, object] = _summarize_segment(values, list(column_payload))
        mileage = float(metadata["mileage"])
        if not math.isfinite(mileage):
            raise ValueError(f"segment {path.name} has invalid mileage")
        row.update(
            {
                "vehicle_id": car,
                "vehicle_label": labels[car],
                "mileage": mileage,
                "charge_segment": str(metadata["charge_segment"]),
                "source_file": path.name,
                "source_label": "paper_vehicle_level_label",
            }
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(
        ["vehicle_id", "mileage", "charge_segment", "source_file"], kind="stable"
    ).reset_index(drop=True)
    frame["relative_segment_order"] = frame.groupby("vehicle_id", sort=False).cumcount()
    frame["timestamp"] = pd.Timestamp("2000-01-01") + pd.to_timedelta(
        frame.relative_segment_order, unit="s"
    )
    return frame


def _split_vehicles(
    frame: pd.DataFrame, *, seed: int, validation_fraction: float, test_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[int]]]:
    if validation_fraction <= 0 or test_fraction <= 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("validation and test fractions must be positive and sum to less than one")
    labels = frame.groupby("vehicle_id").vehicle_label.first()
    normal_ids = np.asarray(sorted(labels[labels == 0].index), dtype=int)
    fault_ids = sorted(labels[labels == 1].index.astype(int).tolist())
    if len(normal_ids) < 10 or not fault_ids:
        raise ValueError("benchmark requires at least ten labelled-normal and one fault vehicle")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(normal_ids)
    val_count = max(1, int(round(len(shuffled) * validation_fraction)))
    test_count = max(1, int(round(len(shuffled) * test_fraction)))
    train_count = len(shuffled) - val_count - test_count
    if train_count < 2:
        raise ValueError("not enough normal vehicles for disjoint train/validation/test groups")
    train_ids = sorted(shuffled[:train_count].astype(int).tolist())
    val_ids = sorted(shuffled[train_count : train_count + val_count].astype(int).tolist())
    test_normal_ids = sorted(shuffled[train_count + val_count :].astype(int).tolist())
    test_ids = sorted(test_normal_ids + fault_ids)

    def chronological_tail(ids: list[int], fraction: float) -> pd.DataFrame:
        parts = []
        for _, group in frame[frame.vehicle_id.isin(ids)].groupby("vehicle_id", sort=True):
            group = group.sort_values(["mileage", "charge_segment", "source_file"], kind="stable")
            count = max(1, int(math.ceil(len(group) * fraction)))
            parts.append(group.tail(count))
        return pd.concat(parts, ignore_index=True) if parts else frame.iloc[:0].copy()

    train_parts = []
    for _, group in frame[frame.vehicle_id.isin(train_ids)].groupby("vehicle_id", sort=True):
        group = group.sort_values(["mileage", "charge_segment", "source_file"], kind="stable")
        count = max(1, int(math.floor(len(group) * 0.70)))
        train_parts.append(group.head(count))
    train = pd.concat(train_parts, ignore_index=True)
    validation = chronological_tail(val_ids, 0.20)
    test = chronological_tail(test_ids, 0.20)
    groups = {"train": train_ids, "validation": val_ids, "test": test_ids}
    return train, validation, test, groups


def _top_fraction_vehicle_score(
    frame: pd.DataFrame,
    scores: np.ndarray,
    *,
    score_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if len(frame) != len(scores):
        raise ValueError("score count does not match snippet rows")
    if score_frame is None:
        aligned_scores = np.asarray(scores, dtype=float)
    else:
        if "source_file" not in frame or "source_file" not in score_frame:
            raise ValueError("source_file is required to realign detector scores")
        if frame.source_file.duplicated().any() or score_frame.source_file.duplicated().any():
            raise ValueError("source_file must uniquely identify every battery segment")
        score_by_file = pd.Series(np.asarray(scores, dtype=float), index=score_frame.source_file)
        aligned_scores = frame.source_file.map(score_by_file)
        if aligned_scores.isna().any():
            raise ValueError("some source segments have no aligned detector score")
        aligned_scores = aligned_scores.to_numpy(dtype=float)
    scored = frame[["vehicle_id", "vehicle_label"]].copy()
    scored["score"] = aligned_scores
    values = []
    for car, group in scored.groupby("vehicle_id", sort=True):
        top_count = max(1, int(math.ceil(len(group) * 0.10)))
        values.append(
            {
                "vehicle_id": int(car),
                "vehicle_label": int(group.vehicle_label.iloc[0]),
                "score": float(group.nlargest(top_count, "score").score.mean()),
                "evaluated_segments": len(group),
            }
        )
    return pd.DataFrame(values)


def evaluate_labeled_battery_vehicles(
    frame: pd.DataFrame,
    *,
    seed: int = 42,
    epochs: int = 20,
    n_estimators: int = 200,
    validation_fraction: float = 0.20,
    test_fraction: float = 0.20,
    threshold_quantile: float = 0.95,
) -> dict[str, object]:
    """Evaluate existing IF/AE detectors at held-out, vehicle-label granularity.

    Vehicle groups are disjoint. For each vehicle, training uses the earliest 70%
    of mileage-ordered segments; validation and test use the latest 20% from
    separate vehicle groups. Validation contains source-labelled normal vehicles
    only and determines the vehicle-score cutoff. All labelled fault vehicles are
    held out for final test; no test scores or labels affect fitting/cutoff choice.
    """
    if not 0.5 < threshold_quantile < 1:
        raise ValueError("threshold_quantile must be between 0.5 and 1")
    missing = set(BATTERY_FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"battery frame is missing features: {sorted(missing)}")
    train, validation, test, vehicle_groups = _split_vehicles(
        frame,
        seed=seed,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    if len(train) < 4 or validation.empty or test.empty:
        raise ValueError("train, validation, and test partitions must be nonempty")
    train_model = train.assign(sensor_id=train.vehicle_id)
    validation_model = validation.assign(sensor_id=validation.vehicle_id)
    test_model = test.assign(sensor_id=test.vehicle_id)

    iso = IsolationForestDetector(
        random_state=seed,
        n_estimators=n_estimators,
        contamination="auto",
        measurement_features=BATTERY_FEATURES,
        include_time_features=False,
        use_sensor_baselines=False,
        include_sensor_id_feature=False,
    ).fit(train_model)
    ae = AutoencoderDetector(
        seed=seed,
        epochs=epochs,
        patience=max(2, min(8, epochs // 3)),
        measurement_features=BATTERY_FEATURES,
    ).fit(train_model, validation_model)

    val_if_data, val_if_scores = iso.score_samples(validation_model)
    _, val_ae_scores = ae.score_samples(validation_model)
    validation_scores = {
        "isolation_forest": _top_fraction_vehicle_score(
            validation, val_if_scores, score_frame=val_if_data
        ),
        "autoencoder": _top_fraction_vehicle_score(validation, val_ae_scores),
    }
    thresholds = {
        detector: float(scores.score.quantile(threshold_quantile))
        for detector, scores in validation_scores.items()
    }

    test_if_data, test_if_scores = iso.score_samples(test_model)
    _, test_ae_scores = ae.score_samples(test_model)
    test_scores = {
        "isolation_forest": _top_fraction_vehicle_score(
            test, test_if_scores, score_frame=test_if_data
        ),
        "autoencoder": _top_fraction_vehicle_score(test, test_ae_scores),
    }
    metrics = []
    vehicle_results = []
    for detector, values in test_scores.items():
        cutoff = thresholds[detector]
        values = values.copy()
        values["prediction"] = values.score >= cutoff
        y = values.vehicle_label.to_numpy(dtype=bool)
        predictions = values.prediction.to_numpy(dtype=bool)
        raw_metric = binary_metrics(y, predictions)
        metric = {
            "evaluation_unit": "held-out vehicle",
            "evaluated_vehicles": raw_metric["evaluated_pairs"],
            "positive_fault_vehicles": raw_metric["positive_pairs"],
            "true_positive_fault_vehicles": raw_metric["true_positives"],
            "false_positive_normal_vehicles": raw_metric["false_positives"],
            "false_negative_fault_vehicles": raw_metric["false_negatives"],
            "true_negative_normal_vehicles": raw_metric["true_negatives"],
            "precision": raw_metric["precision"],
            "recall": raw_metric["recall"],
            "f1": raw_metric["f1"],
            "false_positive_rate": raw_metric["false_positive_rate"],
        }
        metrics.append(
            {
                "detector": detector,
                **metric,
                "threshold": cutoff,
                "threshold_quantile_validation_normals": threshold_quantile,
                "training_vehicles": len(vehicle_groups["train"]),
                "validation_vehicles": len(vehicle_groups["validation"]),
                "test_labeled_normal_vehicles": raw_metric["true_negatives"]
                + raw_metric["false_positives"],
            }
        )
        values["detector"] = detector
        values["threshold"] = cutoff
        values["prediction"] = values.prediction.astype(bool)
        values["label_kind"] = np.where(
            values.vehicle_label.eq(1), "paper_labeled_battery_fault", "paper_labeled_normal"
        )
        vehicle_results.append(values)
    return {
        "metrics": pd.DataFrame(metrics),
        "per_vehicle": pd.concat(vehicle_results, ignore_index=True),
        "vehicle_groups": vehicle_groups,
        "split_rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "thresholds": thresholds,
        "validation_scores": validation_scores,
        "detectors": {"isolation_forest": iso, "autoencoder": ae},
    }


def sha256_file(path: str | Path) -> str:
    """Hash an external archive incrementally for reproducible provenance."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_battery_evaluation_report(result, output_dir: str | Path, metadata: dict) -> Path:
    """Write reproducible CSV and Markdown summaries without modifying inputs."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    result["metrics"].to_csv(output / "real_battery_metrics.csv", index=False)
    result["per_vehicle"].to_csv(output / "real_battery_per_vehicle.csv", index=False)
    (output / "real_battery_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metric_frame = result["metrics"]
    columns = list(metric_frame.columns)
    table_lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in metric_frame.itertuples(index=False, name=None):
        cells = [f"{value:.4f}" if isinstance(value, float) else str(value) for value in row]
        table_lines.append("| " + " | ".join(cells) + " |")
    table = "\n".join(table_lines)
    lines = [
        "# Published real-world EV battery fault evaluation",
        "",
        "This is separate from SensorGuard's synthetic Intel Lab fault experiment.",
        "The evaluation unit is a held-out vehicle, not a charging sample or a timestamped failure event.",
        "Vehicle labels are taken from the study's vehicle-level CSV and were described by the study authors as based on driver reports and engineer confirmation.",
        "The source does not include physical battery service records or exact failure times; this is a published vehicle-fault label, not a timestamped hardware intervention record.",
        "",
        "## Results",
        "",
        table,
        "",
        "Thresholds were selected from the configured quantile of validation-normal vehicle scores only. Training, validation, and test vehicle IDs are disjoint. Each group's data are ordered by source mileage and the model sees only its designated chronological segment. Test labels were not used for fitting or threshold selection.",
        "False-positive counts/rates use only held-out vehicles labelled normal by the source dataset. Fault counts represent source-labelled fault vehicles. `real_battery_per_vehicle.csv` contains every evaluated source vehicle and its prediction so both missed and detected cases can be reviewed. Event dates, event-level recall, and fault-type metrics are unavailable because the release labels vehicles rather than precisely bounded events.",
        "Only the included brand subset is evaluated. Results do not establish transfer to Intel Lab motes or other battery populations. Very small fault counts and the source's coarse vehicle-level labels limit generalization.",
        "",
        "## Reproduction metadata",
        "",
        "```json",
        json.dumps(metadata, indent=2, sort_keys=True),
        "```",
        "",
    ]
    report_path = output / "real_battery_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
