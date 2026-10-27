"""Run bounded-memory fault injection, comparisons, plots and voltage cases."""

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sensorguard.experiment import evaluate_track
from sensorguard.historical import voltage_case_studies, voltage_window_candidates
from sensorguard.ingestion import iter_records


def load_reservoir(path, sensors, limit, seed):
    # Keep a contiguous prefix per sensor in source order rather than random
    # reservoir points: random thinning would manufacture sampling gaps.
    buckets = defaultdict(list)
    seen = defaultdict(int)
    for row in iter_records(path):
        sid = row.get("moteid")
        if sid not in sensors or row.get("timestamp") is None:
            continue
        seen[sid] += 1
        bucket = buckets[sid]
        if len(bucket) < limit:
            bucket.append(row)
    rows = [r for sid in sorted(buckets) for r in buckets[sid]]
    return pd.DataFrame(rows), dict(seen)


def markdown_table(frame):
    if frame.empty:
        return "_(no rows)_"

    def cell(value):
        if pd.isna(value):
            return "n/a"
        return str(value).replace("|", "\\|").replace("\n", " ")

    cols = list(frame.columns)
    rows = [[cell(v) for v in row] for row in frame.itertuples(index=False, name=None)]
    return (
        "| "
        + " | ".join(cols)
        + " |\n| "
        + " | ".join(["---"] * len(cols))
        + " |\n"
        + "\n".join("| " + " | ".join(row) + " |" for row in rows)
    )


def alert_breakdowns(predictions):
    """Count deduplicated alert pairs and individual rule contributions."""
    alerts = predictions[predictions.predicted].drop_duplicates(
        ["detector", "timestamp", "sensor_id", "feature"]
    )
    by_sensor_feature = (
        alerts.groupby(["detector", "sensor_id", "feature"], as_index=False)
        .size()
        .rename(columns={"size": "alert_pairs"})
    )
    # Count distinct sensor-timestamp readings explicitly because multivariate
    # detector scores are projected across feature rows for investigation.
    reading_counts = []
    for detector, group in alerts.groupby("detector", sort=True):
        feature_pairs = group[["timestamp", "sensor_id", "feature"]].drop_duplicates().shape[0]
        readings = group[["timestamp", "sensor_id"]].drop_duplicates().shape[0]
        reading_counts.append(
            {
                "detector": detector,
                "feature_alert_pairs": feature_pairs,
                "unique_alerted_readings": readings,
            }
        )
    by_detector_reading = pd.DataFrame(
        reading_counts,
        columns=["detector", "feature_alert_pairs", "unique_alerted_readings"],
    )
    by_detector_reading["feature_pairs_per_reading"] = (
        by_detector_reading.feature_alert_pairs / by_detector_reading.unique_alerted_readings
    )
    rules = predictions[
        predictions.detector.eq("rules") & predictions.predicted & predictions.rule_id.notna()
    ]
    contributions = []
    for row in rules.itertuples():
        for rule_id in str(row.rule_id).split("; "):
            contributions.append(
                {
                    "rule_id": rule_id,
                    "sensor_id": row.sensor_id,
                    "feature": row.feature,
                    "timestamp": row.timestamp,
                }
            )
    rule_rows = pd.DataFrame(contributions)
    if rule_rows.empty:
        by_rule = pd.DataFrame(columns=["rule_id", "sensor_id", "feature", "alert_pairs"])
    else:
        by_rule = (
            rule_rows.drop_duplicates(["rule_id", "sensor_id", "feature", "timestamp"])
            .groupby(["rule_id", "sensor_id", "feature"], as_index=False)
            .size()
            .rename(columns={"size": "alert_pairs"})
        )
    return by_sensor_feature, by_rule, by_detector_reading


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data.txt/data.txt")
    parser.add_argument("--output-dir", default="artifacts/experiment")
    parser.add_argument("--sensors", default="1,2,3,4")
    parser.add_argument("--max-per-sensor", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--threshold-quantile", type=float, default=0.99)
    parser.add_argument(
        "--contextual",
        action="store_true",
        help="add causal, training-baseline-adjusted statistical peer features",
    )
    parser.add_argument(
        "--peer-map",
        help="optional JSON mapping target sensor IDs to allowed peer sensor IDs",
    )
    parser.add_argument("--peer-radius-meters", type=float, default=8.0)
    parser.add_argument("--peer-minimum", type=int, default=3)
    parser.add_argument("--peer-tolerance", default="120s")
    args = parser.parse_args()
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Output directory already contains files: {out}; choose a new path.")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    digest = hashlib.sha256()
    with Path(args.input).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    sensors = {int(x.strip()) for x in args.sensors.split(",") if x.strip()}
    frame, counts = load_reservoir(args.input, sensors, args.max_per_sensor, args.seed)
    if frame.empty:
        raise SystemExit("No rows for requested sensors")
    frame["sensor_id"] = frame["moteid"]
    # The source contains isolated far-away timestamps. Bound this controlled
    # subset to a dense temporal neighborhood so a single bad clock value
    # cannot define all chronological partitions or manufacture month-long gaps.
    center = pd.to_datetime(frame.timestamp).median()
    lo = center - pd.Timedelta(days=1)
    hi = center + pd.Timedelta(days=1)
    before_filter = len(frame)
    frame = frame[pd.to_datetime(frame.timestamp).between(lo, hi)].copy()
    excluded_outside_dense_window = before_filter - len(frame)
    frame = frame.sort_values(["timestamp", "sensor_id"], kind="stable").reset_index(drop=True)
    peer_map = None
    if args.peer_map:
        peer_map_payload = json.loads(Path(args.peer_map).read_text(encoding="utf-8"))
        peer_map = {
            int(sensor): tuple(int(peer) for peer in peers)
            for sensor, peers in peer_map_payload.items()
        }
        if not args.contextual:
            parser.error("--peer-map requires --contextual")
    result = evaluate_track(
        frame,
        seed=args.seed,
        threshold_quantile=args.threshold_quantile,
        autoencoder_epochs=args.epochs,
        n_estimators=100,
        use_contextual_features=args.contextual,
        peer_neighborhoods=peer_map,
        peer_radius_m=args.peer_radius_meters,
        peer_tolerance=args.peer_tolerance,
        minimum_peers=args.peer_minimum,
    )
    alert_breakdown, rule_breakdown, reading_alert_summary = alert_breakdowns(
        result["prediction_details"]
    )
    historical = voltage_case_studies(frame)
    voltage_windows = voltage_window_candidates(
        frame.to_dict(orient="records"), window_days=7, minimum_rows=50
    )
    tables = {
        # Reuse this bounded, unmodified chronological sample with the training
        # CLI. Keeping it beside the evaluation artifacts makes the exact model
        # input inspectable and avoids materializing the complete raw source.
        "sampled_records": frame,
        "metrics": result["metrics"],
        "per_fault_type": result["per_fault"],
        "per_sensor": result["per_sensor"],
        "alert_breakdown": alert_breakdown,
        "rule_breakdown": rule_breakdown,
        "reading_alert_summary": reading_alert_summary,
        "clean_baseline": result["clean_baseline"],
        "fault_manifest": result["fault_manifest"],
        "injection_observations": result["injection_observations"],
        "prediction_details": result["prediction_details"],
        "clean_test_prediction_details": result["clean_prediction_details"],
        "test_records": result["test_frame"],
        "injected_test_records": result["injected_frame"],
        "errors": result["error_cases"],
        "detection_delay": result["detection_delay"],
        "historical_voltage_candidates": historical,
        "historical_voltage_windows": voltage_windows,
        "contextual_coverage": result["contextual_coverage"],
    }
    curves = []
    for detector, curve in result["pr_curves"].items():
        item = curve.copy()
        item.insert(0, "detector", detector)
        curves.append(item)
    tables["precision_recall_curves"] = (
        pd.concat(curves, ignore_index=True)
        if curves
        else pd.DataFrame(columns=["detector", "precision", "recall", "threshold"])
    )
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    (out / "experiment_metadata.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "dataset_sha256": digest.hexdigest(),
                "contextual_features_enabled": args.contextual,
                "contextual_feature_columns": result["contextual_features"],
                "contextual_peer_sensors_used": len(
                    set(
                        result["contextual_coverage"].loc[
                            result["contextual_coverage"].context_available > 0,
                            "sensor_id",
                        ]
                    )
                    if args.contextual and not result["contextual_coverage"].empty
                    else set()
                ),
                "peer_neighborhoods": {
                    str(sensor): list(peers)
                    for sensor, peers in result["peer_neighborhoods"].items()
                },
                "contextual_reference_method": (
                    "user-supplied sensor neighborhood mapping"
                    if args.peer_map
                    else f"official physical mote coordinates, Euclidean radius {args.peer_radius_meters:g} m"
                    if args.contextual
                    else "disabled"
                ),
                "peer_radius_meters": args.peer_radius_meters,
                "peer_minimum": args.peer_minimum,
                "peer_tolerance": args.peer_tolerance,
                "end_to_end_seconds": None,
                "sample_per_sensor_cap": args.max_per_sensor,
                "sampling_method": "first N source-order records per selected sensor, then chronological sort; preserves local sampling cadence better than random thinning",
                "source_rows_seen_per_sensor": counts,
                "selected_sensors": sorted(sensors),
                "sampled_rows": len(frame),
                "excluded_outside_dense_window": excluded_outside_dense_window,
                "dense_window": [str(lo), str(hi)],
                "split_sizes": result["split_sizes"],
                "thresholds": result["thresholds"],
                "training_range": result["training_range"],
                "validation_range": result["validation_range"],
                "test_range": result["test_range"],
                "evaluation_unit": "sensor-feature-reading pair including record marker",
                "threshold_method": "Isolation Forest and autoencoder validation-score quantiles; per-feature robust rule-deviation quantiles using training-only scale floors; no injected labels used in calibration.",
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    # High-resolution paper-ready comparison plots.
    metrics = result["metrics"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(metrics))
    width = 0.32
    for i, col in enumerate(["recall", "unlabeled_alert_rate"]):
        ax.bar(x + (i - 1) * width, metrics[col], width, label=col.title())
    ax.set_xticks(x, metrics.detector, rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Known injected-pair recall and unlabeled alert rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "detector_comparison.png", dpi=300)
    plt.close(fig)
    if not result["per_fault"].empty:
        pivot = (
            result["per_fault"]
            .pivot(index="fault_type", columns="detector", values="injection_recall")
            .fillna(0)
        )
        ax = pivot.plot(
            kind="bar",
            figsize=(10, 5),
            ylim=(0, 1),
            ylabel="Recall",
            title="Recall by injected fault type",
        )
        ax.figure.tight_layout()
        ax.figure.savefig(out / "fault_type_recall.png", dpi=300)
        plt.close(ax.figure)
    if result["pr_curves"]:
        fig, ax = plt.subplots(figsize=(7, 6))
        for detector, curve in result["pr_curves"].items():
            ax.plot(curve.recall, curve.precision, label=detector)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-recall curves on injected test pairs")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / "precision_recall.png", dpi=300)
        plt.close(fig)
    if not frame.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        for sid, g in frame.groupby("sensor_id"):
            ax.scatter(g.temperature, g.voltage, s=8, alpha=0.45, label=f"Sensor {sid}")
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Voltage (V)")
        ax.set_title("Historical voltage and temperature (unlabeled sample)")
        ax.legend(ncol=2)
        fig.tight_layout()
        fig.savefig(out / "historical_voltage_temperature.png", dpi=300)
        plt.close(fig)
    report = [
        "# SensorGuard controlled evaluation",
        "",
        "## Method",
        "",
        f"Seed: {args.seed}. Streamed up to the first {args.max_per_sensor} source-order records per requested sensor, then retained {len(frame):,} records within one day of the sample median timestamp (excluded {excluded_outside_dense_window} isolated records outside that dense window) and sorted chronologically. This bounded subset better preserves local sampling cadence than random thinning. Source data remains untouched.",
        f"Chronological partitions: {result['split_sizes']}. Training range: {result['training_range']}; validation: {result['validation_range']}; held-out test: {result['test_range']}.",
        "Injected faults are applied only to a copy of the held-out test set. Detector fitting uses unmodified training rows. Isolation Forest, autoencoder, and statistical-rule cutoffs are calibrated using unmodified validation score distributions only; injected labels are not used for fitting or calibration. A validation quantile is an operating-point choice on unlabeled data, not a claim that validation rows are normal.",
        (
            "Contextual mode uses backward-aligned peers within the configured physical radius from official coordinates (or the explicit user map), training-only per-sensor/feature medians, the configured maximum age, and the configured minimum peer count. Each reading is excluded from its own reference; same-time peer values are allowed; future peer observations are excluded. This is physical proximity, not proof of shared environment. Peer availability and alignment summaries are exported separately."
            if args.contextual
            else "Contextual peer features are disabled for this run. Use --contextual for the causal physical-neighborhood reference ablation."
        ),
        "Evaluation unit is the sensor-feature-reading pair; timestamp-only edits use a `record` marker. Only directly modified pairs have known positive labels. All other historical pairs remain unlabeled and are not called true negatives. Alert rows are deduplicated per detector and pair. Injected-pair recall and missed injected pairs are measurable, but precision, F1, false-positive rate, true negatives, and PR curves are not reported for this partly labeled test set. Alerts on unlabeled pairs are counted descriptively, not called false positives. Synthetic distributions do not represent natural failure prevalence or failure dynamics.",
        "Historical cases below are exploratory candidates from the observed records, not labeled faults. Voltage is regressed against temperature per sensor before estimating the residual time trend; this linear adjustment is not causal. Dataset documentation: https://db.csail.mit.edu/labdata/labdata.html.",
        "",
        "## Overall metrics",
        "",
        markdown_table(metrics),
        "",
        "## Alerts on unmodified unlabeled test readings",
        "",
        markdown_table(result["clean_baseline"]),
        "",
        "Unmodified pairs are unlabeled. Their alert counts and rates are descriptive only, not false positives or false-positive rates. Natural anomalies may be present. Treat these results as a diagnostic of this short sample and selected synthetic edits, not a deployable operating point.",
        "## Alerts by sensor and feature",
        "",
        markdown_table(alert_breakdown),
        "",
        "## Rule contributions",
        "",
        "Rule contribution counts may overlap when multiple rules fire on one reading-feature pair.",
        markdown_table(rule_breakdown),
        "",
        "## Feature alert pairs and distinct alerted readings",
        "",
        "Multivariate detector decisions are record-level scores projected to their present feature rows for investigation. Feature-pair counts can therefore exceed distinct sensor-timestamp readings; this table makes both counts explicit.",
        markdown_table(reading_alert_summary),
        "",
        "",
        "## Per-fault-type metrics",
        "",
        markdown_table(result["per_fault"]),
        "",
        "## Drift/outage detection delay",
        "",
        markdown_table(result["detection_delay"]),
        "",
        "## Injected fault events",
        "",
        markdown_table(result["fault_manifest"]),
        "",
    ]
    if args.contextual:
        report += [
            "## Contextual reference availability",
            "",
            markdown_table(result["contextual_coverage"]),
            "",
            "Contextual scores are only calculated when the target has a training baseline and at least three prior peer sensors within 120 seconds. A missing score means insufficient context; it is not scored as normal.",
            "",
        ]
    report += ["## Historical voltage candidate summary", ""]
    if historical.empty:
        report.append(
            "No sensor had enough complete voltage and temperature records in the sample."
        )
    else:
        report.append(markdown_table(historical))
    if not historical.empty:
        declines = int(historical.candidate_decline.sum())
        abrupt = int(historical.candidate_abrupt_change.sum())
        report.append(
            f"In this sample, {declines} sensors met the exploratory temperature-adjusted decline threshold and {abrupt} met the abrupt-step threshold. These flags remain unverified candidate cases."
        )
    report += ["", "## Temperature-adjusted voltage candidate windows", ""]
    if voltage_windows.empty:
        report.append(
            "No seven-day sensor window met the minimum of 50 complete readings in this bounded sample."
        )
    else:
        report.append(markdown_table(voltage_windows))
    report += [
        "",
        "No historical pattern is called a confirmed battery failure. Temperature, sensor load, environmental conditions, calibration drift, missing intervals, and data truncation are alternative explanations. Independent battery or maintenance ground truth is not available.",
        "",
        "Synthetic fault injection enables measured detection recall against known edits but has a distribution mismatch: injected spans, amplitudes and fault types are chosen by the experimenter, whereas real failures can be gradual, correlated, intermittent, or absent from this dataset. No natural anomaly labels are inferred.",
        "",
    ]
    (out / "experiment_report.md").write_text("\n".join(report), encoding="utf-8")
    metadata_path = out / "experiment_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["end_to_end_seconds"] = time.perf_counter() - started
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(out),
                "sampled_rows": len(frame),
                "split_sizes": result["split_sizes"],
                "faults": len(result["fault_manifest"]),
                "metrics": result["metrics"].to_dict(orient="records"),
                "voltage_candidates": len(historical),
            },
            default=str,
        )
    )


if __name__ == "__main__":
    main()
