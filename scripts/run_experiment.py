"""Run bounded-memory fault injection, comparisons, plots and voltage cases."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sensorguard.experiment import evaluate_track
from sensorguard.historical import voltage_case_studies
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data.txt/data.txt")
    parser.add_argument("--output-dir", default="artifacts/experiment")
    parser.add_argument("--sensors", default="1,2,3,4")
    parser.add_argument("--max-per-sensor", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--threshold-quantile", type=float, default=0.99)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
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
    result = evaluate_track(
        frame,
        seed=args.seed,
        threshold_quantile=args.threshold_quantile,
        autoencoder_epochs=args.epochs,
        n_estimators=100,
    )
    historical = voltage_case_studies(frame)
    tables = {
        "metrics": result["metrics"],
        "per_fault_type": result["per_fault"],
        "per_sensor": result["per_sensor"],
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
                "threshold_method": "Isolation Forest validation-score quantile; autoencoder aggregate validation reconstruction-error quantile; no injected labels used in calibration.",
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
    width = 0.24
    for i, col in enumerate(["precision", "recall", "f1"]):
        ax.bar(x + (i - 1) * width, metrics[col], width, label=col.title())
    ax.set_xticks(x, metrics.detector, rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Controlled injection detection (feature-reading pairs)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "detector_comparison.png", dpi=300)
    plt.close(fig)
    if not result["per_fault"].empty:
        pivot = (
            result["per_fault"]
            .pivot(index="fault_type", columns="detector", values="recall")
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
        "Injected faults are applied only to a copy of the held-out test set. Rule and model fitting use unmodified training rows. Isolation Forest cutoff and autoencoder reconstruction threshold use the unmodified validation score distribution only. Unmodified does not mean verified fault-free because natural anomaly labels are unavailable.",
        "Evaluation unit is the sensor-feature-reading pair; timestamp-only faults use the `record` feature marker for one reading. Missing predictions count as negative. Injected points have known ground truth; every unmodified pair is treated as negative for this controlled score, although natural anomalies may be present. Alert rows are reduced to one decision per pair before counting. Thus precision and FPR are relative to injected labels and FPR is apparent on unlabeled negatives. Combined systems use OR rules, but their detectors have incomparable score scales, so combined PR-AUC is not reported. Synthetic distributions do not represent natural failure prevalence or failure dynamics.",
        "Historical cases below are exploratory candidates from the observed records, not labeled faults. Voltage is regressed against temperature per sensor before estimating the residual time trend; this linear adjustment is not causal. Dataset documentation: https://db.csail.mit.edu/labdata/labdata.html.",
        "",
        "## Overall metrics",
        "",
        markdown_table(metrics),
        "",
        "## Unmodified test alert baseline (unlabeled)",
        "",
        markdown_table(result["clean_baseline"]),
        "",
        "The unmodified subset has substantial apparent alert rates. These pairs have no injected labels, but they are not verified true negatives; natural anomalies can appear in this set. The combined union recovers more injected positives at the cost of more alerts. Treat the metrics as a diagnostic of this short sample and selected synthetic edits, not a deployable operating point or a verified historical FPR.",
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
        "## Historical voltage candidate summary",
        "",
    ]
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
    report += [
        "",
        "No historical pattern is called a confirmed battery failure. Temperature, sensor load, environmental conditions, calibration drift, missing intervals, and data truncation are alternative explanations. Independent battery or maintenance ground truth is not available.",
        "",
        "Synthetic fault injection enables measured recall against known edits but has a distribution mismatch: injected spans, amplitudes and fault types are chosen by the experimenter, whereas real failures can be gradual, correlated, intermittent, or absent from this dataset. No natural anomaly labels are inferred.",
        "",
    ]
    (out / "experiment_report.md").write_text("\n".join(report), encoding="utf-8")
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
