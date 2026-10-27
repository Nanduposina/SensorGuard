"""Evaluate SensorGuard detectors on the published vehicle-labelled EV subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import pandas as pd

from sensorguard.battery_evaluation import (
    evaluate_labeled_battery_vehicles,
    load_vehicle_labeled_segments,
    sha256_file,
    write_battery_evaluation_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path, help="extracted battery_brand3 directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, help="source Figshare archive for provenance")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--estimators", type=int, default=200)
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"output directory already exists; refusing to overwrite: {args.output_dir}")

    started = time.perf_counter()
    labels = pd.read_csv(args.dataset_dir / "label" / "all_label.csv")
    frame = load_vehicle_labeled_segments(args.dataset_dir)
    load_seconds = time.perf_counter() - started
    result = evaluate_labeled_battery_vehicles(
        frame,
        seed=args.seed,
        epochs=args.epochs,
        n_estimators=args.estimators,
        threshold_quantile=args.threshold_quantile,
    )
    elapsed = time.perf_counter() - started
    archive_metadata = {}
    if args.archive:
        archive_metadata = {
            "archive_path": str(args.archive.resolve()),
            "archive_sha256": sha256_file(args.archive),
            "archive_md5": hashlib.md5(args.archive.read_bytes()).hexdigest(),
        }
    metadata = {
        "dataset": "Realistic fault detection of Li-ion battery via dynamical deep learning approach, brand 3 subset",
        "dataset_doi": "10.6084/m9.figshare.23659323.v1",
        "paper_doi": "10.1038/s41467-023-41226-5",
        "label_unit": "vehicle",
        "label_source": "all_label.csv from cited release; paper states vehicle faults are based on driver reports and engineer confirmation",
        "n_vehicles_in_label_file": int(labels.car.nunique()),
        "n_source_fault_vehicles": int(labels.label.sum()),
        "n_source_normal_vehicles": int((labels.label == 0).sum()),
        "n_charging_segments": int(len(frame)),
        "features": list(
            frame.columns.intersection(
                [
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
                ]
            )
        ),
        "ordering": "within vehicle by source mileage, charge_segment, and source filename; synthetic timestamp is an internal order key only",
        "split_strategy": "vehicle-disjoint; train uses earliest 70% mileage-ordered segments of train vehicles; validation and test use latest 20% segments of distinct vehicle groups; all fault-labelled vehicles held out for test",
        "seed": args.seed,
        "epochs": args.epochs,
        "n_estimators": args.estimators,
        "threshold_quantile_on_validation_normal_vehicles": args.threshold_quantile,
        "thresholds": result["thresholds"],
        "vehicle_groups": result["vehicle_groups"],
        "split_rows": result["split_rows"],
        "load_seconds": load_seconds,
        "total_seconds": elapsed,
        **archive_metadata,
    }
    report = write_battery_evaluation_report(result, args.output_dir, metadata)
    print(f"Loaded {len(frame):,} snippets for {labels.car.nunique()} vehicles")
    print(
        f"Source labels: {int(labels.label.sum())} fault, {int((labels.label == 0).sum())} normal"
    )
    print(f"Split snippet counts: {json.dumps(result['split_rows'], sort_keys=True)}")
    print(
        f"Vehicle groups: {json.dumps({key: len(value) for key, value in result['vehicle_groups'].items()}, sort_keys=True)}"
    )
    print(result["metrics"].to_string(index=False))
    print(f"Elapsed: {elapsed:.2f}s; report: {report}")


if __name__ == "__main__":
    main()
