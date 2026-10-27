"""Compare paired baseline/context experiment outputs from identical inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def compare_runs(baseline_dir: Path, contextual_dir: Path, output: Path) -> pd.DataFrame:
    """Validate paired run identity, then compare known recall and unknown alert rate."""
    if output.exists():
        raise FileExistsError(f"comparison output already exists: {output}")
    baseline_meta = json.loads(
        (baseline_dir / "experiment_metadata.json").read_text(encoding="utf-8")
    )
    context_meta = json.loads(
        (contextual_dir / "experiment_metadata.json").read_text(encoding="utf-8")
    )
    for key in ("seed", "dataset_sha256", "selected_sensors", "sampled_rows", "split_sizes"):
        if baseline_meta.get(key) != context_meta.get(key):
            raise ValueError(f"runs are not paired; metadata differs for {key}")
    base_sample = (baseline_dir / "sampled_records.csv").read_bytes()
    context_sample = (contextual_dir / "sampled_records.csv").read_bytes()
    if base_sample != context_sample:
        raise ValueError("runs are not paired; sampled_records.csv differs")
    base_faults = pd.read_csv(baseline_dir / "fault_manifest.csv")
    context_faults = pd.read_csv(contextual_dir / "fault_manifest.csv")
    pd.testing.assert_frame_equal(base_faults, context_faults, check_dtype=False)

    metrics = [
        "detector",
        "evaluated_records",
        "evaluated_pairs",
        "fault_events",
        "positive_pairs",
        "true_positives",
        "false_negatives",
        "recall",
        "total_alerts",
        "unlabeled_pairs",
        "alerts_on_unlabeled_pairs",
        "unlabeled_alert_rate",
    ]
    base = pd.read_csv(baseline_dir / "metrics.csv")[metrics].copy()
    context = pd.read_csv(contextual_dir / "metrics.csv")[metrics].copy()
    context["detector"] = context.detector.str.removesuffix("_contextual")
    base = base.add_prefix("baseline_").rename(columns={"baseline_detector": "detector"})
    context = context.add_prefix("contextual_").rename(columns={"contextual_detector": "detector"})
    result = base.merge(context, on="detector", how="inner", validate="one_to_one")
    result["recall_change"] = result.contextual_recall - result.baseline_recall
    result["unlabeled_alert_rate_change"] = (
        result.contextual_unlabeled_alert_rate - result.baseline_unlabeled_alert_rate
    )
    coverage = pd.read_csv(contextual_dir / "contextual_coverage.csv")
    result["contextual_sensors_scored"] = int(coverage.sensor_id.nunique())
    result["contextual_sensors_with_any_context"] = int(
        coverage.loc[coverage.context_available > 0, "sensor_id"].nunique()
    )
    result["contextual_feature_availability_mean"] = float(coverage.availability_rate.mean())
    result["baseline_runtime_seconds"] = float(baseline_meta["end_to_end_seconds"])
    result["contextual_runtime_seconds"] = float(context_meta["end_to_end_seconds"])
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--contextual-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        result = compare_runs(Path(args.baseline_dir), Path(args.contextual_dir), Path(args.output))
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        parser.exit(2, f"contextual comparison failed: {exc}\n")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
