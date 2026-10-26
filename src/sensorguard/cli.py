"""Command-line entry point for fitting detectors and exporting alerts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .autoencoder import AutoencoderDetector
from .detection import merge_alerts
from .evaluation import chronological_split
from .ingestion import iter_records
from .isolation import IsolationForestDetector
from .rules import RuleDetector


def main() -> None:
    """Fit detectors on an early chronological split and write model artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Plain/gzip sensor text or parsed CSV")
    parser.add_argument("--output", default="artifacts/detections.csv")
    parser.add_argument("--model-dir", default="artifacts/models")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    source = Path(args.input)
    frame = (
        pd.read_csv(source)
        if source.suffix.lower() == ".csv"
        else pd.DataFrame(iter_records(source))
    )
    if frame.empty:
        raise SystemExit("Input contains no records")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    train, validation, test = chronological_split(frame)

    isolation_forest = IsolationForestDetector(random_state=42).fit(train)
    autoencoder = AutoencoderDetector(epochs=args.epochs).fit(train, validation=validation)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    isolation_forest.save(model_dir / "isolation_forest.joblib")
    autoencoder.save(model_dir / "autoencoder.pt")

    detectors = [
        RuleDetector().fit(train).predict(test),
        isolation_forest.predict(test),
        autoencoder.predict(test),
    ]
    result = merge_alerts(*detectors)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(target, index=False)
    print(
        json.dumps(
            {
                "train_rows": len(train),
                "validation_rows": len(validation),
                "test_rows": len(test),
                "alerts": int(result.is_anomaly.sum()),
                "output": str(target),
                "model_dir": str(model_dir),
            }
        )
    )
