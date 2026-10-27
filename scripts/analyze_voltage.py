"""Run a streaming, temperature-adjusted voltage candidate investigation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sensorguard.historical import voltage_window_candidates
from sensorguard.ingestion import iter_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="data.txt/data.txt")
    parser.add_argument("--output-dir", default="artifacts/voltage-investigation")
    parser.add_argument("--window-days", type=int, default=7)
    parser.add_argument("--minimum-observations", type=int, default=100)
    parser.add_argument("--decline-threshold", type=float, default=-0.005)
    args = parser.parse_args()
    source = Path(args.input)
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        parser.error(f"output directory already contains files: {output}; choose a new path")
    output.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    cases = voltage_window_candidates(
        iter_records(source),
        window_days=args.window_days,
        minimum_rows=args.minimum_observations,
        decline_threshold_v_per_day=args.decline_threshold,
    )
    csv_path = output / "historical_voltage_windows.csv"
    report_path = output / "voltage_investigation.md"
    metadata_path = output / "voltage_investigation.json"
    cases.to_csv(csv_path, index=False)
    metadata = {
        "input": str(source.resolve()),
        "input_bytes": source.stat().st_size,
        "dataset_sha256": digest.hexdigest(),
        "window_days": args.window_days,
        "minimum_observations": args.minimum_observations,
        "decline_threshold_v_per_day": args.decline_threshold,
        "windows_evaluated": len(cases),
        "sensor_ids": sorted(int(value) for value in cases.sensor_id.unique())
        if not cases.empty
        else [],
        "candidate_windows": int(cases.candidate_decline.sum()) if not cases.empty else 0,
        "ground_truth": "No independently verified battery-failure labels are available.",
        "result_csv": csv_path.name,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Historical voltage-trend investigation",
        "",
        "This is an unlabeled candidate analysis, not battery-failure detection. The official Intel Lab documentation states that voltage variation is strongly correlated with temperature and that these lithium-ion cells generally maintain fairly constant voltage. The analysis fits `voltage ~ temperature + time` independently inside each fixed calendar window and reports the partial time coefficient after adjusting linearly for temperature.",
        "",
        f"Dataset SHA-256: `{digest.hexdigest()}`",
        f"Window length: {args.window_days} days; minimum observations per sensor/window: {args.minimum_observations}.",
        f"Candidate rule: adjusted voltage slope < {args.decline_threshold:g} V/day.",
        f"Evaluated windows: {len(cases)}; candidate-decline windows: {metadata['candidate_windows']}.",
        "",
        "No independent battery, maintenance, or replacement labels were found. Temperature effects may be nonlinear; load, sensor calibration, gaps, source truncation, and other environmental changes remain alternatives. The threshold is an exploratory screen and does not establish a failure.",
        "",
        "## Windows",
        "",
        "```csv\n" + cases.to_csv(index=False).rstrip() + "\n```"
        if not cases.empty
        else "_(no sensor windows met the minimum-observation requirement)_",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
