"""Stream the source file through explicit cleaning policies and provenance exports."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sensorguard.cleaning import clean_frame, cleaning_summary
from sensorguard.config import SensorGuardConfig
from sensorguard.ingestion import iter_records


def _read_statistics(path: Path | None) -> dict[str, dict[int | None, float]] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for feature, values in payload.items():
        result[feature] = {
            None if sensor in {"global", "null", "None"} else int(sensor): float(value)
            for sensor, value in values.items()
        }
    return result


def run_cleaning(
    input_path: Path,
    output_dir: Path,
    *,
    chunk_size: int = 50_000,
    max_rows: int | None = None,
    policies: dict[str, str] | None = None,
    imputation_statistics: dict[str, dict[int | None, float]] | None = None,
) -> dict[str, object]:
    """Process any-size input with chunk-bounded frames and cross-chunk issue state."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory already contains files: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = SensorGuardConfig()
    policy = policies or {}
    clean_path = output_dir / "cleaned.csv"
    provenance_path = output_dir / "provenance.csv"
    report_json = output_dir / "cleaning_report.json"
    report_md = output_dir / "cleaning_report.md"
    counts: Counter[str] = Counter()
    policy_counts: Counter[str] = Counter()
    rows_total = rows_retained = 0
    truncated = False
    last_by_sensor: dict[int, pd.Timestamp] = {}
    chunks: list[dict] = []
    headers_written = False

    with tempfile.TemporaryDirectory(prefix="sensorguard-clean-") as temporary:
        state = sqlite3.connect(Path(temporary) / "seen.sqlite3")
        state.execute("CREATE TABLE seen_keys (sensor INTEGER, ts TEXT, PRIMARY KEY(sensor, ts))")
        state.execute("CREATE TABLE seen_raw (digest TEXT PRIMARY KEY)")
        state.execute(
            "CREATE TEMP TABLE batch_records "
            "(pos INTEGER PRIMARY KEY, digest TEXT NOT NULL, sensor INTEGER, ts TEXT)"
        )
        try:
            for record in iter_records(input_path):
                if max_rows is not None and rows_total + len(chunks) >= max_rows:
                    truncated = True
                    break
                chunks.append(record)
                if len(chunks) < chunk_size:
                    continue
                batch = pd.DataFrame.from_records(chunks)
                _add_cross_chunk_issues(batch, state, last_by_sensor, cfg)
                _write_clean_chunk(
                    batch,
                    output_dir,
                    policy,
                    imputation_statistics,
                    counts,
                    policy_counts,
                    clean_path,
                    provenance_path,
                    not headers_written,
                )
                rows_total += len(batch)
                rows_retained += int(batch.attrs.get("retained_rows", 0))
                headers_written = True
                chunks.clear()
            if chunks:
                batch = pd.DataFrame.from_records(chunks)
                _add_cross_chunk_issues(batch, state, last_by_sensor, cfg)
                _write_clean_chunk(
                    batch,
                    output_dir,
                    policy,
                    imputation_statistics,
                    counts,
                    policy_counts,
                    clean_path,
                    provenance_path,
                    not headers_written,
                )
                rows_total += len(batch)
                rows_retained += int(batch.attrs.get("retained_rows", 0))
                headers_written = True
        finally:
            state.close()

    if not headers_written:
        raise ValueError("input contained no records")
    result = {
        "input": str(input_path.resolve()),
        "input_bytes": input_path.stat().st_size,
        "input_rows": rows_total,
        "input_truncated": truncated,
        "retained_rows": rows_retained,
        "excluded_rows": rows_total - rows_retained,
        "policies": dict(sorted(policy.items())),
        "issue_counts": dict(sorted(counts.items())),
        "policy_counts": dict(sorted(policy_counts.items())),
        "cleaned_csv": clean_path.name,
        "provenance_csv": provenance_path.name,
        "duplicate_scope": "exact raw-line duplicates and duplicate (timestamp, sensor_id) keys across the full input",
        "timestamp_order_scope": "per-sensor source order across chunk boundaries",
    }
    report_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# SensorGuard cleaning report",
        "",
        f"Source: `{input_path.resolve()}` ({input_path.stat().st_size:,} bytes)",
        f"Input rows: {rows_total:,}",
        f"Input truncated at configured row cap: {truncated}",
        f"Retained rows: {rows_retained:,}",
        f"Excluded rows: {rows_total - rows_retained:,}",
        f"Chunk size: {chunk_size:,}",
        f"Policies: `{json.dumps(dict(sorted(policy.items())), sort_keys=True)}`",
        "",
        "Every source row is present in `provenance.csv`; excluded rows remain auditable there. `cleaned.csv` contains retained rows. Imputation, when explicitly selected, uses only the supplied training-fitted statistics and does not interpolate across time.",
        "",
        "## Issues",
        "",
        "| Issue | Count |",
        "|---|---:|",
        *[f"| {issue} | {count:,} |" for issue, count in sorted(counts.items())],
        "",
        "## Policy actions",
        "",
        "| Action | Affected issue observations |",
        "|---|---:|",
        *[f"| {action} | {count:,} |" for action, count in sorted(policy_counts.items())],
        "",
        "No measurements were interpolated. Implausibility limits are configurable screening rules, not ground-truth failure labels.",
        "",
    ]
    report_md.write_text("\n".join(lines), encoding="utf-8")
    return result


def _add_cross_chunk_issues(
    batch: pd.DataFrame,
    state: sqlite3.Connection,
    last_by_sensor: dict[int, pd.Timestamp],
    config: SensorGuardConfig,
) -> None:
    """Flag cross-chunk duplicates and source-order events using batch operations."""
    issues = [list(value) if isinstance(value, list) else [] for value in batch["issues"]]
    gap_values = [float("nan")] * len(batch)
    raw_values = batch.get("raw", pd.Series([""] * len(batch))).fillna("").astype(str)
    timestamp_values = pd.to_datetime(batch["timestamp"], errors="coerce")
    sensor_values = batch["moteid"].to_numpy()
    stage_rows = []
    for position, (raw, timestamp, sensor) in enumerate(
        zip(raw_values, timestamp_values, sensor_values, strict=True)
    ):
        digest = hashlib.blake2b(raw.encode("utf-8"), digest_size=16).hexdigest()
        if pd.isna(timestamp) or pd.isna(sensor):
            stage_rows.append((position, digest, None, None))
        else:
            stage_rows.append((position, digest, int(sensor), pd.Timestamp(timestamp).isoformat()))

    state.execute("DELETE FROM batch_records")
    state.executemany("INSERT INTO batch_records VALUES (?, ?, ?, ?)", stage_rows)
    duplicate_raw = state.execute(
        "WITH ranked AS ("
        " SELECT pos, digest, ROW_NUMBER() OVER (PARTITION BY digest ORDER BY pos) AS rank "
        " FROM batch_records) "
        "SELECT pos FROM ranked WHERE rank > 1 "
        "OR EXISTS (SELECT 1 FROM seen_raw s WHERE s.digest = ranked.digest)"
    ).fetchall()
    for (position,) in duplicate_raw:
        issues[position].append("duplicate_record")
    state.execute("INSERT OR IGNORE INTO seen_raw SELECT digest FROM batch_records")

    duplicate_keys = state.execute(
        "WITH ranked AS ("
        " SELECT pos, sensor, ts, ROW_NUMBER() OVER "
        " (PARTITION BY sensor, ts ORDER BY pos) AS rank "
        " FROM batch_records WHERE sensor IS NOT NULL AND ts IS NOT NULL) "
        "SELECT pos FROM ranked WHERE rank > 1 OR EXISTS ("
        " SELECT 1 FROM seen_keys s WHERE s.sensor = ranked.sensor AND s.ts = ranked.ts)"
    ).fetchall()
    for (position,) in duplicate_keys:
        issues[position].append("duplicate_key")
    state.execute(
        "INSERT OR IGNORE INTO seen_keys "
        "SELECT sensor, ts FROM batch_records WHERE sensor IS NOT NULL AND ts IS NOT NULL"
    )

    valid_positions = [
        pos
        for pos, (sensor, timestamp) in enumerate(zip(sensor_values, timestamp_values, strict=True))
        if pd.notna(sensor) and pd.notna(timestamp)
    ]
    if valid_positions:
        valid = pd.DataFrame(
            {
                "position": valid_positions,
                "sensor": [int(sensor_values[pos]) for pos in valid_positions],
                "timestamp": [pd.Timestamp(timestamp_values.iloc[pos]) for pos in valid_positions],
            }
        )
        for sensor, group in valid.groupby("sensor", sort=False):
            deltas = group.timestamp.diff().dt.total_seconds().tolist()
            previous = last_by_sensor.get(int(sensor))
            if previous is not None:
                deltas[0] = (group.timestamp.iloc[0] - previous).total_seconds()
            for position, delta in zip(group.position.tolist(), deltas, strict=True):
                if pd.isna(delta):
                    continue
                if delta < 0:
                    issues[position].append("out_of_order")
                else:
                    gap_values[position] = delta
                    if delta > config.sampling_gap_seconds:
                        issues[position].append("sampling_gap")
            last_by_sensor[int(sensor)] = group.timestamp.iloc[-1]

    batch["issues"] = issues
    batch["sampling_gap_seconds"] = gap_values
    state.commit()


def _write_clean_chunk(
    batch: pd.DataFrame,
    output_dir: Path,
    policies: dict[str, str],
    statistics: dict[str, dict[int | None, float]] | None,
    issue_counts: Counter[str],
    policy_counts: Counter[str],
    clean_path: Path,
    provenance_path: Path,
    write_header: bool,
) -> None:
    cleaned = clean_frame(
        batch,
        policies=policies,
        imputation_statistics=statistics,
    )
    summary = cleaning_summary(cleaned)
    issue_counts.update(summary["issue_counts"])
    policy_counts.update(summary["policy_counts"])
    cleaned.loc[cleaned.retained].to_csv(
        clean_path,
        mode="w" if write_header else "a",
        header=write_header,
        index=False,
        date_format="%Y-%m-%dT%H:%M:%S.%f",
    )
    cleaned.to_csv(
        provenance_path,
        mode="w" if write_header else "a",
        header=write_header,
        index=False,
        date_format="%Y-%m-%dT%H:%M:%S.%f",
    )
    batch.attrs["retained_rows"] = int(cleaned.retained.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="data.txt/data.txt")
    parser.add_argument("--output-dir", default="artifacts/cleaning")
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument(
        "--max-rows", type=int, help="optional source-order cap for a bounded smoke run"
    )
    parser.add_argument(
        "--policy",
        action="append",
        default=[],
        metavar="CATEGORY=ACTION",
        help="policy action preserve|flag|drop|impute; repeat as needed",
    )
    parser.add_argument(
        "--imputation-statistics",
        help="JSON of training-only per-sensor/global medians; required when using impute",
    )
    args = parser.parse_args()
    policies = {}
    for item in args.policy:
        if "=" not in item:
            parser.error(f"invalid --policy {item!r}; expected CATEGORY=ACTION")
        category, action = item.split("=", 1)
        policies[category] = action
    try:
        result = run_cleaning(
            Path(args.input),
            Path(args.output_dir),
            chunk_size=args.chunk_size,
            max_rows=args.max_rows,
            policies=policies,
            imputation_statistics=_read_statistics(
                Path(args.imputation_statistics) if args.imputation_statistics else None
            ),
        )
    except (OSError, ValueError, sqlite3.Error) as exc:
        parser.exit(2, f"cleaning failed: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
