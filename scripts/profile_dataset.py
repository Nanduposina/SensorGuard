"""Streaming dataset inspection and measured Markdown report generation."""

import gzip
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sensorguard.ingestion import iter_records

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data.txt/data.txt")
count = 0
unlocated_rows = 0
whitespace_token_counts = Counter()
schema_slot_counts = Counter()
sensor_rows = Counter()
issue_counts = Counter()
feature_stats = defaultdict(
    lambda: {"n": 0, "sum": 0.0, "min": float("inf"), "max": float("-inf"), "missing": 0}
)
first = last = None
last_per_sensor = {}
gap_count = 0
gap_max = 0.0
line_count = 0
current_prefix_only_run = 0
trailing_prefix_only_start = None
prefix_only_records = 0
open_text = gzip.open if path.suffix == ".gz" else open
with open_text(path, "rt", encoding="utf-8", errors="replace") as stream:
    for line_count, raw in enumerate(stream, 1):
        whitespace_token_counts[len(raw.split())] += 1
        line = raw.rstrip("\r\n")
        slots = line.split(" ") if "\t" not in line else line.split()
        schema_slot_counts[len(slots)] += 1
        prefix_only = len(slots) == 8 and all(not token for token in slots[3:])
        if prefix_only:
            if current_prefix_only_run == 0:
                trailing_prefix_only_start = line_count
            current_prefix_only_run += 1
            prefix_only_records += 1
        else:
            current_prefix_only_run = 0
for row in iter_records(path):
    count += 1
    issue_counts.update(row["issues"])
    if row["moteid"] is None:
        unlocated_rows += 1
    if row["timestamp"]:
        first = min(first, row["timestamp"]) if first else row["timestamp"]
        last = max(last, row["timestamp"]) if last else row["timestamp"]
    sid = row["moteid"]
    if sid is not None:
        sensor_rows[sid] += 1
        if row["timestamp"]:
            previous = last_per_sensor.get(sid)
            if previous is not None:
                gap = (row["timestamp"] - previous).total_seconds()
                if gap > 300:
                    gap_count += 1
                    gap_max = max(gap_max, gap)
            last_per_sensor[sid] = row["timestamp"]
    for name in ("temperature", "humidity", "light", "voltage"):
        value = row[name]
        stat = feature_stats[name]
        if value is None:
            stat["missing"] += 1
        else:
            stat["n"] += 1
            stat["sum"] += value
            stat["min"] = min(stat["min"], value)
            stat["max"] = max(stat["max"], value)

report = [
    "# SensorGuard data profile",
    "",
    f"Source: `{path}` ({path.stat().st_size:,} bytes)",
    "",
    "## Integrity and coverage",
    "",
    f"- Rows: {count:,}",
    f"- Whitespace token counts (collapsed): {dict(sorted(whitespace_token_counts.items()))}",
    f"- Positional field-slot counts (repeated spaces preserved): {dict(sorted(schema_slot_counts.items()))}",
    f"- Rows with mote ID and all measurement slots explicitly blank: {prefix_only_records:,}; contiguous trailing run: {current_prefix_only_run:,} starting at line {trailing_prefix_only_start if current_prefix_only_run else 'none'}",
    f"- Rows with no parsed sensor ID: {unlocated_rows:,}; source records are retained with raw text and line number.",
    "- The blank-ID suffix is structurally positional but its physical cause is unknown; it is not assigned to a sensor.",
    f"- Parsed timestamp range: {first} to {last}",
    f"- Distinct sensor IDs: {len(sensor_rows)} (minimum {min(sensor_rows)}, maximum {max(sensor_rows)})",
    f"- Per-sensor counts: {dict(sorted(sensor_rows.items()))}",
    f"- Per-sensor consecutive input-order timestamp intervals over 300 seconds: {gap_count:,}; maximum: {gap_max:,.1f} seconds",
    "- This count measures consecutive records as stored in the file; reversed intervals are not counted as gaps. Epoch is treated as a per-sensor sequence.",
    "",
    "## Measurement statistics",
    "",
    "| Feature | Present | Missing/unparsed | Minimum | Maximum | Mean |",
    "|---|---:|---:|---:|---:|---:|",
]
for name, s in feature_stats.items():
    if s["n"]:
        report.append(
            f"| {name} | {s['n']:,} | {s['missing']:,} | {s['min']:.6g} | {s['max']:.6g} | {s['sum'] / s['n']:.6g} |"
        )
report += ["", "## Parse issue counts", "", "| Issue | Count |", "|---|---:|"]
report += [f"| {issue} | {n:,} |" for issue, n in sorted(issue_counts.items())]
report += [
    "",
    "Repeated and trailing literal spaces are preserved as empty positional slots. Records whose collapsed whitespace-token count is below eight can still map to eight source columns; these are not automatically rejected. Raw text and line number are preserved.",
    "",
]
report.insert(
    report.index("## Measurement statistics"),
    "Numeric statistics use positionally recoverable values. Explicit empty slots count as missing/unparsed measurements; rows without a sensor ID remain outside per-sensor coverage summaries.",
)
target = Path("artifacts/data_profile.md")
target.parent.mkdir(exist_ok=True)
target.write_text("\n".join(report), encoding="utf-8")
print(
    f"Wrote {target}; {count:,} rows, {len(sensor_rows)} sensor IDs, {dict(schema_slot_counts)} positional field-slot counts"
)
