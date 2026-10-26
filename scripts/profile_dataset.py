"""One-pass streaming dataset profile with duplicate and validity summaries."""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sensorguard.config import SensorGuardConfig
from sensorguard.ingestion import iter_records


def peak_working_set_mib() -> float | None:
    """Return process peak working set on Windows, or peak RSS where available."""
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = (
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCounters),
                wintypes.DWORD,
            )
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            ok = psapi.GetProcessMemoryInfo(
                kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
            )
            if ok:
                return counters.PeakWorkingSetSize / (1024 * 1024)
        except (AttributeError, OSError):
            return None
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / (1024 if sys.platform != "darwin" else 1024 * 1024)
    except (ImportError, OSError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="data.txt/data.txt")
    parser.add_argument("--output", default="artifacts/data_profile.md")
    args = parser.parse_args()
    path = Path(args.input)
    config = SensorGuardConfig()
    started = time.perf_counter()

    row_count = 0
    unlocated_rows = 0
    collapsed_token_counts = Counter()
    schema_slot_counts = Counter()
    sensor_rows = Counter()
    issue_counts = Counter()
    physical_counts = Counter()
    feature_stats = defaultdict(
        lambda: {"n": 0, "sum": 0.0, "min": float("inf"), "max": float("-inf"), "missing": 0}
    )
    first_timestamp = last_timestamp = None
    last_per_sensor = {}
    duplicate_keys: set[bytes] = set()
    duplicate_timestamp_sensor_records = 0
    out_of_order_records = 0
    gap_count = 0
    gap_max = 0.0
    prefix_only_records = 0
    prefix_only_run = 0
    prefix_run_start = None
    incomplete_positional_records = 0
    invalid_examples = []

    for row in iter_records(path):
        row_count += 1
        raw = row["raw"]
        collapsed_token_counts[len(raw.split())] += 1
        slots = raw.split(" ") if "\t" not in raw else raw.split()
        schema_slot_counts[len(slots)] += 1
        if len(slots) != 8:
            incomplete_positional_records += 1
        prefix_only = len(slots) == 8 and all(not token for token in slots[3:])
        if prefix_only:
            if prefix_only_run == 0:
                prefix_run_start = row["line_number"]
            prefix_only_run += 1
            prefix_only_records += 1
        else:
            prefix_only_run = 0

        issue_counts.update(row["issues"])
        if (
            any(
                issue in row["issues"]
                for issue in (
                    "malformed_field_count",
                    "measurement_alignment_unknown",
                    "invalid_timestamp",
                    "invalid_epoch",
                    "invalid_moteid",
                )
            )
            and len(invalid_examples) < 10
        ):
            invalid_examples.append(
                {"line_number": row["line_number"], "issues": row["issues"], "raw": raw}
            )

        timestamp = row["timestamp"]
        if timestamp is not None:
            first_timestamp = (
                timestamp if first_timestamp is None else min(first_timestamp, timestamp)
            )
            last_timestamp = timestamp if last_timestamp is None else max(last_timestamp, timestamp)

        sensor_id = row["moteid"]
        if sensor_id is None:
            unlocated_rows += 1
        else:
            sensor_rows[sensor_id] += 1
            if timestamp is not None:
                # Hash the detector's duplicate identity to keep the set compact.
                key = hashlib.blake2b(
                    f"{sensor_id}\0{timestamp.isoformat()}".encode("ascii"), digest_size=12
                ).digest()
                if key in duplicate_keys:
                    duplicate_timestamp_sensor_records += 1
                else:
                    duplicate_keys.add(key)
                previous = last_per_sensor.get(sensor_id)
                if previous is not None:
                    gap = (timestamp - previous).total_seconds()
                    if gap < 0:
                        out_of_order_records += 1
                    elif gap > config.sampling_gap_seconds:
                        gap_count += 1
                        gap_max = max(gap_max, gap)
                last_per_sensor[sensor_id] = timestamp

        for feature, (lower, upper) in config.physical_limits.items():
            value = row[feature]
            stats = feature_stats[feature]
            if value is None:
                stats["missing"] += 1
            else:
                stats["n"] += 1
                stats["sum"] += value
                stats["min"] = min(stats["min"], value)
                stats["max"] = max(stats["max"], value)
                if value < lower or value > upper:
                    physical_counts[feature] += 1

    elapsed = time.perf_counter() - started
    sensors = sorted(sensor_rows)
    report = [
        "# SensorGuard data profile",
        "",
        f"Source: `{path}` ({path.stat().st_size:,} bytes)",
        "",
        "## Integrity and coverage",
        "",
        f"- Rows parsed: {row_count:,}",
        f"- Whitespace token counts after collapsing: {dict(sorted(collapsed_token_counts.items()))}",
        f"- Positional field-slot counts (repeated spaces preserved): {dict(sorted(schema_slot_counts.items()))}",
        f"- Incomplete positional records (field slots other than eight): {incomplete_positional_records:,}",
        f"- Rows with mote ID and all measurement slots explicitly blank: {prefix_only_records:,}; contiguous trailing run: {prefix_only_run:,} starting at line {prefix_run_start if prefix_only_run else 'none'}",
        f"- Rows with no parsed sensor ID: {unlocated_rows:,}; source records are retained with raw text and line number.",
        "- The blank-ID suffix is structurally positional but its physical cause is unknown; it is not assigned to a sensor.",
        f"- Parsed timestamp range: {first_timestamp} to {last_timestamp}",
        f"- Distinct sensor IDs: {len(sensors)} (minimum {min(sensors) if sensors else 'n/a'}, maximum {max(sensors) if sensors else 'n/a'})",
        f"- IDs outside configured expected set: {sorted(set(sensors) - config.sensor_ids)}",
        f"- Per-sensor counts: {dict(sorted(sensor_rows.items()))}",
        f"- Repeated (timestamp, sensor ID) record keys after the first: {duplicate_timestamp_sensor_records:,}",
        f"- Out-of-order records relative to prior input record for that sensor: {out_of_order_records:,}",
        f"- Per-sensor consecutive input-order intervals over {config.sampling_gap_seconds:g} seconds: {gap_count:,}; maximum positive interval: {gap_max:,.1f} seconds",
        "- Epoch is treated as a per-sensor sequence, not a globally unique timestamp.",
        "",
        "## Measurement and configured physical-limit statistics",
        "",
        f"Screening bounds (heuristics, values retained): {config.physical_limits}",
        "",
        "| Feature | Present | Missing/unparsed | Below/above configured bounds | Minimum | Maximum | Mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for feature, stats in feature_stats.items():
        if stats["n"]:
            report.append(
                f"| {feature} | {stats['n']:,} | {stats['missing']:,} | {physical_counts[feature]:,} | {stats['min']:.6g} | {stats['max']:.6g} | {stats['sum'] / stats['n']:.6g} |"
            )
    report += ["", "## Parse issue counts", "", "| Issue | Count |", "|---|---:|"]
    report += [f"| {issue} | {count:,} |" for issue, count in sorted(issue_counts.items())]
    peak_memory = peak_working_set_mib()
    report += [
        "",
        f"- Elapsed processing time: {elapsed:.2f} seconds (single streaming pass).",
        f"- Peak process working set: {peak_memory:.1f} MiB."
        if peak_memory is not None
        else "- Peak process memory was not available from this platform.",
        "- Configured physical limits are screening heuristics, not verified failure labels; out-of-range values are preserved.",
        "- Repeated/trailing literal spaces remain empty positional slots. A collapsed token count below eight is not by itself evidence of corruption.",
        "- Raw text and line number are preserved by the parser. If malformed examples are found, the report retains their source-line provenance.",
        "",
    ]
    if invalid_examples:
        report += [
            "## Malformed record examples",
            "",
            "| Line | Issues | Raw source line |",
            "|---:|---|---|",
        ]
        for item in invalid_examples:
            escaped_raw = item["raw"].replace("|", "\\|")
            report.append(
                f"| {item['line_number']} | {', '.join(item['issues'])} | `{escaped_raw}` |"
            )
        report.append("")

    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(report), encoding="utf-8")
    print(
        f"Wrote {target}; {row_count:,} rows, {len(sensors)} sensor IDs, "
        f"{duplicate_timestamp_sensor_records:,} duplicate keys in {elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()
