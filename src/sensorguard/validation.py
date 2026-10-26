"""Parsing and row-level validation."""

from __future__ import annotations

import math
from datetime import datetime

FIELDS = ("date", "time", "epoch", "moteid", "temperature", "humidity", "light", "voltage")
NUMERIC = FIELDS[2:]


def parse_line(raw: str, line_number: int = 0) -> dict:
    """Parse a whitespace record, retaining invalid rows and reason flags."""
    line = raw.rstrip("\r\n")
    # The Intel file uses literal spaces between columns; repeated and trailing
    # spaces encode blank positional fields. Preserve those slots. Tabs or other
    # whitespace use generic splitting because their blank positions are ambiguous.
    tokens = line.split(" ") if "\t" not in line else line.split()
    row = {
        "line_number": line_number,
        "raw": raw.rstrip("\r\n"),
        "timestamp": None,
        "epoch": None,
        "moteid": None,
        "temperature": None,
        "humidity": None,
        "light": None,
        "voltage": None,
        "issues": [],
    }
    if len(tokens) != 8:
        row["issues"].append("malformed_field_count")
    if len(tokens) >= 2:
        try:
            row["timestamp"] = datetime.fromisoformat(f"{tokens[0]}T{tokens[1]}")
        except ValueError:
            row["issues"].append("invalid_timestamp")
    else:
        row["issues"].append("invalid_timestamp")
    # The sequence and sensor columns are positionally identifiable when the
    # source uses literal spaces; generic tab whitespace cannot retain blanks.
    for index, name in ((2, "epoch"), (3, "moteid")):
        if index < len(tokens):
            if not tokens[index].strip():
                if len(tokens) != 8:
                    row["issues"].append(f"missing_{name}")
                continue
            try:
                value = float(tokens[index])
                if not math.isfinite(value) or not value.is_integer():
                    raise ValueError
                row[name] = int(value)
            except ValueError:
                row["issues"].append(f"invalid_{name}")
    # Incomplete lines remain auditable but their trailing columns cannot be
    # aligned safely. Do not shift later tokens into measurement columns.
    if len(tokens) != 8:
        if len(tokens) < 8:
            row["issues"].append("measurement_alignment_unknown")
        return row
    for index, name in enumerate(NUMERIC, 2):
        if index >= len(tokens) or tokens[index].strip().lower() in {"", "na", "nan", "null"}:
            row[name] = None
            row["issues"].append(f"missing_{name}")
            continue
        try:
            value = float(tokens[index])
            if not math.isfinite(value):
                raise ValueError
            row[name] = int(value) if name in {"epoch", "moteid"} and value.is_integer() else value
            if name in {"epoch", "moteid"} and not value.is_integer():
                row["issues"].append(f"invalid_{name}")
        except ValueError:
            row[name] = None
            row["issues"].append(f"invalid_{name}")
    return row
