"""Line-wise ingestion with original text preserved for audit."""

from __future__ import annotations

import gzip
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO

import pandas as pd

from .validation import parse_line

logger = logging.getLogger(__name__)


def _open_text(path: Path) -> TextIO:
    return (
        gzip.open(path, "rt", encoding="utf-8", errors="replace")
        if path.suffix == ".gz"
        else path.open("rt", encoding="utf-8", errors="replace")
    )


def iter_records(path: str | Path) -> Iterator[dict]:
    """Yield parsed records one line at a time from gzip or plain text."""
    source = Path(path)
    logger.info("Reading sensor data from %s", source)
    with _open_text(source) as stream:
        for line_number, raw in enumerate(stream, 1):
            yield parse_line(raw, line_number)


def read_chunks(path: str | Path, chunk_size: int = 100_000) -> Iterator[pd.DataFrame]:
    """Yield bounded-size data frames; malformed records remain represented."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    batch = []
    for record in iter_records(path):
        batch.append(record)
        if len(batch) == chunk_size:
            yield pd.DataFrame.from_records(batch)
            batch.clear()
    if batch:
        yield pd.DataFrame.from_records(batch)
