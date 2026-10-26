import subprocess
import sys
from pathlib import Path


def test_profile_cli_reports_duplicates_limits_and_source_provenance(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "sample.txt"
    report = tmp_path / "profile.md"
    source.write_text(
        "\n".join(
            [
                "2004-02-28 01:00:00 1 1 20 50 9 2.5",
                "2004-02-28 01:00:00 2 1 21 50 9 2.5",
                "2004-02-28 01:00:01 3 55 90 101 9 6",
                "bad-date 01:00:02 4",
            ]
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "profile_dataset.py"),
            str(source),
            "--output",
            str(report),
        ],
        check=True,
        cwd=root,
        capture_output=True,
        text=True,
    )
    result = report.read_text(encoding="utf-8")
    assert "Rows parsed: 4" in result
    assert "Repeated (timestamp, sensor ID) record keys after the first: 1" in result
    assert "Out-of-order records" in result
    assert "IDs outside configured expected set: [55]" in result
    assert "| temperature | 3 | 1 | 1 |" in result
    assert "| humidity | 3 | 1 | 1 |" in result
    assert "| voltage | 3 | 1 | 1 |" in result
    assert "Line | Issues | Raw source line" in result
    assert "bad-date 01:00:02 4" in result
