from sensorguard.cleaning import validate_frame
from sensorguard.config import SensorGuardConfig
from sensorguard.validation import parse_line


def test_parse_combines_timestamp_and_numbers():
    row = parse_line("2004-02-28 01:06:16.013453 17 1 19.1652 38.8039 45.08 2.68742")
    assert row["timestamp"].isoformat() == "2004-02-28T01:06:16.013453"
    assert row["epoch"] == 17 and row["moteid"] == 1


def test_malformed_line_preserved():
    row = parse_line("2004-02-28 01:00:00 1 2")
    assert "malformed_field_count" in row["issues"]
    assert row["raw"].endswith("1 2")


def test_real_format_blank_measurement_slots_are_recovered_positionally():
    light_missing = parse_line("2004-03-02 02:55:32.980639 8875 8 17.7932 42.5178  2.65143", 276858)
    assert light_missing["moteid"] == 8
    assert light_missing["temperature"] == 17.7932
    assert light_missing["humidity"] == 42.5178
    assert light_missing["light"] is None
    assert light_missing["voltage"] == 2.65143
    assert "missing_light" in light_missing["issues"]
    assert "malformed_field_count" not in light_missing["issues"]
    assert light_missing["line_number"] == 276858

    three_missing = parse_line("2004-03-30 08:11:56.681052 53671 3    2.1997", 129123)
    assert [three_missing[name] for name in ("temperature", "humidity", "light")] == [
        None,
        None,
        None,
    ]
    assert three_missing["voltage"] == 2.1997
    assert three_missing["raw"].endswith("    2.1997")


def test_trailing_blank_columns_are_retained_with_missing_sensor_evidence():
    raw = "2004-02-28 00:58:15.315133 1     "
    row = parse_line(raw, 2_313_157)
    assert row["timestamp"].isoformat() == "2004-02-28T00:58:15.315133"
    assert row["epoch"] == 1
    assert row["moteid"] is None
    assert "missing_moteid" in row["issues"]
    assert all(row[name] is None for name in ("temperature", "humidity", "light", "voltage"))
    assert "malformed_field_count" not in row["issues"]
    assert "measurement_alignment_unknown" not in row["issues"]
    assert row["line_number"] == 2_313_157
    assert row["raw"] == raw


def test_missing_value_and_invalid_numeric():
    row = parse_line("2004-02-28 01:00:00 1 2 bad 50 9 2.5")
    assert row["temperature"] is None
    assert "invalid_temperature" in row["issues"]
    short = parse_line("2004-02-28 01:00:00 1 2")
    assert short["temperature"] is None


def test_invalid_timestamp_and_sensor_id():
    row = parse_line("bad-date 01:00:00 1 99 20 50 9 2.5")
    assert "invalid_timestamp" in row["issues"]
    valid_time = parse_line("2004-02-28 01:00:00 1 99 20 50 9 2.5")
    checked = validate_frame(__import__("pandas").DataFrame([valid_time]))
    assert "invalid_sensor_id" in checked.iloc[0].issues


def test_duplicates_and_sampling_gap():
    import pandas as pd

    records = [
        parse_line(f"2004-02-28 01:00:0{i} 1 {sid} 20 50 9 2.5")
        for i, sid in [(0, 1), (0, 1), (9, 1)]
    ]
    checked = validate_frame(pd.DataFrame(records), SensorGuardConfig(sampling_gap_seconds=5))
    assert checked.duplicate.iloc[:2].all()
    assert "sampling_gap" in checked.iloc[2].issues


def test_gzip_and_plain_stream(tmp_path):
    import gzip

    from sensorguard.ingestion import iter_records

    line = b"2004-02-28 01:00:00 1 2 20 50 9 2.5\n"
    plain = tmp_path / "data.txt"
    zipped = tmp_path / "data.txt.gz"
    plain.write_bytes(line)
    with gzip.open(zipped, "wb") as stream:
        stream.write(line)
    assert list(iter_records(plain))[0]["moteid"] == list(iter_records(zipped))[0]["moteid"] == 2
