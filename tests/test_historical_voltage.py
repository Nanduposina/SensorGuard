import numpy as np
import pandas as pd
import pytest

from sensorguard.historical import voltage_window_candidates


def test_streaming_voltage_windows_adjust_temperature_and_mark_only_candidates():
    records = []
    timestamps = pd.date_range("2004-02-02", periods=240, freq="30min")
    for index, timestamp in enumerate(timestamps):
        elapsed_days = (timestamp - timestamps[0]).total_seconds() / 86400
        temperature = 20 + 3 * np.sin(index / 9)
        voltage = 2.7 + 0.002 * temperature - 0.02 * elapsed_days
        records.append(
            {
                "timestamp": timestamp,
                "moteid": 1,
                "temperature": temperature,
                "voltage": voltage,
            }
        )
    cases = voltage_window_candidates(
        records,
        window_days=7,
        minimum_rows=100,
        decline_threshold_v_per_day=-0.005,
    )
    assert len(cases) == 1
    assert cases.sensor_id.iloc[0] == 1
    assert cases.observations.iloc[0] == 240
    assert cases.candidate_decline.iloc[0]
    assert cases.temperature_adjusted_voltage_slope_v_per_day.iloc[0] == pytest.approx(-0.02)
    assert "Unverified" in cases.interpretation.iloc[0]


def test_voltage_windows_handle_sparse_constant_and_invalid_input():
    sparse = [
        {
            "timestamp": pd.Timestamp("2004-01-01") + pd.Timedelta(minutes=i),
            "moteid": 7,
            "temperature": 20.0,
            "voltage": 2.5,
        }
        for i in range(2)
    ]
    assert voltage_window_candidates(sparse, minimum_rows=10).empty
    assert voltage_window_candidates([], minimum_rows=3).empty
    with pytest.raises(ValueError, match="window_days"):
        voltage_window_candidates([], window_days=0)
