"""Configurable validation assumptions."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SensorGuardConfig:
    """Bounds are screening rules; out-of-range values are retained and flagged."""

    sensor_ids: frozenset[int] = field(default_factory=lambda: frozenset(range(1, 55)))
    physical_limits: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "temperature": (-40.0, 85.0),
            "humidity": (0.0, 100.0),
            "light": (0.0, 200000.0),
            "voltage": (0.0, 5.0),
        }
    )
    sampling_gap_seconds: float = 300.0
    chunk_size: int = 100_000
    contextual_tolerance_seconds: float = 120.0
    minimum_peer_sensors: int = 3
    abrupt_change_limits: dict[str, float] = field(
        default_factory=lambda: {
            "temperature": 10.0,
            "humidity": 25.0,
            "light": 1000.0,
            "voltage": 0.5,
        }
    )
    constant_window: int = 20
    constant_minimum_observations: int = 20
    constant_tolerance: float = 1e-9
    sensor_baseline_minimum_rows: int = 20
    sensor_deviation_z_threshold: float = 5.0
