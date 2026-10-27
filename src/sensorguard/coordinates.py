"""Official Intel Berkeley mote coordinates and derived physical neighborhoods.

Coordinates are copied from the three-column mote location file linked by the
Intel Lab data page: https://db.csail.mit.edu/labdata/mote_locs.txt. Values are
meters relative to the upper-right lab corner. They describe all 54 documented
motes; additional IDs in some source files have no coordinate assignment here.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import hypot

MOTE_COORDINATES: dict[int, tuple[float, float]] = {
    1: (21.5, 23),
    2: (24.5, 20),
    3: (19.5, 19),
    4: (22.5, 15),
    5: (24.5, 12),
    6: (19.5, 12),
    7: (22.5, 8),
    8: (24.5, 4),
    9: (21.5, 2),
    10: (19.5, 5),
    11: (16.5, 3),
    12: (13.5, 1),
    13: (12.5, 5),
    14: (8.5, 6),
    15: (5.5, 3),
    16: (1.5, 2),
    17: (1.5, 8),
    18: (5.5, 10),
    19: (3.5, 13),
    20: (0.5, 17),
    21: (4.5, 18),
    22: (1.5, 23),
    23: (6, 24),
    24: (1.5, 30),
    25: (4.5, 30),
    26: (7.5, 31),
    27: (8.5, 26),
    28: (10.5, 31),
    29: (12.5, 26),
    30: (13.5, 31),
    31: (15.5, 28),
    32: (17.5, 31),
    33: (19.5, 26),
    34: (21.5, 30),
    35: (24.5, 27),
    36: (26.5, 31),
    37: (27.5, 26),
    38: (30.5, 31),
    39: (30.5, 26),
    40: (33.5, 28),
    41: (36.5, 30),
    42: (39.5, 30),
    43: (35.5, 24),
    44: (40.5, 22),
    45: (37.5, 19),
    46: (34.5, 16),
    47: (39.5, 14),
    48: (35.5, 10),
    49: (39.5, 6),
    50: (38.5, 1),
    51: (35.5, 4),
    52: (31.5, 6),
    53: (28.5, 5),
    54: (26.5, 2),
}


def physical_neighborhoods(
    radius_m: float = 8.0,
    coordinates: Mapping[int, tuple[float, float]] | None = None,
) -> dict[int, tuple[int, ...]]:
    """Return physical-neighbor IDs within a Euclidean distance in meters."""
    if radius_m <= 0:
        raise ValueError("radius_m must be positive")
    locations = coordinates or MOTE_COORDINATES
    return {
        int(sensor): tuple(
            sorted(
                int(peer)
                for peer, peer_xy in locations.items()
                if peer != sensor and hypot(xy[0] - peer_xy[0], xy[1] - peer_xy[1]) <= radius_m
            )
        )
        for sensor, xy in locations.items()
    }
