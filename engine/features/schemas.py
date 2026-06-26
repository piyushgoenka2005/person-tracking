"""Feature schemas used by behaviour analysis."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class TrajectorySample:
    recorded_at: datetime
    x: float
    y: float
    identity_id: uuid.UUID
    track_id: uuid.UUID


@dataclass
class KinematicStats:
    speed_mean: float = 0.0
    speed_max: float = 0.0
    speed_std: float = 0.0
    acceleration_mean: float = 0.0
    acceleration_max: float = 0.0
    acceleration_std: float = 0.0
    heading_mean_rad: float = 0.0
    heading_mean_deg: float = 0.0
