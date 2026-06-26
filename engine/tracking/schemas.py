"""Tracking schemas."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DetectionInput:
    frame_index: int
    timestamp_ms: float
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def centroid_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def centroid_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass(frozen=True)
class TrackedDetection:
    frame_index: int
    timestamp_ms: float
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    local_track_id: int

    @property
    def centroid_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def centroid_y(self) -> float:
        return (self.y1 + self.y2) / 2.0


@dataclass
class TrackAggregate:
    local_track_id: int
    observations: list[TrackedDetection] = field(default_factory=list)
    gap_count: int = 0
    quality_score: float = 0.0

    @property
    def frame_count(self) -> int:
        return len(self.observations)

    @property
    def started_at_ms(self) -> float:
        return self.observations[0].timestamp_ms if self.observations else 0.0

    @property
    def ended_at_ms(self) -> float:
        return self.observations[-1].timestamp_ms if self.observations else 0.0
