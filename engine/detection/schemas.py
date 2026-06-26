"""Detection schemas."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DetectionBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass(frozen=True)
class FrameDetections:
    frame_index: int
    timestamp_ms: float
    frame_width: int
    frame_height: int
    boxes: list[DetectionBox]
