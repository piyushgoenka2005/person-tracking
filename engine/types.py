"""Domain types for the Behaviour Timeline Engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TimelineBehaviour(str, Enum):
    WALKING = "Walking"
    STANDING = "Standing"
    WAITING = "Waiting"
    QUEUEING = "Queueing"
    SITTING = "Sitting"
    SLEEPING = "Sleeping"
    PHONE_USAGE = "Phone Usage"
    EATING = "Eating"
    UNKNOWN = "Unknown"


COCO_SKELETON: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)


@dataclass(frozen=True)
class VideoInfo:
    path: str
    fps: float
    frame_width: int
    frame_height: int
    total_frames: int
    duration_s: float


@dataclass(frozen=True)
class PoseResult:
    frame_index: int
    track_id: int
    keypoints: list[tuple[float, float]]
    scores: list[float]

    @property
    def mean_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(self.scores) / len(self.scores)


@dataclass(frozen=True)
class BehaviourLabel:
    track_id: int
    start_ms: float
    end_ms: float
    behaviour: TimelineBehaviour
    confidence: float


@dataclass(frozen=True)
class BehaviourTransition:
    person_id: int
    transition_time: str
    transition_ms: float
    from_behaviour: TimelineBehaviour
    to_behaviour: TimelineBehaviour
    confidence: float


@dataclass(frozen=True)
class TimelineSegment:
    person_id: int
    start_time: str
    end_time: str
    behaviour: TimelineBehaviour
    confidence: float
    duration_s: float
    start_ms: float
    end_ms: float


@dataclass
class SessionContext:
    video: VideoInfo
    yolo_model: str
    pose_model: str
    device: str
    frame_stride: int
    person_count: int = 0
    behaviour_distribution: dict[str, float] = field(default_factory=dict)
    qa_warnings: list[str] = field(default_factory=list)
    qa_errors: list[str] = field(default_factory=list)
    started_at: str = ""
    completed_at: str = ""
