"""Per-window kinematic and pose feature extraction."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from engine.features.kinematics import compute_kinematics
from engine.features.schemas import TrajectorySample
from engine.tracking.schemas import TrackedDetection

from engine.config import TimelineSettings
from engine.pose.keypoints import mean_keypoint_score
from engine.types import PoseResult


@dataclass(frozen=True)
class WindowFeatures:
    speed_mean: float
    speed_std: float
    speed_max: float
    acceleration_mean: float
    acceleration_std: float
    stop_duration_s: float
    trajectory_variance: float
    pose_motion: float
    turn_frequency: float
    dwell_duration_s: float
    trajectory_smoothness: float
    window_duration_s: float
    observation_count: int
    pose_confidence: float


def extract_window_features(
    observations: list[TrackedDetection],
    poses: dict[tuple[int, int], PoseResult],
    *,
    track_id: int,
    samples: list[TrajectorySample],
    settings: TimelineSettings,
) -> WindowFeatures:
    """Compute behaviour features for one sliding window."""
    if not observations:
        return WindowFeatures(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0)

    stats, turn_freq, smoothness, dwell = compute_kinematics(
        samples,
        stationary_speed_threshold=settings.standing_speed_threshold,
        turn_angle_threshold_rad=0.52,
    )

    xs = np.array([o.centroid_x for o in observations], dtype=np.float32)
    ys = np.array([o.centroid_y for o in observations], dtype=np.float32)
    trajectory_variance = float(xs.var() + ys.var()) if len(observations) >= 2 else 0.0

    stop_duration_s = _stop_duration(observations, settings.standing_speed_threshold)
    pose_motion, pose_confidence = _pose_motion(observations, poses, track_id, settings.pose_keypoint_threshold)

    window_duration_s = max(
        (observations[-1].timestamp_ms - observations[0].timestamp_ms) / 1000.0,
        1.0 / 30.0,
    )

    return WindowFeatures(
        speed_mean=stats.speed_mean,
        speed_std=stats.speed_std,
        speed_max=stats.speed_max,
        acceleration_mean=abs(stats.acceleration_mean),
        acceleration_std=stats.acceleration_std,
        stop_duration_s=stop_duration_s,
        trajectory_variance=trajectory_variance,
        pose_motion=pose_motion,
        turn_frequency=turn_freq,
        dwell_duration_s=dwell,
        trajectory_smoothness=smoothness,
        window_duration_s=window_duration_s,
        observation_count=len(observations),
        pose_confidence=pose_confidence,
    )


def _stop_duration(observations: list[TrackedDetection], speed_threshold: float) -> float:
    """Seconds spent nearly stationary within the window."""
    if len(observations) < 2:
        return 0.0

    ordered = sorted(observations, key=lambda o: o.timestamp_ms)
    stopped_s = 0.0
    for i in range(1, len(ordered)):
        prev, curr = ordered[i - 1], ordered[i]
        dt = (curr.timestamp_ms - prev.timestamp_ms) / 1000.0
        if dt <= 0:
            continue
        dx = curr.centroid_x - prev.centroid_x
        dy = curr.centroid_y - prev.centroid_y
        speed = math.hypot(dx, dy) / dt
        if speed < speed_threshold:
            stopped_s += dt
    return stopped_s


def _pose_motion(
    observations: list[TrackedDetection],
    poses: dict[tuple[int, int], PoseResult],
    track_id: int,
    score_threshold: float,
) -> tuple[float, float]:
    """Mean per-frame keypoint displacement and average pose confidence."""
    ordered = sorted(observations, key=lambda o: o.frame_index)
    displacements: list[float] = []
    confidences: list[float] = []

    prev_kps: list[tuple[float, float]] | None = None
    for obs in ordered:
        pose = poses.get((obs.frame_index, track_id))
        if pose is None or not pose.keypoints:
            prev_kps = None
            continue
        confidences.append(mean_keypoint_score(pose.scores))
        if prev_kps is not None and len(prev_kps) == len(pose.keypoints):
            dists = []
            for (x1, y1), (x2, y2), score in zip(prev_kps, pose.keypoints, pose.scores):
                if score >= score_threshold:
                    dists.append(math.hypot(x2 - x1, y2 - y1))
            if dists:
                displacements.append(float(np.mean(dists)))
        prev_kps = pose.keypoints

    motion = float(np.mean(displacements)) if displacements else 0.0
    confidence = float(np.mean(confidences)) if confidences else 0.0
    return motion, confidence
