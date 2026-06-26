"""Trajectory kinematic feature computation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from engine.features.schemas import KinematicStats, TrajectorySample


@dataclass(frozen=True)
class _Segment:
    dt_s: float
    speed: float
    acceleration: float
    heading_rad: float


def compute_kinematics(
    samples: list[TrajectorySample],
    *,
    stationary_speed_threshold: float,
    turn_angle_threshold_rad: float,
) -> tuple[KinematicStats, float, float, float]:
    if len(samples) < 2:
        return KinematicStats(), 0.0, 1.0, 0.0

    ordered = sorted(samples, key=lambda s: s.recorded_at)
    segments = _build_segments(ordered)
    if not segments:
        return KinematicStats(), 0.0, 1.0, 0.0

    speeds = np.array([s.speed for s in segments], dtype=np.float32)
    accelerations = np.array([s.acceleration for s in segments], dtype=np.float32)
    headings = np.array([s.heading_rad for s in segments], dtype=np.float32)

    stats = KinematicStats(
        speed_mean=float(speeds.mean()),
        speed_max=float(speeds.max()),
        speed_std=float(speeds.std()) if len(speeds) > 1 else 0.0,
        acceleration_mean=float(accelerations.mean()),
        acceleration_max=float(abs(accelerations).max()),
        acceleration_std=float(accelerations.std()) if len(accelerations) > 1 else 0.0,
        heading_mean_rad=_circular_mean(headings),
        heading_mean_deg=math.degrees(_circular_mean(headings)),
    )

    total_time = sum(s.dt_s for s in segments)
    turns = _count_turns(headings, turn_angle_threshold_rad)
    turn_frequency = turns / total_time if total_time > 0 else 0.0
    smoothness = _trajectory_smoothness(headings)
    dwell_duration = sum(s.dt_s for s in segments if s.speed < stationary_speed_threshold)

    return stats, turn_frequency, smoothness, dwell_duration


def _build_segments(samples: list[TrajectorySample]) -> list[_Segment]:
    segments: list[_Segment] = []
    prev_speed = 0.0

    for i in range(1, len(samples)):
        prev = samples[i - 1]
        curr = samples[i]
        dt = (curr.recorded_at - prev.recorded_at).total_seconds()
        if dt <= 0:
            continue

        dx = curr.x - prev.x
        dy = curr.y - prev.y
        distance = math.hypot(dx, dy)
        speed = distance / dt
        acceleration = (speed - prev_speed) / dt
        heading = math.atan2(dy, dx)

        segments.append(
            _Segment(
                dt_s=dt,
                speed=speed,
                acceleration=acceleration,
                heading_rad=heading,
            )
        )
        prev_speed = speed

    return segments


def _count_turns(headings: np.ndarray, threshold_rad: float) -> int:
    if len(headings) < 2:
        return 0

    turns = 0
    for i in range(1, len(headings)):
        delta = _angle_delta(headings[i - 1], headings[i])
        if abs(delta) >= threshold_rad:
            turns += 1
    return turns


def _trajectory_smoothness(headings: np.ndarray) -> float:
    if len(headings) < 2:
        return 1.0

    deltas = [_abs_angle_delta(headings[i - 1], headings[i]) for i in range(1, len(headings))]
    mean_delta = float(np.mean(deltas))
    return round(max(0.0, min(1.0, 1.0 - mean_delta / math.pi)), 4)


def _circular_mean(headings: np.ndarray) -> float:
    if len(headings) == 0:
        return 0.0
    sin_sum = float(np.sin(headings).mean())
    cos_sum = float(np.cos(headings).mean())
    return math.atan2(sin_sum, cos_sum)


def _angle_delta(a: float, b: float) -> float:
    return math.atan2(math.sin(b - a), math.cos(b - a))


def _abs_angle_delta(a: float, b: float) -> float:
    return abs(_angle_delta(a, b))
