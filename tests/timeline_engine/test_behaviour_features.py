"""Tests for window feature extraction."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from engine.features.schemas import TrajectorySample
from engine.tracking.schemas import TrackedDetection

from engine.behaviour.features import extract_window_features
from engine.config import TimelineSettings
from engine.types import PoseResult


def test_extract_window_features_stop_duration():
    observations = [
        TrackedDetection(0, 0.0, 10, 10, 20, 30, 0.9, 1),
        TrackedDetection(1, 500.0, 10, 10, 20, 30, 0.9, 1),
        TrackedDetection(2, 1000.0, 10.2, 10.1, 20, 30, 0.9, 1),
    ]
    base = datetime(2020, 1, 1, tzinfo=UTC)
    samples = [
        TrajectorySample(
            recorded_at=base + timedelta(milliseconds=o.timestamp_ms),
            x=o.centroid_x,
            y=o.centroid_y,
            identity_id=uuid.UUID(int=0),
            track_id=uuid.UUID(int=1),
        )
        for o in observations
    ]
    features = extract_window_features(
        observations,
        {},
        track_id=1,
        samples=samples,
        settings=TimelineSettings(),
    )
    assert features.stop_duration_s > 0.4
    assert features.observation_count == 3


def test_pose_motion_detects_movement():
    observations = [
        TrackedDetection(0, 0.0, 10, 10, 50, 100, 0.9, 1),
        TrackedDetection(1, 500.0, 10, 10, 50, 100, 0.9, 1),
    ]
    kps_a = [(20.0, 30.0)] * 17
    kps_b = [(40.0, 50.0)] * 17
    poses = {
        (0, 1): PoseResult(0, 1, kps_a, [0.9] * 17),
        (1, 1): PoseResult(1, 1, kps_b, [0.9] * 17),
    }
    base = datetime(2020, 1, 1, tzinfo=UTC)
    samples = [
        TrajectorySample(
            recorded_at=base + timedelta(milliseconds=o.timestamp_ms),
            x=o.centroid_x,
            y=o.centroid_y,
            identity_id=uuid.UUID(int=0),
            track_id=uuid.UUID(int=1),
        )
        for o in observations
    ]
    features = extract_window_features(
        observations,
        poses,
        track_id=1,
        samples=samples,
        settings=TimelineSettings(),
    )
    assert features.pose_motion > 0.0
