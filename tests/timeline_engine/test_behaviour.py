"""Tests for behaviour rules."""

from __future__ import annotations

from engine.behaviour.features import WindowFeatures
from engine.behaviour.rules import WindowSignals, classify_window
from engine.config import TimelineSettings
from engine.types import PoseResult, TimelineBehaviour


def _features(**kwargs) -> WindowFeatures:
    defaults = dict(
        speed_mean=0.0,
        speed_std=0.0,
        speed_max=0.0,
        acceleration_mean=0.0,
        acceleration_std=0.0,
        stop_duration_s=0.0,
        trajectory_variance=0.0,
        pose_motion=0.0,
        turn_frequency=0.0,
        dwell_duration_s=0.0,
        trajectory_smoothness=1.0,
        window_duration_s=1.0,
        observation_count=3,
        pose_confidence=0.0,
    )
    defaults.update(kwargs)
    return WindowFeatures(**defaults)


def test_classify_walking():
    settings = TimelineSettings()
    behaviour, conf = classify_window(
        WindowSignals(
            features=_features(
                speed_mean=30.0,
                speed_std=8.0,
                acceleration_std=12.0,
                trajectory_variance=80.0,
                pose_motion=8.0,
                stop_duration_s=0.1,
            ),
            pose=None,
            bbox_height=100.0,
            speed_ratio=1.4,
        ),
        settings,
    )
    assert behaviour == TimelineBehaviour.WALKING
    assert conf > 0


def test_classify_standing():
    settings = TimelineSettings()
    behaviour, _ = classify_window(
        WindowSignals(
            features=_features(
                speed_mean=2.0,
                stop_duration_s=0.7,
                trajectory_variance=5.0,
                pose_motion=1.0,
            ),
            pose=None,
            bbox_height=100.0,
            speed_ratio=0.5,
        ),
        settings,
    )
    assert behaviour in (TimelineBehaviour.STANDING, TimelineBehaviour.WAITING)


def test_classify_phone_usage_with_pose():
    settings = TimelineSettings()
    kps = [
        (100, 50),
        (90, 48), (110, 48), (85, 50), (115, 50),
        (80, 80), (120, 80),
        (75, 110), (125, 110),
        (78, 55), (122, 110),
        (90, 140), (110, 140),
        (88, 170), (112, 170),
        (90, 200), (110, 200),
    ]
    scores = [0.9] * 17
    pose = PoseResult(0, 1, kps, scores)
    behaviour, _ = classify_window(
        WindowSignals(
            features=_features(
                speed_mean=2.0,
                stop_duration_s=0.8,
                trajectory_variance=4.0,
                pose_motion=1.0,
                pose_confidence=0.9,
            ),
            pose=pose,
            bbox_height=200.0,
            speed_ratio=0.4,
        ),
        settings,
    )
    assert behaviour in (TimelineBehaviour.PHONE_USAGE, TimelineBehaviour.STANDING, TimelineBehaviour.WAITING)
