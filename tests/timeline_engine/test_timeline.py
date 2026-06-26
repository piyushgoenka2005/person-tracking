"""Tests for timeline generator."""

from __future__ import annotations

from engine.config import TimelineSettings
from engine.timeline.generator import TimelineGenerator, format_timestamp, format_timestamp_end
from engine.types import BehaviourLabel, TimelineBehaviour


def test_format_timestamp():
    assert format_timestamp(10_000) == "00:00:10"
    assert format_timestamp(65_000) == "00:01:05"
    assert format_timestamp(3_661_000) == "01:01:01"
    assert format_timestamp_end(500) == "00:00:01"
    assert format_timestamp_end(2000) == "00:00:02"


def test_merge_adjacent_segments():
    settings = TimelineSettings(min_segment_s=0.0, window_hop_s=0.5)
    gen = TimelineGenerator(settings)
    labels = [
        BehaviourLabel(1, 0, 1000, TimelineBehaviour.STANDING, 0.8),
        BehaviourLabel(1, 500, 1500, TimelineBehaviour.STANDING, 0.85),
        BehaviourLabel(1, 1000, 2000, TimelineBehaviour.WALKING, 0.9),
    ]
    segments = gen.generate(labels)
    assert len(segments) == 2
    assert segments[0].behaviour == TimelineBehaviour.STANDING
    assert segments[1].behaviour == TimelineBehaviour.WALKING


def test_walking_standing_walking_transitions():
    settings = TimelineSettings(min_segment_s=0.0, window_hop_s=0.5)
    gen = TimelineGenerator(settings)
    labels = [
        BehaviourLabel(1, 0, 1000, TimelineBehaviour.WALKING, 0.9),
        BehaviourLabel(1, 500, 1500, TimelineBehaviour.STANDING, 0.85),
        BehaviourLabel(1, 1000, 2000, TimelineBehaviour.WALKING, 0.92),
    ]
    segments = gen.generate(labels)
    behaviours = [s.behaviour for s in segments]
    assert behaviours == [
        TimelineBehaviour.WALKING,
        TimelineBehaviour.STANDING,
        TimelineBehaviour.WALKING,
    ]
    transitions = gen.extract_transitions(segments)
    assert len(transitions) == 2
    assert transitions[0].from_behaviour == TimelineBehaviour.WALKING
    assert transitions[0].to_behaviour == TimelineBehaviour.STANDING
    assert transitions[1].to_behaviour == TimelineBehaviour.WALKING


def test_min_segment_filter():
    settings = TimelineSettings(min_segment_s=2.0, window_hop_s=0.5)
    gen = TimelineGenerator(settings)
    labels = [
        BehaviourLabel(1, 0, 1000, TimelineBehaviour.STANDING, 0.8),
        BehaviourLabel(1, 500, 1500, TimelineBehaviour.WALKING, 0.9),
        BehaviourLabel(1, 1000, 2000, TimelineBehaviour.WALKING, 0.9),
        BehaviourLabel(1, 1500, 2500, TimelineBehaviour.WALKING, 0.9),
        BehaviourLabel(1, 2000, 3000, TimelineBehaviour.WALKING, 0.9),
    ]
    segments = gen.generate(labels)
    behaviours = [s.behaviour for s in segments]
    assert TimelineBehaviour.WALKING in behaviours
