"""Merge behaviour windows into per-person timeline segments."""

from __future__ import annotations

from collections import defaultdict

from engine.config import TimelineSettings
from engine.types import BehaviourLabel, BehaviourTransition, TimelineBehaviour, TimelineSegment


def format_timestamp(ms: float) -> str:
    total_s = max(0, int(ms // 1000))
    hours = total_s // 3600
    minutes = (total_s % 3600) // 60
    seconds = total_s % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_timestamp_end(ms: float) -> str:
    """Ceil to the next whole second so sub-second segments are visible in HH:MM:SS."""
    total_s = max(0, int((ms + 999) // 1000))
    hours = total_s // 3600
    minutes = (total_s % 3600) // 60
    seconds = total_s % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class TimelineGenerator:
    """Convert overlapping 1s window labels into non-overlapping behaviour segments."""

    def __init__(self, settings: TimelineSettings) -> None:
        self._settings = settings

    def generate(self, labels: list[BehaviourLabel]) -> list[TimelineSegment]:
        by_track: dict[int, list[BehaviourLabel]] = defaultdict(list)
        for label in sorted(labels, key=lambda l: (l.track_id, l.start_ms)):
            by_track[label.track_id].append(label)

        segments: list[TimelineSegment] = []
        sorted_track_ids = sorted(by_track.keys())
        person_map = {tid: idx + 1 for idx, tid in enumerate(sorted_track_ids)}

        hop_ms = self._settings.window_hop_s * 1000.0
        for track_id in sorted_track_ids:
            hop_slices = self._windows_to_hop_slices(by_track[track_id], hop_ms)
            merged = self._merge_consecutive(hop_slices)
            filtered = self._filter_short_flicker(merged)
            person_id = person_map[track_id]
            for label in filtered:
                duration_s = max(0.0, (label.end_ms - label.start_ms) / 1000.0)
                segments.append(
                    TimelineSegment(
                        person_id=person_id,
                        start_time=format_timestamp(label.start_ms),
                        end_time=format_timestamp_end(label.end_ms),
                        behaviour=label.behaviour,
                        confidence=label.confidence,
                        duration_s=round(duration_s, 3),
                        start_ms=label.start_ms,
                        end_ms=label.end_ms,
                    )
                )
        return sorted(segments, key=lambda s: (s.person_id, s.start_ms))

    def extract_transitions(self, segments: list[TimelineSegment]) -> list[BehaviourTransition]:
        """Build behaviour transition events from merged segments."""
        by_person: dict[int, list[TimelineSegment]] = defaultdict(list)
        for seg in segments:
            by_person[seg.person_id].append(seg)

        transitions: list[BehaviourTransition] = []
        for person_id in sorted(by_person):
            ordered = sorted(by_person[person_id], key=lambda s: s.start_ms)
            for prev, curr in zip(ordered, ordered[1:]):
                if prev.behaviour == curr.behaviour:
                    continue
                transitions.append(
                    BehaviourTransition(
                        person_id=person_id,
                        transition_time=curr.start_time,
                        transition_ms=curr.start_ms,
                        from_behaviour=prev.behaviour,
                        to_behaviour=curr.behaviour,
                        confidence=round((prev.confidence + curr.confidence) / 2.0, 4),
                    )
                )
        return transitions

    def _windows_to_hop_slices(
        self, labels: list[BehaviourLabel], hop_ms: float
    ) -> list[BehaviourLabel]:
        """Map each 1s window label onto its hop interval [start, start + hop)."""
        slices: list[BehaviourLabel] = []
        for label in labels:
            slices.append(
                BehaviourLabel(
                    track_id=label.track_id,
                    start_ms=label.start_ms,
                    end_ms=label.start_ms + hop_ms,
                    behaviour=label.behaviour,
                    confidence=label.confidence,
                )
            )
        return slices

    def _merge_consecutive(self, labels: list[BehaviourLabel]) -> list[BehaviourLabel]:
        if not labels:
            return []
        merged: list[BehaviourLabel] = [labels[0]]
        for label in labels[1:]:
            prev = merged[-1]
            if label.behaviour == prev.behaviour:
                merged[-1] = BehaviourLabel(
                    track_id=prev.track_id,
                    start_ms=prev.start_ms,
                    end_ms=label.end_ms,
                    behaviour=prev.behaviour,
                    confidence=max(prev.confidence, label.confidence),
                )
            else:
                merged.append(label)
        return merged

    def _filter_short_flicker(self, labels: list[BehaviourLabel]) -> list[BehaviourLabel]:
        """Drop unknown flicker and absorb sub-minimum segments into neighbours."""
        min_ms = self._settings.min_segment_s * 1000.0
        if not labels or min_ms <= 0:
            return labels

        result = list(labels)
        changed = True
        while changed:
            changed = False
            filtered: list[BehaviourLabel] = []
            idx = 0
            while idx < len(result):
                label = result[idx]
                duration_ms = label.end_ms - label.start_ms
                if duration_ms >= min_ms:
                    filtered.append(label)
                    idx += 1
                    continue
                if label.behaviour == TimelineBehaviour.UNKNOWN:
                    idx += 1
                    changed = True
                    continue
                if filtered:
                    prev = filtered[-1]
                    filtered[-1] = BehaviourLabel(
                        track_id=prev.track_id,
                        start_ms=prev.start_ms,
                        end_ms=max(prev.end_ms, label.end_ms),
                        behaviour=prev.behaviour,
                        confidence=max(prev.confidence, label.confidence),
                    )
                    idx += 1
                    changed = True
                    continue
                if idx + 1 < len(result):
                    nxt = result[idx + 1]
                    filtered.append(
                        BehaviourLabel(
                            track_id=nxt.track_id,
                            start_ms=label.start_ms,
                            end_ms=nxt.end_ms,
                            behaviour=nxt.behaviour,
                            confidence=max(label.confidence, nxt.confidence),
                        )
                    )
                    idx += 2
                    changed = True
                    continue
                filtered.append(label)
                idx += 1
            result = filtered
        return self._merge_consecutive(result)
