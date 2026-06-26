"""CSV timeline export."""

from __future__ import annotations

import csv
from pathlib import Path

from engine.types import BehaviourTransition, TimelineSegment


def export_timeline_csv(path: Path, segments: list[TimelineSegment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["person_id", "start_time", "end_time", "behaviour", "confidence"])
        for seg in segments:
            writer.writerow(
                [
                    seg.person_id,
                    seg.start_time,
                    seg.end_time,
                    seg.behaviour.value,
                    f"{seg.confidence:.4f}",
                ]
            )


def export_transitions_csv(path: Path, transitions: list[BehaviourTransition]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["person_id", "transition_time", "from_behaviour", "to_behaviour", "confidence"]
        )
        for tr in transitions:
            writer.writerow(
                [
                    tr.person_id,
                    tr.transition_time,
                    tr.from_behaviour.value,
                    tr.to_behaviour.value,
                    f"{tr.confidence:.4f}",
                ]
            )
