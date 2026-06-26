"""JSON session summary export."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from engine.types import BehaviourTransition, SessionContext, TimelineSegment


def export_session_json(
    path: Path,
    context: SessionContext,
    segments: list[TimelineSegment],
    transitions: list[BehaviourTransition] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    per_person: dict[int, list[dict]] = defaultdict(list)
    for seg in segments:
        per_person[seg.person_id].append(
            {
                "start": seg.start_time,
                "end": seg.end_time,
                "behaviour": seg.behaviour.value,
                "confidence": seg.confidence,
                "duration_s": seg.duration_s,
            }
        )

    behaviour_totals: dict[str, float] = defaultdict(float)
    for seg in segments:
        behaviour_totals[seg.behaviour.value] += seg.duration_s

    payload = {
        "video": {
            "path": context.video.path,
            "fps": context.video.fps,
            "width": context.video.frame_width,
            "height": context.video.frame_height,
            "duration_s": round(context.video.duration_s, 3),
            "total_frames": context.video.total_frames,
        },
        "models": {
            "yolo": context.yolo_model,
            "pose": context.pose_model,
            "device": context.device,
            "frame_stride": context.frame_stride,
        },
        "session": {
            "person_count": context.person_count,
            "segment_count": len(segments),
            "behaviour_distribution_s": {k: round(v, 3) for k, v in sorted(behaviour_totals.items())},
            "started_at": context.started_at,
            "completed_at": context.completed_at,
        },
        "person_timelines": {
            f"Person {pid}": entries for pid, entries in sorted(per_person.items())
        },
        "behaviour_transitions": [
            {
                "person_id": tr.person_id,
                "time": tr.transition_time,
                "from": tr.from_behaviour.value,
                "to": tr.to_behaviour.value,
                "confidence": tr.confidence,
            }
            for tr in (transitions or [])
        ],
        "qa": {
            "warnings": context.qa_warnings,
            "errors": context.qa_errors,
            "status": "failed" if context.qa_errors else ("warning" if context.qa_warnings else "ok"),
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
