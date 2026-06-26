"""Excel per-person behaviour summary."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from openpyxl import Workbook

from engine.types import TimelineSegment


def export_person_summary_xlsx(path: Path, segments: list[TimelineSegment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()

    # Sheet 1: duration totals per person per behaviour
    ws_totals = wb.active
    ws_totals.title = "Duration Totals"
    ws_totals.append(["person_id", "behaviour", "total_duration_s"])

    totals: dict[tuple[int, str], float] = defaultdict(float)
    for seg in segments:
        totals[(seg.person_id, seg.behaviour.value)] += seg.duration_s
    for (person_id, behaviour), duration in sorted(totals.items()):
        ws_totals.append([person_id, behaviour, round(duration, 3)])

    # Sheet 2: full segment list
    ws_segments = wb.create_sheet("Segments")
    ws_segments.append(
        ["person_id", "start_time", "end_time", "behaviour", "confidence", "duration_s"]
    )
    for seg in segments:
        ws_segments.append(
            [
                seg.person_id,
                seg.start_time,
                seg.end_time,
                seg.behaviour.value,
                seg.confidence,
                seg.duration_s,
            ]
        )

    wb.save(path)
