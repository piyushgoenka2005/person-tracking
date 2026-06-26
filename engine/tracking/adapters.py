"""Group tracker inputs by frame."""

from __future__ import annotations

from collections import defaultdict

from engine.tracking.schemas import DetectionInput


def group_detections_by_frame(
    detections: list[DetectionInput],
) -> dict[int, list[DetectionInput]]:
    grouped: dict[int, list[DetectionInput]] = defaultdict(list)
    for det in detections:
        grouped[det.frame_index].append(det)
    return dict(sorted(grouped.items()))
