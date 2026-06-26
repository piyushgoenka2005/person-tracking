"""Detection result filtering."""

from __future__ import annotations

from engine.core.settings import EngineSettings, get_settings
from engine.detection.schemas import DetectionBox, FrameDetections


def filter_person_detections(
    frame: FrameDetections,
    *,
    confidence_threshold: float,
    min_box_area: float,
    person_class_id: int = 0,
) -> FrameDetections:
    filtered_boxes: list[DetectionBox] = []
    for box in frame.boxes:
        if box.class_id != person_class_id:
            continue
        if box.confidence < confidence_threshold:
            continue
        if box.area < min_box_area:
            continue
        filtered_boxes.append(box)

    return FrameDetections(
        frame_index=frame.frame_index,
        timestamp_ms=frame.timestamp_ms,
        frame_width=frame.frame_width,
        frame_height=frame.frame_height,
        boxes=filtered_boxes,
    )


def filter_batch(
    frames: list[FrameDetections],
    settings: EngineSettings | None = None,
    *,
    confidence_threshold: float | None = None,
    min_box_area: float | None = None,
) -> list[FrameDetections]:
    cfg = settings or get_settings()
    threshold = confidence_threshold if confidence_threshold is not None else cfg.detection_confidence_threshold
    min_area = min_box_area if min_box_area is not None else cfg.detection_min_box_area
    return [
        filter_person_detections(
            frame,
            confidence_threshold=threshold,
            min_box_area=min_area,
            person_class_id=cfg.detection_person_class_id,
        )
        for frame in frames
    ]
