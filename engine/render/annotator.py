"""Annotated video renderer with boxes, skeleton, and behaviour labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from engine.types import COCO_SKELETON, PoseResult, TimelineBehaviour, TimelineSegment


@dataclass
class _TrackOverlay:
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    timestamp_ms: float
    pose: PoseResult | None
    last_seen_frame: int


def _track_color(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(track_id)
    return int(rng.integers(60, 255)), int(rng.integers(60, 255)), int(rng.integers(60, 255))


def _behaviour_at_time(
    segments: list[TimelineSegment],
    person_id: int,
    timestamp_ms: float,
) -> TimelineBehaviour | None:
    for seg in segments:
        if seg.person_id == person_id and seg.start_ms <= timestamp_ms <= seg.end_ms:
            return seg.behaviour
    return None


def _draw_track(
    frame,
    overlay: _TrackOverlay,
    *,
    person_id: int,
    segments: list[TimelineSegment],
    timestamp_ms: float,
    pose_threshold: float,
) -> None:
    color = _track_color(overlay.track_id)
    ix1, iy1, ix2, iy2 = map(int, [overlay.x1, overlay.y1, overlay.x2, overlay.y2])
    cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), color, 2)

    behaviour = _behaviour_at_time(segments, person_id, timestamp_ms)
    beh_text = behaviour.value if behaviour else "?"
    label = f"P{person_id} ID{overlay.track_id} {beh_text}"
    cv2.putText(
        frame,
        label,
        (ix1, max(iy1 - 8, 16)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
        cv2.LINE_AA,
    )

    pose = overlay.pose
    if pose and pose.keypoints:
        for i, j in COCO_SKELETON:
            if i >= len(pose.keypoints) or j >= len(pose.keypoints):
                continue
            if pose.scores[i] < pose_threshold or pose.scores[j] < pose_threshold:
                continue
            p1 = tuple(map(int, pose.keypoints[i]))
            p2 = tuple(map(int, pose.keypoints[j]))
            cv2.line(frame, p1, p2, color, 2)
        for idx, (px, py) in enumerate(pose.keypoints):
            if idx < len(pose.scores) and pose.scores[idx] >= pose_threshold:
                cv2.circle(frame, (int(px), int(py)), 3, color, -1)


def render_annotated_video(
    *,
    source_path: Path,
    output_path: Path,
    observations_by_frame: dict[int, list[tuple[int, float, float, float, float, float]]],
    poses: dict[tuple[int, int], PoseResult],
    segments: list[TimelineSegment],
    track_to_person: dict[int, int],
    fps: float,
    pose_threshold: float = 0.3,
    frame_stride: int = 1,
    track_hold_frames: int | None = None,
) -> None:
    """
    Render annotated video with stable overlays.

    Detections may be sparse (frame_stride > 1). Each track's box and pose are
    held on screen until the next update or until the track expires.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {source_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    effective_fps = fps if fps > 0 else 25.0
    hold_frames = track_hold_frames if track_hold_frames is not None else max(30, frame_stride * 6)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        effective_fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to create output video: {output_path}")

    active: dict[int, _TrackOverlay] = {}
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            timestamp_ms = (frame_index / effective_fps) * 1000.0

            if frame_index in observations_by_frame:
                for track_id, x1, y1, x2, y2, obs_ts_ms in observations_by_frame[frame_index]:
                    pose = poses.get((frame_index, track_id))
                    active[track_id] = _TrackOverlay(
                        track_id=track_id,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        timestamp_ms=obs_ts_ms,
                        pose=pose,
                        last_seen_frame=frame_index,
                    )

            expired = [
                track_id
                for track_id, overlay in active.items()
                if frame_index - overlay.last_seen_frame > hold_frames
            ]
            for track_id in expired:
                del active[track_id]

            for overlay in active.values():
                person_id = track_to_person.get(overlay.track_id, overlay.track_id)
                _draw_track(
                    frame,
                    overlay,
                    person_id=person_id,
                    segments=segments,
                    timestamp_ms=timestamp_ms,
                    pose_threshold=pose_threshold,
                )

            writer.write(frame)
            frame_index += 1
    finally:
        capture.release()
        writer.release()
