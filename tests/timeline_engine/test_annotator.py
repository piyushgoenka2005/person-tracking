"""Tests for annotated video renderer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from engine.render.annotator import render_annotated_video
from engine.types import PoseResult, TimelineBehaviour, TimelineSegment


def _write_test_video(path: Path, frames: int = 10) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (64, 64),
    )
    for _ in range(frames):
        writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    writer.release()


def test_render_holds_sparse_observations(tmp_path: Path):
    source = tmp_path / "src.mp4"
    output = tmp_path / "out.mp4"
    _write_test_video(source, frames=10)

    observations = {
        0: [(1, 10.0, 10.0, 30.0, 50.0, 0.0)],
        5: [(1, 12.0, 12.0, 32.0, 52.0, 500.0)],
    }
    segments = [
        TimelineSegment(
            person_id=1,
            start_time="00:00:00",
            end_time="00:00:01",
            behaviour=TimelineBehaviour.WALKING,
            confidence=0.9,
            duration_s=1.0,
            start_ms=0,
            end_ms=1000,
        )
    ]
    draw_calls: list[int] = []

    def _spy_draw(frame, overlay, **kwargs):
        draw_calls.append(1)

    with patch("engine.render.annotator._draw_track", side_effect=_spy_draw):
        render_annotated_video(
            source_path=source,
            output_path=output,
            observations_by_frame=observations,
            poses={},
            segments=segments,
            track_to_person={1: 1},
            fps=10.0,
            frame_stride=5,
            track_hold_frames=10,
        )

    assert output.is_file()
    # Frames 0-5 should draw (hold between sparse updates); not just frames 0 and 5.
    assert len(draw_calls) >= 6
