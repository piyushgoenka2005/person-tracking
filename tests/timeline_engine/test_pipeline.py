"""Integration tests for timeline pipeline with mocked CV."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from engine.detection.schemas import DetectionBox, FrameDetections
from engine.tracking.schemas import DetectionInput, TrackedDetection, TrackAggregate
from engine.config import TimelineSettings
from engine.pipeline import TimelinePipeline
from engine.types import PoseResult


@pytest.fixture
def tiny_video(tmp_path: Path) -> Path:
    import cv2

    path = tmp_path / "test.mp4"
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (64, 64),
    )
    for _ in range(5):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_pipeline_with_mocks(tiny_video: Path, tmp_path: Path):
    settings = TimelineSettings(
        input_path=tiny_video,
        output_dir=tmp_path / "output",
        yolo_model_name="yolo11x.pt",
        frame_stride=1,
        batch_size=2,
        min_segment_s=0.1,
    )

    fake_frames = [
        FrameDetections(
            frame_index=i,
            timestamp_ms=i * 100.0,
            frame_width=64,
            frame_height=64,
            boxes=[
                DetectionBox(10.0, 10.0, 30.0, 50.0, 0.9, 0),
            ],
        )
        for i in range(5)
    ]

    track_obs = [
        TrackedDetection(i, i * 100.0, 10.0, 10.0, 30.0, 50.0, 0.9, 1)
        for i in range(5)
    ]
    fake_tracks = [TrackAggregate(local_track_id=1, observations=track_obs)]

    fake_pose = PoseResult(0, 1, [(20, 15)] * 17, [0.8] * 17)

    with (
        patch.object(TimelinePipeline, "_run_detection", return_value=fake_frames),
        patch("engine.pipeline.TrackingService") as mock_tracking_cls,
        patch.object(TimelinePipeline, "_run_pose", return_value=({(i, 1): fake_pose for i in range(5)}, {})),
    ):
        mock_tracking_cls.return_value.track_detections.return_value = (fake_tracks, 5)
        pipeline = TimelinePipeline(settings)
        segments, context = pipeline.run()

    out = settings.output_dir
    assert (out / "person_timeline.csv").is_file()
    assert (out / "person_summary.xlsx").is_file()
    assert (out / "session_summary.json").is_file()
    assert context.video.path.endswith("test.mp4")
