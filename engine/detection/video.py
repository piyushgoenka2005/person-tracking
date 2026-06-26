"""Video frame extraction for the timeline engine."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from engine.core.exceptions import IngestionError
from engine.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class VideoMetadata:
    fps: float
    frame_width: int
    frame_height: int
    total_frames: int
    duration_ms: float


class VideoIngestionService:
    """Read local video files and yield sampled frames."""

    def resolve_local_path(self, video_path: str | Path) -> tuple[Path, bool]:
        path = Path(video_path)
        if path.is_file():
            return path.resolve(), False
        raise IngestionError(f"Video file not found: {video_path}")

    def probe_video(self, local_path: Path) -> VideoMetadata:
        capture = cv2.VideoCapture(str(local_path))
        if not capture.isOpened():
            raise IngestionError(f"Unable to open video: {local_path}")
        try:
            fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
            frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_ms = (total_frames / fps) * 1000.0 if fps > 0 else 0.0
            return VideoMetadata(
                fps=fps,
                frame_width=frame_width,
                frame_height=frame_height,
                total_frames=total_frames,
                duration_ms=duration_ms,
            )
        finally:
            capture.release()

    def extract_frames(
        self,
        local_path: Path,
        *,
        frame_stride: int = 1,
    ) -> Generator[tuple[int, np.ndarray, float], None, None]:
        if frame_stride < 1:
            raise ValueError("frame_stride must be >= 1")

        capture = cv2.VideoCapture(str(local_path))
        if not capture.isOpened():
            raise IngestionError(f"Unable to open video: {local_path}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        sampled = 0
        try:
            frame_index = 0
            while True:
                success, frame = capture.read()
                if not success:
                    break
                if frame_index % frame_stride == 0:
                    timestamp_ms = (frame_index / fps) * 1000.0
                    sampled += 1
                    yield frame_index, frame, timestamp_ms
                frame_index += 1
        finally:
            capture.release()

        logger.info(
            "frames_extracted",
            video=str(local_path),
            stride=frame_stride,
            sampled_frames=sampled,
        )
