"""ByteTrack multi-object tracking service."""

from __future__ import annotations

from engine.core.logging import get_logger
from engine.core.settings import EngineSettings, get_settings
from engine.tracking.adapters import group_detections_by_frame
from engine.tracking.bytetrack import ByteTrackTracker
from engine.tracking.quality import compute_track_quality
from engine.tracking.schemas import DetectionInput, TrackAggregate

logger = get_logger(__name__)


class TrackingService:
    """Associates person detections into persistent local track IDs using ByteTrack."""

    def __init__(self, settings: EngineSettings | None = None) -> None:
        self._settings = settings or get_settings()

    def track_detections(
        self,
        detections: list[DetectionInput],
        *,
        video_fps: float | None = None,
        track_activation_threshold: float | None = None,
        lost_track_buffer: int | None = None,
        minimum_matching_threshold: float | None = None,
    ) -> tuple[list[TrackAggregate], int]:
        if not detections:
            return [], 0

        _ = video_fps
        activation = (
            track_activation_threshold
            if track_activation_threshold is not None
            else self._settings.tracking_activation_threshold
        )
        buffer = (
            lost_track_buffer
            if lost_track_buffer is not None
            else self._settings.tracking_lost_buffer_frames
        )
        match_thresh = (
            minimum_matching_threshold
            if minimum_matching_threshold is not None
            else self._settings.tracking_match_threshold
        )

        tracker = ByteTrackTracker(
            track_activation_threshold=activation,
            low_threshold=self._settings.tracking_low_threshold,
            match_threshold=match_thresh,
            lost_track_buffer=buffer,
            min_consecutive_frames=self._settings.tracking_min_consecutive_frames,
        )

        frames = group_detections_by_frame(detections)
        aggregates: dict[int, TrackAggregate] = {}
        last_seen_frame: dict[int, int] = {}
        frames_processed = 0

        for frame_index, frame_dets in frames.items():
            frames_processed += 1
            tracked = tracker.update(frame_index, frame_dets)

            seen_ids = {t.local_track_id for t in tracked}
            for local_id in list(last_seen_frame):
                if local_id not in seen_ids and local_id in aggregates:
                    gap_start = last_seen_frame[local_id] + 1
                    if gap_start <= frame_index:
                        aggregates[local_id].gap_count += frame_index - gap_start + 1

            for observation in tracked:
                local_id = observation.local_track_id
                if local_id not in aggregates:
                    aggregates[local_id] = TrackAggregate(local_track_id=local_id)

                prev_frame = last_seen_frame.get(local_id)
                if prev_frame is not None and frame_index - prev_frame > 1:
                    aggregates[local_id].gap_count += frame_index - prev_frame - 1

                aggregates[local_id].observations.append(observation)
                last_seen_frame[local_id] = frame_index

        results: list[TrackAggregate] = []
        for aggregate in aggregates.values():
            aggregate.quality_score = compute_track_quality(aggregate)
            if aggregate.frame_count >= self._settings.tracking_min_track_length:
                results.append(aggregate)

        logger.info(
            "bytetrack_completed",
            input_detections=len(detections),
            frames=frames_processed,
            tracks=len(results),
        )
        return results, frames_processed
