"""End-to-end behaviour timeline pipeline orchestration."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from engine.core.exceptions import PerceptionError
from engine.detection.filters import filter_batch
from engine.detection.service import DetectionService
from engine.detection.locateanything import (
    LocateAnythingDetectionService,
    is_la_remote_unavailable_error,
    resolve_locateanything_model,
)
from engine.detection.video import VideoIngestionService
from engine.tracking.schemas import DetectionInput, TrackAggregate
from engine.tracking.service import TrackingService

from engine.behaviour.engine import BehaviourEngine
from engine.config import TimelineSettings
from engine.export.csv_exporter import export_timeline_csv, export_transitions_csv
from engine.export.json_exporter import export_session_json
from engine.export.xlsx_exporter import export_person_summary_xlsx
from engine.pose.service import PoseService
from engine.qa import validators as qa
from engine.render.annotator import render_annotated_video
from engine.timeline.generator import TimelineGenerator
from engine.types import SessionContext, TimelineSegment, VideoInfo


class TimelinePipeline:
    """Runs detection → tracking → pose → behaviour → timeline → export."""

    def __init__(self, settings: TimelineSettings) -> None:
        self._settings = settings
        self._engine_settings = settings.to_engine_settings()
        self._qa = qa.QAReport()

    @property
    def qa_report(self) -> qa.QAReport:
        return self._qa

    def run(self) -> tuple[list[TimelineSegment], SessionContext]:
        started_at = datetime.now(timezone.utc).isoformat()
        input_path = self._settings.input_path.resolve()
        output_dir = self._settings.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        ingestion = VideoIngestionService()
        if input_path.is_file():
            local_path = input_path
            is_temp = False
        else:
            local_path, is_temp = ingestion.resolve_local_path(input_path)
        meta = ingestion.probe_video(local_path)

        video_info = VideoInfo(
            path=str(input_path),
            fps=meta.fps,
            frame_width=meta.frame_width,
            frame_height=meta.frame_height,
            total_frames=meta.total_frames,
            duration_s=meta.duration_ms / 1000.0,
        )

        # ---------------------------------------------------------------------
        # DETECTOR SWITCH — person detection only (tracking/pose/behaviour unchanged)
        #
        # ACTIVE: Hugging Face LocateAnything-3B (open-vocabulary detector)
        # TO USE YOLO AGAIN: comment the LocateAnything block below and uncomment
        # the YOLOv11x block.
        # ---------------------------------------------------------------------

        detection_service = LocateAnythingDetectionService(
            model_path=resolve_locateanything_model(self._settings.locateanything_model),
            device=self._settings.device,
            remote=self._settings.locateanything_remote,
            min_box_area=self._settings.min_box_area,
        )
        detection_model_name = detection_service.model_name

        # --- YOLOv11x (comment LocateAnything block above, uncomment below to switch back) ---
        # detection_service = DetectionService(self._engine_settings)
        # detection_model_name = detection_service.model_name

        try:
            frame_detections = self._run_detection(ingestion, local_path, detection_service)
        except PerceptionError as exc:
            if (
                isinstance(detection_service, LocateAnythingDetectionService)
                and self._settings.locateanything_remote
                and is_la_remote_unavailable_error(exc)
            ):
                self._qa.warn(
                    "LocateAnything HF Space unavailable (ZeroGPU limit); using YOLOv11x for this run"
                )
                detection_service = DetectionService(self._engine_settings)
                detection_model_name = f"{detection_service.model_name} (YOLO fallback)"
                frame_detections = self._run_detection(ingestion, local_path, detection_service)
            else:
                raise
        person_frames = sum(1 for f in frame_detections if f.boxes)
        qa.validate_detection(len(frame_detections), person_frames, self._qa)

        detections = self._to_detection_inputs(frame_detections)
        tracking_service = TrackingService(self._engine_settings)
        tracks, _ = tracking_service.track_detections(detections, video_fps=meta.fps)

        observations_by_frame = self._build_observation_index(tracks)
        qa.validate_tracking(observations_by_frame, self._qa)

        poses, _ = self._run_pose(local_path, tracks)
        qa.validate_pose(poses, self._qa)
        if PoseService(self._settings).uses_fallback:
            self._qa.warn("RTMPose ONNX model not found; using geometric pose fallback")

        behaviour_engine = BehaviourEngine(self._settings)
        labels = behaviour_engine.classify_tracks(tracks, poses, video_fps=meta.fps)

        generator = TimelineGenerator(self._settings)
        segments = generator.generate(labels)
        transitions = generator.extract_transitions(segments)
        qa.validate_timeline(segments, self._qa)

        track_to_person = self._track_person_map(tracks)

        self._export_all(
            output_dir,
            segments,
            transitions,
            local_path,
            observations_by_frame,
            poses,
            track_to_person,
            meta.fps,
        )

        if is_temp:
            local_path.unlink(missing_ok=True)

        completed_at = datetime.now(timezone.utc).isoformat()
        context = SessionContext(
            video=video_info,
            yolo_model=detection_model_name,
            pose_model=self._settings.pose_model_path,
            device=self._settings.device,
            frame_stride=self._settings.frame_stride,
            person_count=len({s.person_id for s in segments}),
            behaviour_distribution=self._behaviour_distribution(segments),
            qa_warnings=list(self._qa.warnings),
            qa_errors=list(self._qa.errors),
            started_at=started_at,
            completed_at=completed_at,
        )
        export_session_json(output_dir / "session_summary.json", context, segments, transitions)
        qa.validate_exports(output_dir, segments, self._qa)
        return segments, context

    def _run_detection(self, ingestion, local_path, detection_service):
        frame_buffer: list[tuple[int, np.ndarray, float]] = []
        all_frames: list = []
        batch_size = self._settings.batch_size
        if isinstance(detection_service, LocateAnythingDetectionService):
            batch_size = 1
        stride = self._settings.frame_stride

        for frame_index, frame_array, timestamp_ms in ingestion.extract_frames(
            local_path, frame_stride=stride
        ):
            frame_buffer.append((frame_index, frame_array, timestamp_ms))
            if len(frame_buffer) >= batch_size:
                raw = detection_service.predict_batch(frame_buffer)
                all_frames.extend(filter_batch(raw, self._engine_settings))
                frame_buffer.clear()

        if frame_buffer:
            raw = detection_service.predict_batch(frame_buffer)
            all_frames.extend(filter_batch(raw, self._engine_settings))

        return all_frames

    def _to_detection_inputs(self, frame_detections) -> list[DetectionInput]:
        inputs: list[DetectionInput] = []
        for frame in frame_detections:
            for box in frame.boxes:
                inputs.append(
                    DetectionInput(
                        frame_index=frame.frame_index,
                        timestamp_ms=frame.timestamp_ms,
                        x1=box.x1,
                        y1=box.y1,
                        x2=box.x2,
                        y2=box.y2,
                        confidence=box.confidence,
                    )
                )
        return inputs

    def _build_observation_index(
        self, tracks: list[TrackAggregate]
    ) -> dict[int, list[tuple[int, float, float, float, float, float]]]:
        by_frame: dict[int, list] = defaultdict(list)
        for track in tracks:
            for obs in track.observations:
                by_frame[obs.frame_index].append(
                    (obs.local_track_id, obs.x1, obs.y1, obs.x2, obs.y2, obs.timestamp_ms)
                )
        return dict(by_frame)

    def _run_pose(self, local_path, tracks):
        import cv2

        pose_service = PoseService(self._settings)
        poses: dict = {}

        obs_by_frame: dict[int, list] = defaultdict(list)
        for track in tracks:
            for obs in track.observations:
                obs_by_frame[obs.frame_index].append((track.local_track_id, obs))

        capture = cv2.VideoCapture(str(local_path))
        if not capture.isOpened():
            return poses, {}

        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index in obs_by_frame:
                    for track_id, obs in obs_by_frame[frame_index]:
                        result = pose_service.estimate(
                            frame,
                            frame_index=frame_index,
                            track_id=track_id,
                            x1=obs.x1,
                            y1=obs.y1,
                            x2=obs.x2,
                            y2=obs.y2,
                        )
                        poses[(frame_index, track_id)] = result
                frame_index += 1
        finally:
            capture.release()
        return poses, {}

    def _track_person_map(self, tracks: list[TrackAggregate]) -> dict[int, int]:
        sorted_track_ids = sorted({t.local_track_id for t in tracks})
        return {tid: idx + 1 for idx, tid in enumerate(sorted_track_ids)}

    def _behaviour_distribution(self, segments: list[TimelineSegment]) -> dict[str, float]:
        dist: dict[str, float] = defaultdict(float)
        for seg in segments:
            dist[seg.behaviour.value] += seg.duration_s
        return dict(dist)

    def _export_all(
        self,
        output_dir,
        segments,
        transitions,
        local_path,
        observations_by_frame,
        poses,
        track_to_person,
        fps,
    ) -> None:
        export_timeline_csv(output_dir / "person_timeline.csv", segments)
        export_transitions_csv(output_dir / "person_transitions.csv", transitions)
        export_person_summary_xlsx(output_dir / "person_summary.xlsx", segments)

        render_annotated_video(
            source_path=local_path,
            output_path=output_dir / "annotated_video.mp4",
            observations_by_frame=observations_by_frame,
            poses=poses,
            segments=segments,
            track_to_person=track_to_person,
            fps=fps,
            pose_threshold=self._settings.pose_keypoint_threshold,
            frame_stride=self._settings.frame_stride,
            track_hold_frames=self._engine_settings.tracking_lost_buffer_frames,
        )
