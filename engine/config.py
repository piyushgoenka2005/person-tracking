"""Timeline engine configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from engine.core.settings import EngineSettings, get_settings


@dataclass(frozen=True)
class TimelineSettings:
    """Settings for the standalone behaviour timeline pipeline."""

    input_path: Path = Path("assets/input.mp4")
    output_dir: Path = Path("output")
    yolo_model_name: str = "models/yolo11x.pt"
    pose_model_path: str = "models/rtmpose-m.onnx"
    device: str = "cpu"
    frame_stride: int = 1
    batch_size: int = 8
    confidence_threshold: float = 0.5
    min_box_area: float = 400.0
    pose_crop_padding: float = 0.15
    pose_keypoint_threshold: float = 0.3
    window_s: float = 1.0
    window_hop_s: float = 0.5
    min_segment_s: float = 0.5
    behaviour_confidence_threshold: float = 0.35
    walking_speed_threshold: float = 15.0
    standing_speed_threshold: float = 5.0
    waiting_dwell_s: float = 2.0
    waiting_stop_ratio: float = 0.55
    speed_std_threshold: float = 3.0
    acceleration_std_threshold: float = 8.0
    trajectory_variance_threshold: float = 25.0
    pose_motion_threshold: float = 4.0
    queue_score_threshold: float = 0.35
    queue_turn_frequency_cap: float = 0.25
    sitting_knee_angle_deg: float = 100.0
    sleeping_torso_angle_deg: float = 35.0
    phone_wrist_face_ratio: float = 0.35
    eating_hand_mouth_ratio: float = 0.25
    eating_cycle_min: int = 2

    # LocateAnything-3B (active detector in pipeline.py — see DETECTOR SWITCH comments)
    locateanything_model: str = "nvidia/LocateAnything-3B"
    locateanything_remote: bool = False  # False = local weights; True = HF Space API

    def to_engine_settings(self) -> EngineSettings:
        """Map timeline CLI settings onto shared engine settings."""
        base = get_settings()
        return base.model_copy(
            update={
                "yolo_model_name": self.yolo_model_name,
                "detection_confidence_threshold": self.confidence_threshold,
                "detection_min_box_area": self.min_box_area,
                "detection_batch_size": self.batch_size,
                "detection_frame_stride": self.frame_stride,
                "tracking_min_track_length": 1,
                "behaviour_walking_speed_threshold": self.walking_speed_threshold,
                "behaviour_standing_speed_threshold": self.standing_speed_threshold,
                "behaviour_waiting_dwell_s": self.waiting_dwell_s,
                "behaviour_confidence_threshold": self.behaviour_confidence_threshold,
                "queue_score_threshold": self.queue_score_threshold,
                "queue_turn_frequency_cap": self.queue_turn_frequency_cap,
            }
        )
