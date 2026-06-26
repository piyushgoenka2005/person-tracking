"""Engine configuration."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class EngineSettings(BaseSettings):
    """Settings for the behaviour timeline engine."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "console"

    yolo_model_name: str = "models/yolo11x.pt"
    detection_confidence_threshold: float = 0.5
    detection_min_box_area: float = 400.0
    detection_batch_size: int = 8
    detection_frame_stride: int = 1
    detection_person_class_id: int = 0

    tracking_activation_threshold: float = 0.5
    tracking_low_threshold: float = 0.1
    tracking_lost_buffer_frames: int = 30
    tracking_match_threshold: float = 0.55
    tracking_min_consecutive_frames: int = 1
    tracking_min_track_length: int = 1

    behaviour_walking_speed_threshold: float = 15.0
    behaviour_standing_speed_threshold: float = 5.0
    behaviour_waiting_dwell_s: float = 2.0
    behaviour_confidence_threshold: float = 0.35

    queue_score_threshold: float = 0.35
    queue_turn_frequency_cap: float = 0.25

@lru_cache
def get_settings() -> EngineSettings:
    return EngineSettings()
