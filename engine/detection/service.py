"""YOLO batch inference service."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from engine.core.exceptions import PerceptionError
from engine.core.logging import get_logger
from engine.core.settings import EngineSettings, get_settings
from engine.detection.schemas import DetectionBox, FrameDetections

if TYPE_CHECKING:
    from ultralytics import YOLO

logger = get_logger(__name__)


class DetectionService:
    """Wraps YOLO model loading and batch person detection inference."""

    def __init__(self, settings: EngineSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model: YOLO | None = None

    @property
    def model_name(self) -> str:
        return self._settings.yolo_model_name

    def _load_model(self) -> YOLO:
        if self._model is None:
            try:
                from ultralytics import YOLO

                self._model = YOLO(self._settings.yolo_model_name)
                logger.info("yolo_model_loaded", model=self._settings.yolo_model_name)
            except Exception as exc:
                raise PerceptionError(
                    f"Failed to load YOLO model {self._settings.yolo_model_name}"
                ) from exc
        return self._model

    def predict_batch(
        self,
        frames: list[tuple[int, np.ndarray, float]],
    ) -> list[FrameDetections]:
        if not frames:
            return []

        model = self._load_model()
        images = [frame[1] for frame in frames]
        frame_meta = [(f[0], f[2], img.shape[1], img.shape[0]) for f, img in zip(frames, images)]

        try:
            results = model.predict(
                images,
                verbose=False,
                conf=self._settings.detection_confidence_threshold,
                classes=[self._settings.detection_person_class_id],
            )
        except Exception as exc:
            raise PerceptionError("YOLO batch inference failed") from exc

        output: list[FrameDetections] = []
        for (frame_index, timestamp_ms, width, height), result in zip(frame_meta, results):
            boxes: list[DetectionBox] = []
            if result.boxes is not None:
                for box in result.boxes:
                    xyxy = box.xyxy[0].tolist()
                    conf_val = box.conf[0]
                    cls_val = box.cls[0]
                    confidence = float(conf_val.item() if hasattr(conf_val, "item") else conf_val)
                    class_id = int(cls_val.item() if hasattr(cls_val, "item") else cls_val)
                    boxes.append(
                        DetectionBox(
                            x1=float(xyxy[0]),
                            y1=float(xyxy[1]),
                            x2=float(xyxy[2]),
                            y2=float(xyxy[3]),
                            confidence=confidence,
                            class_id=class_id,
                        )
                    )
            output.append(
                FrameDetections(
                    frame_index=frame_index,
                    timestamp_ms=timestamp_ms,
                    frame_width=width,
                    frame_height=height,
                    boxes=boxes,
                )
            )

        logger.info(
            "yolo_batch_inference",
            batch_size=len(frames),
            total_boxes=sum(len(f.boxes) for f in output),
        )
        return output
