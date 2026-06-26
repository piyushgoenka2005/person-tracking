"""RTMPose inference on person crops via ONNX Runtime."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from engine.config import TimelineSettings
from engine.types import PoseResult

# RTMPose-m typical input size (width, height)
_INPUT_W = 192
_INPUT_H = 256
_NUM_KEYPOINTS = 17


class PoseService:
    """Top-down pose estimation on person bounding boxes."""

    def __init__(self, settings: TimelineSettings) -> None:
        self._settings = settings
        self._session = None
        self._use_fallback = False
        self._load_model()

    @property
    def model_path(self) -> str:
        return self._settings.pose_model_path

    @property
    def uses_fallback(self) -> bool:
        return self._use_fallback

    def _load_model(self) -> None:
        model_path = Path(self._settings.pose_model_path)
        if not model_path.is_file():
            self._use_fallback = True
            return
        try:
            import onnxruntime as ort

            providers = ["CPUExecutionProvider"]
            if self._settings.device.startswith("cuda"):
                providers.insert(0, "CUDAExecutionProvider")
            self._session = ort.InferenceSession(str(model_path), providers=providers)
        except Exception:
            self._use_fallback = True

    def estimate(
        self,
        frame: np.ndarray,
        *,
        frame_index: int,
        track_id: int,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> PoseResult:
        crop, offset_x, offset_y, scale_x, scale_y = self._crop_person(
            frame, x1, y1, x2, y2
        )
        if crop.size == 0:
            return PoseResult(frame_index, track_id, [], [])

        if self._session is not None:
            keypoints, scores = self._infer_onnx(crop, offset_x, offset_y, scale_x, scale_y)
        else:
            keypoints, scores = self._fallback_keypoints(
                x1, y1, x2, y2, frame.shape[1], frame.shape[0]
            )

        return PoseResult(frame_index, track_id, keypoints, scores)

    def _crop_person(
        self,
        frame: np.ndarray,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> tuple[np.ndarray, float, float, float, float]:
        h, w = frame.shape[:2]
        pad = self._settings.pose_crop_padding
        bw = x2 - x1
        bh = y2 - y1
        px = bw * pad
        py = bh * pad
        cx1 = int(max(0, x1 - px))
        cy1 = int(max(0, y1 - py))
        cx2 = int(min(w, x2 + px))
        cy2 = int(min(h, y2 + py))
        crop = frame[cy1:cy2, cx1:cx2]
        scale_x = (cx2 - cx1) / _INPUT_W if cx2 > cx1 else 1.0
        scale_y = (cy2 - cy1) / _INPUT_H if cy2 > cy1 else 1.0
        return crop, float(cx1), float(cy1), scale_x, scale_y

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        resized = cv2.resize(crop, (_INPUT_W, _INPUT_H))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - np.array([123.675, 116.28, 103.53], dtype=np.float32)) / np.array(
            [58.395, 57.12, 57.375], dtype=np.float32
        )
        return np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]

    def _decode_simcc(
        self,
        simcc_x: np.ndarray,
        simcc_y: np.ndarray,
        offset_x: float,
        offset_y: float,
        scale_x: float,
        scale_y: float,
    ) -> tuple[list[tuple[float, float]], list[float]]:
        keypoints: list[tuple[float, float]] = []
        scores: list[float] = []
        n_kpts = min(simcc_x.shape[1], simcc_y.shape[1], _NUM_KEYPOINTS)
        for i in range(n_kpts):
            ix = int(simcc_x[0, i].argmax())
            iy = int(simcc_y[0, i].argmax())
            sx = float(simcc_x[0, i, ix])
            sy = float(simcc_y[0, i, iy])
            # SimCC bins map to input resolution
            x = (ix / max(simcc_x.shape[-1] - 1, 1)) * _INPUT_W * scale_x + offset_x
            y = (iy / max(simcc_y.shape[-1] - 1, 1)) * _INPUT_H * scale_y + offset_y
            keypoints.append((x, y))
            scores.append(float(min(1.0, (sx + sy) / 2.0)))
        return keypoints, scores

    def _infer_onnx(
        self,
        crop: np.ndarray,
        offset_x: float,
        offset_y: float,
        scale_x: float,
        scale_y: float,
    ) -> tuple[list[tuple[float, float]], list[float]]:
        assert self._session is not None
        tensor = self._preprocess(crop)
        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: tensor})
        if len(outputs) >= 2:
            return self._decode_simcc(outputs[0], outputs[1], offset_x, offset_y, scale_x, scale_y)
        # Single heatmap output fallback
        return self._fallback_from_heatmap(outputs[0], offset_x, offset_y, scale_x, scale_y)

    def _fallback_from_heatmap(
        self,
        heatmaps: np.ndarray,
        offset_x: float,
        offset_y: float,
        scale_x: float,
        scale_y: float,
    ) -> tuple[list[tuple[float, float]], list[float]]:
        keypoints: list[tuple[float, float]] = []
        scores: list[float] = []
        hm = heatmaps[0] if heatmaps.ndim == 4 else heatmaps
        n = min(hm.shape[0], _NUM_KEYPOINTS)
        for i in range(n):
            plane = hm[i]
            iy, ix = np.unravel_index(int(plane.argmax()), plane.shape)
            score = float(plane[iy, ix])
            x = (ix / max(plane.shape[1] - 1, 1)) * _INPUT_W * scale_x + offset_x
            y = (iy / max(plane.shape[0] - 1, 1)) * _INPUT_H * scale_y + offset_y
            keypoints.append((x, y))
            scores.append(min(1.0, score))
        return keypoints, scores

    def _fallback_keypoints(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        frame_w: int,
        frame_h: int,
    ) -> tuple[list[tuple[float, float]], list[float]]:
        """Geometric skeleton from bbox when ONNX model is unavailable."""
        cx = (x1 + x2) / 2.0
        w = x2 - x1
        h = y2 - y1
        def pt(rx: float, ry: float) -> tuple[float, float]:
            return (cx + rx * w, y1 + ry * h)

        layout = [
            pt(0, 0.08),    # nose
            pt(-0.08, 0.06), pt(0.08, 0.06),  # eyes
            pt(-0.12, 0.08), pt(0.12, 0.08),  # ears
            pt(-0.22, 0.22), pt(0.22, 0.22),  # shoulders
            pt(-0.28, 0.42), pt(0.28, 0.42),  # elbows
            pt(-0.30, 0.55), pt(0.30, 0.55),  # wrists
            pt(-0.18, 0.52), pt(0.18, 0.52),  # hips
            pt(-0.16, 0.72), pt(0.16, 0.72),  # knees
            pt(-0.14, 0.92), pt(0.14, 0.92),  # ankles
        ]
        keypoints = [(max(0, min(frame_w, x)), max(0, min(frame_h, y))) for x, y in layout]
        scores = [0.5] * len(keypoints)
        return keypoints, scores
