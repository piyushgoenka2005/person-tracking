"""NVIDIA LocateAnything-3B detection integration.

LocateAnything is a vision-language model for open-vocabulary detection and
grounding. It can replace YOLO at the detection stage but does not provide
tracking, pose, or behaviour analysis on its own.

Model: https://huggingface.co/nvidia/LocateAnything-3B
Demo:  https://huggingface.co/spaces/nvidia/LocateAnything
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from engine.core.exceptions import PerceptionError
from engine.core.logging import get_logger
from engine.detection.schemas import DetectionBox, FrameDetections

logger = get_logger(__name__)

BOX_PATTERN = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")


def resolve_locateanything_model(path: str) -> str:
    """Prefer a local models/ directory when the HF hub id is not cached yet."""
    candidate = Path(path)
    if candidate.is_dir() and any(candidate.iterdir()):
        return str(candidate.resolve())
    default_dir = Path("models") / "LocateAnything-3B"
    if default_dir.is_dir() and any(default_dir.iterdir()):
        return str(default_dir.resolve())
    return path


def parse_boxes(answer: str, image_width: int, image_height: int) -> list[dict[str, float]]:
    """Parse LocateAnything coordinate tokens into pixel boxes."""
    boxes: list[dict[str, float]] = []
    for match in BOX_PATTERN.finditer(answer):
        x1, y1, x2, y2 = (int(g) for g in match.groups())
        boxes.append(
            {
                "x1": x1 / 1000 * image_width,
                "y1": y1 / 1000 * image_height,
                "x2": x2 / 1000 * image_width,
                "y2": y2 / 1000 * image_height,
            }
        )
    return boxes


def _resize_for_api(image: Image.Image, max_side: int = 512) -> tuple[Image.Image, float, float]:
    """Downscale frames for HF Space to stay within ZeroGPU time limits."""
    width, height = image.size
    if max(width, height) <= max_side:
        return image, 1.0, 1.0
    scale = max_side / max(width, height)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    resized = image.resize(new_size, Image.Resampling.BILINEAR)
    sx = width / resized.size[0]
    sy = height / resized.size[1]
    return resized, sx, sy


def _scale_boxes(boxes: list[dict[str, float]], sx: float, sy: float) -> list[dict[str, float]]:
    if sx == 1.0 and sy == 1.0:
        return boxes
    return [
        {
            "x1": b["x1"] * sx,
            "y1": b["y1"] * sy,
            "x2": b["x2"] * sx,
            "y2": b["y2"] * sy,
        }
        for b in boxes
    ]


def is_la_remote_unavailable_error(exc: BaseException) -> bool:
    """True when the public HF Space cannot serve inference (quota / duration limits)."""
    msg = str(exc).lower()
    markers = (
        "gpu duration",
        "gpu quota",
        "quota exceeded",
        "zerogpu",
        "no cuda gpus",
        "retry later",
    )
    return any(m in msg for m in markers)


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    import cv2

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


@dataclass(frozen=True)
class LocateAnythingResult:
    answer: str
    boxes: list[dict[str, float]]
    inference_ms: float
    backend: str


class LocateAnythingWorker:
    """Local LocateAnything-3B worker (GPU recommended; CPU supported but slow)."""

    def __init__(
        self,
        model_path: str = "nvidia/LocateAnything-3B",
        *,
        device: str = "cuda",
    ) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        model_path = resolve_locateanything_model(model_path)
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("locateanything_cuda_unavailable", fallback="cpu")
            device = "cpu"

        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.device = device
        self.dtype = dtype
        self.model_path = model_path
        self._cpu_mode = device == "cpu"

        logger.info("locateanything_loading", model=model_path, device=device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        load_kwargs: dict[str, object] = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
        }
        if self._cpu_mode:
            load_kwargs["low_cpu_mem_usage"] = True
        self.model = AutoModel.from_pretrained(model_path, **load_kwargs).to(device).eval()
        logger.info("locateanything_model_loaded", model=model_path, device=device)

    def detect(
        self,
        image: Image.Image,
        categories: list[str] | None = None,
        *,
        generation_mode: str | None = None,
        max_new_tokens: int | None = None,
    ) -> dict[str, Any]:
        cats = categories or ["person"]
        prompt = f"Locate all the instances that matches the following description: {'</c>'.join(cats)}."
        if generation_mode is None:
            generation_mode = "fast" if self._cpu_mode else "hybrid"
        if max_new_tokens is None:
            max_new_tokens = 512 if self._cpu_mode else 2048
        return self.predict(
            image,
            prompt,
            generation_mode=generation_mode,
            max_new_tokens=max_new_tokens,
        )

    def ground_multi(
        self,
        image: Image.Image,
        phrase: str,
        *,
        generation_mode: str = "hybrid",
        max_new_tokens: int = 2048,
    ) -> dict[str, Any]:
        prompt = f"Locate all the instances that match the following description: {phrase}."
        return self.predict(image, prompt, generation_mode=generation_mode, max_new_tokens=max_new_tokens)

    def predict(
        self,
        image: Image.Image,
        question: str,
        *,
        generation_mode: str = "hybrid",
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        verbose: bool = False,
    ) -> dict[str, Any]:
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = self.processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=images, videos=videos, return_tensors="pt"
        )

        pixel_values = inputs["pixel_values"].to(device=self.device, dtype=self.dtype)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        image_grid_hws = inputs.get("image_grid_hws")
        if image_grid_hws is not None:
            if not isinstance(image_grid_hws, torch.Tensor):
                image_grid_hws = torch.tensor(image_grid_hws)
            image_grid_hws = image_grid_hws.to(self.device)

        with torch.no_grad():
            response = self.model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_grid_hws=image_grid_hws,
                tokenizer=self.tokenizer,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                generation_mode=generation_mode,
                temperature=temperature,
                do_sample=generation_mode != "fast",
                top_p=0.9,
                repetition_penalty=1.1,
                verbose=verbose,
            )

        answer = response[0] if isinstance(response, tuple) else response
        if isinstance(answer, torch.Tensor):
            answer = self.tokenizer.decode(answer, skip_special_tokens=False)
        return {"answer": str(answer)}


class LocateAnythingRemoteClient:
    """Call the public Hugging Face Space when local GPU inference is unavailable."""

    SPACE_ID = "nvidia/LocateAnything"

    def __init__(self, hf_token: str | None = None) -> None:
        from gradio_client import Client

        self._client = Client(self.SPACE_ID, token=hf_token)

    def detect(self, image: Image.Image, categories: list[str] | None = None) -> dict[str, Any]:
        from gradio_client import handle_file
        import tempfile

        categories = categories or ["person"]
        category = categories[0] if len(categories) == 1 else "</c>".join(categories)
        api_image, sx, sy = _resize_for_api(image, max_side=512)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            api_image.save(tmp.name, format="JPEG", quality=85)
            path = tmp.name

        started = time.perf_counter()
        try:
            result = self._client.predict(
                "Image",
                handle_file(path),
                None,
                "Detection",
                category,
                "fast",
                0.7,
                0.9,
                20,
                512,
                None,
                4,
                api_name="/run_inference",
            )
        finally:
            Path(path).unlink(missing_ok=True)

        elapsed_ms = (time.perf_counter() - started) * 1000
        answer, meta = self._extract_response(result)
        if meta.get("success") is False:
            error = str(meta.get("error", "LocateAnything Space inference failed"))
            hint = (
                "The NVIDIA HF Space is misconfigured (requests 240s GPU; HF max ~120s). "
                "Use local weights: python -m engine.run --no-la-remote --device cpu --frame-stride 10"
            )
            if "gpu duration" in error.lower():
                hint = (
                    "HF Space ZeroGPU duration exceeds platform limit (not your video). "
                    "Pipeline will fall back to YOLO if enabled, or run: "
                    "python -m engine.run --no-la-remote --device cpu --frame-stride 10"
                )
            raise PerceptionError(f"{error}. {hint}")
        return {
            "answer": answer,
            "inference_ms": elapsed_ms,
            "raw": result,
            "meta": meta,
            "scale": (sx, sy),
        }

    @staticmethod
    def _extract_response(result: Any) -> tuple[str, dict[str, Any]]:
        meta: dict[str, Any] = {}
        if isinstance(result, (list, tuple)) and len(result) >= 3 and isinstance(result[2], dict):
            meta = result[2]
            answer = meta.get("answer") or meta.get("text") or meta.get("output") or ""
            if isinstance(answer, str):
                return answer, meta
        if isinstance(result, str):
            return result, meta
        if isinstance(result, (list, tuple)):
            for item in result:
                if isinstance(item, str) and "<box>" in item:
                    return item, meta
                if isinstance(item, dict):
                    meta = item
                    answer = item.get("answer") or item.get("text") or ""
                    if answer:
                        return str(answer), meta
        return str(result), meta


class LocateAnythingDetectionService:
    """Detection service compatible with the timeline pipeline's frame batch API."""

    def __init__(
        self,
        *,
        model_path: str = "nvidia/LocateAnything-3B",
        device: str = "cuda",
        remote: bool = False,
        hf_token: str | None = None,
        categories: list[str] | None = None,
        min_box_area: float = 400.0,
        default_confidence: float = 1.0,
    ) -> None:
        model_path = resolve_locateanything_model(model_path)
        self.model_name = model_path if not remote else self.SPACE_ID
        self._categories = categories or ["person"]
        self._min_box_area = min_box_area
        self._default_confidence = default_confidence
        self._remote = remote
        self._worker: LocateAnythingWorker | LocateAnythingRemoteClient | None = None
        self._worker_kwargs = {
            "model_path": model_path,
            "device": device,
            "hf_token": hf_token,
        }

    SPACE_ID = LocateAnythingRemoteClient.SPACE_ID

    def _load_backend(self) -> LocateAnythingWorker | LocateAnythingRemoteClient:
        if self._worker is None:
            if self._remote:
                self._worker = LocateAnythingRemoteClient(hf_token=self._worker_kwargs.get("hf_token"))
            else:
                try:
                    self._worker = LocateAnythingWorker(
                        self._worker_kwargs["model_path"],
                        device=self._worker_kwargs["device"],
                    )
                except Exception as exc:
                    raise PerceptionError(
                        "Failed to load LocateAnything locally. Download weights first: "
                        "python scripts/download_locateanything.py"
                    ) from exc
        return self._worker

    def predict_frame(self, frame_index: int, bgr: np.ndarray, timestamp_ms: float) -> FrameDetections:
        image = bgr_to_pil(bgr)
        width, height = image.size
        backend = self._load_backend()
        started = time.perf_counter()

        if isinstance(backend, LocateAnythingRemoteClient):
            api_image, sx, sy = _resize_for_api(image, max_side=512)
            raw = backend.detect(api_image, self._categories)
            inference_ms = float(raw.get("inference_ms", 0))
            answer = raw["answer"]
            sx, sy = raw.get("scale", (sx, sy))
            parsed = parse_boxes(answer, api_image.size[0], api_image.size[1])
            parsed = _scale_boxes(parsed, sx, sy)
        else:
            raw = backend.detect(image, self._categories)
            inference_ms = (time.perf_counter() - started) * 1000
            answer = raw["answer"]
            parsed = parse_boxes(answer, width, height)
        boxes = self._to_detection_boxes(parsed)
        logger.info(
            "locateanything_inference",
            frame_index=frame_index,
            boxes=len(boxes),
            inference_ms=round(inference_ms, 1),
            remote=self._remote,
        )
        return FrameDetections(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            frame_width=width,
            frame_height=height,
            boxes=boxes,
        )

    def predict_batch(
        self,
        frames: list[tuple[int, np.ndarray, float]],
    ) -> list[FrameDetections]:
        return [self.predict_frame(idx, bgr, ts) for idx, bgr, ts in frames]

    def _to_detection_boxes(self, parsed: list[dict[str, float]]) -> list[DetectionBox]:
        boxes: list[DetectionBox] = []
        for item in parsed:
            x1, y1, x2, y2 = item["x1"], item["y1"], item["x2"], item["y2"]
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if area < self._min_box_area:
                continue
            boxes.append(
                DetectionBox(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=self._default_confidence,
                    class_id=0,
                )
            )
        return boxes
