#!/usr/bin/env python3
"""Compare YOLOv11x vs NVIDIA LocateAnything-3B on sample video frames.

Usage:
  python scripts/compare_detection_models.py --input assets/input.mp4
  python scripts/compare_detection_models.py --input assets/input.mp4 --locateanything remote
  python scripts/compare_detection_models.py --input assets/input.mp4 --locateanything local --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.core.settings import get_settings
from engine.detection.service import DetectionService
from engine.detection.locateanything import LocateAnythingDetectionService, parse_boxes


@dataclass
class FrameComparison:
    frame_index: int
    timestamp_s: float
    yolo_boxes: int
    yolo_ms: float
    locateanything_boxes: int
    locateanything_ms: float
    locateanything_answer: str


def extract_frames(
    video_path: Path,
    *,
    frame_indices: list[int] | None = None,
    max_frames: int = 5,
    stride: int = 50,
) -> list[tuple[int, np.ndarray, float]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if frame_indices is None:
        if total <= max_frames:
            frame_indices = list(range(total))
        else:
            step = max(1, total // max_frames)
            frame_indices = [min(i * step, total - 1) for i in range(max_frames)]

    frames: list[tuple[int, np.ndarray, float]] = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        if not ok:
            continue
        frames.append((idx, bgr, idx / fps * 1000.0))
    cap.release()
    return frames


def run_yolo(frames: list[tuple[int, np.ndarray, float]], yolo_model: str) -> list[tuple[int, float, int]]:
    settings = get_settings().model_copy(update={"yolo_model_name": yolo_model})
    service = DetectionService(settings)
    started = time.perf_counter()
    detections = service.predict_batch(frames)
    total_ms = (time.perf_counter() - started) * 1000
    per_frame_ms = total_ms / max(len(frames), 1)
    return [(d.frame_index, per_frame_ms, len(d.boxes)) for d in detections]


def run_locateanything(
    frames: list[tuple[int, np.ndarray, float]],
    *,
    mode: str,
    device: str,
    model_path: str,
    hf_token: str | None,
) -> list[tuple[int, float, int, str]]:
    remote = mode == "remote"
    service = LocateAnythingDetectionService(
        model_path=model_path,
        device=device,
        remote=remote,
        hf_token=hf_token,
    )
    results: list[tuple[int, float, int, str]] = []
    for frame_index, bgr, ts in frames:
        started = time.perf_counter()
        det = service.predict_frame(frame_index, bgr, ts)
        elapsed = (time.perf_counter() - started) * 1000
        answer = ""
        if remote:
            answer = "(remote)"
        results.append((frame_index, elapsed, len(det.boxes), answer))
    return results


def save_overlay(
    bgr: np.ndarray,
    yolo_boxes: list,
    la_boxes: list[dict[str, float]],
    out_path: Path,
) -> None:
    vis = bgr.copy()
    for box in yolo_boxes:
        cv2.rectangle(
            vis,
            (int(box.x1), int(box.y1)),
            (int(box.x2), int(box.y2)),
            (0, 255, 0),
            2,
        )
    for box in la_boxes:
        cv2.rectangle(
            vis,
            (int(box["x1"]), int(box["y1"])),
            (int(box["x2"]), int(box["y2"])),
            (0, 128, 255),
            2,
        )
    cv2.putText(vis, "Green=YOLO  Orange=LocateAnything", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare YOLOv11x and LocateAnything detection")
    parser.add_argument("--input", default="assets/input.mp4")
    parser.add_argument("--output", default="output/model_comparison")
    parser.add_argument("--yolo-model", default="models/yolo11x.pt")
    parser.add_argument("--locateanything", choices=["skip", "remote", "local"], default="remote")
    parser.add_argument("--la-model", default="nvidia/LocateAnything-3B")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--max-frames", type=int, default=3)
    parser.add_argument("--hf-token", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    video_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = extract_frames(video_path, max_frames=args.max_frames)
    if not frames:
        print("No frames extracted.")
        return 1

    print(f"Video: {video_path}")
    print(f"Frames tested: {[f[0] for f in frames]}")
    print()

    yolo_results = run_yolo(frames, args.yolo_model)
    yolo_map = {idx: (ms, count) for idx, ms, count in yolo_results}

    la_results: list[tuple[int, float, int, str]] = []
    if args.locateanything != "skip":
        print(f"Running LocateAnything ({args.locateanything})...")
        try:
            la_results = run_locateanything(
                frames,
                mode=args.locateanything,
                device=args.device,
                model_path=args.la_model,
                hf_token=args.hf_token,
            )
        except Exception as exc:
            print(f"LocateAnything failed: {exc}")
            if "GPU duration" in str(exc):
                print(
                    "  Note: The public HF Space has GPU time limits. "
                    "Use --locateanything local --device cuda on a GPU machine, "
                    "or duplicate the Space to your HF account."
                )
            la_results = []

    la_map = {idx: (ms, count, ans) for idx, ms, count, ans in la_results}

    comparisons: list[FrameComparison] = []
    settings = get_settings().model_copy(update={"yolo_model_name": args.yolo_model})
    yolo_service = DetectionService(settings)

    for frame_index, bgr, ts in frames:
        yolo_ms, yolo_count = yolo_map.get(frame_index, (0.0, 0))
        la_ms, la_count, la_answer = la_map.get(frame_index, (0.0, 0, ""))
        comparisons.append(
            FrameComparison(
                frame_index=frame_index,
                timestamp_s=round(ts / 1000.0, 2),
                yolo_boxes=yolo_count,
                yolo_ms=round(yolo_ms, 1),
                locateanything_boxes=la_count,
                locateanything_ms=round(la_ms, 1),
                locateanything_answer=la_answer[:200],
            )
        )

        if la_results and args.locateanything == "local":
            from engine.detection.locateanything import bgr_to_pil, LocateAnythingWorker

            worker = LocateAnythingWorker(args.la_model, device=args.device)
            pil = bgr_to_pil(bgr)
            answer = worker.detect(pil, ["person"])["answer"]
            la_boxes = parse_boxes(answer, pil.size[0], pil.size[1])
        elif la_results:
            la_boxes = []
        else:
            la_boxes = []

        yolo_det = yolo_service.predict_batch([(frame_index, bgr, ts)])[0]
        save_overlay(bgr, yolo_det.boxes, la_boxes, output_dir / f"frame_{frame_index:04d}.jpg")

    summary = {
        "video": str(video_path.resolve()),
        "yolo_model": args.yolo_model,
        "locateanything_model": args.la_model if args.locateanything != "skip" else None,
        "locateanything_backend": args.locateanything,
        "frames": [asdict(c) for c in comparisons],
        "totals": {
            "yolo_boxes": sum(c.yolo_boxes for c in comparisons),
            "locateanything_boxes": sum(c.locateanything_boxes for c in comparisons),
            "avg_yolo_ms": round(sum(c.yolo_ms for c in comparisons) / len(comparisons), 1),
            "avg_locateanything_ms": round(
                sum(c.locateanything_ms for c in comparisons) / len(comparisons), 1
            )
            if la_results
            else None,
        },
        "differences": {
            "task": "Detection / grounding only — behaviour analysis still requires YOLO+ByteTrack+RTMPose pipeline",
            "yolo": {
                "type": "Dedicated object detector (CNN)",
                "params": "~56M (YOLOv11x)",
                "speed": "Fast batch inference on CPU/GPU",
                "strengths": ["Fixed person class", "High FPS", "Confidence scores", "Proven tracking stack"],
                "limits": ["Closed vocabulary", "No natural-language queries"],
            },
            "locateanything": {
                "type": "Vision-language model (3B parameters)",
                "architecture": "MoonViT + Qwen2.5-3B + Parallel Box Decoding",
                "speed": "Slow on CPU; designed for NVIDIA GPU (H100/A100/RTX)",
                "strengths": [
                    "Open-vocabulary detection",
                    "Phrase grounding ('person with red bag')",
                    "GUI/text/pointing tasks",
                ],
                "limits": [
                    "No built-in tracking or behaviour",
                    "Heavy memory (~12-35GB GPU)",
                    "Non-commercial license",
                    "Replaces detection only",
                ],
            },
        },
    }

    report_path = output_dir / "comparison_report.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 60)
    print("DETECTION MODEL COMPARISON")
    print("=" * 60)
    for c in comparisons:
        print(
            f"Frame {c.frame_index:4d} @ {c.timestamp_s:5.2f}s | "
            f"YOLO: {c.yolo_boxes:2d} boxes ({c.yolo_ms:6.0f} ms) | "
            f"LocateAnything: {c.locateanything_boxes:2d} boxes ({c.locateanything_ms:6.0f} ms)"
        )
    print()
    print(f"Report: {report_path}")
    print(f"Overlays: {output_dir}/frame_*.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
