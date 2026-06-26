"""CLI entry point for the Behaviour Timeline Engine."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if sys.version_info < (3, 11):
    print(
        "Error: Python 3.11+ is required (current: "
        f"{sys.version_info.major}.{sys.version_info.minor}).\n"
        "Recreate the venv: py -3.11 -m venv .venv",
        file=sys.stderr,
    )
    raise SystemExit(1)

from engine.config import TimelineSettings
from engine.core.logging import setup_logging
from engine.detection.locateanything import resolve_locateanything_model
from engine.pipeline import TimelinePipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Behaviour Timeline Engine — detection + ByteTrack + RTMPose → timelines",
    )
    parser.add_argument("--input", default="assets/input.mp4", help="Input video path")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--yolo-model", default="models/yolo11x.pt", help="YOLO weights (when YOLO block is active in pipeline.py)")
    parser.add_argument(
        "--la-model",
        default="nvidia/LocateAnything-3B",
        help="LocateAnything model id/path (when LocateAnything block is active in pipeline.py)",
    )
    parser.add_argument(
        "--la-remote",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use Hugging Face Space API. For local weights on CPU use --no-la-remote (default).",
    )
    parser.add_argument("--pose-model", default="models/rtmpose-m.onnx", help="RTMPose ONNX model")
    parser.add_argument("--frame-stride", type=int, default=1, help="Process every Nth frame")
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda"],
        help="Inference device (default: cuda if available, else cpu)",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="YOLO batch size")
    parser.add_argument("--min-segment-s", type=float, default=2.0, help="Minimum segment duration")
    parser.add_argument(
        "--detector",
        choices=["locateanything", "yolo"],
        default="locateanything",
        help="Person detector: locateanything (default) or yolo",
    )
    parser.add_argument(
        "--no-la-fallback",
        action="store_true",
        help="Do not fall back to YOLO when LocateAnything fails",
    )
    return parser


def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _resolve_yolo_model(path: str) -> str:
    """Use local weights file if present, otherwise let ultralytics resolve the name."""
    candidate = Path(path)
    if candidate.is_file():
        return str(candidate.resolve())
    fallback = Path("yolo11x.pt")
    if fallback.is_file():
        return str(fallback.resolve())
    return path


def _resolve_la_model(path: str) -> str:
    return resolve_locateanything_model(path)


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = build_parser().parse_args(argv)
    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"Error: input video not found: {input_path}", file=sys.stderr)
        return 1

    device = args.device or _default_device()

    settings = TimelineSettings(
        input_path=input_path,
        output_dir=Path(args.output),
        yolo_model_name=_resolve_yolo_model(args.yolo_model),
        detector=args.detector,
        locateanything_model=_resolve_la_model(args.la_model),
        locateanything_remote=args.la_remote,
        la_fallback_yolo=not args.no_la_fallback,
        pose_model_path=args.pose_model,
        device=device,
        frame_stride=args.frame_stride,
        batch_size=args.batch_size,
        min_segment_s=args.min_segment_s,
    )

    if settings.detector == "locateanything" and not settings.locateanything_remote:
        print(
            "LocateAnything: local mode (install requirements-locateanything.txt; "
            "~6GB download on first run). Use --device cuda on Colab/GPU."
        )
    if settings.detector == "yolo":
        print(f"Detector: YOLOv11x on {device}")

    pipeline = TimelinePipeline(settings)
    segments, context = pipeline.run()

    print(f"Processed {context.video.path}")
    print(f"Detector: {context.yolo_model}")
    print(f"Persons: {context.person_count}, Segments: {len(segments)}")
    print(f"Output: {settings.output_dir.resolve()}")
    if context.qa_warnings:
        print("QA warnings:")
        for w in context.qa_warnings:
            print(f"  - {w}")
    if context.qa_errors:
        print("QA errors:")
        for e in context.qa_errors:
            print(f"  - {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
