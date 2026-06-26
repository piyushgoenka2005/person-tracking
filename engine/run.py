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
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Inference device")
    parser.add_argument("--batch-size", type=int, default=8, help="YOLO batch size")
    parser.add_argument("--min-segment-s", type=float, default=2.0, help="Minimum segment duration")
    return parser


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

    settings = TimelineSettings(
        input_path=input_path,
        output_dir=Path(args.output),
        yolo_model_name=_resolve_yolo_model(args.yolo_model),
        locateanything_model=_resolve_la_model(args.la_model),
        locateanything_remote=args.la_remote,
        pose_model_path=args.pose_model,
        device=args.device,
        frame_stride=args.frame_stride,
        batch_size=args.batch_size,
        min_segment_s=args.min_segment_s,
    )

    if not settings.locateanything_remote:
        print(
            "LocateAnything: local mode (first run downloads ~6GB if not cached). "
            "Use --frame-stride 10+ on CPU/i3 for faster runs."
        )

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
