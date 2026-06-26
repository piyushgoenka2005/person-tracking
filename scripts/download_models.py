#!/usr/bin/env python3
"""Download YOLOv11x and RTMPose-m ONNX weights for the timeline engine."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

# RTMPose-m ONNX SDK (zip contains end2end.onnx)
RTMPOSE_ZIP_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip"
)


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        print(f"Already exists: {dest}")
        return
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    print("Done.")


def download_rtmpose_onnx(dest: Path) -> None:
    if dest.is_file():
        print(f"Already exists: {dest}")
        return

    print(f"Downloading RTMPose SDK zip from OpenMMLab...")
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "rtmpose-m.zip"
        urllib.request.urlretrieve(RTMPOSE_ZIP_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            onnx_members = [n for n in zf.namelist() if n.endswith(".onnx")]
            if not onnx_members:
                raise RuntimeError("No .onnx file found in RTMPose SDK zip")
            # Prefer end2end.onnx from MMDeploy export
            preferred = next((n for n in onnx_members if "end2end" in n.lower()), onnx_members[0])
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(preferred) as src, dest.open("wb") as out:
                shutil.copyfileobj(src, out)
    print(f"Extracted ONNX to {dest}")


def download_yolo(dest: Path) -> None:
    if dest.is_file():
        print(f"Already exists: {dest}")
        return
    print("Downloading YOLOv11x via ultralytics...")
    from ultralytics import YOLO

    model = YOLO("yolo11x.pt")
    _ = model  # trigger download
    cached = Path("yolo11x.pt")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if cached.is_file():
        if cached.resolve() != dest.resolve():
            shutil.copy2(cached, dest)
        print(f"YOLO weights at {dest}")
    elif not dest.is_file():
        raise FileNotFoundError("ultralytics did not cache yolo11x.pt in working directory")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download timeline engine models")
    parser.add_argument("--models-dir", default=str(MODELS_DIR))
    parser.add_argument("--skip-yolo", action="store_true")
    parser.add_argument("--skip-pose", action="store_true")
    args = parser.parse_args()
    models_dir = Path(args.models_dir)

    if not args.skip_pose:
        try:
            download_rtmpose_onnx(models_dir / "rtmpose-m.onnx")
        except Exception as exc:
            print(f"Warning: RTMPose download failed: {exc}", file=sys.stderr)
            print("Engine will use geometric pose fallback until model is available.", file=sys.stderr)

    if not args.skip_yolo:
        try:
            download_yolo(models_dir / "yolo11x.pt")
        except Exception as exc:
            print(f"Warning: YOLO download failed: {exc}", file=sys.stderr)
            print("Place yolo11x.pt in models/ or project root manually.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
