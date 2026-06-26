#!/usr/bin/env python3
"""Download NVIDIA LocateAnything-3B weights for local inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "models" / "LocateAnything-3B"
MODEL_ID = "nvidia/LocateAnything-3B"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download LocateAnything-3B from Hugging Face")
    parser.add_argument("--output", default=str(DEFAULT_DIR), help="Local model directory")
    parser.add_argument("--model-id", default=MODEL_ID)
    args = parser.parse_args()
    dest = Path(args.output)
    dest.mkdir(parents=True, exist_ok=True)

    if any(dest.iterdir()):
        print(f"Model directory already populated: {dest}")
        return 0

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        print("Install requirements first: pip install -r requirements-locateanything.txt", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"Downloading {args.model_id} -> {dest}")
    print("This is several GB and may take a while...")
    snapshot_download(
        repo_id=args.model_id,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
    )
    print(f"Done. Run with: python -m engine.run --la-model {dest} --no-la-remote --device cpu")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
