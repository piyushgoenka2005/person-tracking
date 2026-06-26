#!/usr/bin/env python3
"""Download NVIDIA LocateAnything-3B weights for local inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "models" / "LocateAnything-3B"
MODEL_ID = "nvidia/LocateAnything-3B"


def has_weights(dest: Path) -> bool:
    return any(dest.glob("*.safetensors")) or any(dest.glob("**/*.safetensors"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Download LocateAnything-3B from Hugging Face")
    parser.add_argument("--output", default=str(DEFAULT_DIR), help="Local model directory")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if partial files exist without weights",
    )
    args = parser.parse_args()
    dest = Path(args.output)
    dest.mkdir(parents=True, exist_ok=True)

    if has_weights(dest) and not args.force:
        print(f"Model weights already present: {dest}")
        return 0

    if dest.is_dir() and any(dest.iterdir()) and not has_weights(dest):
        print(f"Incomplete model directory (no .safetensors): {dest} — downloading weights...")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        print("Install requirements first: pip install -r requirements-locateanything.txt", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"Downloading {args.model_id} -> {dest}")
    print("This is several GB and may take 10–20 minutes on Colab...")
    snapshot_download(
        repo_id=args.model_id,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
    )
    if not has_weights(dest):
        print("ERROR: download finished but no .safetensors files found.", file=sys.stderr)
        return 1
    print(f"Done. Run with: python -m engine.run --detector locateanything --no-la-remote --device cuda")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
