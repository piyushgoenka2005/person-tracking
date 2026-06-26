#!/usr/bin/env bash
# Colab: install deps, download models, run YOLO + LocateAnything with same settings.
set -euo pipefail

REPO_DIR="${1:-person-tracking}"
FRAME_STRIDE="${2:-10}"

if [ ! -d "$REPO_DIR" ]; then
  git clone https://github.com/piyushgoenka2005/person-tracking.git "$REPO_DIR"
fi

cd "$REPO_DIR"
git pull 2>/dev/null || true

echo "=== Installing Colab dependencies (includes decord, lmdb, transformers) ==="
pip install -q -r requirements-colab.txt

if ! python -c "import decord, lmdb, transformers; print('LA deps OK')"; then
  echo "ERROR: LocateAnything dependencies missing. Re-run: pip install -r requirements-colab.txt" >&2
  exit 1
fi

python scripts/download_models.py
python scripts/download_locateanything.py

if python -c "import torch; assert torch.cuda.is_available()"; then
  DEVICE=cuda
else
  echo "ERROR: No GPU. Runtime → Change runtime type → T4 GPU" >&2
  exit 1
fi

echo "=== Run 1/2: YOLOv11x → output_yolo/ ==="
python -m engine.run \
  --input assets/input.mp4 \
  --output output_yolo/ \
  --detector yolo \
  --device "$DEVICE" \
  --frame-stride "$FRAME_STRIDE" \
  --no-la-fallback

echo "=== Run 2/2: LocateAnything-3B → output_la/ ==="
python -m engine.run \
  --input assets/input.mp4 \
  --output output_la/ \
  --detector locateanything \
  --no-la-remote \
  --device "$DEVICE" \
  --frame-stride "$FRAME_STRIDE" \
  --no-la-fallback

echo "=== Done ==="
echo "Compare: output_yolo/ vs output_la/"
ls -la output_yolo/ output_la/
