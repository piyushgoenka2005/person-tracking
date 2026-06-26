#!/usr/bin/env bash
# Google Colab setup for person-tracking
set -euo pipefail

REPO_DIR="${1:-person-tracking}"
MODE="${2:-yolo}"  # yolo | locateanything

if [ ! -d "$REPO_DIR" ]; then
  echo "Cloning repository..."
  git clone https://github.com/piyushgoenka2005/person-tracking.git "$REPO_DIR"
fi

cd "$REPO_DIR"

echo "Installing dependencies..."
pip install -q -r requirements.txt

if [ "$MODE" = "locateanything" ]; then
  pip install -q -r requirements-locateanything.txt huggingface_hub
  python scripts/download_locateanything.py
fi

python scripts/download_models.py

if python -c "import torch; assert torch.cuda.is_available()"; then
  DEVICE=cuda
  echo "GPU detected — using cuda"
else
  DEVICE=cpu
  echo "WARNING: No GPU. In Colab: Runtime → Change runtime type → T4 GPU"
fi

if [ "$MODE" = "yolo" ]; then
  python -m engine.run \
    --input assets/input.mp4 \
    --output output/ \
    --detector yolo \
    --device "$DEVICE" \
    --frame-stride 10
else
  python -m engine.run \
    --input assets/input.mp4 \
    --output output_la/ \
    --detector locateanything \
    --no-la-remote \
    --device "$DEVICE" \
    --frame-stride 5
fi

echo "Done. Outputs in $(pwd)/output*"
