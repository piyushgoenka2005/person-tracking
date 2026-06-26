# Person Behaviour Timeline Engine

Standalone CLI pipeline for person behaviour timelines from CCTV video.

```
Video → YOLOv11x → ByteTrack → RTMPose → Behaviour Engine → Timeline → Exports
```

## Behaviours

Walking, Standing, Waiting, Queueing, Sitting, Sleeping, Phone Usage, Eating, Unknown

## Outputs (`output/`)

| File | Description |
|------|-------------|
| `annotated_video.mp4` | Boxes, skeleton, behaviour labels |
| `person_timeline.csv` | `person_id,start_time,end_time,behaviour,confidence` |
| `person_transitions.csv` | Behaviour change events per person |
| `person_summary.xlsx` | Duration totals + segment sheets |
| `session_summary.json` | Video metadata, models, QA status |

## Setup

Requires **Python 3.11+**.

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_models.py
```

Place input video at `assets/input.mp4`.

## Run

```bash
python -m engine.run --input assets/input.mp4 --output output/
```

**Detector:** use `--detector yolo` or `--detector locateanything` (default). No need to edit `pipeline.py`.

```bash
# YOLO (fast, works everywhere including Colab CPU)
python -m engine.run --input assets/input.mp4 --output output_yolo/ --detector yolo --frame-stride 10

# LocateAnything local (GPU recommended)
python -m engine.run --input assets/input.mp4 --output output_la/ --detector locateanything --no-la-remote --device cuda --frame-stride 5

# LocateAnything via HF Space (often blocked — auto-falls back to YOLO)
python -m engine.run --input assets/input.mp4 --output output_la/ --detector locateanything --la-remote
```

Options:

```bash
python -m engine.run --yolo-model models/yolo11x.pt --pose-model models/rtmpose-m.onnx --frame-stride 5 --min-segment-s 0.5
```

## Project layout

```
engine/
  core/          Settings, logging, exceptions
  detection/     YOLO, video ingestion, LocateAnything integration
  tracking/      ByteTrack
  features/      Kinematics for behaviour rules
  pose/          RTMPose ONNX
  behaviour/     Sliding-window behaviour classifier
  timeline/      Segment merge + transitions
  export/        CSV, XLSX, JSON
  render/        Annotated video
  qa/            Pipeline validators
  run.py         CLI entry point
scripts/
  download_models.py
  compare_detection_models.py
tests/timeline_engine/
```

## Google Colab

**Before running:** Runtime → **Change runtime type** → **T4 GPU**.

### Quick run (YOLO — works immediately)

```python
%%bash
set -e
if [ ! -d "person-tracking" ]; then
  git clone https://github.com/piyushgoenka2005/person-tracking.git
fi
cd person-tracking
pip install -q -r requirements.txt
python scripts/download_models.py
python -m engine.run --input assets/input.mp4 --output output/ --detector yolo --device cuda --frame-stride 10
```

Or use the setup script:

```bash
bash scripts/colab_setup.sh person-tracking yolo
```

### LocateAnything on Colab (GPU required)

```python
%%bash
set -e
cd person-tracking  # after clone
pip install -q -r requirements.txt -r requirements-locateanything.txt huggingface_hub
python scripts/download_models.py
python scripts/download_locateanything.py
python -m engine.run --input assets/input.mp4 --output output_la/ --detector locateanything --no-la-remote --device cuda --frame-stride 5
```

**Do not use `--la-remote`** on Colab — the public HF Space API is blocked (ZeroGPU limit).

If LocateAnything fails to load, the pipeline **auto-falls back to YOLO** (unless you pass `--no-la-fallback`).

### Download results

```python
!zip -r output.zip output/
from google.colab import files
files.download("output.zip")
```

## Tests

```bash
pytest tests/timeline_engine/
```

## Optional: LocateAnything comparison

```bash
pip install -r requirements-locateanything.txt
python scripts/compare_detection_models.py --input assets/input.mp4
```
