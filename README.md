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

**Detector switch:** `engine/pipeline.py` has a `DETECTOR SWITCH` block. By default **LocateAnything-3B** (Hugging Face) is active; **YOLOv11x** is commented out. Swap the two blocks to change detector. Compare runs with different output folders, e.g. `--output output_yolo` vs `--output output_locateanything`.

```bash
# LocateAnything via HF Space (often blocked — see README)
python -m engine.run --input assets/input.mp4 --output output_la/ --la-remote

# LocateAnything local (GPU — use Colab or CUDA machine)
python -m engine.run --input assets/input.mp4 --output output_la/ --no-la-remote --device cuda

# After uncommenting YOLO in pipeline.py:
python -m engine.run --input assets/input.mp4 --output output_yolo/ --yolo-model models/yolo11x.pt
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

## Google Colab (LocateAnything on GPU)

The public HF Space API is often blocked (ZeroGPU duration limit). For a real LocateAnything run, use Colab with a T4 GPU:

```bash
pip install -r requirements.txt -r requirements-locateanything.txt huggingface_hub
python scripts/download_models.py
python scripts/download_locateanything.py
python -m engine.run --input assets/input.mp4 --output output_la/ --no-la-remote --device cuda --frame-stride 5
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
