# DanceBattle Documentation

## Overview
DanceBattle aligns two dance videos by audio, extracts poses, computes frame-wise similarity, and renders a side-by-side comparison video with annotated scores. Two pose backends are supported:
- MediaPipe Pose (default, lightweight)
- YOLO11m-pose via Ultralytics (higher accuracy, GPU recommended)

## Installation
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
For YOLO backend, install matching torch/torchvision for your CPU/CUDA platform before or alongside requirements.

## Usage
CLI (installed or from repo):
```bash
python -m dancebattle.cli --video-a assets/chuu.mov --video-b assets/cyves.mov --output-dir output
```
Local wrapper (adds `src/` to `PYTHONPATH`):
```bash
python main.py --video-a assets/chuu.mov --video-b assets/cyves.mov
```

### Key CLI options
- `--pose-backend` `mediapipe` (default) or `yolo`
- `--align-method` `chroma` (default) or `onset`
- `--target-height` output height per video pane
- MediaPipe tuning: `--model-complexity`, `--static-mode`, `--min-detection-confidence`
- YOLO tuning: `--yolo-model` (default `yolo11m-pose.pt`), `--yolo-device` (`cuda`/`cpu`), `--yolo-conf`

## Project Layout
- `src/dancebattle/`
  - `audio_aligner.py` audio offset (chroma/onset)
  - `pose_estimation/` MediaPipe + YOLO pose estimators
  - `similarity_analysis/` Procrustes similarity
  - `visualizer.py` side-by-side video with overlays and scores
  - `pipeline.py` orchestration
  - `cli.py` CLI entrypoint
- `assets/` sample videos
- `output/`, `tests/output/` generated artifacts (gitignored)
- `tests/` unit tests and optional smoke tests

## Running Tests
```bash
pytest
```
Smoke tests (optional, may download models / need GPU):
```bash
RUN_MEDIAPIPE_TEST=1 pytest tests/test_mediapipe_pipeline.py
RUN_YOLO_TEST=1 pytest tests/test_yolo_pipeline.py
```
Or run scripts directly from IDE: execute the test files; they emit outputs to `tests/output`.

## Models & Performance Notes
- MediaPipe: good speed/accuracy balance on CPU; reduce resolution/FPS on mobile; prefer TFLite GPU delegate when available.
- YOLO11m-pose: higher accuracy, GPU recommended. For faster runs, lower resolution/FPS or switch device to CPU (slower).
- Video decode/encode cost scales with input resolution; downsample if needed.

## Troubleshooting
- Missing torch/torchvision for YOLO: install platform-appropriate wheels before running YOLO backend.
- Model downloads (Ultralytics) require network access; pre-download weights if running offline.
- If sample assets are missing, provide your own `--video-a/--video-b` paths.

## License
MIT License (see project root `LICENSE`).
