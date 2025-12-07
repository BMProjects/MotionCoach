# DanceBattle

Audio-aligned dance pose comparison demo. The pipeline extracts audio, aligns two videos, runs pose estimation, scores frame-wise similarity with Procrustes analysis, and exports a side-by-side video annotated with per-frame scores.

## Quickstart
1) Install ffmpeg (moviepy depends on it).  
2) Install dependencies (YOLO backend also needs torch/torchvision for your CPU/CUDA platform):
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
3) Run the demo with the sample assets:
```bash
python -m dancebattle.cli --video-a assets/chuu.mov --video-b assets/cyves.mov --output-dir output
```
This produces `output/comparison_video.mp4` and prints the detected offset and average similarity. Defaults already point to the sample assets, so you can also run `python -m dancebattle.cli --output-dir output`.

You can also run the wrapper script (adds `src/` to `PYTHONPATH` for local runs):
```bash
python main.py --video-a assets/chuu.mov --video-b assets/cyves.mov
```

## Project Layout
- `src/dancebattle/` core modules  
  - `audio_aligner.py` chroma/onset audio alignment  
  - `pose_estimation/` MediaPipe pose estimator + YOLO11m-pose estimator  
  - `similarity_analysis/` Procrustes similarity scoring  
  - `visualizer.py` side-by-side video renderer with skeleton overlay  
  - `pipeline.py` end-to-end orchestration  
  - `cli.py` argparse CLI entrypoint  
- `assets/` sample videos for the demo  
- `output/`, `tests/output/` generated artifacts (gitignored)  
- `tests/` automated unit tests and diagnostic scripts (visual/chroma/self-alignment, optional smoke tests)

## Development
- Lint/format: choose your preferred tools (e.g., black/ruff/isort) and add to pre-commit.  
- Tests: `pytest` (after `pip install -r requirements.txt`). Smoke tests need env vars `RUN_MEDIAPIPE_TEST` / `RUN_YOLO_TEST`.  
- Entrypoints: `python -m dancebattle.cli ...` or `python main.py ...` (adds `src/` when not installed).  
- Version history: see `CHANGELOG.md` (target release 0.1.0).

## Dependency Notes
- Dependencies are managed in `requirements.txt`; install via `pip install -r requirements.txt`.  
- High-accuracy YOLO pose requires torch/torchvision (pick the wheel for your CPU/CUDA platform), plus `ultralytics` (listed in requirements).

## Models & Performance
- Default backend: MediaPipe Pose (BlazePose, `model_complexity=1`). Compute is dominated by per-frame pose inference; audio alignment and Procrustes scoring are lightweight. Desktop CPU/GPU at 256–368px can approach real time; on mobile, lower resolution/FPS and prefer the TFLite GPU delegate.  
- Video decode/encode cost scales with input resolution; downsample or subsample frames when needed.  
- High-accuracy backend: YOLO11m-pose via `--pose-backend yolo --yolo-device cuda --yolo-model yolo11m-pose.pt`. Requires GPU for good speed; CPU works but is slower. Tune resolution/FPS to trade speed for quality.  
- Poses are cached in memory only; serialize keypoints if you need reuse across runs.

## License
- MIT License (see `LICENSE`).

## Acknowledgments
- Inspired by [Mruchus/dance-sync-analysis](https://github.com/Mruchus/dance-sync-analysis).
