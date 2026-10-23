# DanceBattle (legacy dance demo) Documentation

## Overview
DanceBattle aligns two dance videos by audio, extracts poses, computes frame-wise similarity, and renders a side-by-side comparison video with annotated scores.

## Architecture

### Core Module (`dancebattle.core`)
Platform-independent algorithms that can be integrated into any application:

- **Audio Alignment**: `find_time_offset()` - Chroma/onset cross-correlation
- **Pose Estimation**: `MediaPipePoseEstimator`, `YOLOPoseEstimator`
- **Similarity Analysis**: `ProcrustesSimilarityAnalyzer`
- **Pipeline**: `run_pipeline()` - Full end-to-end processing
- **Visualization**: `create_comparison_video()`, `draw_pose()`
- **Utilities**: `PoseCacher` - Pre-compute poses for real-time comparison

### Interfaces
- **CLI** (`dancebattle.cli`): Command-line interface
- **GUI** (`dancebattle.gui`): PySide6-based PC demo

## Installation

> Legacy module, restored on the MediaPipe Tasks API (same output format and scoring as before). New work lives in `motioncoach` (see [ARCHITECTURE.md](ARCHITECTURE.md)).
```bash
uv sync --extra dance --extra gui   # legacy dance demo (CLI + PySide6 GUI)
```

For YOLO backend, install matching torch/torchvision for your CPU/CUDA platform.

## Usage

### CLI
```bash
python -m dancebattle.cli --video-a assets/chuu.mov --video-b assets/cyves.mov --output-dir output
```

### GUI
```bash
python main_gui.py
```

### Programmatic Usage
```python
from dancebattle.core import (
    run_pipeline,
    MediaPipePoseEstimator,
    ProcrustesSimilarityAnalyzer,
    PoseCacher,
)

# Option 1: Full pipeline
results = run_pipeline(
    video_a="reference.mp4",
    video_b="user.mp4",
    output_dir="output",
)
print(f"Average similarity: {results['average_score']:.1f}%")

# Option 2: Real-time comparison
cacher = PoseCacher()
cacher.load_video("reference.mp4")

estimator = MediaPipePoseEstimator()
analyzer = ProcrustesSimilarityAnalyzer()

# In your frame loop:
user_pose = estimator.estimate(camera_frame)
ref_pose = cacher.get_pose_at_time(elapsed_seconds)
score = analyzer.compare(user_pose, ref_pose)
```

### Key CLI Options
- `--pose-backend` `mediapipe` (default) or `yolo`
- `--align-method` `chroma` (default) or `onset`
- `--target-height` output height per video pane
- MediaPipe tuning: `--model-complexity`, `--static-mode`, `--min-detection-confidence`
- YOLO tuning: `--yolo-model`, `--yolo-device`, `--yolo-conf`

## Integration Guide

### Mobile App Integration
The `dancebattle.core` module is designed for cross-platform reuse:

1. **Extract core algorithms**: Use the classes directly without GUI dependencies
2. **Pose estimation**: MediaPipe has mobile SDKs; YOLO can run on mobile with CoreML/TFLite
3. **Similarity scoring**: Pure NumPy/SciPy, portable to any platform
4. **Audio alignment**: librosa-based, may need adaptation for mobile

### API Reference

#### `MediaPipePoseEstimator`
```python
estimator = MediaPipePoseEstimator(
    static_mode=False,      # Video mode (faster)
    model_complexity=1,     # 0=lite, 1=full, 2=heavy
    min_detection_confidence=0.5,
)
pose = estimator.estimate(frame)  # Returns (33, 3) array or None
```

#### `ProcrustesSimilarityAnalyzer`
```python
analyzer = ProcrustesSimilarityAnalyzer()
score = analyzer.compare(pose1, pose2)  # Returns 0-100% similarity
```

#### `PoseCacher`
```python
cacher = PoseCacher()
cacher.load_video("video.mp4")  # Pre-compute all poses
pose = cacher.get_pose_at_time(1.5)  # Get pose at 1.5 seconds
```

## Running Tests
```bash
pytest
```

Smoke tests (optional):
```bash
RUN_MEDIAPIPE_TEST=1 pytest tests/test_mediapipe_pipeline.py
RUN_YOLO_TEST=1 pytest tests/test_yolo_pipeline.py
```

## Troubleshooting
- **ffmpeg not found**: Install ffmpeg (`apt install ffmpeg` or `brew install ffmpeg`)
- **Missing torch**: Install for your platform from pytorch.org
- **Camera not working**: Check permissions and device ID

## License
MIT License (see project root `LICENSE`).
