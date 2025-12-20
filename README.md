# DanceBattle

Audio-aligned dance pose comparison tool. The pipeline extracts audio, aligns two videos, runs pose estimation, scores frame-wise similarity with Procrustes analysis, and exports a side-by-side video annotated with per-frame scores.

## Architecture

The project is structured to separate **core algorithms** from **platform-specific interfaces**:

```
src/dancebattle/
├── core/                 # Platform-independent algorithms (reusable in mobile apps)
│   ├── audio_aligner.py  # Chroma/onset audio alignment
│   ├── pipeline.py       # End-to-end orchestration
│   ├── visualizer.py     # Side-by-side video renderer
│   ├── utils.py          # PoseCacher and utilities
│   ├── pose_estimation/  # MediaPipe + YOLO pose estimators
│   └── similarity_analysis/  # Procrustes similarity scoring
├── cli.py               # Command-line interface
└── gui/                 # PC demo frontend (PySide6)
    ├── app.py           # Main application window
    └── workers.py       # Background threads for camera/video
```

### Using Core Algorithms in Other Platforms

The `dancebattle.core` module is designed to be platform-independent:

```python
from dancebattle.core import (
    run_pipeline,           # Full pipeline
    MediaPipePoseEstimator, # Pose estimation
    ProcrustesSimilarityAnalyzer,  # Similarity scoring
    find_time_offset,       # Audio alignment
    PoseCacher,             # Pre-compute poses for real-time comparison
)

# Example: Real-time comparison
estimator = MediaPipePoseEstimator()
analyzer = ProcrustesSimilarityAnalyzer()

user_pose = estimator.estimate(frame)
score = analyzer.compare(user_pose, reference_pose)
```

## Quickstart

### CLI Usage
```bash
# Install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run comparison
python -m dancebattle.cli --video-a assets/chuu.mov --video-b assets/cyves.mov --output-dir output
```

### GUI Demo (PC)
```bash
# Install PySide6 (included in requirements.txt)
pip install -r requirements.txt

# Launch GUI
python main_gui.py
```

## GUI Features
- **Video conference layout**: Full-screen camera view + reference video PiP
- **3-2-1 countdown** before recording
- **Real-time similarity scoring** during dance
- **Audio-synced playback** of reference video
- **Final comparison video** generation
- **Direct video comparison** without recording

## Dependency Notes
- **ffmpeg**: Required for audio extraction and video recording with audio
- **PySide6**: Required for GUI demo only
- **torch/torchvision**: Required for YOLO backend only

## Models & Performance
- **MediaPipe** (default): Lightweight, good for CPU
- **YOLO11m-pose**: Higher accuracy, GPU recommended

## License
MIT License (see `LICENSE`).

## Acknowledgments
- Inspired by [Mruchus/dance-sync-analysis](https://github.com/Mruchus/dance-sync-analysis)
