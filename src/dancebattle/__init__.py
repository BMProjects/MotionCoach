"""
DanceBattle - Audio-aligned dance pose comparison tool.

This package provides:
- Core algorithms (audio alignment, pose estimation, similarity analysis)
- CLI interface for command-line usage
- GUI demo for PC testing

Architecture:
- dancebattle.core: Platform-independent algorithms (can be used in mobile apps)
- dancebattle.cli: Command-line interface
- dancebattle.gui: PySide6-based PC demo

Quick Start:
    # Using core algorithms directly
    from dancebattle.core import run_pipeline, MediaPipePoseEstimator

    # Using CLI
    python -m dancebattle.cli --video-a ref.mp4 --video-b user.mp4

    # Using GUI
    python main_gui.py
"""

# Re-export core APIs for backward compatibility
from .core import (
    find_time_offset,
    run_pipeline,
    extract_audio,
    process_video,
    BasePoseEstimator,
    MediaPipePoseEstimator,
    YOLOPoseEstimator,
    BaseSimilarityAnalyzer,
    ProcrustesSimilarityAnalyzer,
    create_comparison_video,
    draw_pose,
    PoseCacher,
)

__version__ = "0.2.0"

__all__ = [
    # Audio
    "find_time_offset",
    # Pipeline
    "run_pipeline",
    "extract_audio",
    "process_video",
    # Pose Estimation
    "BasePoseEstimator",
    "MediaPipePoseEstimator",
    "YOLOPoseEstimator",
    # Similarity Analysis
    "BaseSimilarityAnalyzer",
    "ProcrustesSimilarityAnalyzer",
    # Visualization
    "create_comparison_video",
    "draw_pose",
    # Utilities
    "PoseCacher",
]
