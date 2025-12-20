"""
DanceBattle Core Algorithm Package.

This package contains the platform-independent core algorithms for:
- Audio alignment (chroma/onset cross-correlation)
- Pose estimation (MediaPipe, YOLO backends)
- Similarity analysis (Procrustes)
- Video visualization

These modules can be used independently of any GUI framework,
making them suitable for integration into mobile apps or other platforms.
"""

from .audio_aligner import find_time_offset
from .visualizer import create_comparison_video, draw_pose
from .pipeline import run_pipeline, extract_audio, process_video
from .utils import PoseCacher
from .pose_estimation.mediapipe_estimator import MediaPipePoseEstimator
from .pose_estimation.yolo_estimator import YOLOPoseEstimator
from .pose_estimation.base_estimator import BasePoseEstimator
from .similarity_analysis.procrustes_analyzer import ProcrustesSimilarityAnalyzer
from .similarity_analysis.base_analyzer import BaseSimilarityAnalyzer

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
