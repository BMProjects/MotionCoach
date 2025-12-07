"""
Core dance battle alignment package.

This package exposes audio alignment, pose estimation, similarity scoring,
and visualization utilities that can be reused in downstream applications.
"""

from .audio_aligner import find_time_offset
from .visualizer import create_comparison_video
from .pose_estimation.mediapipe_estimator import MediaPipePoseEstimator
from .pose_estimation.yolo_estimator import YOLOPoseEstimator
from .similarity_analysis.procrustes_analyzer import ProcrustesSimilarityAnalyzer

__all__ = [
    "find_time_offset",
    "create_comparison_video",
    "MediaPipePoseEstimator",
    "YOLOPoseEstimator",
    "ProcrustesSimilarityAnalyzer",
]
