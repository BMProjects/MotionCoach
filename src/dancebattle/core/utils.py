"""
Utility for pre-computing and caching pose landmarks from a reference video.
"""

from pathlib import Path
from typing import List

import cv2
import numpy as np
from tqdm import tqdm

from .pose_estimation.mediapipe_estimator import MediaPipePoseEstimator


class PoseCacher:
    """
    Pre-computes pose landmarks for a video and stores them in memory.
    This allows real-time comparison without running inference twice.
    """

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
    ):
        self.estimator = MediaPipePoseEstimator(
            static_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
        )
        self._cached_poses: List[np.ndarray | None] = []
        self._fps: float = 30.0
        self._frame_count: int = 0

    def load_video(self, video_path: Path) -> None:
        """
        Processes the entire video and caches all pose landmarks.

        Args:
            video_path: Path to the video file.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        self._fps = cap.get(cv2.CAP_PROP_FPS)
        self._frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._cached_poses = []

        for _ in tqdm(range(self._frame_count), desc=f"Caching poses for {video_path.name}"):
            ret, frame = cap.read()
            if not ret:
                break
            pose = self.estimator.estimate(frame)
            self._cached_poses.append(pose)

        cap.release()

    def get_pose_at_frame(self, frame_index: int) -> np.ndarray | None:
        """
        Retrieves the cached pose for a specific frame index.

        Args:
            frame_index: Zero-based frame index.

        Returns:
            Pose landmarks array or None if not detected or out of bounds.
        """
        if 0 <= frame_index < len(self._cached_poses):
            return self._cached_poses[frame_index]
        return None

    def get_pose_at_time(self, time_seconds: float) -> np.ndarray | None:
        """
        Retrieves the cached pose for a specific timestamp.

        Args:
            time_seconds: Time in seconds from the start of the video.

        Returns:
            Pose landmarks array or None.
        """
        frame_index = int(time_seconds * self._fps)
        return self.get_pose_at_frame(frame_index)

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def duration(self) -> float:
        """Duration of the video in seconds."""
        return self._frame_count / self._fps if self._fps > 0 else 0.0

    @property
    def poses(self) -> List[np.ndarray | None]:
        return self._cached_poses
