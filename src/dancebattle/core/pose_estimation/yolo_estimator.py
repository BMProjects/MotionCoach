from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from .base_estimator import BasePoseEstimator


class YOLOPoseEstimator(BasePoseEstimator):
    """
    High-accuracy whole-body pose estimator using YOLO11m-pose.

    Uses Ultralytics YOLO pose model to detect keypoints. Expects a GPU by default.
    """

    def __init__(self, model_path: str = "yolo11m-pose.pt", device: str = "cuda", conf: float = 0.25):
        self.model = YOLO(model_path)
        self.device = device
        self.conf = conf

    def estimate(self, frame: np.ndarray) -> Optional[np.ndarray]:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model.predict(
            source=frame_rgb,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )
        if not results or len(results[0].keypoints) == 0:
            return None

        # Take the top detection
        kpts = results[0].keypoints.xy[0].cpu().numpy()  # shape (num_kpts, 2)
        scores = results[0].keypoints.conf[0].cpu().numpy()  # shape (num_kpts,)

        h, w, _ = frame.shape
        keypoints = np.zeros((kpts.shape[0], 3), dtype=float)
        # Normalize x/y into [0,1] relative to frame size to match other estimators
        keypoints[:, 0] = kpts[:, 0] / w
        keypoints[:, 1] = kpts[:, 1] / h
        keypoints[:, 2] = scores
        return keypoints
