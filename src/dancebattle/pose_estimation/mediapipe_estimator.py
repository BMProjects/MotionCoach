import cv2
import mediapipe as mp
import numpy as np
from .base_estimator import BasePoseEstimator

class MediaPipePoseEstimator(BasePoseEstimator):
    def __init__(self, static_mode=False, model_complexity=1, min_detection_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence
        )

    def estimate(self, frame: np.ndarray) -> np.ndarray | None:
        """使用MediaPipe进行姿态估计"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if not results.pose_landmarks:
            return None
            
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.visibility])
            
        return np.array(landmarks)