import cv2
import numpy as np
from .base_estimator import BasePoseEstimator

# mediapipe>=1.0 removed the legacy ``mp.solutions`` API; this keeps the original interface and output
# ((33, 3) normalised x, y, visibility) on top of the Tasks Pose Landmarker.
_VARIANTS = {0: "lite", 1: "full", 2: "heavy"}


class MediaPipePoseEstimator(BasePoseEstimator):
    def __init__(self, static_mode=False, model_complexity=1, min_detection_confidence=0.5):
        from mediapipe.tasks.python import BaseOptions, vision
        from motioncoach import settings
        from motioncoach.perception.mediapipe_tasks import ensure_model

        model_path = ensure_model(_VARIANTS.get(model_complexity, "full"), settings.MODELS_DIR)
        self.static_mode = static_mode
        self._ts_ms = 0
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE if static_mode else vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
        ))

    def estimate(self, frame: np.ndarray) -> np.ndarray | None:
        """使用MediaPipe进行姿态估计"""
        import mediapipe as mp

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if self.static_mode:
            results = self.pose.detect(image)
        else:
            self._ts_ms += 33   # VIDEO mode needs strictly increasing timestamps; callers pass frames in order
            results = self.pose.detect_for_video(image, self._ts_ms)

        if not results.pose_landmarks:
            return None

        return np.array([[lm.x, lm.y, lm.visibility] for lm in results.pose_landmarks[0]])
