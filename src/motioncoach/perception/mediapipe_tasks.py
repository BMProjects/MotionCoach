"""MediaPipe Tasks Pose Landmarker adapter (the legacy ``mp.solutions`` API is gone in mediapipe>=1.0)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

MODEL_URLS = {
    v: f"https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task"
    for v in ("lite", "full", "heavy")
}


def ensure_model(variant: str, models_dir: Path) -> Path:
    path = Path(models_dir) / f"pose_landmarker_{variant}.task"
    if not path.exists():
        import urllib.request

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URLS[variant], tmp)
        tmp.replace(path)
    return path


def file_digest(path: Path, n: int = 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:n]


class MediaPipeTasksBackend:
    skeleton_name = "mediapipe33"
    conf_threshold = 0.5

    def __init__(self, variant: str, models_dir: Path, device: str | None = None,
                 min_pose_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        from .gpu import forced_cpu, prepare_mediapipe_gpu

        self.name = f"mediapipe-{variant}"
        self.model_path = ensure_model(variant, models_dir)
        self.min_det = min_pose_detection_confidence
        self.min_track = min_tracking_confidence
        self._lm = None
        self.device, self.device_note = "cpu", ""
        if device != "cpu" and not forced_cpu():
            prepare_mediapipe_gpu()
            try:   # probe once: GPU delegate needs a working EGL/GLES context
                self._create(gpu=True).close()
                self.device = "gpu"
            except Exception as exc:
                self.device_note = f"GPU 委托不可用，已用 CPU：{str(exc)[-160:]}"
        self.model_id = f"mediapipe-tasks/{self.model_path.stem}@{file_digest(self.model_path, 8)}/{self.device}"

    def _create(self, gpu: bool):
        from mediapipe.tasks.python import BaseOptions, vision

        opts = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.model_path),
                                     delegate=BaseOptions.Delegate.GPU if gpu else BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO,   # temporal tracking; needs increasing timestamps
            num_poses=1,
            min_pose_detection_confidence=self.min_det,
            min_tracking_confidence=self.min_track,
        )
        return vision.PoseLandmarker.create_from_options(opts)

    def open(self) -> None:
        self._lm = self._create(gpu=self.device == "gpu")

    def estimate(self, frame_bgr: np.ndarray, ts_ms: int):
        import mediapipe as mp

        h, w = frame_bgr.shape[:2]
        res = self._lm.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)), ts_ms)
        if not res.pose_landmarks:
            return None, None, None
        lms = res.pose_landmarks[0]
        xy = np.array([(p.x * w, p.y * h) for p in lms], np.float32)
        cf = np.array([min(p.visibility, p.presence) for p in lms], np.float32)
        wd = np.array([(p.x, p.y, p.z) for p in res.pose_world_landmarks[0]], np.float32) if res.pose_world_landmarks else None
        return xy, cf, wd

    def close(self) -> None:
        if self._lm is not None:
            self._lm.close()
            self._lm = None
