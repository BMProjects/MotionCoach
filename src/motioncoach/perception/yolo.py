"""Ultralytics YOLO pose adapter (YOLO26 / YOLO11, COCO-17). AGPL-3.0: commercial use needs a licence."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import select_target


class UltralyticsBackend:
    skeleton_name = "coco17"
    conf_threshold = 0.5

    def __init__(self, name: str, weights: str, models_dir: Path, device: str | None = None):
        from .gpu import torch_cuda

        self.name = name
        self.weights = Path(models_dir) / weights
        self.device = device if device in ("cpu",) else ("cuda:0" if torch_cuda() else "cpu")
        self.device_note = "" if self.device != "cpu" or device == "cpu" else "PyTorch 未检测到 CUDA，已用 CPU"
        self.model_id = f"ultralytics/{weights}/{'cuda' if self.device.startswith('cuda') else 'cpu'}"
        self._model = None
        self._prev_box = None

    def open(self) -> None:
        if self._model is None:
            from ultralytics import YOLO

            self.weights.parent.mkdir(parents=True, exist_ok=True)
            self._model = YOLO(str(self.weights))   # downloads the named asset to this path if missing
        self._prev_box = None

    def estimate(self, frame_bgr: np.ndarray, ts_ms: int):
        # numpy input must be BGR for Ultralytics (it converts internally)
        res = self._model.predict(frame_bgr, device=self.device, conf=0.25, verbose=False)
        if not res or res[0].keypoints is None or len(res[0].keypoints) == 0:
            return None, None, None
        kp = res[0].keypoints
        kpts = kp.xy.cpu().numpy()
        scores = kp.conf.cpu().numpy() if kp.conf is not None else np.ones(kpts.shape[:2], np.float32)
        i, self._prev_box = select_target(kpts, scores, self._prev_box, self.conf_threshold)
        if i < 0:
            return None, None, None
        return kpts[i].astype(np.float32), scores[i].astype(np.float32), None

    def close(self) -> None:
        self._prev_box = None
