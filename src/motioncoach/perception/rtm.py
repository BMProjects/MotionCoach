"""rtmlib adapters: RTMPose (COCO-17, top-down), RTMO (one-stage), RTMW (COCO-WholeBody 133), RTMW3D.

rtmlib runs ONNX models without MMCV/MMPose and downloads weights on first use
(~/.cache/rtmlib). Input frames are BGR, as rtmlib expects. SimCC confidences are not
probabilities, so they are clipped to [0, 1] and use a lower threshold than MediaPipe.
GPU: onnxruntime-gpu CUDA execution provider (see perception/gpu.py).
RTMW3D's 3D output is in model-crop pixel units for x/y plus a relative depth z — not metric;
it is stored in ``world`` for experiments, not used by the 2D features.
"""

from __future__ import annotations

import numpy as np

from .base import select_target

KINDS = {
    # name: (rtmlib class, mode, skeleton, extra kwargs)
    "rtmpose-m": ("Body", "balanced", "coco17", {}),
    "rtmpose-x": ("Body", "performance", "coco17", {}),
    "rtmo-m": ("Body", "balanced", "coco17", {"pose": "rtmo"}),
    "rtmw-l": ("Wholebody", "lightweight", "coco_wholebody133", {}),
    "rtmw-x": ("Wholebody", "balanced", "coco_wholebody133", {}),
    "rtmw3d-x": ("Wholebody3d", "balanced", "coco_wholebody133", {}),
}


class RtmlibBackend:
    conf_threshold = 0.3

    def __init__(self, name: str, device: str | None = None):
        from .gpu import ort_cuda

        if name not in KINDS:
            raise ValueError(f"Unknown rtmlib backend {name}")
        self.name = name
        self.cls_name, self.mode, self.skeleton_name, self.kwargs = KINDS[name]
        self.device = "cuda" if (device != "cpu" and ort_cuda()) else "cpu"
        self.device_note = "" if self.device == "cuda" or device == "cpu" else "未检测到 onnxruntime CUDA 执行器，已用 CPU"
        self.model_id = (f"rtmlib/{self.cls_name}:{self.mode}{':rtmo' if self.kwargs.get('pose') == 'rtmo' else ''}"
                         f"/{self.device}")
        self._model = None
        self._prev_box = None

    def open(self) -> None:
        import rtmlib

        cls = getattr(rtmlib, self.cls_name)
        kw = dict(mode=self.mode, backend="onnxruntime", device=self.device)
        if self.kwargs.get("pose") == "rtmo":
            kw["pose"] = cls.RTMO_MODE[self.mode]["pose"]
            kw["pose_input_size"] = cls.RTMO_MODE[self.mode]["pose_input_size"]
        if self._model is None:   # weights load once; tracking state resets per run
            self._model = cls(**kw)
            sess = getattr(getattr(self._model, "pose_model", None), "session", None)
            if self.device == "cuda" and sess is not None and "CUDAExecutionProvider" not in sess.get_providers():
                raise RuntimeError(f"{self.name}: CUDA requested but the session runs on {sess.get_providers()}")
        self._prev_box = None

    def estimate(self, frame_bgr: np.ndarray, ts_ms: int):
        out = self._model(frame_bgr)
        world = None
        if self.cls_name == "Wholebody3d":
            k3d, scores, _, k2d = out
            kpts = k2d
        else:
            kpts, scores = out
            k3d = None
        kpts, scores = np.asarray(kpts), np.asarray(scores)
        if kpts.ndim != 3 or len(kpts) == 0:
            return None, None, None
        i, self._prev_box = select_target(kpts, scores, self._prev_box, self.conf_threshold)
        if i < 0:
            return None, None, None
        if k3d is not None:
            world = np.asarray(k3d)[i].astype(np.float32)
        return kpts[i].astype(np.float32), np.clip(scores[i], 0, 1).astype(np.float32), world

    def close(self) -> None:
        self._prev_box = None
