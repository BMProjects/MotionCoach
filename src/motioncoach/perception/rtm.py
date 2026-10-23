"""rtmlib adapters: RTMPose (COCO-17, top-down), RTMO (one-stage), RTMW (COCO-WholeBody 133), RTMW3D.

rtmlib runs ONNX models without MMCV/MMPose and downloads weights on first use
(~/.cache/rtmlib). Input frames are BGR, as rtmlib expects. SimCC confidences are not
probabilities, so they are clipped to [0, 1] and use a lower threshold than MediaPipe.
GPU: onnxruntime-gpu CUDA execution provider (see perception/gpu.py).
RTMW3D's 3D output is in model-crop pixel units for x/y plus a relative depth z — not metric;
it is stored in ``world`` for experiments, not used by the 2D features.

Person boxes (top-down RTMPose / RTMW): the detector can be chosen (``MOTIONCOACH_RTM_DETECTOR``: native = rtmlib's
own choice per mode, or tiny / m / x, YOLOX trained on HumanArt) and the target's box is tracked across
frames (``BoxTracker``; ``MOTIONCOACH_RTM_TRACK=0`` turns it off). rtmlib alone runs the pose model on the
whole frame when the detector finds nobody, which in deep bends gave a scrambled or no skeleton; with the
tracker the last box is carried through such frames, and a skeleton reaching outside its detection box is
re-estimated once on the enlarged box (docs/EXPERIMENT_PLAN.md E2, docs/ITERATION_LOG.md I18).
"""

from __future__ import annotations

import os

import numpy as np

from .base import BoxTracker, expand_box, select_target

_OMM = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
DETECTORS = {
    "tiny": (_OMM + "yolox_tiny_8xb8-300e_humanart-6f3252f9.zip", (416, 416)),
    "m": (_OMM + "yolox_m_8xb8-300e_humanart-c2c7a14a.zip", (640, 640)),
    "x": (_OMM + "yolox_x_8xb8-300e_humanart-a39d44ed.zip", (640, 640)),
}
# 2026-10-23 (research/detector_bench.py): box tracking is what fixes deep bends; larger detectors cost 2-3x
# the time and change the crops enough to raise the indicator error in calm forms, so the default keeps
# rtmlib's own detector per mode ("native") and turns tracking on.
DEFAULT_DETECTOR = os.environ.get("MOTIONCOACH_RTM_DETECTOR", "native")
DEFAULT_TRACK = os.environ.get("MOTIONCOACH_RTM_TRACK", "1") != "0"
TRACKER_VERSION = "trk1"
BODY = list(range(5, 17))      # shoulders .. ankles: the same indices in COCO-17 and COCO-WholeBody
MIN_BODY_JOINTS = 6            # confident body joints for an estimate to move the tracked box

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

    def __init__(self, name: str, device: str | None = None, detector: str | None = None, track: bool | None = None):
        from .gpu import ort_cuda

        if name not in KINDS:
            raise ValueError(f"Unknown rtmlib backend {name}")
        self.name = name
        self.cls_name, self.mode, self.skeleton_name, self.kwargs = KINDS[name]
        self.device = "cuda" if (device != "cpu" and ort_cuda()) else "cpu"
        self.device_note = "" if self.device == "cuda" or device == "cpu" else "未检测到 onnxruntime CUDA 执行器，已用 CPU"
        top_down = self.cls_name in ("Body", "Wholebody") and self.kwargs.get("pose") != "rtmo"
        self.detector = (DEFAULT_DETECTOR if detector is None else detector) if top_down else None
        if self.detector is not None and self.detector != "native" and self.detector not in DETECTORS:
            raise ValueError(f"Unknown detector {self.detector!r}; known: native, {', '.join(DETECTORS)}")
        self.track = top_down and (DEFAULT_TRACK if track is None else track)
        boxes = ""
        if self.detector not in (None, "native") or self.track:      # the plain rtmlib id keeps old caches valid
            boxes = f"+det-{self.detector}" + (f"+{TRACKER_VERSION}" if self.track else "")
        self.model_id = (f"rtmlib/{self.cls_name}:{self.mode}{':rtmo' if self.kwargs.get('pose') == 'rtmo' else ''}"
                         f"{boxes}/{self.device}")
        self._model = None
        self._prev_box = None
        self._tracker = BoxTracker() if self.track else None

    def open(self) -> None:
        import rtmlib

        cls = getattr(rtmlib, self.cls_name)
        kw = dict(mode=self.mode, backend="onnxruntime", device=self.device)
        if self.kwargs.get("pose") == "rtmo":
            kw["pose"] = cls.RTMO_MODE[self.mode]["pose"]
            kw["pose_input_size"] = cls.RTMO_MODE[self.mode]["pose_input_size"]
        elif self.detector not in (None, "native"):
            kw["det"], kw["det_input_size"] = DETECTORS[self.detector]
            kw["pose"] = cls.MODE[self.mode]["pose"]
            kw["pose_input_size"] = cls.MODE[self.mode]["pose_input_size"]
        if self._model is None:   # weights load once; tracking state resets per run
            self._model = cls(**kw)
            sess = getattr(getattr(self._model, "pose_model", None), "session", None)
            if self.device == "cuda" and sess is not None and "CUDAExecutionProvider" not in sess.get_providers():
                raise RuntimeError(f"{self.name}: CUDA requested but the session runs on {sess.get_providers()}")
        self._prev_box = None
        if self._tracker is not None:
            self._tracker.reset()

    @property
    def tracking_stats(self) -> dict | None:
        return dict(self._tracker.stats) if self._tracker is not None else None

    def _pose(self, frame_bgr: np.ndarray, box: np.ndarray):
        k, s = self._model.pose_model(frame_bgr, bboxes=[box.tolist()])
        return np.asarray(k)[0], np.clip(np.asarray(s)[0], 0, 1)

    def _body_box(self, kpts: np.ndarray, scores: np.ndarray) -> np.ndarray | None:
        ok = scores[BODY] >= self.conf_threshold
        if ok.sum() < MIN_BODY_JOINTS:
            return None
        p = kpts[BODY][ok]
        return np.array([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()])

    def _estimate_tracked(self, frame_bgr: np.ndarray, ts_ms: int):
        h, w = frame_bgr.shape[:2]
        boxes = np.asarray(self._model.det_model(frame_bgr), dtype=np.float64).reshape(-1, 4)
        box, source = self._tracker.choose(boxes, ts_ms)
        if box is None:
            self._tracker.update(None, None, "", ts_ms, (w, h))
            return None, None, None
        kpts, scores = self._pose(frame_bgr, box)
        kb = self._body_box(kpts, scores)
        if kb is not None and source == "detected":
            margin = 0.03 * (box[2:] - box[:2])
            if np.any(kb[:2] < box[:2] - margin) or np.any(kb[2:] > box[2:] + margin):   # limbs outside the box
                big = expand_box(np.concatenate([np.minimum(box[:2], kb[:2]), np.maximum(box[2:], kb[2:])]), 1.1, (w, h))
                k2, s2 = self._pose(frame_bgr, big)
                if s2[BODY].mean() >= scores[BODY].mean() - 0.02:
                    kpts, scores, box = k2, s2, big
                    kb = self._body_box(kpts, scores)
                    self._tracker.stats["enlarged"] += 1
        self._tracker.update(box, kb, source, ts_ms, (w, h))
        if (scores >= self.conf_threshold).sum() < 3:
            return None, None, None
        return kpts.astype(np.float32), scores.astype(np.float32), None

    def estimate(self, frame_bgr: np.ndarray, ts_ms: int):
        if self._tracker is not None:
            return self._estimate_tracked(frame_bgr, ts_ms)
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

    def estimate_all(self, frame_bgr: np.ndarray, ts_ms: int):
        """Every detected person: (kpts (N,J,2), conf (N,J), boxes (N,4) or None). Top-down models expose the
        detector's boxes; the one-stage RTMO gives keypoints only."""
        det = getattr(self._model, "det_model", None)
        if det is not None and self.cls_name != "Wholebody3d":
            boxes = np.asarray(det(frame_bgr), dtype=np.float32).reshape(-1, 4)
            if len(boxes) == 0:
                return np.zeros((0, 0, 2), np.float32), np.zeros((0, 0), np.float32), boxes
            kpts, scores = self._model.pose_model(frame_bgr, bboxes=boxes)
        else:
            out = self._model(frame_bgr)
            kpts, scores = (out[3], out[1]) if self.cls_name == "Wholebody3d" else out
            boxes = None
        kpts, scores = np.asarray(kpts, np.float32), np.asarray(scores, np.float32)
        if kpts.ndim != 3:
            return np.zeros((0, 0, 2), np.float32), np.zeros((0, 0), np.float32), None
        return kpts, np.clip(scores, 0, 1), boxes

    def close(self) -> None:
        self._prev_box = None   # the tracker keeps its counts for run_on_video's metadata until the next open()
