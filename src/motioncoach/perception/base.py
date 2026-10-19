"""Common backend interface and the shared decoding loop.

A backend only implements ``open / estimate / close`` on single frames. Decoding, real
timestamps, time windows and sampling are handled once here, so every model sees exactly the
same frames — a precondition for comparing pose algorithms fairly.
"""

from __future__ import annotations

import math
import time
from typing import Callable, Protocol

import cv2
import numpy as np

from ..engine.sequence import PoseSequence
from ..engine.skeleton import get_skeleton

WARMUP = 3
SAMPLING_VERSION = "grid-k/fps"
Estimate = tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]   # xy (J,2) px, conf (J,), world (J,3)


class PoseBackend(Protocol):
    name: str
    model_id: str
    skeleton_name: str
    device: str
    conf_threshold: float      # confidence below which a joint counts as missing (model-specific calibration)

    def open(self) -> None: ...
    def estimate(self, frame_bgr: np.ndarray, ts_ms: int) -> Estimate: ...
    def close(self) -> None: ...


def select_target(kpts: np.ndarray, scores: np.ndarray, prev_box: np.ndarray | None,
                  thr: float = 0.3) -> tuple[int, np.ndarray | None]:
    """Pick one person among detections: best IoU with the previous box, else the largest."""
    boxes = []
    for k, s in zip(kpts, scores):
        ok = s >= thr
        if ok.sum() < 3:
            boxes.append(None)
            continue
        p = k[ok]
        boxes.append(np.array([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()]))
    cand = [i for i, b in enumerate(boxes) if b is not None]
    if not cand:
        return -1, prev_box

    def area(b):
        return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    if prev_box is not None:
        def iou(b):
            ix = max(0.0, min(b[2], prev_box[2]) - max(b[0], prev_box[0]))
            iy = max(0.0, min(b[3], prev_box[3]) - max(b[1], prev_box[1]))
            inter = ix * iy
            return inter / (area(b) + area(prev_box) - inter + 1e-9)
        best = max(cand, key=lambda i: iou(boxes[i]))
        if iou(boxes[best]) > 0.1:
            return best, boxes[best]
    best = max(cand, key=lambda i: area(boxes[i]))
    return best, boxes[best]


def run_on_video(backend: PoseBackend, video_path, sample_fps: float, t0: float = 0.0, t1: float | None = None,
                 progress: Callable[[float], None] | None = None) -> PoseSequence:
    """Estimate poses on frames sampled at ``sample_fps`` within [t0, t1) seconds."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    duration = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / (cap.get(cv2.CAP_PROP_FPS) or 30.0)
    t1 = duration if t1 is None else min(t1, duration + 1.0)
    if t0 > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t0 - 0.5) * 1000.0)
    J = len(get_skeleton(backend.skeleton_name).joints)
    ts, xys, confs, worlds = [], [], [], []
    size = None
    # absolute sampling grid (k / fps): a window run and a slice of a full-length run pick the same frames
    k = math.ceil(t0 * sample_fps - 1e-6)
    next_t = k / sample_fps
    last_ms = -1
    infer_s = 0.0
    backend.open()
    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if t >= t1:
                break
            if t + 1e-6 < next_t:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            k = max(k + 1, math.floor(t * sample_fps + 1e-6) + 1)   # next grid point strictly after t
            next_t = k / sample_fps
            h, w = frame.shape[:2]
            size = (w, h)
            ms = max(int(round(t * 1000)), last_ms + 1)
            last_ms = ms
            s = time.perf_counter()
            xy, cf, wd = backend.estimate(frame, ms)
            if len(ts) >= WARMUP:   # first inferences include lazy init / CUDA warm-up
                infer_s += time.perf_counter() - s
            ts.append(ms / 1000.0)
            xys.append(xy if xy is not None else np.full((J, 2), np.nan, np.float32))
            confs.append(cf if cf is not None else np.zeros(J, np.float32))
            worlds.append(wd if wd is not None else np.full((J, 3), np.nan, np.float32))
            if progress and len(ts) % 20 == 0 and t1 > t0:
                progress(min((t - t0) / (t1 - t0), 1.0))
    finally:
        backend.close()
        cap.release()
    if not ts:
        raise ValueError(f"No frames decoded from {video_path} in [{t0}, {t1})")
    return PoseSequence(
        t=np.asarray(ts, np.float64), xy=np.stack(xys).astype(np.float32), conf=np.stack(confs).astype(np.float32),
        world=np.stack(worlds).astype(np.float32), frame_size=size, skeleton_name=backend.skeleton_name,
        meta={"model": backend.model_id, "backend": backend.name, "device": backend.device,
              "conf_thr": backend.conf_threshold, "sample_fps": sample_fps, "window": [t0, t1],
              "sampling": SAMPLING_VERSION, "device_note": getattr(backend, "device_note", ""),
              "ms_per_frame": 1000.0 * infer_s / max(1, len(ts) - WARMUP), "source": str(video_path)},
    )
