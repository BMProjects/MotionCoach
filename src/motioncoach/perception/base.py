"""Common backend interface and the shared decoding loop.

A backend only implements ``open / estimate / close`` on single frames. Decoding, real
timestamps, time windows and sampling are handled once here, so every model sees exactly the
same frames — a precondition for comparing pose algorithms fairly.
"""

from __future__ import annotations

import math
import time
from typing import Callable, Iterator, Protocol

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


def _area(b: np.ndarray) -> float:
    return max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))


def box_iou(boxes: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of each row of ``boxes`` (N,4 xyxy) with ``b``."""
    ix = np.clip(np.minimum(boxes[:, 2], b[2]) - np.maximum(boxes[:, 0], b[0]), 0, None)
    iy = np.clip(np.minimum(boxes[:, 3], b[3]) - np.maximum(boxes[:, 1], b[1]), 0, None)
    inter = ix * iy
    areas = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    return inter / (areas + _area(b) - inter + 1e-9)


def expand_box(b: np.ndarray, factor: float, size: tuple[int, int] | None = None) -> np.ndarray:
    """Scale a xyxy box about its centre; clip to the frame (width, height) when given."""
    c, half = 0.5 * (b[:2] + b[2:]), 0.5 * factor * (b[2:] - b[:2])
    out = np.concatenate([c - half, c + half])
    if size is not None:
        out = np.clip(out, 0, [size[0] - 1, size[1] - 1, size[0] - 1, size[1] - 1])
    return out


class BoxTracker:
    """The one target person's box across sampled frames, for top-down models (docs/EXPERIMENT_PLAN.md E2).

    Per frame the detection overlapping the predicted box is used. When there is none - a deep bend or a fast
    turn the detector misses, or only other people are visible - the predicted box is used instead
    ("bridged") for up to ``max_bridge_s``, and every confident estimate updates the prediction from the
    skeleton itself, so the box follows the body through the miss. Without a target the largest detection
    starts one. Counts per box source are kept in ``stats``."""

    def __init__(self, max_bridge_s: float = 1.0, min_iou: float = 0.3, kp_pad: float = 1.2):
        self.max_bridge_s, self.min_iou, self.kp_pad = max_bridge_s, min_iou, kp_pad
        self.reset()

    def reset(self) -> None:
        self.box: np.ndarray | None = None
        self.last_ok_ms: float | None = None
        self.stats = {"detected": 0, "bridged": 0, "enlarged": 0, "lost": 0}

    def choose(self, boxes: np.ndarray, ts_ms: float) -> tuple[np.ndarray | None, str]:
        """Box to run the pose model on, and its source: "detected", "bridged" or "" (nothing to run)."""
        boxes = np.asarray(boxes, np.float64).reshape(-1, 4)
        if self.box is not None and self.last_ok_ms is not None and ts_ms - self.last_ok_ms > 1000 * self.max_bridge_s:
            self.box = None                                    # bridged too long: let the detector re-acquire
        if self.box is not None:
            if len(boxes):
                iou = box_iou(boxes, self.box)
                if iou.max() >= self.min_iou:
                    return boxes[int(np.argmax(iou))], "detected"
            return self.box.copy(), "bridged"
        if len(boxes):
            return boxes[int(np.argmax([_area(b) for b in boxes]))], "detected"
        return None, ""

    def update(self, used: np.ndarray, kp_box: np.ndarray | None, source: str, ts_ms: float,
               size: tuple[int, int]) -> None:
        """``kp_box``: extent of the confident body joints, or None when the estimate was rejected."""
        if kp_box is None:
            self.stats["lost"] += 1
            return
        self.stats[source] += 1
        pred = expand_box(kp_box, self.kp_pad, size)
        self.box = pred if source == "bridged" else np.concatenate([np.minimum(used[:2], pred[:2]), np.maximum(used[2:], pred[2:])])
        self.last_ok_ms = ts_ms


def sampled_frames(video_path, sample_fps: float, t0: float = 0.0, t1: float | None = None,
                   progress: Callable[[float], None] | None = None) -> Iterator[tuple[int, np.ndarray]]:
    """(timestamp ms, BGR frame) on the absolute k / fps grid within [t0, t1) seconds.

    A window run and a slice of a full-length run pick the same frames; timestamps strictly increase."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    try:
        duration = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / (cap.get(cv2.CAP_PROP_FPS) or 30.0)
        t1 = duration if t1 is None else min(t1, duration + 1.0)
        if t0 > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t0 - 0.5) * 1000.0)
        k = math.ceil(t0 * sample_fps - 1e-6)
        next_t = k / sample_fps
        last_ms, n = -1, 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if t + 1e-6 >= t1:   # same tolerance as the grid test below: a grid point belongs to [t0, t1) exactly once
                break
            if t + 1e-6 < next_t:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            k = max(k + 1, math.floor(t * sample_fps + 1e-6) + 1)   # next grid point strictly after t
            next_t = k / sample_fps
            ms = max(int(round(t * 1000)), last_ms + 1)
            last_ms = ms
            yield ms, frame
            n += 1
            if progress and n % 20 == 0 and t1 > t0:
                progress(min((t - t0) / (t1 - t0), 1.0))
    finally:
        cap.release()


def run_on_video(backend: PoseBackend, video_path, sample_fps: float, t0: float = 0.0, t1: float | None = None,
                 progress: Callable[[float], None] | None = None) -> PoseSequence:
    """Estimate poses on frames sampled at ``sample_fps`` within [t0, t1) seconds."""
    J = len(get_skeleton(backend.skeleton_name).joints)
    ts, xys, confs, worlds = [], [], [], []
    size = None
    infer_s = 0.0
    backend.open()
    try:
        for ms, frame in sampled_frames(video_path, sample_fps, t0, t1, progress):
            h, w = frame.shape[:2]
            size = (w, h)
            s = time.perf_counter()
            xy, cf, wd = backend.estimate(frame, ms)
            if len(ts) >= WARMUP:   # first inferences include lazy init / CUDA warm-up
                infer_s += time.perf_counter() - s
            ts.append(ms / 1000.0)
            xys.append(xy if xy is not None else np.full((J, 2), np.nan, np.float32))
            confs.append(cf if cf is not None else np.zeros(J, np.float32))
            worlds.append(wd if wd is not None else np.full((J, 3), np.nan, np.float32))
    finally:
        backend.close()
    if not ts:
        raise ValueError(f"No frames decoded from {video_path} in [{t0}, {t1})")
    return PoseSequence(
        t=np.asarray(ts, np.float64), xy=np.stack(xys).astype(np.float32), conf=np.stack(confs).astype(np.float32),
        world=np.stack(worlds).astype(np.float32), frame_size=size, skeleton_name=backend.skeleton_name,
        meta={"model": backend.model_id, "backend": backend.name, "device": backend.device,
              "conf_thr": backend.conf_threshold, "sample_fps": sample_fps, "window": [t0, t1],
              "sampling": SAMPLING_VERSION, "device_note": getattr(backend, "device_note", ""),
              "ms_per_frame": 1000.0 * infer_s / max(1, len(ts) - WARMUP), "source": str(video_path),
              "boxes": getattr(backend, "tracking_stats", None)},
    )
