"""Every person in every sampled frame (multi-person input for ``engine.tracking``).

Same decoding and sampling grid as the single-person path (``base.sampled_frames``). Per detection:
keypoints, confidences, a box (the detector's, else the keypoints' extent), a detection score (mean
confidence of the body joints) and an appearance descriptor - HSV colour histograms of the torso and the
legs, which separate people of different clothing and are cheap to compute while the frame is decoded.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..engine.skeleton import get_skeleton
from .base import WARMUP, sampled_frames

H_BINS, S_BINS, V_BINS = 8, 3, 2
APP_DIM = 2 * H_BINS * S_BINS * V_BINS          # torso + legs
BODY = ("left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_knee", "right_knee",
        "left_ankle", "right_ankle", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "nose")


@dataclass
class MultiPoses:
    t: np.ndarray                   # (T,) seconds
    frame: np.ndarray               # (D,) frame index of each detection
    xy: np.ndarray                  # (D, J, 2) pixels
    conf: np.ndarray                # (D, J)
    box: np.ndarray                 # (D, 4) x0 y0 x1 y1
    score: np.ndarray               # (D,) detection score
    app: np.ndarray                 # (D, APP_DIM) appearance histograms (each half sums to 1, or 0 if not visible)
    frame_size: tuple[int, int]
    skeleton_name: str
    meta: dict = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part.npz")
        np.savez_compressed(tmp, t=self.t, frame=self.frame, xy=self.xy, conf=self.conf, box=self.box,
                            score=self.score, app=self.app, frame_size=np.asarray(self.frame_size),
                            skeleton=np.asarray(self.skeleton_name), meta=np.asarray(repr(self.meta)))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "MultiPoses":
        import ast

        z = np.load(path, allow_pickle=False)
        return cls(t=z["t"], frame=z["frame"], xy=z["xy"], conf=z["conf"], box=z["box"], score=z["score"],
                   app=z["app"], frame_size=tuple(int(v) for v in z["frame_size"]),
                   skeleton_name=str(z["skeleton"]), meta=ast.literal_eval(str(z["meta"])))


def _region_hist(hsv: np.ndarray, pts: np.ndarray, shrink: float = 0.15) -> np.ndarray:
    """Colour histogram inside the bounding rectangle of ``pts`` (shrunk to stay on the body)."""
    out = np.zeros(H_BINS * S_BINS * V_BINS, np.float32)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 2:
        return out
    (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
    dx, dy = (x1 - x0) * shrink, (y1 - y0) * shrink
    h, w = hsv.shape[:2]
    xa, xb = int(max(0, x0 + dx)), int(min(w, x1 - dx))
    ya, yb = int(max(0, y0 + dy)), int(min(h, y1 - dy))
    if xb - xa < 3 or yb - ya < 3:
        return out
    hist = cv2.calcHist([hsv[ya:yb, xa:xb]], [0, 1, 2], None, [H_BINS, S_BINS, V_BINS], [0, 180, 0, 256, 0, 256])
    hist = hist.ravel().astype(np.float32)
    total = hist.sum()
    return hist / total if total > 0 else out


def appearance(frame_bgr: np.ndarray, xy: np.ndarray, ok: np.ndarray, idx: dict[str, int]) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    pick = lambda names: np.array([xy[idx[n]] if ok[idx[n]] else (np.nan, np.nan) for n in names])   # noqa: E731
    torso = pick(("left_shoulder", "right_shoulder", "left_hip", "right_hip"))
    legs = pick(("left_hip", "right_hip", "left_knee", "right_knee"))
    return np.concatenate([_region_hist(hsv, torso), _region_hist(hsv, legs)])


def run_multi_on_video(backend, video_path, sample_fps: float, t0: float = 0.0, t1: float | None = None,
                       progress: Callable[[float], None] | None = None, max_persons: int = 30) -> MultiPoses:
    """All persons per sampled frame. ``backend`` must implement ``estimate_all``."""
    if not hasattr(backend, "estimate_all"):
        raise ValueError(f"{backend.name} does not report several persons per frame (no estimate_all)")
    sk = get_skeleton(backend.skeleton_name)
    idx = {j: i for i, j in enumerate(sk.joints)}
    body = [idx[j] for j in BODY if j in idx]
    ts, rows = [], {k: [] for k in ("frame", "xy", "conf", "box", "score", "app")}
    size, infer_s = None, 0.0
    backend.open()
    try:
        for fi, (ms, frame) in enumerate(sampled_frames(video_path, sample_fps, t0, t1, progress)):
            size = (frame.shape[1], frame.shape[0])
            s = time.perf_counter()
            kpts, conf, boxes = backend.estimate_all(frame, ms)
            if fi >= WARMUP:
                infer_s += time.perf_counter() - s
            ts.append(ms / 1000.0)
            n = len(kpts)
            if n == 0:
                continue
            score = conf[:, body].mean(axis=1)
            order = np.argsort(-score)[:max_persons]
            for i in order:
                ok = conf[i] >= backend.conf_threshold
                if ok[body].sum() < 4:
                    continue
                p = kpts[i][ok]
                kb = np.array([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()], np.float32)
                rows["frame"].append(fi)
                rows["xy"].append(kpts[i])
                rows["conf"].append(conf[i])
                rows["box"].append(boxes[i] if boxes is not None and len(boxes) == n else kb)
                rows["score"].append(float(score[i]))
                rows["app"].append(appearance(frame, kpts[i], ok, idx))
    finally:
        backend.close()
    if not ts:
        raise ValueError(f"No frames decoded from {video_path}")
    J = len(sk.joints)
    D = len(rows["frame"])
    return MultiPoses(
        t=np.asarray(ts, np.float64),
        frame=np.asarray(rows["frame"], np.int32),
        xy=np.asarray(rows["xy"], np.float32).reshape(D, J, 2),
        conf=np.asarray(rows["conf"], np.float32).reshape(D, J),
        box=np.asarray(rows["box"], np.float32).reshape(D, 4),
        score=np.asarray(rows["score"], np.float32),
        app=np.asarray(rows["app"], np.float32).reshape(D, APP_DIM),
        frame_size=size, skeleton_name=backend.skeleton_name,
        meta={"model": backend.model_id, "backend": backend.name, "device": backend.device,
              "conf_thr": backend.conf_threshold, "sample_fps": sample_fps, "window": [t0, t1],
              "ms_per_frame": 1000.0 * infer_s / max(1, len(ts) - WARMUP), "source": str(video_path)})
