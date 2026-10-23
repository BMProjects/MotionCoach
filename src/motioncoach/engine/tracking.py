"""Online multi-person tracking: ByteTrack association with a pose term.

ByteTrack (Zhang et al., arXiv:2110.06864): a constant-velocity Kalman filter per track on (cx, cy, aspect,
height); detections split by score; high-score detections are matched first (Hungarian on 1 - similarity),
the remaining tracks are then matched to low-score detections, so a person who is briefly occluded (low
score) keeps the track. Lost tracks are kept ``buffer_s`` seconds.

Similarity = mean of box IoU and object keypoint similarity (OKS, COCO definition with uniform per-joint
constants): two people in similar clothes standing side by side have distinct skeletons, and OKS stays high
for the same person even when the box changes with the pose.

Implemented here rather than imported: the maintained Python packages either pull in ``opencv-python``
(which clashes with MediaPipe's ``opencv-contrib-python``) or are AGPL-3.0.
Output is short, trustworthy tracklets (``min_iou`` keeps matching strict); ``engine.group`` stitches them
into identities offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

OKS_K = 0.08          # per-joint falloff (COCO uses 0.025-0.107; uniform keeps it model independent)


@dataclass
class TrackParams:
    high: float = 0.5           # detection score for the first association and for starting a track
    low: float = 0.2            # low-score detections only extend existing tracks
    min_sim_high: float = 0.3   # minimal similarity of a first-stage match
    min_sim_low: float = 0.5    # stricter for low-score detections
    buffer_s: float = 3.0       # how long a lost track can be recovered (occlusion in a crowd lasts seconds)
    min_len: int = 3            # samples; shorter tracklets are discarded


class _Kalman:
    """Constant-velocity Kalman filter on (cx, cy, a, h), as in SORT / ByteTrack."""

    def __init__(self, z: np.ndarray):
        self.x = np.r_[z, np.zeros(4)]
        h = z[3]
        self.P = np.diag([2 * 0.05 * h, 2 * 0.05 * h, 1e-2, 2 * 0.05 * h,
                          10 * 0.00625 * h, 10 * 0.00625 * h, 1e-5, 10 * 0.00625 * h]) ** 2
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)
        self.H = np.eye(4, 8)

    def predict(self) -> None:
        h = self.x[3]
        q = np.r_[0.05 * h, 0.05 * h, 1e-2, 0.05 * h, 0.00625 * h, 0.00625 * h, 1e-5, 0.00625 * h] ** 2
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + np.diag(q)

    def update(self, z: np.ndarray) -> None:
        h = self.x[3]
        R = np.diag(np.r_[0.05 * h, 0.05 * h, 1e-1, 0.05 * h] ** 2)
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(8) - K @ self.H) @ self.P

    def box(self) -> np.ndarray:
        cx, cy, a, h = self.x[:4]
        w = a * h
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def _xyah(b: np.ndarray) -> np.ndarray:
    w, h = max(1.0, b[2] - b[0]), max(1.0, b[3] - b[1])
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, w / h, h])


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N,4) x (M,4) -> (N,M)."""
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    area = lambda x: np.clip(x[:, 2] - x[:, 0], 0, None) * np.clip(x[:, 3] - x[:, 1], 0, None)   # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def oks(xa: np.ndarray, va: np.ndarray, xb: np.ndarray, vb: np.ndarray, area: np.ndarray) -> np.ndarray:
    """Keypoint similarity: (N,J,2),(N,J) bool vs (M,J,2),(M,J) -> (N,M); ``area`` (N,) object scale^2."""
    d2 = ((xa[:, None] - xb[None]) ** 2).sum(axis=3)                     # (N,M,J)
    both = va[:, None] & vb[None]
    e = np.exp(-d2 / (2 * area[:, None, None] * (2 * OKS_K) ** 2 + 1e-9))
    n = both.sum(axis=2)
    return np.where(n > 0, (e * both).sum(axis=2) / np.maximum(n, 1), 0.0)


@dataclass
class _Track:
    tid: int
    kf: _Kalman
    xy: np.ndarray
    vis: np.ndarray
    last: int                                     # last frame with a detection
    dets: list[tuple[int, int]] = field(default_factory=list)   # (frame, detection index)


def track(frame: np.ndarray, box: np.ndarray, score: np.ndarray, xy: np.ndarray, conf: np.ndarray,
          n_frames: int, fps: float, conf_thr: float, p: TrackParams = TrackParams()) -> list[list[tuple[int, int]]]:
    """Per-detection arrays (``frame`` index of each detection) -> tracklets [(frame, detection index), ...]."""
    vis_all = conf >= conf_thr
    by_frame = [[] for _ in range(n_frames)]
    for d, f in enumerate(frame):
        by_frame[int(f)].append(d)
    active: list[_Track] = []
    finished: list[_Track] = []
    next_id, buffer = 0, max(1, int(round(p.buffer_s * fps)))

    def sim(tracks: list[_Track], dets: list[int]) -> np.ndarray:
        if not tracks or not dets:
            return np.zeros((len(tracks), len(dets)))
        tb = np.array([t.kf.box() for t in tracks])
        db = box[dets]
        area = np.array([max(1.0, (b[2] - b[0]) * (b[3] - b[1])) for b in tb])
        s_oks = oks(np.array([t.xy for t in tracks]), np.array([t.vis for t in tracks]), xy[dets], vis_all[dets], area)
        return 0.5 * iou(tb, db) + 0.5 * s_oks

    def match(tracks, dets, min_sim):
        S = sim(tracks, dets)
        if S.size == 0:
            return [], list(range(len(tracks))), list(range(len(dets)))
        r, c = linear_sum_assignment(-S)
        pairs = [(i, j) for i, j in zip(r, c) if S[i, j] >= min_sim]
        mt, md = {i for i, _ in pairs}, {j for _, j in pairs}
        return pairs, [i for i in range(len(tracks)) if i not in mt], [j for j in range(len(dets)) if j not in md]

    for f in range(n_frames):
        for t in active:
            t.kf.predict()
        dets = by_frame[f]
        hi = [d for d in dets if score[d] >= p.high]
        lo = [d for d in dets if p.low <= score[d] < p.high]
        pairs, rest_t, rest_d = match(active, hi, p.min_sim_high)
        upd = [(active[i], hi[j]) for i, j in pairs]
        remaining = [active[i] for i in rest_t]
        pairs2, rest_t2, _ = match(remaining, lo, p.min_sim_low)
        upd += [(remaining[i], lo[j]) for i, j in pairs2]
        for t, d in upd:
            t.kf.update(_xyah(box[d]))
            t.xy, t.vis, t.last = xy[d], vis_all[d], f
            t.dets.append((f, d))
        for j in rest_d:                              # unmatched confident detections start tracks
            d = hi[j]
            t = _Track(next_id, _Kalman(_xyah(box[d])), xy[d], vis_all[d], f, [(f, d)])
            next_id += 1
            active.append(t)
        still = []
        for t in active:
            (finished if f - t.last > buffer else still).append(t)
        active = still
    finished += active
    return [t.dets for t in sorted(finished, key=lambda t: t.dets[0]) if len(t.dets) >= p.min_len]
