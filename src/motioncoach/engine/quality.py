"""Capture-quality measurements of a raw PoseSequence (numbers only; thresholds live in apps)."""

from __future__ import annotations

import numpy as np

from .sequence import PoseSequence

GROUPS = {
    "arms": ("left_elbow", "right_elbow", "left_wrist", "right_wrist"),
    "torso": ("left_shoulder", "right_shoulder", "left_hip", "right_hip"),
    "legs": ("left_knee", "right_knee", "left_ankle", "right_ankle"),
    "feet": ("left_heel", "right_heel", "left_foot_index", "right_foot_index"),
}


def measure(seq: PoseSequence, conf_thr: float = 0.5) -> dict:
    det = seq.detected
    out: dict = {
        "samples": len(seq),
        "duration_s": float(seq.t[-1] - seq.t[0]) if len(seq) > 1 else 0.0,
        "frame_size": list(seq.frame_size),
        "detection_rate": float(det.mean()) if len(seq) else 0.0,
        "visible_rate": {},
    }
    sk = seq.skeleton
    for g, joints in GROUPS.items():
        idx = [sk.index[j] for j in joints if sk.has(j)]
        if idx:
            out["visible_rate"][g] = float((seq.conf[:, idx] >= conf_thr).mean())
    w, h = seq.frame_size
    xy = seq.xy[det]
    if len(xy):
        with np.errstate(all="ignore"):
            ys, xs = xy[..., 1], xy[..., 0]
            height = np.nanmax(ys, axis=1) - np.nanmin(ys, axis=1)
            out["person_height_frac"] = float(np.nanmedian(height) / h)
            margin = 0.01
            outside = (xs < w * margin) | (xs > w * (1 - margin)) | (ys < h * margin) | (ys > h * (1 - margin))
            out["touching_edge_rate"] = float(np.mean(outside.any(axis=1)))
    else:
        out["person_height_frac"] = 0.0
        out["touching_edge_rate"] = 1.0
    # longest run without detection
    longest, run = 0, 0
    dt = float(np.median(np.diff(seq.t))) if len(seq) > 1 else 0.0
    for d in det:
        run = 0 if d else run + 1
        longest = max(longest, run)
    out["longest_gap_s"] = longest * dt
    return out
