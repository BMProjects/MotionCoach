"""Camera motion of a hand-held or panning recording, from the background only.

For every sampled frame (the pose sampling grid), a similarity transform (shift, rotation, uniform scale)
that maps it onto the first frame: sparse corners on a downscaled grey image, Lucas-Kanade flow to the
next sampled frame, RANSAC fit, with the people masked out (their boxes from the multi-person pass) so
moving bodies do not pull the estimate. Chained frame to frame. Positions mapped through it live in one
fixed coordinate system, which is what identity stitching and the floor layout assume.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..perception.base import sampled_frames

WORK_W = 640                      # analysis width; transforms are rescaled to full resolution


def camera_motion(video: Path, sample_fps: float, boxes_by_frame: list[np.ndarray], frame_size: tuple[int, int],
                  cache: Path | None = None) -> np.ndarray:
    """(T, 2, 3) affine matrices mapping each sampled frame's pixels into the first frame's pixels."""
    if cache is not None and Path(cache).exists():
        return np.load(cache)["A"]
    W, H = frame_size
    s = WORK_W / W
    out, prev, prev_pts, acc = [], None, None, np.eye(3)
    for k, (_, frame) in enumerate(sampled_frames(video, sample_fps)):
        g = cv2.cvtColor(cv2.resize(frame, (WORK_W, int(round(H * s)))), cv2.COLOR_BGR2GRAY)
        mask = np.full(g.shape, 255, np.uint8)
        for b in boxes_by_frame[k] if k < len(boxes_by_frame) else []:
            x0, y0, x1, y1 = (np.asarray(b) * s).astype(int)
            pad = int(0.1 * (x1 - x0))
            mask[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad] = 0
        if prev is not None and prev_pts is not None and len(prev_pts) >= 8:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, prev_pts, None, winSize=(21, 21), maxLevel=3)
            ok = st.ravel() == 1
            if ok.sum() >= 8:
                M, inl = cv2.estimateAffinePartial2D(nxt[ok], prev_pts[ok], method=cv2.RANSAC, ransacReprojThreshold=2.0)
                if M is not None and inl is not None and inl.sum() >= 8:
                    acc = acc @ np.vstack([M, [0, 0, 1]])         # current -> previous -> ... -> first
        out.append(acc[:2].copy())
        prev = g
        prev_pts = cv2.goodFeaturesToTrack(g, maxCorners=400, qualityLevel=0.01, minDistance=8, mask=mask)
    A = np.asarray(out, np.float64)
    # back to full-resolution pixels: x_full = x_work / s on both sides -> translation scales, linear part does not
    A[:, :, 2] /= s
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, A=A)
    return A


def apply(A: np.ndarray, frame_idx: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Map points (D, J, 2) seen in frames ``frame_idx`` (D,) into first-frame pixels."""
    M = A[frame_idx]                                   # (D, 2, 3)
    return np.einsum("dij,dkj->dki", M[:, :, :2], xy) + M[:, None, :, 2]


def drift(A: np.ndarray) -> dict:
    """Summary for the report: largest shift (px), rotation (deg) and scale change over the video."""
    shift = np.linalg.norm(A[:, :, 2], axis=1)
    rot = np.degrees(np.arctan2(A[:, 1, 0], A[:, 0, 0]))
    sc = np.hypot(A[:, 0, 0], A[:, 1, 0])
    return {"max_shift_px": round(float(shift.max()), 1), "max_rotation_deg": round(float(np.abs(rot).max()), 2),
            "scale_range": [round(float(sc.min()), 3), round(float(sc.max()), 3)]}
