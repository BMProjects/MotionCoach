"""Body-normalised skeletons for visual comparison ("skeleton difference vector field").

Both skeletons are expressed on one standard body: origin at the hip midpoint, unit = the
reference's median torso length. The user's bones keep their own 2D direction and
foreshortening; each bone is only rescaled by (reference median length / user median length).
Remaining joint offsets therefore reflect posture, not body size or camera distance.
"""

from __future__ import annotations

import numpy as np

from .sequence import PoseSequence

CANON = ("nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
         "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle")
# (parent, child); "hip_mid"/"shoulder_mid" are virtual joints
BONES = (
    ("hip_mid", "left_hip"), ("hip_mid", "right_hip"), ("hip_mid", "shoulder_mid"),
    ("shoulder_mid", "left_shoulder"), ("shoulder_mid", "right_shoulder"), ("shoulder_mid", "nose"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
)
EDGES = (("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"), ("left_shoulder", "left_hip"),
         ("right_shoulder", "right_hip"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
         ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"), ("left_hip", "left_knee"),
         ("left_knee", "left_ankle"), ("right_hip", "right_knee"), ("right_knee", "right_ankle"))


def _points(seq: PoseSequence) -> dict[str, np.ndarray]:
    P = {j: seq.joint(j)[0].astype(np.float64) for j in CANON}
    P["hip_mid"] = 0.5 * (P["left_hip"] + P["right_hip"])
    P["shoulder_mid"] = 0.5 * (P["left_shoulder"] + P["right_shoulder"])
    return P


def bone_lengths(seq: PoseSequence) -> dict[tuple[str, str], float]:
    P = _points(seq)
    out = {}
    for a, b in BONES:
        L = np.linalg.norm(P[b] - P[a], axis=1)
        out[(a, b)] = float(np.nanmedian(L)) if np.isfinite(L).any() else np.nan
    return out


def normalized(seq: PoseSequence, target: dict | None = None, own: dict | None = None,
               unit: float | None = None) -> np.ndarray:
    """(T, len(CANON), 2) in torso units, hip-midpoint origin, image orientation (y down).

    ``target``/``own``: bone length tables (pixels) of the standard body and of ``seq``; when
    given, every bone vector is scaled by target/own before chaining from the root.
    ``unit``: pixel length of one torso unit of the standard body.
    """
    P = _points(seq)
    own = own or bone_lengths(seq)
    target = target or own
    unit = unit or target[("hip_mid", "shoulder_mid")]
    T = len(seq)
    Q = {"hip_mid": np.zeros((T, 2))}
    for a, b in BONES:   # BONES is ordered parent-before-child
        s = target[(a, b)] / own[(a, b)] if own[(a, b)] and np.isfinite(own[(a, b)]) else np.nan
        Q[b] = Q[a] + (P[b] - P[a]) * s
    return np.stack([Q[j] for j in CANON], axis=1) / unit


def ellipse(d: np.ndarray) -> dict | None:
    """1-sigma ellipse of 2D difference vectors (n,2): centre, semi-axes, angle (deg)."""
    d = d[np.isfinite(d).all(axis=1)]
    if len(d) < 3:
        return None
    c = d.mean(axis=0)
    w, v = np.linalg.eigh(np.cov(d.T))
    w = np.sqrt(np.maximum(w, 0))
    return {"cx": float(c[0]), "cy": float(c[1]), "a": float(w[1]), "b": float(w[0]),
            "angle": float(np.degrees(np.arctan2(v[1, 1], v[0, 1])))}
