"""Keypoint filtering candidates (scheme D): confidence-weighted One Euro, bone-length outlier rejection.

The production path uses mask -> short-gap fill -> Savitzky-Golay (``engine.sequence.clean``).
Here the same masking and gap fill are followed by a zero-phase (forward + backward) One Euro
filter whose cutoff drops with keypoint confidence, after rejecting joints whose bone becomes
implausibly *longer* than its longest well-supported projection (2D projection can shorten a limb, never
lengthen it beyond its true length).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .normalize import _points
from .sequence import PoseSequence, clean

LIMBS = (("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"),
         ("right_elbow", "right_wrist"), ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
         ("right_hip", "right_knee"), ("right_knee", "right_ankle"))


def _alpha(cutoff: np.ndarray | float, dt: float) -> np.ndarray:
    tau = 1.0 / (2 * np.pi * np.asarray(cutoff, dtype=np.float64))
    return 1.0 / (1.0 + tau / dt)


def _one_euro_1d(x: np.ndarray, w: np.ndarray, dt: float, min_cutoff: float, beta: float, d_cutoff: float,
                 conf_weight: float = 0.7) -> np.ndarray:
    """Causal One Euro on one coordinate; NaN runs restart the filter. ``w`` in [0,1] scales min_cutoff
    (by up to ``conf_weight``: low confidence -> stronger smoothing)."""
    out = np.full_like(x, np.nan)
    prev = dprev = np.nan
    a_d = float(_alpha(d_cutoff, dt))
    for i, v in enumerate(x):
        if not np.isfinite(v):
            prev = dprev = np.nan
            continue
        if not np.isfinite(prev):
            out[i] = prev = v
            dprev = 0.0
            continue
        dx = (v - prev) / dt
        dprev = a_d * dx + (1 - a_d) * dprev
        cutoff = min_cutoff * (1 - conf_weight + conf_weight * w[i]) + beta * abs(dprev)
        a = float(_alpha(cutoff, dt))
        out[i] = prev = a * v + (1 - a) * prev
    return out


TWIN = {("left_shoulder", "left_elbow"): ("right_shoulder", "right_elbow"), ("left_elbow", "left_wrist"): ("right_elbow", "right_wrist"),
        ("left_hip", "left_knee"): ("right_hip", "right_knee"), ("left_knee", "left_ankle"): ("right_knee", "right_ankle")}
TWIN.update({v: k for k, v in list(TWIN.items())})


def reject_long_bones(seq: PoseSequence, max_ratio: float = 1.35, ref_q: float = 95.0) -> tuple[PoseSequence, int]:
    """Mask the distal joint of a limb bone longer than ``max_ratio`` x the bone's *longest well-supported*
    projected length (``ref_q`` percentile over the sequence, the larger of the bone and its left/right twin).

    A 2D projection can shorten a limb but never lengthen it beyond its true length, and the longest
    projections (limb parallel to the image) approach that length. The previous reference, the *median*
    projected length, deleted correct joints whenever the performer turned or reached so that a limb that is
    usually foreshortened came into the image plane (docs/ITERATION_LOG.md I16)."""
    sk = seq.skeleton
    xy = seq.xy.astype(np.float64).copy()
    lengths = {}
    for a, b in LIMBS:
        L = np.linalg.norm(xy[:, sk.index[b]] - xy[:, sk.index[a]], axis=1)
        lengths[(a, b)] = L
    n = 0
    for (a, b), L in lengths.items():
        refs = [np.nanpercentile(x, ref_q) for x in (L, lengths.get(TWIN.get((a, b)), L)) if np.isfinite(x).sum() >= 10]
        if not refs:
            continue
        bad = np.isfinite(L) & (L > max_ratio * max(refs))
        xy[bad, sk.index[b]] = np.nan
        n += int(bad.sum())
    conf = seq.conf.copy()
    conf[~np.isfinite(xy).all(axis=2)] = 0.0
    return replace(seq, xy=xy.astype(np.float32), conf=conf), n


def one_euro_clean(seq: PoseSequence, conf_thr: float = 0.5, max_gap_s: float = 0.4,
                   min_cutoff: float = 0.6, beta: float = 0.004, d_cutoff: float = 1.0,
                   conf_weight: float = 0.7) -> PoseSequence:
    """Mask + gap fill (as production, no smoothing) + bone check + zero-phase One Euro.

    ``beta`` is per pixel/s of the sequence's own frame; the default suits ~300-1000 px frames at 10 fps.
    """
    seq, _ = reject_long_bones(seq)
    filled, interp = clean(seq, conf_thr=conf_thr, max_gap_s=max_gap_s, smooth_window_s=0.0)   # win 3 -> SG(3,2) = identity
    dt = float(np.median(np.diff(seq.t))) if len(seq) > 1 else 0.1
    scale = max(seq.frame_size) / 640.0        # beta in "640 px frame" units
    xy = filled.xy.astype(np.float64)
    w = np.clip((seq.conf - conf_thr) / max(1e-6, 1 - conf_thr), 0, 1)
    out = np.full_like(xy, np.nan)
    for j in range(xy.shape[1]):
        for c in range(2):
            x = xy[:, j, c]
            f = _one_euro_1d(x, w[:, j], dt, min_cutoff, beta / scale, d_cutoff, conf_weight)
            b = _one_euro_1d(x[::-1], w[::-1, j], dt, min_cutoff, beta / scale, d_cutoff, conf_weight)[::-1]
            out[:, j, c] = 0.5 * (f + b)
    meta = {**filled.meta, "preprocess": {"method": "mask+gap_fill+bone_check+one_euro_zero_phase",
                                          "min_cutoff": min_cutoff, "beta": beta, "conf_weight": conf_weight,
                                          "max_gap_s": max_gap_s},
            # provenance per (sample, joint): True = filled across a short gap, not observed (shown dashed)
            "filled": interp}
    return replace(filled, xy=out.astype(np.float32), meta=meta)


def jitter(seq: PoseSequence) -> float:
    """Median |second difference| of the 13 body joints, in torso lengths per sample^2."""
    P = _points(seq)
    torso = np.nanmedian(np.linalg.norm(P["shoulder_mid"] - P["hip_mid"], axis=1))
    acc = []
    for j in ("left_wrist", "right_wrist", "left_elbow", "right_elbow", "nose", "left_knee", "right_knee"):
        x = P[j]
        a = np.linalg.norm(x[2:] - 2 * x[1:-1] + x[:-2], axis=1)
        acc.append(a[np.isfinite(a)])
    return float(np.median(np.concatenate(acc)) / torso)

