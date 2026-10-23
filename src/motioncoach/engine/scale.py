"""Pixel -> centimetre scale for presenting length indicators (comparison itself stays in body units).

Stature in pixels is measured on upright, straight-legged frames from the eyes to the ankles and
completed with body-segment proportions (Drillis & Contini 1966, as reproduced in Winter,
*Biomechanics and Motor Control of Human Movement*, Fig. 4.1): eye height 0.936 H, shoulder 0.818 H,
hip joint 0.530 H, ankle 0.039 H. With no stated height the adult average is used.

A second, independent proportion (torso = shoulder - hip = 0.288 H) checks the geometry: if the measured
torso disagrees with the stature by more than ``MAX_DEVIATION`` (camera tilt, not upright, cropped
body) the scale is marked unreliable and no centimetres are shown.
"""

from __future__ import annotations

import numpy as np

from .sequence import PoseSequence

EYE_H, SHOULDER_H, HIP_H, ANKLE_H = 0.936, 0.818, 0.530, 0.039
# Mean of the 18-44 y adult averages, men 169.7 cm and women 158.0 cm
# (《中国居民营养与慢性病状况报告（2020年）》, as announced 2020-12-23).
DEFAULT_HEIGHT_CM = 163.9   # (169.7 + 158.0) / 2, rounded
MAX_DEVIATION = 0.15
REL_UNCERTAINTY = {"average_height": 0.06, "stated_height": 0.04}   # population height spread + proportion spread
LENGTH_FEATURES = ("l_wrist_height", "r_wrist_height", "stance_width", "pelvis_height")   # in torso lengths


def _mid(seq: PoseSequence, a: str, b: str) -> np.ndarray:
    return 0.5 * (seq.joint(a)[0].astype(float) + seq.joint(b)[0].astype(float))


def _angle(a, b, c) -> np.ndarray:
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", v1, v2) / n, -1, 1)))


def estimate_scale(seq: PoseSequence, torso_px: float, height_cm: float | None = None, min_frames: int = 10) -> dict:
    """``torso_px``: the body scale the features were divided by (``features.body_scale``), so that
    ``value_in_torso_lengths * torso_cm`` is the same length in centimetres."""
    method = "stated_height" if height_cm else "average_height"
    H = float(height_cm or DEFAULT_HEIGHT_CM)
    out = {"method": method, "height_cm": H, "reliable": False, "cm_per_px": None, "torso_cm": None,
           "frames_used": 0, "rel_uncertainty": REL_UNCERTAINTY[method],
           "source": "成年人平均身高（男 169.7 cm、女 158.0 cm 的平均）" if method == "average_height" else "用户填写的身高"}
    need = ("left_eye", "right_eye", "left_ankle", "right_ankle", "left_shoulder", "right_shoulder",
            "left_hip", "right_hip", "left_knee", "right_knee")
    if not all(seq.skeleton.has(j) for j in need):
        out["reason"] = "骨架缺少眼或踝关键点"
        return out
    eye, ank = _mid(seq, "left_eye", "right_eye"), _mid(seq, "left_ankle", "right_ankle")
    sh, hip = _mid(seq, "left_shoulder", "right_shoulder"), _mid(seq, "left_hip", "right_hip")
    v = sh - hip
    lean = np.abs(np.degrees(np.arctan2(v[:, 0], -v[:, 1])))
    knees = [_angle(seq.joint(f"{s}_hip")[0].astype(float), seq.joint(f"{s}_knee")[0].astype(float),
                    seq.joint(f"{s}_ankle")[0].astype(float)) for s in ("left", "right")]
    ok = (lean < 10) & (knees[0] > 165) & (knees[1] > 165) & np.isfinite(eye).all(1) & np.isfinite(ank).all(1)
    out["frames_used"] = int(ok.sum())
    if ok.sum() < min_frames:
        out["reason"] = "直立、双腿伸直且眼与双踝可见的画面不足 1 秒"
        return out
    H_px = float(np.median((ank[ok, 1] - eye[ok, 1]) / (EYE_H - ANKLE_H)))
    torso_meas = float(np.median(np.linalg.norm(v[ok], axis=1)))
    deviation = torso_meas / ((SHOULDER_H - HIP_H) * H_px) - 1.0
    out.update(stature_px=H_px, cm_per_px=H / H_px, torso_cm=float(torso_px) * H / H_px,
               check={"torso_measured_px": torso_meas, "torso_expected_px": (SHOULDER_H - HIP_H) * H_px,
                      "deviation": deviation})
    if abs(deviation) > MAX_DEVIATION:
        out["reason"] = f"躯干与身高比例不符（偏差 {deviation * 100:.0f}%），可能镜头倾斜或未全身入画"
        return out
    out["reliable"] = True
    return out


def to_cm(feature: str, value: float | None, scale: dict | None) -> float | None:
    """Length indicator (torso lengths) -> centimetres, or None when not convertible."""
    if value is None or not scale or not scale.get("reliable") or feature not in LENGTH_FEATURES:
        return None
    return float(value) * scale["torso_cm"]
