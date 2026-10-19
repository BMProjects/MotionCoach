"""Interpretable kinematic features computed from a cleaned PoseSequence.

All geometry runs in pixel space (isotropic). Ratio features are divided by a per-sequence
body scale (median upright torso length), so they are independent of camera distance and of
the performer's size, without per-frame normalisation erasing stance width or squat depth.

Sign conventions are explicit per feature; ``lateral`` features use the performer's own
left/right axis (from right hip to left hip), so they stay meaningful after un-mirroring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .sequence import PoseSequence


@dataclass(frozen=True)
class FeatureDef:
    name: str
    label: str            # short Chinese label for UI
    unit: str             # "deg" or "ratio"
    joints: tuple[str, ...]
    more: str             # wording when the user's value is larger than the reference
    less: str             # wording when the user's value is smaller
    lr_twin: str | None = None   # feature that swaps with this one under left/right mirroring
    lateral: bool = False        # sign flips under mirroring (performer-left positive)
    desc: str = ""               # plain-language definition for non-specialist readers


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Interior angle ABC in degrees (180 = straight)."""
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)
    cos = np.einsum("...i,...i->...", v1, v2) / np.where(n > 0, n, np.nan)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


FEATURES: dict[str, FeatureDef] = {}


def _reg(f: FeatureDef) -> None:
    FEATURES[f.name] = f


for side, cn in (("left", "左"), ("right", "右")):
    s = side[0]
    o = "r" if s == "l" else "l"
    _reg(FeatureDef(f"{s}_elbow", f"{cn}肘角", "deg", (f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"),
                    "更伸直", "更弯曲", lr_twin=f"{o}_elbow",
                    desc=f"{cn}肩—{cn}肘—{cn}腕三点在画面中的夹角；180° 为手臂伸直"))
    _reg(FeatureDef(f"{s}_shoulder", f"{cn}臂上举角", "deg", (f"{side}_hip", f"{side}_shoulder", f"{side}_elbow"),
                    "举得更高", "举得更低", lr_twin=f"{o}_shoulder",
                    desc=f"{cn}髋—{cn}肩—{cn}肘夹角；约 0° 为上臂贴身下垂，180° 为上举过头"))
    _reg(FeatureDef(f"{s}_hip", f"{cn}髋角", "deg", (f"{side}_shoulder", f"{side}_hip", f"{side}_knee"),
                    "更打开", "更屈曲", lr_twin=f"{o}_hip",
                    desc=f"{cn}肩—{cn}髋—{cn}膝夹角；180° 为躯干与大腿成一直线（正面视角对前屈不敏感）"))
    _reg(FeatureDef(f"{s}_knee", f"{cn}膝角", "deg", (f"{side}_hip", f"{side}_knee", f"{side}_ankle"),
                    "更伸直", "更弯曲", lr_twin=f"{o}_knee",
                    desc=f"{cn}髋—{cn}膝—{cn}踝夹角；180° 为腿伸直，马步约 100°–130°"))
    _reg(FeatureDef(f"{s}_wrist_height", f"{cn}腕高度", "ratio", (f"{side}_wrist", "left_shoulder", "right_shoulder"),
                    "更高", "更低", lr_twin=f"{o}_wrist_height",
                    desc=f"{cn}腕相对两肩中点的高度，单位为躯干长；正值表示高于肩"))

_reg(FeatureDef("trunk_lean", "躯干侧倾", "deg", ("left_shoulder", "right_shoulder", "left_hip", "right_hip"),
                "更向左倾", "更向右倾", lateral=True,
                desc="髋中点→肩中点连线相对竖直方向的侧倾角；向练习者自身左侧为正"))
_reg(FeatureDef("head_turn", "头部左右转", "ratio", ("nose", "left_shoulder", "right_shoulder"),
                "更向左", "更向右", lateral=True,
                desc="鼻尖相对两肩中点沿肩线方向的偏移，单位为肩宽；向练习者自身左侧为正"))
_reg(FeatureDef("stance_width", "步宽", "ratio", ("left_ankle", "right_ankle"), "更宽", "更窄",
                desc="两踝之间的距离，单位为躯干长"))
_reg(FeatureDef("pelvis_height", "髋部高度", "ratio", ("left_hip", "right_hip", "left_ankle", "right_ankle"),
                "更高（蹲得浅）", "更低（蹲得深）",
                desc="髋中点到两踝中点的竖直距离，单位为躯干长；越小表示蹲得越深"))


def body_scale(seq: PoseSequence) -> float:
    """Median torso length (shoulder-mid to hip-mid, pixels) over near-upright frames."""
    sm = _mid(seq, "left_shoulder", "right_shoulder")
    hm = _mid(seq, "left_hip", "right_hip")
    v = sm - hm
    length = np.linalg.norm(v, axis=1)
    upright = np.abs(np.degrees(np.arctan2(v[:, 0], -v[:, 1]))) < 20
    ok = np.isfinite(length) & upright
    if ok.sum() < 5:
        ok = np.isfinite(length)
    if ok.sum() == 0:
        raise ValueError("Cannot estimate body scale: torso never visible")
    return float(np.median(length[ok]))


def _mid(seq: PoseSequence, a: str, b: str) -> np.ndarray:
    return 0.5 * (seq.joint(a)[0].astype(np.float64) + seq.joint(b)[0].astype(np.float64))


def _p(seq: PoseSequence, name: str) -> np.ndarray:
    return seq.joint(name)[0].astype(np.float64)


def compute_features(seq: PoseSequence, names: list[str] | None = None) -> tuple[np.ndarray, list[str], float]:
    """Return (F (T,K) float64 with NaN for unmeasurable samples, feature names, body scale)."""
    names = list(names or FEATURES)
    scale = body_scale(seq)
    cols = []
    lateral_axis = _p(seq, "left_hip") - _p(seq, "right_hip")  # performer's right -> left
    lat_unit = lateral_axis / np.linalg.norm(lateral_axis, axis=1, keepdims=True)
    sh_axis = _p(seq, "left_shoulder") - _p(seq, "right_shoulder")
    sh_w = np.linalg.norm(sh_axis, axis=1)
    sm = _mid(seq, "left_shoulder", "right_shoulder")
    hm = _mid(seq, "left_hip", "right_hip")
    am = _mid(seq, "left_ankle", "right_ankle")

    calc: dict[str, Callable[[], np.ndarray]] = {}
    for side in ("left", "right"):
        s = side[0]
        calc[f"{s}_elbow"] = lambda side=side: _angle(_p(seq, f"{side}_shoulder"), _p(seq, f"{side}_elbow"), _p(seq, f"{side}_wrist"))
        calc[f"{s}_shoulder"] = lambda side=side: _angle(_p(seq, f"{side}_hip"), _p(seq, f"{side}_shoulder"), _p(seq, f"{side}_elbow"))
        calc[f"{s}_hip"] = lambda side=side: _angle(_p(seq, f"{side}_shoulder"), _p(seq, f"{side}_hip"), _p(seq, f"{side}_knee"))
        calc[f"{s}_knee"] = lambda side=side: _angle(_p(seq, f"{side}_hip"), _p(seq, f"{side}_knee"), _p(seq, f"{side}_ankle"))
        calc[f"{s}_wrist_height"] = lambda side=side: (sm[:, 1] - _p(seq, f"{side}_wrist")[:, 1]) / scale

    def trunk_lean() -> np.ndarray:
        v = sm - hm
        ang = np.degrees(np.arccos(np.clip(-v[:, 1] / np.linalg.norm(v, axis=1), -1, 1)))
        return ang * np.sign(np.einsum("ij,ij->i", v, lat_unit))

    calc["trunk_lean"] = trunk_lean
    calc["head_turn"] = lambda: np.einsum("ij,ij->i", _p(seq, "nose") - sm, sh_axis / sh_w[:, None]) / sh_w
    calc["stance_width"] = lambda: np.linalg.norm(_p(seq, "left_ankle") - _p(seq, "right_ankle"), axis=1) / scale
    calc["pelvis_height"] = lambda: (am[:, 1] - hm[:, 1]) / scale

    with np.errstate(invalid="ignore", divide="ignore"):
        for n in names:
            if n not in calc:
                raise KeyError(f"Unknown feature '{n}'")
            missing = [j for j in FEATURES[n].joints if not seq.skeleton.has(j)]
            cols.append(np.full(len(seq), np.nan) if missing else calc[n]())
    F = np.stack(cols, axis=1)
    F[~np.isfinite(F)] = np.nan
    return F, names, scale


def lr_swap_features(F: np.ndarray, names: list[str]) -> np.ndarray:
    """Express features as if the performer's left and right were exchanged (mirror-side practice)."""
    out = F.copy()
    pos = {n: i for i, n in enumerate(names)}
    for i, n in enumerate(names):
        d = FEATURES[n]
        if d.lr_twin and d.lr_twin in pos:
            out[:, i] = F[:, pos[d.lr_twin]]
        if d.lateral:
            out[:, i] = -out[:, i]
    return out
