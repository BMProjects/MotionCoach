"""Choosing the analysis interval: the continuous action both videos share.

Full-resolution analysis runs only on the selected interval. To find it, a coarse pass
(2 fps, fast model) aligns the whole routine and scores every reference segment:

    eligible = both videos detect the person, feet are visible, the reference has no camera
               cut/zoom inside the segment, the alignment is confident and its duration is
               consistent with the overall tempo.

``auto`` picks the longest contiguous run of eligible segments; ``segments`` uses the
segments the user chose; ``manual`` uses user-given time ranges and skips the coarse pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..engine.align import two_stage_align
from ..engine.compare import Segment
from ..engine.features import compute_features
from ..engine.sequence import PoseSequence, clean
from .assets import Profile


@dataclass
class ScopeRequest:
    mode: str = "auto"                       # auto | segments | manual
    seg_from: str | None = None
    seg_to: str | None = None
    user_start: float | None = None          # manual only (seconds in the user video)
    user_end: float | None = None


@dataclass
class ScopeResult:
    segments: list[Segment]
    ref_range: tuple[float, float]
    user_range: tuple[float, float]
    reason: str
    candidates: list[dict] = field(default_factory=list)
    tempo_ratio: float | None = None


def _upright_torso(seq: PoseSequence) -> np.ndarray:
    """Torso length (px) on upright frames only; NaN elsewhere (bending shortens the 2D torso)."""
    def mid(a, b):
        return 0.5 * (seq.joint(a)[0] + seq.joint(b)[0])
    v = mid("left_shoulder", "right_shoulder") - mid("left_hip", "right_hip")
    L = np.linalg.norm(v, axis=1)
    lean = np.degrees(np.arctan2(np.abs(v[:, 0]), -v[:, 1]))
    return np.where(lean < 15, L, np.nan)


def _visible(seq: PoseSequence, m: np.ndarray, joints: tuple[str, ...]) -> float | None:
    sk = seq.skeleton
    idx = [sk.index[j] for j in joints if sk.has(j)]
    if not idx or not m.any():
        return None
    return float((seq.conf[m][:, idx] >= seq.meta.get("conf_thr", 0.5)).mean())


def evaluate_segments(ref_c: PoseSequence, user_c: PoseSequence, segments: list[Segment], profile: Profile,
                      swap_mode: str, ref_cuts: list[float] | None = None) -> tuple[list[dict], float]:
    """Coarse alignment of the whole routine + per-segment eligibility."""
    names = profile.features
    scales = np.array([profile.tolerance(n)["good"] for n in names])
    rc, _ = clean(ref_c, conf_thr=ref_c.meta.get("conf_thr", 0.5), smooth_window_s=1.0)
    uc, _ = clean(user_c, conf_thr=user_c.meta.get("conf_thr", 0.5), smooth_window_s=1.0)
    FR, _, _ = compute_features(rc, names)
    FU, _, _ = compute_features(uc, names)
    acfg = profile.raw.get("alignment", {})
    res = two_stage_align(FR, FU, rc.t, names, scales, [(s.start, s.end) for s in segments], swap_mode=swap_mode,
                          coarse_window_s=float(acfg.get("coarse_window_s", 1.0)) * 2,
                          skip_cost=float(acfg.get("skip_cost", 2.0)), ref_skip_cost=acfg.get("ref_skip_cost"))
    q = profile.quality
    torso = _upright_torso(ref_c)
    torso_med = float(np.nanmedian(torso))
    ref_cuts = ref_cuts or []
    # tempo over the recorded part only (a partial recording leaves reference sections unmatched)
    u_span = uc.t[res.alignment.user_idx[-1]] - uc.t[res.alignment.user_idx[0]]
    tempo = u_span / max(1e-6, rc.t[res.alignment.ref_idx[-1]] - rc.t[res.alignment.ref_idx[0]])
    feet = ("left_heel", "right_heel", "left_foot_index", "right_foot_index")
    out = []
    for seg, ev in zip(segments, res.segment_lr):
        mr = (rc.t >= seg.start) & (rc.t < seg.end)
        last = len(uc.t) - 1                         # a section missing at the end has its start at the very end
        u0, u1 = float(uc.t[min(ev["user"][0], last)]), float(uc.t[max(0, min(ev["user"][1], len(uc.t)) - 1)])
        mu = (uc.t >= u0) & (uc.t <= u1)
        tr = torso[mr]
        tr = tr[np.isfinite(tr)]
        zoom = float(np.median(tr) / torso_med) if len(tr) >= 3 else np.nan
        zoom_range = float(np.percentile(tr, 90) / np.percentile(tr, 10)) if len(tr) >= 5 else 1.0
        jumps = sum(seg.start + 0.5 < c < seg.end - 0.5 for c in ref_cuts)
        det_r = float(ref_c.detected[mr].mean()) if mr.any() else 0.0
        det_u = float(user_c.detected[mu].mean()) if mu.any() else 0.0
        feet_r, feet_u = _visible(ref_c, mr, feet), _visible(user_c, mu, feet)
        cost = min(ev["cost"], ev["cost_swap"])
        reasons_missing = ["练习者视频在此段之前已结束"] if ev.get("missing") else []
        ratio = ((u1 - u0) / (seg.end - seg.start)) / tempo if tempo > 0 else np.nan
        reasons = []
        if det_r < q.get("min_detection_rate", 0.9) or det_u < q.get("min_detection_rate", 0.9):
            reasons.append(f"人物检测不足（标准 {det_r:.0%} / 用户 {det_u:.0%}）")
        if (feet_r is not None and feet_r < q.get("min_feet_visible", 0.6)) or (feet_u is not None and feet_u < q.get("min_feet_visible", 0.6)):
            reasons.append(f"脚部可见不足（标准 {feet_r:.0%} / 用户 {feet_u:.0%}）" if feet_r is not None and feet_u is not None else "脚部可见不足")
        if jumps > 0 or zoom_range > 1.3 or (np.isfinite(zoom) and abs(np.log(zoom)) > np.log(1.25)):
            reasons.append(f"标准视频有剪辑或推拉镜头（人物尺度 {zoom:.2f}×，段内变化 {zoom_range:.2f}×，切换 {jumps} 次）")
        reasons += reasons_missing
        if not reasons_missing and cost > q.get("max_coarse_align_cost", 1.6):
            reasons.append(f"动作对应可信度低（代价 {cost:.2f}）")
        if not (0.4 <= ratio <= 2.5):
            reasons.append(f"用时与整体节奏不符（{ratio:.2f}×）")
        out.append({"segment": seg.id, "name": seg.name, "ref_range": [seg.start, seg.end], "user_range": [u0, u1],
                    "eligible": not reasons, "reasons": reasons, "align_cost": cost if np.isfinite(cost) else None, "zoom": zoom, "zoom_range": zoom_range,
                    "detection": [det_r, det_u], "feet_visible": [feet_r, feet_u], "duration_vs_tempo": ratio})
    return out, float(tempo)


def choose(segments: list[Segment], candidates: list[dict], req: ScopeRequest) -> tuple[list[int], str]:
    ids = [s.id for s in segments]
    if req.mode == "segments":
        i0, i1 = ids.index(req.seg_from), ids.index(req.seg_to or req.seg_from)
        if i1 < i0:
            i0, i1 = i1, i0
        return list(range(i0, i1 + 1)), "按选择的动作段"
    runs, cur = [], []
    for i, c in enumerate(candidates):
        if c["eligible"]:
            cur.append(i)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    if runs:
        best = max(runs, key=lambda r: sum(segments[i].end - segments[i].start for i in r))
        return best, "自动选择：两段视频共同、连续且质量合格的最长区间"
    best = int(np.argmin([np.inf if c["align_cost"] is None else c["align_cost"] for c in candidates]))   # None: not recorded
    return [best], "NEEDS_REVIEW：没有合格的动作段；仅为查看而选择了对应可信度最高的一段，结论需人工核查"
