"""End-to-end analysis job: reference + user video -> structured report + overlay data.

Stages (each reported through ``progress(stage, fraction)``):
    ingest -> locate (coarse 2 fps pass, skipped in manual mode) -> poses_ref -> poses_user
    -> features -> align -> compare -> write
Only the chosen interval is analysed at full rate. Video rendering is not part of the job;
the client draws overlays from the pose data.
"""

from __future__ import annotations

import gzip
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .. import __version__
from ..contracts import POSE_PAYLOAD_SCHEMA, REPORT_SCHEMA
from ..engine import quality as quality_mod
from ..engine.align import two_stage_align
from ..engine.compare import Segment
from ..engine.filters import one_euro_clean
from ..engine.templates import PERIOD_BAND, compare_by_templates
from ..engine.features import FEATURES, compute_features, lr_swap_features
from ..engine.normalize import CANON, EDGES, bone_lengths, ellipse, normalized
from ..engine.patterns import cluster_deviations
from ..engine.scale import estimate_scale
from ..engine.sequence import PoseSequence
from ..media.extract import extract_poses, prepared_video, slice_window, subsample
from ..media.ingest import probe, scene_cuts, view_path
from ..media.presence import WARN_SHARE, require_person
from ..perception.registry import BOUNDS_BACKEND, DEFAULT_BACKEND, LOCATE_BACKEND, get_backend
from .assets import Reference, load_profile
from .rules import STATUS_LABEL, evaluate, quality_warnings
from .scope import ScopeRequest, choose, evaluate_segments

Progress = Callable[[str, float], None]


@dataclass
class AnalysisOptions:
    sample_fps: float = 10.0
    backend: str = DEFAULT_BACKEND
    user_mirrored: str = "auto"      # auto | yes | no   (selfie / front-camera recording)
    mirror_practice: bool = False    # performer deliberately does the opposite side (follow-along)
    scope: ScopeRequest = field(default_factory=ScopeRequest)
    locate_backend: str = LOCATE_BACKEND
    locate_fps: float = 2.0
    # one full-length pass with the analysis model, locating on a subsample of it: each video is decoded once,
    # and any later change of interval reuses the cached poses (cheap on a GPU server)
    locate_from_fine: bool = True
    # model for the interval search, section bounds and left/right (None = ``backend``); the repetition
    # search, measurements and rules always use ``backend``
    bounds_backend: str | None = BOUNDS_BACKEND
    height_cm: float | None = None   # practitioner height for cm wording; None = adult average (engine/scale.py)
    # section id -> (start, end) seconds in the practitioner video, fixed from outside (a group's common
    # time axis, apps/group.py); the coarse alignment then only decides left/right
    section_bounds: dict | None = None


def _r(a, nd: int) -> list:
    a = np.round(np.asarray(a, dtype=np.float64), nd)
    return [None if not np.isfinite(x) else float(x) for x in a.ravel()]


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Not JSON serializable: {type(o).__name__}")


def _finite(o):
    """NaN / inf -> None, recursively: browsers' JSON.parse rejects the NaN that Python writes by default."""
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    if isinstance(o, np.ndarray):
        return _finite(o.tolist())
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def to_json(obj, **kw) -> str:
    """Strict JSON (no NaN / Infinity tokens) for every file a browser reads."""
    return json.dumps(_finite(obj), ensure_ascii=False, default=_json_default, allow_nan=False, **kw)


def _pose_payload(seq: PoseSequence, video_url: str) -> dict:
    """Overlay data restricted to body joints (whole-body models have 133 points)."""
    sk = seq.skeleton
    keep = sorted({i for e in sk.edge_indices() for i in e})
    pos = {j: k for k, j in enumerate(keep)}
    xy = np.nan_to_num(seq.xy[:, keep], nan=-1).round(1)
    return {
        "schema": POSE_PAYLOAD_SCHEMA, "skeleton": sk.name,
        "model": {"id": seq.meta.get("model"), "device": seq.meta.get("device"), "runtime": "server"},
        "sampling": {"fps": seq.meta.get("sample_fps"), "grid": seq.meta.get("sampling")},
        "video_url": video_url,
        "frame_size": list(seq.frame_size),
        "t": _r(seq.t, 3),
        "xy": xy.reshape(len(seq), -1).tolist(),
        "conf": (seq.conf[:, keep] * 100).round().astype(int).tolist(),
        "joints": [sk.joints[i] for i in keep],
        "edges": [(pos[a], pos[b]) for a, b in sk.edge_indices()],
        "conf_thr": int(100 * seq.meta.get("conf_thr", 0.5)),
        # samples whose joint was filled across a short gap (not observed): [[sample, joint index], ...]
        "filled": ([[int(a), int(b)] for a, b in zip(*np.nonzero(seq.meta["filled"][:, keep]))]   # b: index into joints
                   if isinstance(seq.meta.get("filled"), np.ndarray) else []),
    }


def _clean(seq: PoseSequence) -> PoseSequence:
    """Mask + short-gap fill + bone-length check + zero-phase One Euro (see docs/ALIGNMENT_STUDY.md §1.5)."""
    return one_euro_clean(seq, conf_thr=seq.meta.get("conf_thr", 0.5))


def resolve_swap_mode(user_mirrored: str, mirror_practice: bool) -> str:
    """Recording mirror (selfie) and performed side (mirror practice) are independent transforms;
    each swaps left/right once, so both together cancel. "auto" can only detect the combined effect."""
    if user_mirrored == "auto":
        return "auto"
    return "yes" if ((user_mirrored == "yes") != mirror_practice) else "no"


def run_analysis(reference: Reference, user_video: Path, out_dir: Path, data_dir: Path, models_dir: Path,
                 options: AnalysisOptions = AnalysisOptions(), progress: Progress | None = None,
                 user_poses: PoseSequence | None = None) -> dict:
    """``user_poses``: the practitioner's raw full-length sequence when it is already known (one identity of a
    multi-person video, ``apps.group``); the video then only provides playback, and the interval search,
    section bounds and comparison all use this sequence."""
    progress = progress or (lambda s, f: None)
    t_start = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    profile = load_profile(reference.raw["profile"])
    names = profile.features
    all_segments = reference.segments(profile)
    swap_mode = resolve_swap_mode(options.user_mirrored, options.mirror_practice)
    timings: dict[str, float] = {}

    # ---- is anyone in the practitioner video? (seconds, before any heavy work) ---------------------
    progress("check", 0.0)
    s = time.time()
    if user_poses is None:
        presence = require_person(Path(user_video), get_backend(LOCATE_BACKEND, models_dir))   # raises NoPersonError
    else:
        seen = float(user_poses.detected.mean())
        presence = {"checked": len(user_poses), "with_person": int(round(seen * len(user_poses))), "share": seen,
                    "source": "identity"}
    timings["check"] = time.time() - s
    progress("check", 1.0)

    # ---- ingest ---------------------------------------------------------------------------
    progress("ingest", 0.0)
    s = time.time()
    ref_info, user_info = probe(reference.video), probe(Path(user_video))
    ref_id, ref_proxy = prepared_video(reference.video, data_dir)
    progress("ingest", 0.5)
    user_id, user_proxy = prepared_video(Path(user_video), data_dir)
    timings["ingest"] = time.time() - s
    progress("ingest", 1.0)

    # ---- scope: which continuous interval to analyse ------------------------------------------
    req = options.scope
    locate_fps = options.locate_fps
    s = time.time()
    candidates, tempo_hint = [], None
    if req.mode == "manual":
        if req.user_start is None or req.user_end is None or not req.seg_from:
            raise ValueError("manual scope needs seg_from/seg_to and user_start/user_end")
        idx, _ = choose(all_segments, [], ScopeRequest("segments", req.seg_from, req.seg_to))
        reason = "手动指定的用户视频区间"
        user_range = (float(req.user_start), float(req.user_end))
    else:
        locate_fps = float(profile.raw.get("alignment", {}).get("locate_fps", options.locate_fps))   # short clips need denser sampling
        if options.locate_from_fine:
            fine = get_backend(options.bounds_backend or options.backend, models_dir)
            step = max(1, int(round(options.sample_fps / locate_fps)))
            ref_c = subsample(extract_poses(ref_proxy, ref_id, fine, options.sample_fps, data_dir,
                                            progress=lambda f: progress("locate", 0.3 * f)), step)
            user_c = subsample(user_poses if user_poses is not None else
                               extract_poses(user_proxy, user_id, fine, options.sample_fps, data_dir,
                                             progress=lambda f: progress("locate", 0.3 + 0.6 * f)), step)
            locate_fps = options.sample_fps / step
        else:
            loc = get_backend(options.locate_backend, models_dir)
            ref_c = extract_poses(ref_proxy, ref_id, loc, locate_fps, data_dir, progress=lambda f: progress("locate", 0.3 * f))
            user_c = extract_poses(user_proxy, user_id, loc, locate_fps, data_dir, progress=lambda f: progress("locate", 0.3 + 0.6 * f))
        candidates, tempo_hint = evaluate_segments(ref_c, user_c, all_segments, profile, swap_mode, scene_cuts(ref_proxy))
        idx, reason = choose(all_segments, candidates, req)
        margin = 2.0 * max(1.0, tempo_hint)
        user_range = (max(0.0, candidates[idx[0]]["user_range"][0] - margin),
                      min(user_info["duration"], candidates[idx[-1]]["user_range"][1] + margin))
    segments: list[Segment] = [all_segments[i] for i in idx]
    ref_range = (segments[0].start, segments[-1].end)
    timings["locate"] = time.time() - s
    progress("locate", 1.0)

    # ---- fine pose extraction on the interval only -------------------------------------------
    backend = get_backend(options.backend, models_dir)
    s = time.time()
    ref_raw = extract_poses(ref_proxy, ref_id, backend, options.sample_fps, data_dir, *ref_range,
                            progress=lambda f: progress("poses_ref", f))
    timings["poses_ref"] = time.time() - s
    progress("poses_ref", 1.0)
    s = time.time()
    user_raw = (slice_window(user_poses, *user_range) if user_poses is not None else
                extract_poses(user_proxy, user_id, backend, options.sample_fps, data_dir, *user_range,
                              progress=lambda f: progress("poses_user", f)))
    timings["poses_user"] = time.time() - s
    progress("poses_user", 1.0)

    # ---- features --------------------------------------------------------------------------------
    progress("features", 0.0)
    ref_seq = ref_raw.unmirror() if reference.raw.get("mirrored_video") else ref_raw
    ref_clean, user_clean = _clean(ref_seq), _clean(user_raw)
    F_ref, _, ref_scale = compute_features(ref_clean, names)
    F_user, _, user_scale = compute_features(user_clean, names)
    scales = np.array([profile.tolerance(n)["good"] for n in names], dtype=np.float64)
    F_ref_b, F_user_b, bounds_model = F_ref, F_user, backend.name
    if options.bounds_backend and options.bounds_backend != options.backend and user_poses is None:
        bb = get_backend(options.bounds_backend, models_dir)      # full-length poses already cached by the locate pass
        rb = extract_poses(ref_proxy, ref_id, bb, options.sample_fps, data_dir, *ref_range)
        ub = extract_poses(user_proxy, user_id, bb, options.sample_fps, data_dir, *user_range)
        rb = rb.unmirror() if reference.raw.get("mirrored_video") else rb
        if np.allclose(rb.t, ref_clean.t) and np.allclose(ub.t, user_clean.t):   # same absolute sampling grid
            F_ref_b, F_user_b, bounds_model = (compute_features(_clean(rb), names)[0],
                                               compute_features(_clean(ub), names)[0], bb.name)
    progress("features", 1.0)

    # ---- alignment + left/right ----------------------------------------------------------------------
    progress("align", 0.0)
    acfg = profile.raw.get("alignment", {})
    lr_threshold = float(profile.quality.get("lr_swap_ratio", 0.95))
    res = two_stage_align(F_ref_b, F_user_b, ref_clean.t, names, scales, [(g.start, g.end) for g in segments],
                          swap_mode=swap_mode, coarse_window_s=float(acfg.get("coarse_window_s", 1.0)),
                          skip_cost=float(acfg.get("skip_cost", 2.0)), swap_threshold=lr_threshold,
                          ref_skip_cost=acfg.get("ref_skip_cost"))
    progress("align", 1.0)
    ratio, swap, al = res.swap_cost_ratio, bool(res.swap), res.alignment
    if options.user_mirrored == "yes" and options.mirror_practice:
        swap_reason = "mirrored_video_and_practice"
    elif options.mirror_practice:
        swap_reason = "mirror_practice"
    elif swap:
        swap_reason = "mirrored_video_declared" if options.user_mirrored == "yes" else "mirrored_video_auto"
    else:
        swap_reason = None
    F_u = lr_swap_features(F_user, names) if swap else F_user

    # ---- comparison + rules --------------------------------------------------------------------------
    progress("compare", 0.0)
    # interval ↔ interval: section bounds from the coarse alignment, then every practitioner repetition is found
    # by searching the reference repetition templates (annotated in the reference asset, else detected)
    seg_ok = [g for g in segments if ((ref_clean.t >= g.start) & (ref_clean.t < g.end)).sum() >= 2]
    user_bounds = [tuple(e["user"]) for e in res.segment_lr]
    if options.section_bounds:                     # common bounds: map seconds to this sequence's samples
        tt = user_clean.t
        user_bounds = [(int(np.searchsorted(tt, options.section_bounds[g.id][0])),
                        int(np.searchsorted(tt, options.section_bounds[g.id][1])))
                       if g.id in options.section_bounds else ub for g, ub in zip(seg_ok, user_bounds)]
    cmp = compare_by_templates(ref_clean.t, F_ref, user_clean.t, F_u, seg_ok, user_bounds, names, scales,
                               ref_reps=reference.rep_templates(),
                               period_band=tuple(acfg["period_band"]) if acfg.get("period_band") else
                               (None if "period_band" in acfg else PERIOD_BAND))
    scale = estimate_scale(user_clean, user_scale, options.height_cm)   # presentation only; comparison stays in body units
    # centimetres are worded only from the practitioner's own stated height; the adult-average estimate is kept in
    # the report for reference but too coarse to tell someone to move "N cm"
    scale["display"] = bool(scale.get("reliable") and scale.get("method") == "stated_height")
    ev = evaluate(cmp, profile, mirror_practice=options.mirror_practice, scale=scale if scale["display"] else None,
                  backend=backend.name)
    q_ref, q_user = quality_mod.measure(ref_raw), quality_mod.measure(user_raw)
    warnings = quality_warnings(q_user, profile, "用户视频") + quality_warnings(q_ref, profile, "标准视频")
    if presence["share"] < WARN_SHARE:
        warnings.insert(0, f"抽查画面中只有 {presence['with_person']}/{presence['checked']} 个能看到练习者完整躯干，"
                           "部分时段可能人不在画面内或被遮挡，结果只覆盖能看到人的部分")
    lr_text = {
        "mirrored_video_auto": f"左右交换后动作匹配明显更好（代价比 {ratio:.2f}），已自动按“前置摄像头镜像录制”还原左右；"
                               "若实际是对着示范镜像练习，请改选“镜像练习”。",
        "mirrored_video_declared": "已按您选择的“前置摄像头（画面镜像）”还原左右。",
        "mirror_practice": "已按“镜像练习”处理：练习者的右侧与示范左侧比较。",
        "mirrored_video_and_practice": "画面为镜像且练习者按镜像方向练习：两次左右交换相互抵消；文字中的侧别按练习者自身左右，并注明对应示范的哪一侧。",
    }.get(swap_reason)
    if lr_text is None and options.user_mirrored == "auto":
        lr_text = (f"无法从动作自动判断左右方向（交换前后匹配代价比 {ratio:.2f}；左右交替的套路对此不敏感）。"
                   "当前按“未镜像”处理；若用前置摄像头录制，请重新选择录制方式，否则左右关节的文字说明会对调。")
    span = cmp["user_span"][1] - cmp["user_span"][0]
    if span < 0.2 * (ref_clean.t[-1] - ref_clean.t[0]):
        warnings.insert(0, f"对齐失败：标准动作仅匹配到用户视频中 {span:.1f} 秒，两段视频可能不是同一套动作或拍摄条件差异过大")
    needs_review = reason.startswith("NEEDS_REVIEW")
    if needs_review:
        reason = reason.split("：", 1)[1]
        warnings.insert(0, "区间待核查：" + reason)
    if reference.raw.get("status") == "draft" or profile.raw.get("status") == "draft":
        warnings.append("参考分段与评价阈值为草案，尚未经教练确认；结果仅供算法测试")
    progress("compare", 0.6)

    # ---- normalised skeletons + deviation patterns -------------------------------------------------
    mid_user = cmp["series"]["user_idx_for_ref"]
    user_vis = user_clean.unmirror() if swap else user_clean   # same orientation/identity as the reference
    L_ref, L_user = bone_lengths(ref_clean), bone_lengths(user_vis)
    unit = L_ref[("hip_mid", "shoulder_mid")]
    N_ref = normalized(ref_clean, L_ref, L_ref, unit)
    N_user_all = normalized(user_vis, L_ref, L_user, unit)
    N_user = np.full_like(N_ref, np.nan)
    ok = mid_user >= 0
    N_user[ok] = N_user_all[mid_user[ok]]
    D = N_user - N_ref
    seg_fields = []
    for seg in cmp["segments"]:
        hold_idx = [int(np.argmin(np.abs(ref_clean.t - h))) for h in seg["hold_times"]]
        m = (ref_clean.t >= seg["ref_start"]) & (ref_clean.t < seg["ref_end"])
        pick = np.array(hold_idx) if hold_idx else np.flatnonzero(m)
        with np.errstate(all="ignore"):   # pose average, mean vector, spread: one sample set (the key moments)
            seg_fields.append({
                "segment": seg["id"],
                "samples": int(len(pick)),
                "sample_times": [float(ref_clean.t[i]) for i in pick],
                "ref_pose": _r(np.nanmean(N_ref[pick], axis=0), 3),
                "user_pose": _r(np.nanmean(N_user[pick], axis=0), 3),
                "mean_vec": _r(np.nanmean(D[pick], axis=0), 3),
                "ellipses": [ellipse(D[pick][:, j]) for j in range(len(CANON))],
                "magnitude": _r(np.nanmean(np.linalg.norm(D[pick], axis=2), axis=0), 3),
            })
    in_seg = np.zeros(len(ref_clean), bool)
    for g in segments:
        in_seg |= (ref_clean.t >= g.start) & (ref_clean.t < g.end)
    delta = cmp["series"]["delta"].copy()
    for g in segments:   # "not observable from this view" counts as missing, never as "no deviation"
        na = profile.segment_rule(g.id).get("not_applicable", [])
        m = (ref_clean.t >= g.start) & (ref_clean.t < g.end)
        for f in na:
            if f in names:
                delta[m, names.index(f)] = np.nan
    pat = cluster_deviations(delta, scales, in_seg)
    user_weight = cmp["series"]["weight"]   # practitioner samples represented by each reference sample
    n_ref_in, w_user_in = max(1, int(in_seg.sum())), max(1.0, float(user_weight[in_seg].sum()))
    patterns = []
    for k in range(pat["k"] + 1):   # class 0 = close to reference, 1..k = deviation patterns
        cz = pat["centers_z"][k]
        members = np.flatnonzero(pat["labels"] == k)
        if not len(members):
            continue
        rep = int(members[np.argmin(np.linalg.norm(np.nan_to_num(delta[members] / scales) - cz, axis=1))])
        mild = np.max(np.abs(cz)) < 1.0
        tops = [i for i in np.argsort(-np.abs(cz)) if abs(cz[i]) >= (0.5 if mild else 1.0)][:3]
        parts = []
        for i in tops:
            d = FEATURES[names[i]]
            v = cz[i] * scales[i]
            amount = f"{abs(v):.0f}°" if d.unit == "deg" else f"{abs(v):.2f}"
            parts.append(f"{d.label}{'偏大' if v > 0 else '偏小'} {amount}（{d.more if v > 0 else d.less}）")
        seg_share = {}
        for seg in segments:
            m = (ref_clean.t[members] >= seg.start) & (ref_clean.t[members] < seg.end)
            if m.any():
                seg_share[seg.id] = float(m.mean())
        patterns.append({
            "id": k,
            "share": float(len(members) / n_ref_in),                      # share of reference progress
            "user_share": float(user_weight[members].sum() / w_user_in),  # share of the practitioner's (aligned) time
            "near_reference": k == 0,
            "label": "已测指标都在允许范围内（缺失或此视角不可测的指标不计入）" if k == 0 else
                     (("轻度偏差，以" + "、".join(parts) + "为主（平均值未超出允许范围，但各时刻偏差方向和部位不一）") if mild and parts
                      else "；".join(parts) or "多项轻微偏差叠加"),
            "center": {n: float(cz[i] * scales[i]) for i, n in enumerate(names)},
            "center_z": {n: float(cz[i]) for i, n in enumerate(names)},
            "severity": float(np.mean(np.abs(cz))),
            "segments": seg_share,
            "example_t": float(ref_clean.t[rep]),
        })
    progress("compare", 1.0)

    # ---- outputs ----------------------------------------------------------------------------------------
    s_ = cmp["series"]
    ut = s_["user_t_for_ref"].copy()
    okt = np.isfinite(ut)
    ut[okt] = np.maximum.accumulate(ut[okt])   # monotone ref->user time map for playback sync
    report = {
        "schema": REPORT_SCHEMA,
        "engine_version": __version__,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": None,
        "timings_s": None,
        "profile": {"id": profile.id, "title": profile.raw["title"], "status": profile.raw.get("status")},
        "reference": {"id": reference.id, "title": reference.raw.get("title"), "status": reference.raw.get("status"),
                      "video": ref_info, "video_url": f"/media/{view_path(ref_proxy).name}"},
        "user": {"filename": Path(user_video).name, "video": user_info, "video_url": f"/media/{view_path(user_proxy).name}"},
        "options": {**asdict(options)},
        "model": {"id": backend.model_id, "backend": backend.name, "device": backend.device,
                  "device_note": getattr(backend, "device_note", ""),
                  "ms_per_frame": {"reference": ref_raw.meta.get("ms_per_frame"), "user": user_raw.meta.get("ms_per_frame")},
                  "bounds_backend": bounds_model},
        "scope": {"mode": req.mode, "reason": reason, "needs_review": needs_review, "segments": [g.id for g in segments],
                  "ref_range": list(ref_range), "user_range": list(user_range), "candidates": candidates,
                  "locate": None if req.mode == "manual" else {
                      "backend": (options.bounds_backend or options.backend) if options.locate_from_fine
                      else options.locate_backend, "fps": locate_fps}},
        "body_scale_px": {"reference": ref_scale, "user": user_scale},
        "scale": scale,
        "quality": {"reference": q_ref, "user": q_user, "warnings": warnings, "presence": presence},
        "left_right": {"swap_cost_ratio": ratio, "suspected": ratio < lr_threshold, "applied": swap, "reason": swap_reason,
                       "per_segment": [{"segment": g.id, "ratio": round(e["cost_swap"] / e["cost"], 3) if e["cost"] and np.isfinite(e["cost"]) else None}
                                       for g, e in zip(segments, res.segment_lr)],
                       "text": lr_text},
        "timing": {"tempo_ratio": cmp["tempo_ratio"], "user_span": cmp["user_span"], "align_cost": cmp["align_cost"],
                   "coarse_align_cost": res.coarse_cost,
                   "method": "coarse DTW for section bounds; practitioner repetitions found by slope-constrained "
                             "template search of the reference repetitions (features + velocity); loop/jump playback"},
        "playback": cmp["playback"],
        "features": [{"name": n, "label": FEATURES[n].label, "unit": FEATURES[n].unit, "joints": list(FEATURES[n].joints),
                      "more": FEATURES[n].more, "less": FEATURES[n].less, "desc": FEATURES[n].desc,
                      "tolerance": profile.tolerance(n)} for n in names],
        "status_labels": STATUS_LABEL,
        "summary": {k: ev[k] for k in ("pass_rate", "counts", "pass_text", "pass_rate_definition", "top_issues")},
        "segments": ev["segments"],
        "issues": ev["issues"],
        "normalized": {
            "joints": list(CANON), "edges": [(CANON.index(a), CANON.index(b)) for a, b in EDGES],
            "unit": "躯干长（髋中点到肩中点）", "origin": "髋中点", "orientation": "图像方向（y 向下）",
            "ref": [_r(p, 3) for p in N_ref], "user": [_r(p, 3) for p in N_user], "segments": seg_fields,
            # the practitioner's own timeline (practitioner-led playback shows the repetition actually playing)
            "user_own_t0": float(user_clean.t[0]) if len(user_clean) else 0.0,
            "user_own_dt": float(np.median(np.diff(user_clean.t))) if len(user_clean) > 1 else 0.1,
            "user_own": [_r(p, 2) for p in N_user_all],
        },
        "patterns": {"k": pat["k"], "silhouette": pat["silhouette"], "labels": pat["labels"].tolist(), "items": patterns,
                     "unclassified_share": float(((pat["labels"] == -1) & in_seg).sum() / n_ref_in),
                     "outlier_share": float(((pat["labels"] == -2) & in_seg).sum() / n_ref_in),
                     "share_definition": "share = 标准动作进程中的采样点占比；user_share = 按对应关系折算的练习者时长占比"},
        "series": {
            "t_ref": _r(ref_clean.t, 3),
            "user_t": _r(ut, 3),
            "align_cost": _r(s_["align_cost"], 3),
            "user_map": [[round(x, 3) for x in m] for m in cmp["user_map"]],   # every practitioner repetition -> its reference repetition
            "ref": {n: _r(F_ref[:, i], 3) for i, n in enumerate(names)},
            "user": {n: _r(s_["user_aligned"][:, i], 3) for i, n in enumerate(names)},
        },
    }
    timings["total"] = time.time() - t_start
    report["elapsed_s"] = round(timings["total"], 1)
    report["timings_s"] = {k: round(v, 1) for k, v in timings.items()}
    (out_dir / "report.json").write_text(to_json(report), encoding="utf-8")
    for side, seq, url in (("ref", _clean(ref_raw), report["reference"]["video_url"]), ("user", user_clean, report["user"]["video_url"])):
        payload = to_json(_pose_payload(seq, url)).encode()
        (out_dir / f"poses_{side}.json.gz").write_bytes(gzip.compress(payload, 6))
    progress("done", 1.0)
    return report
