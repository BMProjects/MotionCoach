"""Profile-driven evaluation: turn engine evidence into statuses, ranked issues and wording.

Statuses: good / warn / bad (measured), unmeasurable (coverage too low), na (not observable
from this camera view according to the profile). Missing data is never scored as 0.
"""

from __future__ import annotations

from ..engine.features import FEATURES
from .assets import Profile

STATUS_LABEL = {"good": "达标", "warn": "需注意", "bad": "偏差明显", "unmeasurable": "不可评价", "na": "此视角不可测",
                "uncertain": "对应不可靠"}
EVALUATED = ("good", "warn", "bad")


def _fmt(feature: str, value: float) -> str:
    if FEATURES[feature].unit == "deg":
        return f"{value:.0f}°"
    unit = "肩宽" if feature == "head_turn" else "躯干长"
    return f"{value * 100:.0f}% {unit}"


def describe(feature: str, delta: float, side_label: str | None = None, magnitude: float | None = None) -> str:
    """Numeric correction text for delta = user - reference (signed mean at key moments).

    ``magnitude`` is the typical absolute deviation. When the signed mean is much smaller than
    the magnitude, deviations alternate in direction (e.g. left/right repetitions), and a single
    "move this way" instruction would be wrong.
    """
    d = FEATURES[feature]
    label = side_label or d.label
    mag = abs(delta) if magnitude is None else magnitude
    if mag < 1e-9:
        return f"{label}与标准一致"
    if magnitude is not None and abs(delta) < 0.5 * magnitude:
        return f"{label}在重复动作中时而{d.more}、时而{d.less}，典型偏差约 {_fmt(feature, mag)}（方向不固定）"
    if delta > 0:
        return f"{label}比标准{d.more} {_fmt(feature, mag)}，应向“{d.less}”方向调整约 {_fmt(feature, mag)}"
    return f"{label}比标准{d.less} {_fmt(feature, mag)}，应向“{d.more}”方向调整约 {_fmt(feature, mag)}"


def user_side_label(feature: str, mirror_practice: bool) -> str:
    """Under mirror-side practice the user's opposite limb is compared with the reference limb."""
    d = FEATURES[feature]
    if mirror_practice and d.lr_twin:
        twin = FEATURES[d.lr_twin]
        return f"{twin.label}（对应示范{d.label[:1]}侧）"
    return d.label


def evaluate(comparison: dict, profile: Profile, mirror_practice: bool = False) -> dict:
    stat_key = "hold_median_abs" if profile.raw.get("status_statistic", "hold") == "hold" else "median_abs"
    sign_key = "hold_mean_delta" if stat_key == "hold_median_abs" else "mean_delta"
    min_cov = float(profile.raw.get("min_coverage", 0.6))
    q = profile.quality
    w_focus, w_other = float(profile.raw.get("focus_weight", 1.0)), float(profile.raw.get("other_weight", 0.4))

    min_hold_valid = int(profile.raw.get("min_hold_valid", 3))
    max_align = float(q.get("max_align_cost", 1.2))
    issues, segments = [], []
    counts = {k: 0 for k in STATUS_LABEL}
    for seg in comparison["segments"]:
        rule = profile.segment_rule(seg["id"])
        focus = set(rule.get("focus", []))
        na = set(rule.get("not_applicable", []))
        low_align = seg["align_cost"] > max_align
        feats = {}
        for f, ev in seg["features"].items():
            tol = profile.tolerance(f)
            cell = {**ev, "tolerance": tol, "focus": f in focus}
            hold_ok = stat_key != "hold_median_abs" or (ev["hold_coverage"] >= min_cov and ev["hold_valid"] >= min_hold_valid)
            if f in na:
                cell.update(status="na", reason=rule.get("reason", "此视角不可测"))
            elif ev["coverage"] < min_cov or not hold_ok or ev[stat_key] is None:
                cell.update(status="unmeasurable",
                            reason=f"关键点遮挡或置信度低（整段有效 {ev['coverage'] * 100:.0f}%，定势时刻有效 "
                                   f"{ev['hold_valid']}/{ev['hold_n']}）")
            elif low_align:
                cell.update(status="uncertain", value=ev[stat_key],
                            reason=f"该段动作对应可信度低（平均代价 {seg['align_cost']:.2f} > {max_align}），保留数值但暂不判定、不给修正方向")
            else:
                v = ev[stat_key]
                cell["value"] = v
                cell["status"] = "good" if v <= tol["good"] else "warn" if v <= tol["warn"] else "bad"
                cell["excess"] = max(0.0, v - tol["good"])
                signed = ev[sign_key] if ev[sign_key] is not None else ev["mean_delta"]
                cell["signed"] = signed
                cell["text"] = describe(f, signed, user_side_label(f, mirror_practice), magnitude=v)
                if cell["status"] != "good":
                    sev = (v / tol["good"]) * (w_focus if f in focus else w_other)
                    issues.append({"segment": seg["id"], "segment_name": seg["name"], "feature": f,
                                   "label": user_side_label(f, mirror_practice), "status": cell["status"],
                                   "severity": sev, "value": v, "signed": signed, "text": cell["text"],
                                   "time": seg["hold_times"][0] if seg["hold_times"] else seg["ref_start"],
                                   "focus": f in focus})
            feats[f] = cell
            if f in focus:
                counts[cell["status"]] += 1

        flags = []
        r = seg.get("duration_ratio_vs_tempo")
        if r is not None and (r < 0.5 or r > 2.0):
            flags.append(f"该段用时相对整体节奏异常（{r:.2f}×），可能漏做、多做或对齐不确定")
        if low_align:
            flags.append(f"动作对应可信度低（平均代价 {seg['align_cost']:.2f}），本段暂不判定")
        measured_focus = [c for f, c in feats.items() if c["focus"] and c["status"] in ("good", "warn", "bad")]
        segments.append({
            **{k: v for k, v in seg.items() if k != "features"},
            "note": rule.get("note"),
            "flags": flags,
            "pass_rate": (sum(c["status"] == "good" for c in measured_focus) / len(measured_focus)) if measured_focus else None,
            "features": feats,
        })

    issues.sort(key=lambda x: -x["severity"])
    top, seen = [], set()
    for it in issues:
        if not it["focus"] or (it["segment"], it["feature"]) in seen:
            continue
        seen.add((it["segment"], it["feature"]))
        top.append(it)
        if len(top) == 3:
            break
    evaluated = sum(counts[k] for k in EVALUATED)
    not_eval = counts["unmeasurable"] + counts["na"] + counts["uncertain"]
    return {
        "pass_rate": counts["good"] / evaluated if evaluated else None,
        "counts": counts,
        "pass_text": (f"{counts['good']}/{evaluated} 个可评价的重点指标达标" if evaluated else "没有可评价的重点指标")
                     + (f"；另有 {not_eval} 项不可评价（缺失 {counts['unmeasurable']}、视角不可测 {counts['na']}、对应不可靠 {counts['uncertain']}）" if not_eval else ""),
        "pass_rate_definition": "各动作段重点指标中，判定为达标的比例；分母只含可评价的指标，不是动作总分",
        "top_issues": top,
        "issues": issues,
        "segments": segments,
    }


def quality_warnings(qual: dict, profile: Profile, who: str) -> list[str]:
    q = profile.quality
    out = []
    if qual["detection_rate"] < q.get("min_detection_rate", 0.9):
        out.append(f"{who}：人物检测率 {qual['detection_rate'] * 100:.0f}%，部分时段无法评价")
    if qual.get("person_height_frac", 1) < q.get("min_person_height_frac", 0.35):
        out.append(f"{who}：人物在画面中偏小（约占画高 {qual['person_height_frac'] * 100:.0f}%），建议靠近或放大")
    if qual["visible_rate"].get("feet", 1) < q.get("min_feet_visible", 0.6):
        out.append(f"{who}：脚部可见率 {qual['visible_rate']['feet'] * 100:.0f}%，步型相关指标可信度下降")
    if min(qual["frame_size"]) < q.get("min_short_side", 480):
        out.append(f"{who}：分辨率偏低（{qual['frame_size'][0]}×{qual['frame_size'][1]}），关键点噪声较大")
    if qual.get("touching_edge_rate", 0) > q.get("max_edge_rate", 0.2):
        out.append(f"{who}：身体有 {qual['touching_edge_rate'] * 100:.0f}% 的时间触及画面边缘")
    return out
