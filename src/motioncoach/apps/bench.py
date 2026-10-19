"""Pose-backend comparison on the same analysis interval.

The first backend's run decides the interval (auto/segments scope); all other backends then
analyse exactly that interval (manual scope), so every model sees the same frames. Reported:

- speed (ms/frame, excluding warm-up; device noted — CPU and GPU numbers are not comparable),
- detection rate and visibility of arm/leg/foot joints,
- jitter: median second difference of raw (unsmoothed) body keypoints / torso length,
- inter-model angle disagreement on the reference interval (same frames): how much the
  measured angles change just by switching models, to be compared with the tolerances,
- downstream agreement: per (segment, feature) status and value vs the first backend.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np

from ..engine.features import compute_features
from ..engine.quality import measure
from ..engine.sequence import PoseSequence, clean
from ..media.extract import extract_poses, prepared_video
from ..perception.base import run_on_video
from ..perception.registry import SPECS, get_backend
from .analysis import AnalysisOptions, run_analysis
from .assets import Reference, load_profile
from .scope import ScopeRequest

BODY = ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
        "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle")


def jitter(seq: PoseSequence) -> float | None:
    sk = seq.skeleton
    idx = [sk.index[j] for j in BODY]
    xy = seq.xy[:, idx].astype(np.float64)
    xy[seq.conf[:, idx] < seq.meta.get("conf_thr", 0.5)] = np.nan
    acc = np.linalg.norm(xy[2:] - 2 * xy[1:-1] + xy[:-2], axis=2)
    sm = 0.5 * (seq.joint("left_shoulder")[0] + seq.joint("right_shoulder")[0])
    hm = 0.5 * (seq.joint("left_hip")[0] + seq.joint("right_hip")[0])
    torso = np.nanmedian(np.linalg.norm(sm - hm, axis=1))
    v = acc[np.isfinite(acc)]
    return float(np.median(v) / torso) if len(v) and torso > 0 else None


def run_bench(reference: Reference, user_video: Path, backends: list[str], out_dir: Path, data_dir: Path,
              models_dir: Path, base: AnalysisOptions, progress: Callable[[str, float], None] | None = None) -> dict:
    progress = progress or (lambda s, f: None)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    profile = load_profile(reference.raw["profile"])
    names = profile.features
    runs, reports = [], {}
    scope = base.scope
    for k, name in enumerate(backends):
        opts = replace(base, backend=name, scope=scope)
        t0 = time.time()
        try:
            rep = run_analysis(reference, user_video, out_dir / name, data_dir, models_dir, opts,
                               progress=lambda s, f, k=k: progress(f"{name}:{s}", (k + f) / len(backends)))
        except Exception as exc:
            runs.append({"backend": name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        reports[name] = rep
        if k == 0 or scope.mode != "manual":   # freeze the interval chosen by the first successful backend
            sc = rep["scope"]
            scope = ScopeRequest("manual", sc["segments"][0], sc["segments"][-1], sc["user_range"][0], sc["user_range"][1])
        runs.append({"backend": name, "elapsed_s": round(time.time() - t0, 1)})

    # raw sequences of each backend on the same windows (from cache) for model-level metrics
    ref_id, ref_proxy = prepared_video(reference.video, data_dir)
    user_id, user_proxy = prepared_video(Path(user_video), data_dir)
    feats, feats_u, frames, rows = {}, {}, {}, []
    for run in runs:
        name = run["backend"]
        if name not in reports:
            rows.append({**run, "label": SPECS[name].label})
            continue
        rep = reports[name]
        be = get_backend(name, models_dir)
        sc = rep["scope"]
        r = extract_poses(ref_proxy, ref_id, be, base.sample_fps, data_dir, *sc["ref_range"])
        u = extract_poses(user_proxy, user_id, be, base.sample_fps, data_dir, *sc["user_range"])
        # speed is always measured now, on this machine's device: cached poses carry no valid timing
        ms = {}
        for side, proxy, rng in (("reference", ref_proxy, sc["ref_range"]), ("user", user_proxy, sc["user_range"])):
            probe_seq = run_on_video(be, proxy, base.sample_fps, rng[0], min(rng[1], rng[0] + 10.0))
            ms[side] = probe_seq.meta["ms_per_frame"]
        frames[name] = (np.round(r.t, 3), np.round(u.t, 3))
        qr, qu = measure(r, r.meta.get("conf_thr", 0.5)), measure(u, u.meta.get("conf_thr", 0.5))
        F, _, _ = compute_features(clean(r, conf_thr=r.meta.get("conf_thr", 0.5))[0], names)
        Fu, _, _ = compute_features(clean(u, conf_thr=u.meta.get("conf_thr", 0.5))[0], names)
        feats[name] = (r.t, F)
        feats_u[name] = (u.t, Fu)
        rows.append({
            **run, "label": SPECS[name].label, "license": SPECS[name].license, "keypoints": SPECS[name].keypoints,
            "device": rep["model"]["device"], "device_note": r.meta.get("device_note", ""), "ms_per_frame": ms,
            "detection": [qr["detection_rate"], qu["detection_rate"]],
            "visible": {"reference": qr["visible_rate"], "user": qu["visible_rate"]},
            "jitter": [jitter(r), jitter(u)],
            "align_cost": rep["timing"]["align_cost"], "pass_rate": rep["summary"]["pass_rate"],
            "top_issues": [f"{i['segment_name']} · {i['label']}" for i in rep["summary"]["top_issues"]],
        })

    # same-frame check: every backend must have seen exactly the same sampled frames
    names_ok = [n for n in feats]
    frame_check = {}
    if names_ok:
        r0, u0 = frames[names_ok[0]]
        for n in names_ok:
            r1, u1 = frames[n]
            frame_check[n] = {"ref_identical": bool(len(r1) == len(r0) and np.array_equal(r1, r0)),
                              "user_identical": bool(len(u1) == len(u0) and np.array_equal(u1, u0)),
                              "ref_frames": int(len(r1)), "user_frames": int(len(u1))}
    tol = np.array([profile.tolerance(n)["good"] for n in names])

    def pair_stats(store):
        out = {}
        for i, a in enumerate(names_ok):
            for b in names_ok[i + 1:]:
                ta, Fa = store[a]
                tb, Fb = store[b]
                _, ia, ib = np.intersect1d(np.round(ta, 3), np.round(tb, 3), return_indices=True)
                d = np.abs(Fa[ia] - Fb[ib])
                row = {}
                for j, n in enumerate(names):
                    v = d[:, j][np.isfinite(d[:, j])]
                    row[n] = None if not len(v) else {
                        "n": int(len(v)), "valid_rate": float(len(v) / max(1, len(d))),
                        "median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
                        "p95": float(np.percentile(v, 95)), "over_tol": float((v > tol[j]).mean())}
                out[f"{a}|{b}"] = row
        return out

    # fixed correspondence (same frame, no alignment involved): pure model-to-model measurement difference
    disagreement = pair_stats(feats)
    disagreement_user = pair_stats(feats_u)

    # downstream agreement with the first backend
    agreement = {}
    if names_ok:
        base_rep = reports[names_ok[0]]
        def cells(rep):
            return {(s["id"], f): c for s in rep["segments"] for f, c in s["features"].items()}
        b0 = cells(base_rep)
        for n in names_ok[1:]:
            bn = cells(reports[n])
            keys = [k for k in b0 if k in bn and b0[k]["status"] in ("good", "warn", "bad") and bn[k]["status"] in ("good", "warn", "bad")]
            same = sum(b0[k]["status"] == bn[k]["status"] for k in keys)
            vd = [abs(b0[k]["value"] - bn[k]["value"]) / b0[k]["tolerance"]["good"] for k in keys]
            agreement[n] = {"cells": len(keys), "status_agreement": same / len(keys) if keys else None,
                            "median_value_diff_in_tolerance": float(np.median(vd)) if vd else None}

    first = reports[names_ok[0]] if names_ok else None
    result = {
        "schema": "motioncoach.bench/0.1", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "reference": reference.id, "user": Path(user_video).name, "sample_fps": base.sample_fps,
        "scope": first["scope"] if first else None,
        "baseline": names_ok[0] if names_ok else None,
        "rows": rows, "frame_check": frame_check,
        "disagreement": disagreement, "disagreement_user": disagreement_user, "agreement": agreement,
        "definitions": {
            "disagreement": "同一帧上两种算法测得数值的绝对差（固定对应关系，不涉及时间对齐）；over_tol = 超过允许偏差的帧比例",
            "agreement": "各算法独立完成完整流程（含各自的时间对齐）后，动作段×指标判定与基线一致的比例",
            "jitter": "未平滑关键点二阶差分中位数 / 躯干长（10 fps 等间隔）；含真实运动，过度平滑也会变小，只作同视频相对参考"},
        "features": [{"name": n, "label": f["label"], "unit": f["unit"], "tolerance": f["tolerance"]["good"]}
                     for n, f in zip(names, first["features"])] if first else [],
    }
    (out_dir / "bench.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "bench.md").write_text(to_markdown(result), encoding="utf-8")
    progress("done", 1.0)
    return result


def to_markdown(b: dict) -> str:
    sc = b.get("scope") or {}
    lines = [f"# 姿态算法对比 · {b['reference']} vs {b['user']}", "",
             f"区间：标准 {sc.get('ref_range')} s，用户 {sc.get('user_range')} s，段 {sc.get('segments')}；采样 {b['sample_fps']} fps", "",
             "| 后端 | 设备 | ms/帧（标准/用户） | 检测率 | 脚部可见（标/用） | 抖动×1000（标/用） | 对齐代价 | 达标率 | 与基线状态一致 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in b["rows"]:
        if "error" in r:
            lines.append(f"| {r['backend']} | — | 失败：{r['error']} | | | | | | |")
            continue
        ms = r["ms_per_frame"]
        feet = [r["visible"][k].get("feet") for k in ("reference", "user")]
        msf = lambda x: "—" if x is None else f"{x:.0f}"
        ag = b["agreement"].get(r["backend"], {})
        fmt = lambda x, f="{:.0%}": "—" if x is None else f.format(x)
        lines.append(f"| {r['backend']} | {r['device']} | {msf(ms.get('reference'))} / {msf(ms.get('user'))} | "
                     f"{fmt(r['detection'][0])} / {fmt(r['detection'][1])} | {fmt(feet[0])} / {fmt(feet[1])} | "
                     f"{fmt(r['jitter'][0] and r['jitter'][0] * 1000, '{:.1f}')} / {fmt(r['jitter'][1] and r['jitter'][1] * 1000, '{:.1f}')} | "
                     f"{r['align_cost']:.2f} | {fmt(r['pass_rate'])} | {fmt(ag.get('status_agreement')) if ag else '基线'} |")
    fc = b.get("frame_check", {})
    if fc:
        same = all(v["ref_identical"] and v["user_identical"] for v in fc.values())
        lines += ["", f"输入帧一致性：{'全部算法使用完全相同的采样帧' if same else '存在不一致：' + str(fc)}"]
    for key, title in (("disagreement", "标准视频"), ("disagreement_user", "练习者视频")):
        d = b.get(key) or {}
        if not d:
            continue
        lines += ["", f"## 模型间测量差异（{title}，同一帧；中位 / P95 / 超容差比例）", "",
                  "| 指标 | 容差 | " + " | ".join(k.replace("|", " vs ") for k in d) + " |",
                  "|---|---|" + "---|" * len(d)]
        for f in b["features"]:
            cells = []
            for k in d:
                v = d[k][f["name"]]
                cells.append("—" if v is None else f"{v['median']:.2f} / {v['p95']:.2f} / {v['over_tol']:.0%}")
            lines.append(f"| {f['label']} | {f['tolerance']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
