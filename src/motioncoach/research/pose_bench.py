"""Pose-algorithm benchmark for martial-arts analysis (run on the GPU server).

    python -m motioncoach.research.pose_bench --out DIR [--backends a,b,...] [--cache DIR] [--fresh]

Phase 1 - one process per backend (clean timing and GPU memory): full-length extraction of the
reference and the practitioner video at 10 fps into a dedicated cache; wall time, inference ms/frame,
model load time, peak GPU memory (device-level, nvidia-smi) and mean GPU utilisation.
Phase 2 - every backend scored on the same frames:
  coverage    detection and per-joint-group validity, incl. hard sections (forward bend, head turns);
  stability   noise during annotated holds (detrended, torso units), bone-length variation in holds,
              left/right label swaps;
  agreement   distance to a family-balanced consensus skeleton, feature disagreement in tolerance units;
  downstream  the production correspondence (feature template search + annotated templates) scored
              against the annotation, measurement spread across repetitions, status agreement;
  signals     heel-raise (form 8) and head-turn (form 4) signal-to-noise, hand keypoint availability.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

FAMILIES = {"mediapipe": ("mediapipe-lite", "mediapipe-full", "mediapipe-heavy"),
            "rtm-body17": ("rtmpose-m", "rtmpose-x", "rtmo-m"),
            "rtmw-wholebody": ("rtmw-l", "rtmw-x", "rtmw3d-x"),
            "yolo": ("yolo26n-pose", "yolo26m-pose", "yolo11m-pose")}
DEFAULT = [b for fam in FAMILIES.values() for b in fam]
GROUPS = {"arms": ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"),
          "legs": ("left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"),
          "head": ("nose", "left_ear", "right_ear"),
          "feet": ("left_heel", "right_heel", "left_foot_index", "right_foot_index")}
LIMBS = (("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"),
         ("right_elbow", "right_wrist"), ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
         ("right_hip", "right_knee"), ("right_knee", "right_ankle"))


# ----------------------------------------------------------------------------- phase 1 (child process)

class GpuMonitor:
    """Polls device memory / utilisation with nvidia-smi (works for GL, CUDA and ORT alike)."""

    def __init__(self, period: float = 0.2):
        self.period, self.mem, self.util, self._stop = period, [], [], threading.Event()
        self.base = self._read()[0]
        self._t = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def _read():
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5).stdout.strip().split("\n")[0]
            m, u = (float(x) for x in out.split(","))
            return m, u
        except (OSError, ValueError, subprocess.SubprocessError):
            return float("nan"), float("nan")

    def _run(self):
        while not self._stop.is_set():
            m, u = self._read()
            self.mem.append(m)
            self.util.append(u)
            time.sleep(self.period)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()

    def summary(self):
        mem = np.array(self.mem, dtype=float)
        util = np.array(self.util, dtype=float)
        return {"gpu_mem_peak_mb": float(np.nanmax(mem) - self.base) if len(mem) else None,
                "gpu_util_mean": float(np.nanmean(util)) if len(util) else None}


def extract_child(name: str, result: Path) -> None:
    from .. import settings
    from ..apps.assets import get_reference
    from ..media.extract import extract_poses, prepared_video
    from ..perception.registry import get_backend
    import yaml

    ann = yaml.safe_load((settings.ROOT / "references" / "annotations" / "baduanjin_demo_a__weixin_222517.yaml").read_text(encoding="utf-8"))
    ref = get_reference(ann["reference"], settings.REFERENCES_DIR, settings.ROOT)
    videos = {"ref": ref.video, "user": settings.ASSETS_DIR / ann["user_video"]}
    out = {"backend": name}
    with GpuMonitor() as mon:
        t0 = time.time()
        be = get_backend(name, settings.MODELS_DIR)
        out["construct_s"] = time.time() - t0
        for side, v in videos.items():
            vid, proxy = prepared_video(v, settings.DATA_DIR)
            t = time.time()
            seq = extract_poses(proxy, vid, be, 10.0, settings.DATA_DIR)
            el = time.time() - t
            out[side] = {"frames": len(seq), "wall_s": el, "fps": len(seq) / el if el > 0 else None,
                         "ms_per_frame": seq.meta.get("ms_per_frame"), "frame_size": list(seq.frame_size),
                         "device": seq.meta.get("device"), "device_note": seq.meta.get("device_note", "")}
        out["total_s"] = time.time() - t0
    out.update(mon.summary())
    result.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


# ----------------------------------------------------------------------------- phase 2 helpers

def _detrended_std(x: np.ndarray) -> float:
    ok = np.isfinite(x)
    if ok.sum() < 5:
        return np.nan
    i = np.flatnonzero(ok)
    y = x[ok]
    A = np.c_[i, np.ones_like(i)]
    res = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    return float(np.std(res))


def _idx(t: np.ndarray, a: float, b: float) -> slice:
    i = np.flatnonzero((t >= a) & (t < b))
    return slice(int(i[0]), int(i[-1]) + 1) if len(i) else slice(0, 0)


def _valid(seq, joints) -> np.ndarray | None:
    sk = seq.skeleton
    have = [j for j in joints if sk.has(j)]
    if not have:
        return None
    idx = [sk.index[j] for j in have]
    return (seq.conf[:, idx] >= seq.meta.get("conf_thr", 0.5)) & np.isfinite(seq.xy[:, idx]).all(axis=2)


def _holds(gt: dict, side: str) -> list[tuple[str, float, float]]:
    key = "user_reps" if side == "user" else "ref_reps"
    return [(s["id"], q[2], q[3]) for s in gt["sections"] for q in s.get(key, [])]


def _swaps_per_min(seq) -> float:
    """Frames where exchanging the left/right labels of wrists or ankles explains the motion far better."""
    n = 0
    for a, b in (("left_wrist", "right_wrist"), ("left_ankle", "right_ankle")):
        L, R = seq.joint(a)[0].astype(float), seq.joint(b)[0].astype(float)
        dl = np.linalg.norm(L[1:] - L[:-1], axis=1) + np.linalg.norm(R[1:] - R[:-1], axis=1)
        dx = np.linalg.norm(L[1:] - R[:-1], axis=1) + np.linalg.norm(R[1:] - L[:-1], axis=1)
        sep = np.linalg.norm(L[1:] - R[1:], axis=1)
        n += int(np.sum(np.isfinite(dl) & np.isfinite(dx) & (dx < 0.4 * dl) & (sep > 0.15 * np.nanmedian(sep))))
    minutes = (seq.t[-1] - seq.t[0]) / 60.0
    return n / max(minutes, 1e-6)


def _band_amp(x: np.ndarray, dt: float = 0.1, lo_s: float = 0.4, hi_s: float = 4.0) -> float:
    """Std of the 0.4-4 s scale component (smoothed minus slow trend) of a 1-D signal."""
    ok = np.isfinite(x)
    if ok.sum() < 20:
        return np.nan
    y = np.interp(np.arange(len(x)), np.flatnonzero(ok), x[ok])
    k1, k2 = max(1, int(lo_s / dt)), max(2, int(hi_s / dt))
    sm = np.convolve(y, np.ones(k1) / k1, mode="same")
    tr = np.convolve(y, np.ones(k2) / k2, mode="same")
    band = (sm - tr)[k2:-k2] if len(y) > 2 * k2 + 5 else sm - tr
    return float(np.std(band))


def score_backend(name: str, pair, from_bounds) -> dict:
    from ..apps.assets import get_reference, load_profile
    from ..apps.rules import evaluate
    from ..engine.compare import Segment
    from ..engine.templates import compare_by_templates
    from . import methods as M
    from . import metrics as X
    from .. import settings

    gt = pair.gt
    out = {"backend": name}
    for side, S in (("ref", pair.ref), ("user", pair.user)):
        raw = S.raw.unmirror() if (side == "user" and pair.swap) else S.raw
        cov = {"detected": float(raw.detected.mean())}
        for g, js in GROUPS.items():
            v = _valid(raw, js)
            cov[g] = None if v is None else float(v.mean())
        hv = _valid(raw, ("left_hand_0", "left_hand_8", "right_hand_0", "right_hand_8"))
        cov["hands"] = None if hv is None else float(hv.mean())
        # hard moments: forward bend holds (form 6) and head-turn holds (form 4)
        for sid, key in (("s6", "bend_body"), ("s4", "turn_head")):
            js = GROUPS["arms"] + GROUPS["legs"] if key == "bend_body" else GROUPS["head"]
            v = _valid(raw, js)
            sl = [_idx(raw.t, a, b) for s_, a, b in _holds(gt, side) if s_ == sid]
            cov[key] = float(np.concatenate([v[s].ravel() for s in sl]).mean()) if sl else None
        # stability in holds: raw (masked only) and after the production filter
        stab = {}
        for filt in ("raw", "oneeuro"):
            N = S.N[filt]
            vals, cvs = [], []
            for _, a, b in _holds(gt, side):
                sl = _idx(S.t, a, b)
                blk = N[sl]
                vals.append(np.nanmean([_detrended_std(blk[:, j, c]) for j in range(blk.shape[1]) for c in (0, 1)]))
                if filt == "raw":
                    seq = S.seq["raw"]
                    for a_, b_ in LIMBS:
                        L = np.linalg.norm(seq.joint(b_)[0][sl].astype(float) - seq.joint(a_)[0][sl].astype(float), axis=1)
                        if np.isfinite(L).sum() >= 5:
                            cvs.append(np.nanstd(L) / np.nanmean(L))
            stab[f"hold_noise_{filt}"] = float(np.nanmedian(vals))
            if filt == "raw":
                stab["bone_cv_hold"] = float(np.nanmedian(cvs)) if cvs else None
        stab["lr_swaps_per_min"] = _swaps_per_min(raw)
        out[side] = {"coverage": cov, "stability": stab}

    # signals relevant to specific forms (practitioner): heel raise (s8), head turn (s4)
    U = pair.user
    s8 = next(s for s in gt["sections"] if s["id"] == "s8")["user"]
    sl8 = _idx(U.t, *s8)
    seq = U.seq["raw"]
    hip_y = 0.5 * (seq.joint("left_hip")[0][:, 1] + seq.joint("right_hip")[0][:, 1]).astype(float)
    torso = np.nanmedian(np.linalg.norm(0.5 * (seq.joint("left_shoulder")[0] + seq.joint("right_shoulder")[0])
                                        - 0.5 * (seq.joint("left_hip")[0] + seq.joint("right_hip")[0]), axis=1))
    noise = out["user"]["stability"]["hold_noise_raw"]
    sig = {}
    for j in ("left_heel", "left_ankle"):           # heel raise lifts the body; heel / ankle move up a little too
        if seq.skeleton.has(j):
            sig[f"s8_{j}_amp"] = _band_amp(seq.joint(j)[0][:, 1].astype(float)[sl8] / torso)
    sig["s8_hip_amp"] = _band_amp(hip_y[sl8] / torso)
    sig["s8_hip_snr"] = sig["s8_hip_amp"] / noise if noise else None
    ht = U.F["oneeuro"][:, pair.names.index("head_turn")]
    turns = [abs(np.nanmedian(ht[_idx(U.t, a, b)])) for s_, a, b in _holds(gt, "user") if s_ == "s4"]
    sig["s4_head_turn_amp"] = float(np.nanmedian(turns)) if turns else None
    d = np.diff(ht[_idx(U.t, 282, 353)])
    d = d[np.isfinite(d)]
    robust = 1.4826 * np.median(np.abs(d - np.median(d))) if len(d) else np.nan   # outlier frames must not dominate
    sig["s4_head_turn_snr"] = float(np.nanmedian(turns) / max(1e-3, robust)) if turns else None
    out["signals"] = sig

    # downstream: production correspondence with annotated templates
    for kind in ("gt", "coarse"):
        b = M.bounds_gt(pair) if kind == "gt" else from_bounds(pair)
        secs = M.METHODS["M3fa"][1](pair, b, "oneeuro")
        sc = X.score(pair, secs, "loop")
        a = sc["annotation"]
        out[f"downstream_{kind}"] = {
            "rep_f1": (a["user_reps"] or {}).get("f1"), "rep_boundary_mae_s": (a["user_reps"] or {}).get("boundary_mae_s"),
            "count_err": a["count_abs_err"]["user"], "hold_hit_user_master": a["hold_hit_user_master"],
            "hold_hit_ref_master": a["hold_hit_ref_master"], "alternation_ok": a["alternation_ok"],
            "residual": sc["residual"], "speed_in_range": sc["balance"]["speed_in_range"],
            "section_start_mae_s": a["section_start_mae_s"]}
    # production comparison + rules (GT bounds): statuses, pass rate, spread across repetitions
    ref = get_reference(gt["reference"], settings.REFERENCES_DIR, settings.ROOT)
    profile = load_profile(ref.raw["profile"])
    bounds = M.bounds_gt(pair)
    segs = [Segment(sid, profile.segment_rule(sid).get("name", sid), float(pair.ref.t[r0]), float(pair.ref.t[r1 - 1]) + 0.05)
            for sid, r0, r1, _, _ in bounds]
    cmp = compare_by_templates(pair.ref.t, pair.ref.F["oneeuro"], pair.user.t, pair.user.F["oneeuro"], segs,
                               [(u0, u1) for *_, u0, u1 in bounds], pair.names, pair.scales, ref_reps=ref.rep_templates())
    ev = evaluate(cmp, profile, backend=name)
    statuses = {f"{s['id']}|{f}": c["status"] for s in ev["segments"] for f, c in s["features"].items()}
    spread = []
    for s in cmp["segments"]:
        rule = profile.segment_rule(s["id"])
        for f in rule.get("focus", []):
            pr = [v for v in s["features"][f]["per_rep"] if v is not None]
            if len(pr) >= 3:
                spread.append(np.std(pr) / profile.tolerance(f)["good"])
    out["rules"] = {"pass_rate": ev["pass_rate"], "statuses": statuses,
                    "rep_spread_tol": float(np.median(spread)) if spread else None}
    return out


def consensus(pairs: dict) -> dict:
    """Distance of every backend to a family-balanced consensus (median of family medians), same frames."""
    out = {b: {} for b in pairs}
    for side in ("ref", "user"):
        n = min(len(p.ref.t if side == "ref" else p.user.t) for p in pairs.values())
        N = {b: (p.ref if side == "ref" else p.user).N["oneeuro"][:n] for b, p in pairs.items()}
        F = {b: (p.ref if side == "ref" else p.user).F["oneeuro"][:n] for b, p in pairs.items()}
        fam_N, fam_F = [], []
        for fam, members in FAMILIES.items():
            have = [m for m in members if m in pairs]
            if have:
                fam_N.append(np.nanmedian(np.stack([N[m] for m in have]), axis=0))
                fam_F.append(np.nanmedian(np.stack([F[m] for m in have]), axis=0))
        CN, CF = np.nanmedian(np.stack(fam_N), axis=0), np.nanmedian(np.stack(fam_F), axis=0)
        scales = next(iter(pairs.values())).scales
        for b in pairs:
            d = np.linalg.norm(N[b] - CN, axis=2)                     # (n, 13) torso units
            fd = np.abs(F[b] - CF) / scales
            out[b][side] = {"joint_dev_median": float(np.nanmedian(d)), "joint_dev_p90": float(np.nanpercentile(d, 90)),
                            "wrist_dev_median": float(np.nanmedian(d[:, 5:7])), "ankle_dev_median": float(np.nanmedian(d[:, 11:13])),
                            "feature_dev_tol_median": float(np.nanmedian(fd)),
                            "feature_over_tol_share": float(np.nanmean(fd > 1.0))}
    return out


def status_agreement(results: dict) -> dict:
    keys = sorted(set.intersection(*[set(r["rules"]["statuses"]) for r in results.values()]))
    major = {}
    for k in keys:
        vals = [r["rules"]["statuses"][k] for r in results.values()]
        major[k] = max(set(vals), key=vals.count)
    return {b: float(np.mean([r["rules"]["statuses"][k] == major[k] for k in keys])) for b, r in results.items()}


# ----------------------------------------------------------------------------- report

def _f(v, pct=False, nd=2):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{100 * v:.0f}%" if pct else f"{v:.{nd}f}"


def to_markdown(meta, eff, res, cons, agree) -> str:
    from ..perception.registry import SPECS
    from ..engine.skeleton import get_skeleton
    L = [f"# 姿态算法横评（{meta['host']} · {meta['gpu']}）", "",
         f"{meta['created']} · 代码 {meta['commit']} · 视频：标准 Weixin…221512（{meta['ref_frames']} 帧 @10 fps）、"
         f"练习者 Weixin…222517（{meta['user_frames']} 帧 @10 fps，{meta['user_size']}）· 每个算法独立进程、空缓存运行", ""]
    order = [b for b in DEFAULT if b in res]
    L += ["## 1. 效率（GPU）", "", "| 算法 | 设备 | 推理 ms/帧 | 端到端 帧/秒（含解码） | 两段视频总耗时 s | 模型加载 s | 显存峰值 MB | GPU 利用率 |", "|---|---|---|---|---|---|---|---|"]
    for b in order:
        e = eff.get(b, {})
        u, r = e.get("user", {}), e.get("ref", {})
        frames = (u.get("frames") or 0) + (r.get("frames") or 0)
        wall = (u.get("wall_s") or 0) + (r.get("wall_s") or 0)
        L.append(f"| {b} | {u.get('device', '—')} | {_f(u.get('ms_per_frame'), nd=1)} | {_f(frames / wall if wall else None, nd=0)} | "
                 f"{_f(e.get('total_s'), nd=0)} | {_f(e.get('construct_s'), nd=1)} | {_f(e.get('gpu_mem_peak_mb'), nd=0)} | "
                 f"{_f((e.get('gpu_util_mean') or 0) / 100, True) if e.get('gpu_util_mean') is not None else '—'} |")
    L += ["", "## 2. 下游动作分析（特征 + 模板搜索 + 标注模板；段边界 = 标注）", "",
          "| 算法 | 重复 F1 | 边界误差 s | 定势命中·练习者为准 | 定势命中·标准为准 | 左右交替 | 达标比例 | 判定与多数一致 | 重复间离散（容差） | 段边界误差 s（自动） |", "|---|---|---|---|---|---|---|---|---|---|"]
    for b in order:
        d, dc, rl = res[b]["downstream_gt"], res[b]["downstream_coarse"], res[b]["rules"]
        L.append(f"| {b} | {_f(d['rep_f1'])} | {_f(d['rep_boundary_mae_s'], nd=1)} | {_f(d['hold_hit_user_master'], True)} | "
                 f"{_f(d['hold_hit_ref_master'], True)} | {_f(d['alternation_ok'], True)} | {_f(rl['pass_rate'], True)} | "
                 f"{_f(agree.get(b), True)} | {_f(rl['rep_spread_tol'])} | {_f(dc['section_start_mae_s'], nd=1)} |")
    L += ["", "## 3. 稳定性与一致性", "",
          "| 算法 | 定势噪声·原始（躯干长） 练习者 / 标准 | 定势噪声·滤波后 练习者 | 骨长变异（定势内） | 左右标签互换 次/分 | 偏离共识·关节中位 练习者 / 标准 | 腕 | 踝 | 指标偏离共识（容差） | 超出容差比例 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for b in order:
        su, sr = res[b]["user"]["stability"], res[b]["ref"]["stability"]
        cu, cr = cons[b]["user"], cons[b]["ref"]
        L.append(f"| {b} | {_f(su['hold_noise_raw'], nd=4)} / {_f(sr['hold_noise_raw'], nd=4)} | {_f(su['hold_noise_oneeuro'], nd=4)} | "
                 f"{_f(su['bone_cv_hold'], True)} | {_f(su['lr_swaps_per_min'], nd=1)} | {_f(cu['joint_dev_median'], nd=3)} / {_f(cr['joint_dev_median'], nd=3)} | "
                 f"{_f(cu['wrist_dev_median'], nd=3)} | {_f(cu['ankle_dev_median'], nd=3)} | {_f(cu['feature_dev_tol_median'])} | {_f(cu['feature_over_tol_share'], True)} |")
    L += ["", "## 4. 覆盖与武术相关信号（练习者视频）", "",
          "| 算法 | 关键点 | 检出率 | 手臂 | 腿 | 头（鼻/耳） | 足（跟/尖） | 手指 | 前屈定势（第六式）有效 | 转头定势（第四式）头部有效 | 转头幅度 / 信噪比 | 提踵（第八式）髋部起伏 / 信噪比 | 3D | 许可 |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for b in order:
        c, s = res[b]["user"]["coverage"], res[b]["signals"]
        sp = SPECS[b]
        L.append(f"| {b} | {sp.keypoints} | {_f(c['detected'], True)} | {_f(c['arms'], True)} | {_f(c['legs'], True)} | {_f(c['head'], True)} | "
                 f"{_f(c['feet'], True)} | {_f(c['hands'], True)} | {_f(c['bend_body'], True)} | {_f(c['turn_head'], True)} | "
                 f"{_f(s['s4_head_turn_amp'])} / {_f(s['s4_head_turn_snr'], nd=1)} | {_f(s['s8_hip_amp'], nd=3)} / {_f(s['s8_hip_snr'], nd=1)} | "
                 f"{'是' if '3D' in sp.dims else '—'} | {sp.license} |")
    L += ["", "说明：", "- 推理 ms/帧 为模型调用本身（不含解码，去掉前 3 帧预热）；端到端 帧/秒 含视频解码与采样，两段视频共约 1 万采样帧。显存峰值为设备级（nvidia-smi），已扣除运行前基线，同机其他进程的波动会混入。",
          "- 定势噪声：标注的定势区间内，同体型骨架 13 个关节坐标去线性趋势后的标准差（躯干长）；定势中人体近似静止，此值主要反映关键点抖动。骨长变异：定势内上臂/前臂/大腿/小腿长度的变异系数。",
          "- 共识：四个算法族（MediaPipe、RTMPose/RTMO、RTMW、YOLO）各取族内中位数后再取中位数；偏离共识只说明“与多数算法不同”，不等于误差。",
          "- 判定与多数一致：每个（动作段 × 指标）的状态与 12 个算法多数结论相同的比例。重复间离散：各式重点指标在 6 次重复上的定势偏差标准差 / 允许偏差（含练习者真实的动作差异）。",
          "- 转头幅度：第四式标注定势内头部左右转（肩宽）中位数；信噪比 = 幅度 / 第四式逐帧差分的稳健标准差（1.4826×MAD）。提踵：第八式内髋部竖直位置 0.4–4 s 尺度波动（躯干长），信噪比相对定势噪声。", ""]
    return "\n".join(L)


# ----------------------------------------------------------------------------- figure

def plot(results_path: Path, out_png: Path) -> None:
    """Two panels sharing the x axis (end-to-end throughput, log): downstream hold hit rate, and hold noise.
    Family = marker shape (four families exceed the three-colour all-pairs limit for a scatter); every point
    is labelled directly, so identity never depends on colour."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401
    plt.rcParams["font.family"] = ["Noto Sans CJK SC", "Noto Sans CJK TC", "DejaVu Sans"]
    R = json.loads(Path(results_path).read_text(encoding="utf-8"))
    eff, sc = R["efficiency"], R["scores"]
    marker = {"mediapipe": "o", "rtm-body17": "s", "rtmw-wholebody": "^", "yolo": "D"}
    fam_of = {b: f for f, ms in FAMILIES.items() for b in ms}
    ink, muted, grid = "#0b0b0b", "#52514e", "#e1e0d9"
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2))
    panels = [("定势对应命中率（以练习者为准，越高越好）", lambda b: sc[b]["downstream_gt"]["hold_hit_user_master"] * 100, "%"),
              ("定势内关键点噪声·原始（练习者，躯干长，越低越好）", lambda b: sc[b]["user"]["stability"]["hold_noise_raw"], "")]
    for ax, (title, f, unit) in zip(axs, panels):
        for b in sc:
            if b not in eff:
                continue
            u, r = eff[b]["user"], eff[b]["ref"]
            x = (u["frames"] + r["frames"]) / (u["wall_s"] + r["wall_s"])
            y = f(b)
            ax.scatter([x], [y], marker=marker[fam_of[b]], s=70, facecolor="#2a78d6", edgecolor="#fcfcfb", linewidth=2, zorder=3)
            off, ha = (6, 4), "left"
            if ax is axs[1] and b in ("yolo26m-pose", "rtmw-x"):   # avoid label collisions in the noise panel
                off = (6, -12)
            elif ax is axs[1] and b == "yolo11m-pose":
                off, ha = (-6, -3), "right"
            ax.annotate(b, (x, y), xytext=off, textcoords="offset points", fontsize=8.5, color=ink, ha=ha)
        ax.set_xscale("log")
        ax.set_xticks([50, 100, 200, 400])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("端到端吞吐（采样帧/秒，含解码，对数轴）", color=muted)
        ax.set_title(title, fontsize=11, color=ink, loc="left")
        ax.grid(color=grid, linewidth=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=muted)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker=m, linestyle="", markerfacecolor="#2a78d6", markeredgecolor="#fcfcfb", markersize=9, label=f)
               for f, m in marker.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_png, dpi=120, facecolor="#fcfcfb")
    plt.close(fig)


# ----------------------------------------------------------------------------- main

def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=False)
    ap.add_argument("--backends", default=",".join(DEFAULT))
    ap.add_argument("--cache", help="dedicated data dir for this benchmark (videos are transcoded once into it)")
    ap.add_argument("--fresh", action="store_true", help="delete cached poses in --cache first (timings from scratch)")
    ap.add_argument("--plot", metavar="RESULTS_JSON", help="only draw the summary figure next to RESULTS_JSON")
    ap.add_argument("--score-only", action="store_true", help="reuse efficiency_*.json in --out and the cached poses")
    ap.add_argument("--extract", help=argparse.SUPPRESS)
    ap.add_argument("--result", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.cache:
        os.environ["MOTIONCOACH_DATA"] = str(Path(a.cache).resolve())
    if a.plot:
        plot(Path(a.plot), Path(a.plot).with_name("summary.png"))
        return
    if a.extract:                                   # child
        extract_child(a.extract, Path(a.result))
        return
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    backends = a.backends.split(",")
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)   # all-NaN medians on undetected frames
    from .. import settings
    if a.fresh and not a.score_only:
        shutil.rmtree(settings.DATA_DIR / "poses", ignore_errors=True)
    from ..apps.assets import get_reference
    from ..media.extract import prepared_video
    import yaml
    ann = yaml.safe_load((settings.ROOT / "references" / "annotations" / "baduanjin_demo_a__weixin_222517.yaml").read_text(encoding="utf-8"))
    for v in (get_reference(ann["reference"], settings.REFERENCES_DIR, settings.ROOT).video, settings.ASSETS_DIR / ann["user_video"]):
        prepared_video(v, settings.DATA_DIR)          # transcode once, outside the timed extraction
    eff = {}
    for b in (backends if a.score_only else []):
        f = out / f"efficiency_{b}.json"
        if f.exists():
            eff[b] = json.loads(f.read_text(encoding="utf-8"))
    for b in ([] if a.score_only else backends):    # phase 1: one clean process per backend
        res_p = out / f"efficiency_{b}.json"
        cmd = [sys.executable, "-m", "motioncoach.research.pose_bench", "--extract", b, "--result", str(res_p)]
        if a.cache:
            cmd += ["--cache", a.cache]
        t = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not res_p.exists():
            print(f"{b}: extraction failed\n{r.stderr[-2000:]}", flush=True)
            continue
        eff[b] = json.loads(res_p.read_text(encoding="utf-8"))
        print(f"{b}: extracted in {time.time() - t:.0f}s  user {eff[b]['user']['fps']:.0f} fps "
              f"({eff[b]['user']['ms_per_frame']:.1f} ms/frame, {eff[b]['user']['device']})", flush=True)
    from .data import load_pair
    from . import methods as M
    pairs = {b: load_pair("baduanjin", backend=b, maps=False) for b in eff}   # phase 2: cached poses
    res = {}
    for b, p in pairs.items():
        res[b] = score_backend(b, p, lambda pr: M.bounds_coarse(pr, "oneeuro"))
        print(f"{b}: scored", flush=True)
    cons = consensus(pairs)
    agree = status_agreement(res)
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    dep = settings.ROOT / "DEPLOYED_COMMIT"
    raw_c = dep.read_text().strip() if dep.exists() else ""
    commit = (raw_c[:7] + ("+dirty" if "dirty" in raw_c else "")) if raw_c else "local"
    p0 = next(iter(pairs.values()))
    meta = {"created": time.strftime("%Y-%m-%d %H:%M"), "host": platform.node(), "gpu": gpu, "commit": commit,
            "ref_frames": len(p0.ref.t), "user_frames": len(p0.user.t), "user_size": "×".join(map(str, p0.user.raw.frame_size))}
    (out / "results.json").write_text(json.dumps({"meta": meta, "efficiency": eff, "scores": res, "consensus": cons,
                                                  "status_agreement": agree}, ensure_ascii=False, indent=1,
                                                 default=lambda o: o.item() if isinstance(o, np.generic) else str(o)),
                                      encoding="utf-8")
    md = to_markdown(meta, eff, res, cons, agree)
    (out / "report.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
