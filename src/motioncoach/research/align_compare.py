"""Run every correspondence method on the same inputs and write a comparison report.

    uv run python -m motioncoach.research.align_compare [--out DIR] [--pairs baduanjin,dance]

Outputs (DIR): results.json (all scores), report.md (tables), timeline.png, playback.png.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from .. import settings
from . import methods as M
from . import metrics as X
from ..perception.registry import DEFAULT_BACKEND
from .data import load_pair

BOUNDS = ("gt", "coarse", "hsmm")


def _bounds(pair, kind, filt, bounds_pair=None):
    if kind == "gt":
        return M.bounds_gt(pair)
    bp = bounds_pair or pair            # section bounds may come from another pose model (same 10 fps grid)
    return M.bounds_coarse(bp, filt) if kind == "coarse" else M.bounds_hsmm(bp, filt)


def run_pair(pair, only: set[str] | None = None, bounds_pair=None) -> dict:
    out = {"jitter": pair.jitter, "runs": {}, "sections": {}}
    cache = {}
    for kind in BOUNDS:
        if kind == "gt" and pair.gt is None and pair.gt_offset is None:
            continue
        for key, (label, fn, filt_default) in M.METHODS.items():
            if only and key not in only:
                continue
            for filt in ("sg", "oneeuro"):
                if kind != "gt" and filt != filt_default:
                    continue                      # end-to-end runs only with each method's own filter
                bk = (kind, filt)
                if bk not in cache:
                    cache[bk] = _bounds(pair, kind, filt, bounds_pair)
                t0 = time.time()
                secs = fn(pair, cache[bk], filt)
                el = time.time() - t0
                policies = ("ui", "loop") if key == "M1" else ("loop",)
                for pol in policies:
                    name = f"{key}|{kind}|{filt}|{pol}"
                    sc = X.score(pair, secs, pol)
                    sc.update({"method": key, "label": label, "bounds": kind, "filter": filt, "policy": pol,
                               "seconds": round(el, 2)})
                    out["runs"][name] = sc
                out["sections"][f"{key}|{kind}|{filt}"] = [
                    {"id": s.id, "ref": [s.r0 * M.DT, s.r1 * M.DT], "user": [s.u0 * M.DT, s.u1 * M.DT],
                     "ref_reps": [[r.start * M.DT, r.end * M.DT] for r in s.ref_reps],
                     "user_reps": [{"t": [u.start * M.DT, u.end * M.DT], "ref": u.ref, "mirror": u.mirror,
                                    "paired": u.paired, "default": u.default, "cost": round(float(u.cost), 3),
                                    "hold": None if u.hold is None else [u.hold[0] * M.DT, u.hold[1] * M.DT]}
                                   for u in s.user_reps]} for s in secs]
    use = {"gt": ("gt", "sg"), "coarse": ("coarse", "sg"), "hsmm": ("hsmm", "oneeuro")}   # production filter for coarse
    out["bounds"] = {k: [list(b) for b in cache[v]] for k, v in use.items() if v in cache}
    return out


# ----------------------------------------------------------------------------- report

def _f(v, pct=False, nd=2):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{100 * v:.0f}%" if pct else f"{v:.{nd}f}"


ORDER = ["M0", "M1", "M2", "M2a", "M3", "M3a", "M3f", "M3fa", "M3ca", "M3fca"]


def _row(name, r):
    a = r.get("annotation", {})
    ur = a.get("user_reps") or {}
    b, pu, pr = r["balance"], r["playback_user_master"], r["playback_ref_master"]
    return ("| " + " | ".join([
        name, _f(ur.get("f1")), _f(ur.get("boundary_mae_s"), nd=1), _f(a.get("count_abs_err", {}).get("user"), nd=1),
        _f(a.get("hold_hit_user_master"), True), _f(a.get("hold_err_user_master_s"), nd=1), _f(a.get("hold_hit_ref_master"), True),
        _f(a.get("alternation_ok"), True), _f(b["speed_in_range"], True), _f(b["longest_stall_s"], nd=1),
        _f(b["one_to_many_share"], True), _f(r["residual"], nd=3),
        _f(pu["in_range"], True), _f(pu["longest_stall_s"], nd=1), _f(pr["in_range"], True), _f(pr["rush"], True),
        str(pr["jumps"])]) + " |")


HEAD = ("| 方法 | 重复 F1 | 边界误差 s | 次数误差 | 定势命中（练习者为准） | 定势误差 s | 定势命中（标准为准） | 左右交替正确 "
        "| 段内速度比 0.5–2× | 段内最长停滞 s | 一对多（练习者时间） | 骨架残差 | 播放·练习者为准 速度在范围 | 最长停滞 s "
        "| 播放·标准为准 速度在范围 | 快进 >4× | 跳转次数 |\n" + "|" + "---|" * 17)


def _name(key, label, pol):
    return f"{key} {label}" + ("（现行界面播放）" if pol == "ui" else "（循环/跳转播放）" if key == "M1" else "")


def report(results: dict, out: Path, meta: dict) -> str:
    L = ["# 动作对应方法对比（同一输入、同一评价）", "",
         f"生成时间 {meta['created']} · 代码 {meta['commit']} · 机器 {meta['host']} · 姿态 {meta['backend']} 10 fps", ""]
    B = results.get("baduanjin")
    if B:
        L += ["## 八段锦样例（标准 Weixin…221512 ↔ 练习者 Weixin…222517）", "",
              "基准：`references/annotations/baduanjin_demo_a__weixin_222517.yaml`（逐秒看片草标，±1 s，待人工核对）。"
              "评价第一式至第七式的 42 次练习者重复与 12 次标准重复；预备势、第八式、收势只参与播放与残差统计。", "",
              "### 1. 段边界取人工标注（只比较段内对应方法）", "", HEAD]
        for key in ORDER:
            label = M.METHODS[key][0]
            for pol in (("ui", "loop") if key == "M1" else ("loop",)):
                filt = M.METHODS[key][2]
                r = B["runs"].get(f"{key}|gt|{filt}|{pol}")
                if r:
                    L.append(_row(_name(key, label, pol), r))
        L += ["", "列说明：重复 F1 = 练习者重复与标注 IoU ≥ 0.5 的匹配；定势命中 = 标注的定势中心经该方法的播放映射后落在对方任一标注定势区间内（±0.5 s）；"
              "左右交替正确 = 左右交替的式中相邻两次重复对应到标准的不同侧；速度比/停滞/一对多在已配对重复内部统计（一对多 = 落到同一标准采样上 ≥4 个练习者采样的练习者时间占比）；"
              "骨架残差 = 对应帧同体型骨架的平均关节距离（躯干长）；播放统计覆盖整个分析区间。M0 没有“重复”概念，重复类指标为 —。", "",
              "### 2. 端到端（段边界自动）", "", HEAD]
        for kind in ("coarse", "hsmm"):
            for key in ORDER:
                label = M.METHODS[key][0]
                filt = M.METHODS[key][2]
                for pol in (("ui",) if key == "M1" else ("loop",)):
                    r = B["runs"].get(f"{key}|{kind}|{filt}|{pol}")
                    if r:
                        L.append(_row(f"{_name(key, label, pol)} · 段边界={'粗 DTW（现行）' if kind == 'coarse' else 'HSMM（方案 C）'}", r))
        L += ["", "### 3. 段边界精度（练习者视频中各式起点与标注之差）", "", "| 段边界方法 | 平均绝对误差 s | 各式误差 s |", "|---|---|---|"]
        gt = {s["id"]: s for s in load_gt()["sections"]}
        for kind in ("coarse", "hsmm"):
            bd = B["bounds"].get(kind)
            if bd:
                errs = [abs(u0 * M.DT - gt[sid]["user"][0]) for sid, _, _, u0, _ in bd]
                L.append(f"| {'粗 DTW（现行）' if kind == 'coarse' else 'HSMM（方案 C）'} | {np.mean(errs):.1f} | "
                         + " ".join(f"{sid} {e:.0f}" for (sid, *_), e in zip(bd, errs)) + " |")
        L += ["", "### 4. 每式检出的练习者重复次数（标注：第一至第七式各 6 次）", "",
              "| 方法 | " + " | ".join(f"s{i}" for i in range(1, 8)) + " |", "|---|" + "---|" * 7]
        for key in ORDER:
            secs = B["sections"].get(f"{key}|gt|{M.METHODS[key][2]}")
            if secs and key != "M0":
                cnt = {s["id"]: sum(u["paired"] for u in s["user_reps"]) for s in secs}
                tot = {s["id"]: len(s["user_reps"]) for s in secs}
                L.append(f"| {key} | " + " | ".join(f"{cnt.get(f's{i}', 0)}/{tot.get(f's{i}', 0)}" for i in range(1, 8)) + " |")
        L += ["", "格式：配对数/检出数。", "", "### 5. 滤波消融（段边界=标注）", "",
              "| 方法 · 滤波 | 重复 F1 | 定势命中（练习者为准） | 定势命中（标准为准） | 段内速度比 0.5–2× | 骨架残差 |", "|---|---|---|---|---|---|"]
        for key in ("M2a", "M3a", "M3fa"):
            for filt in ("sg", "oneeuro"):
                r = B["runs"].get(f"{key}|gt|{filt}|loop")
                if r:
                    a = r["annotation"]
                    L.append(f"| {key} · {'SG（现行）' if filt == 'sg' else 'One Euro'} | {_f((a['user_reps'] or {}).get('f1'))} | "
                             f"{_f(a['hold_hit_user_master'], True)} | {_f(a['hold_hit_ref_master'], True)} | "
                             f"{_f(r['balance']['speed_in_range'], True)} | {_f(r['residual'], nd=3)} |")
        j = B["jitter"]
        L += ["", f"关键点抖动（中位二阶差分，躯干长/采样²）：练习者 原始 {j['user']['raw']:.4f} · SG {j['user']['sg']:.4f} · One Euro {j['user']['oneeuro']:.4f}；"
              f"标准 原始 {j['ref']['raw']:.4f} · SG {j['ref']['sg']:.4f} · One Euro {j['ref']['oneeuro']:.4f}。", ""]
    D = results.get("dance")
    if D:
        L += ["## 舞蹈对照（chuu ↔ cyves，片段 3–5）", "",
              "外部真值：音频对齐测得 cyves = chuu − 4.55 s（同一段音乐，节奏一致）。误差 = 播放映射与真值之差。", "",
              "| 方法 | 练习者为准 中位误差 s | P90 s | 标准为准 中位误差 s | P90 s | 播放·标准为准 速度在范围 |", "|---|---|---|---|---|---|"]
        for key in ORDER:
            filt = M.METHODS[key][2]
            for pol in (("ui", "loop") if key == "M1" else ("loop",)):
                r = D["runs"].get(f"{key}|gt|{filt}|{pol}")
                if r:
                    o = r["offset"]
                    L.append(f"| {_name(key, M.METHODS[key][0], pol)} | {o['user_master']['median_s']:.2f} | {o['user_master']['p90_s']:.2f} | "
                             f"{o['ref_master']['median_s']:.2f} | {o['ref_master']['p90_s']:.2f} | {_f(r['playback_ref_master']['in_range'], True)} |")
    L += ["", "![timeline](timeline.png)", "", "![playback](playback.png)", ""]
    text = "\n".join(L)
    (out / "report.md").write_text(text, encoding="utf-8")
    return text


def load_gt():
    import yaml
    return yaml.safe_load((settings.ROOT / "references" / "annotations" / "baduanjin_demo_a__weixin_222517.yaml").read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- figures

def figures(pair, results: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.rcParams["font.family"] = ["Noto Sans CJK SC", "Noto Sans CJK TC", "DejaVu Sans"]
    gt = load_gt()
    rows = [k for k in ORDER if k != "M0"]
    fig, ax = plt.subplots(figsize=(18, 1.0 + 0.55 * (len(rows) + 1)))
    cols = ["#2a78d6", "#d9822b"]
    y = len(rows)
    for s in gt["sections"]:
        for q in s.get("user_reps", []):
            ax.add_patch(Rectangle((q[0], y - 0.35), q[1] - q[0], 0.7, fc="#bbbbbb", ec="#555555", lw=0.6))
            ax.add_patch(Rectangle((q[2], y - 0.35), q[3] - q[2], 0.7, fc="#777777", ec="none"))
        ax.text(s["user"][0], y + 0.55, s["id"], fontsize=8, color="#444")
    for i, key in enumerate(rows):
        secs = results["sections"].get(f"{key}|gt|{M.METHODS[key][2]}", [])
        yy = len(rows) - 1 - i
        for s in secs:
            for u in s["user_reps"]:
                c = cols[(u["ref"] or 0) % 2] if u["paired"] else "#ffffff"
                ax.add_patch(Rectangle((u["t"][0], yy - 0.3), u["t"][1] - u["t"][0], 0.6, fc=c, ec="#222", lw=0.6,
                                       alpha=0.85, hatch="//" if u["mirror"] else None))
    ax.set_yticks(range(len(rows) + 1), [f"{k} {M.METHODS[k][0]}" for k in rows[::-1]] + ["标注（深灰=定势）"], fontsize=8)
    ax.set_xlim(0, pair.user.t[-1])
    ax.set_ylim(-0.7, len(rows) + 1)
    ax.set_xlabel("练习者视频时间（秒）；颜色 = 对应到标准的第 1/2 次（蓝/橙），斜纹 = 按镜像侧对应，白 = 未配对")
    fig.tight_layout()
    fig.savefig(out / "timeline.png", dpi=110)
    plt.close(fig)

    # playback map: practitioner time -> reference time shown, with annotated hold rectangles
    show = [("M0", "loop"), ("M1", "ui"), ("M3fa", "loop")]
    fig, axs = plt.subplots(1, len(show), figsize=(18, 5.5), sharey=True)
    for ax, (key, pol) in zip(axs, show):
        filt = M.METHODS[key][2]
        secs = M.METHODS[key][1](pair, M.bounds_gt(pair), filt)
        (U0, u2r), _ = X.playback_ui_m1(pair, secs) if pol == "ui" else X.playback_loop(pair, secs)
        for s in gt["sections"]:
            for qu in s.get("user_reps", []):
                for qr in s.get("ref_reps", []):
                    ax.add_patch(Rectangle((qu[2], qr[2]), qu[3] - qu[2], qr[3] - qr[2], fc="#e8a33d", alpha=0.5, ec="none"))
        ax.plot((U0 + np.arange(len(u2r))) * M.DT, u2r * M.DT, lw=0.8, color="#1b3a5c")
        ax.set_title(_name(key, M.METHODS[key][0], pol), fontsize=10)
        ax.set_xlabel("练习者时间 s")
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("此刻显示的标准时间 s（橙块 = 标注的定势×定势）")
    fig.tight_layout()
    fig.savefig(out / "playback.png", dpi=110)
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(settings.DATA_DIR / "research" / "align_compare"))
    ap.add_argument("--pairs", default="baduanjin,dance")
    ap.add_argument("--backend", default=DEFAULT_BACKEND, help="pose model whose cached poses are compared")
    ap.add_argument("--methods", default="", help="comma-separated subset of methods (default: all)")
    ap.add_argument("--bounds-backend", default="", help="pose model for the automatic section bounds (default: --backend)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.run(["git", "-C", str(settings.ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, check=False).stdout.strip() + "+dirty"
    except OSError:
        commit = "unknown"
    meta = {"created": time.strftime("%Y-%m-%d %H:%M"), "commit": commit, "host": platform.node(), "backend": a.backend + (f" · 段边界 {a.bounds_backend}" if a.bounds_backend else "")}
    results, pairs = {}, {}
    for name in a.pairs.split(","):
        t0 = time.time()
        pairs[name] = load_pair(name, backend=a.backend)
        bp = None
        if a.bounds_backend and a.bounds_backend != a.backend:
            bp = load_pair(name, backend=a.bounds_backend, maps=False)
            assert np.allclose(bp.user.t, pairs[name].user.t) and np.allclose(bp.ref.t, pairs[name].ref.t)
        results[name] = run_pair(pairs[name], set(filter(None, a.methods.split(","))) or None, bp)
        results[name]["load_and_run_s"] = round(time.time() - t0, 1)
        print(f"{name}: {len(results[name]['runs'])} runs in {results[name]['load_and_run_s']} s", flush=True)
    (out / "results.json").write_text(json.dumps({"meta": meta, **results}, ensure_ascii=False, indent=1,
                                                 default=lambda o: o.item() if isinstance(o, np.generic) else str(o)),
                                      encoding="utf-8")
    print(report(results, out, meta))
    if "baduanjin" in pairs:
        figures(pairs["baduanjin"], results["baduanjin"], out)


if __name__ == "__main__":
    main()
