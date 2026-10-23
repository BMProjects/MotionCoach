"""Figures for the expert memo (docs/EXPERT_MEMO_PHASE1.html) from the saved experiment results.

    python -m motioncoach.research.memo_figures [--data data/remote] [--out docs/figures]

Inputs (pulled from the GPU server): cleaning-ablation/results.json, recapture-20261021-results.json,
vitpose-trial/results.json, smoothing-sweep/results.json (optional). The figures hold numbers only, no people.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SEC = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]
SEC_ZH = ["一", "二", "三", "四", "五", "六", "七", "八"]
# Okabe-Ito (colour-blind safe), fixed order; grey for the baseline
C = {"grey": "#8c8c8c", "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "pink": "#CC79A7", "sky": "#56B4E9"}
INK, INK2, RULE = "#1d2622", "#5b6660", "#e3e7e4"


def _style(plt):
    plt.rcParams.update({"font.family": ["Noto Sans CJK SC", "Noto Sans CJK TC", "DejaVu Sans"], "font.size": 10.5,
                         "axes.edgecolor": RULE, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
                         "grid.color": RULE, "grid.linewidth": 0.8, "axes.titlesize": 11.5, "axes.titleweight": "bold",
                         "axes.titlecolor": INK, "axes.titlelocation": "left", "legend.frameon": False, "savefig.dpi": 160})


def _bars(ax, groups: list[str], series: list[tuple[str, list[float], str]], fmt="{:.2f}", label_idx=()):
    n = len(series)
    w = 0.8 / n
    x = np.arange(len(groups))
    for i, (name, vals, col) in enumerate(series):
        xs = x - 0.4 + w * (i + 0.5)
        ax.bar(xs, vals, w * 0.86, color=col, label=name, zorder=2)
        for j in label_idx:
            ax.text(xs[j], vals[j], fmt.format(vals[j]), ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.set_xticks(x, groups)
    ax.grid(axis="x", visible=False)


def ablation(data: Path, out: Path, plt) -> Path:
    r = json.loads((data / "cleaning-ablation" / "results.json").read_text())
    rc, rm = r["recapture"], r["removed"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.9), gridspec_kw={"width_ratios": [1.5, 1]})
    st = [("模型原始输出", "raw", C["grey"]), ("+ 骨长过滤（新）", "bone_new", C["blue"]),
          ("+ 平滑（只计观测点）", "full_observed", C["orange"]), ("+ 平滑 + 补缺（现行）", "full", C["pink"])]
    _bars(a1, [f"第{z}式" for z in SEC_ZH], [(n, [rc[k][s]["joint_p90"] for s in SEC], c) for n, k, c in st])
    a1.set_title("翻拍检验：各式关键点误差 P90（躯干长，越低越好）")
    a1.legend(ncol=2, fontsize=9, loc="upper left")
    a1.set_ylim(0, 0.17)
    src = ("recapture_reference", "recapture_practitioner", "sample_practitioner")
    old = [100 * np.mean([rm[s]["bone_old"][k] for s in src]) for k in SEC]
    new = [100 * np.mean([rm[s]["bone_new"][k] for s in src]) for k in SEC]
    _bars(a2, SEC_ZH, [("旧规则（中位骨长）", old, C["grey"]), ("新规则（较长投影）", new, C["blue"])], fmt="{:.1f}%", label_idx=(4, 5))
    a2.set_title("骨长过滤删掉的置信关节比例（%）")
    a2.set_xlabel("动作（第 N 式）")
    a2.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    p = out / "memo_cleaning_ablation.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def recapture_sections(data: Path, out: Path, plt) -> Path:
    r = json.loads((data / "recapture-20261021-results.json").read_text())["configs"]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    models = [("MediaPipe full（轻量）", "mediapipe-full", C["orange"], "o"), ("RTMW-l（现用）", "rtmw-l", C["blue"], "s"),
              ("RTMW-x（大模型）", "rtmw-x", C["green"], "^")]
    x = np.arange(len(SEC))
    for name, k, col, mk in models:
        v = [r[k]["per_seg"][s]["skel_p90"] for s in SEC]
        ax.plot(x, v, color=col, lw=2, marker=mk, ms=7, label=name, zorder=3)
    ax.set_xticks(x, [f"第{z}式" for z in SEC_ZH])
    ax.grid(axis="x", visible=False)
    ax.axvspan(3.5, 5.5, color="#f3e6e6", zorder=0)
    ax.text(4.5, ax.get_ylim()[0] + 0.004, "大幅俯身 / 摇转", ha="center", va="bottom", fontsize=9, color=INK2)
    ax.set_title("翻拍检验：同一动作被测成“不同”的程度（骨架误差 P90，躯干长）")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    p = out / "memo_recapture_sections.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def hard_routes(data: Path, out: Path, plt) -> Path:
    """Tail error of every candidate route so far: form 5, form 6 and the mean of the six calmer forms."""
    ab = json.loads((data / "cleaning-ablation" / "results.json").read_text())["recapture"]
    vt = json.loads((data / "vitpose-trial" / "results.json").read_text())
    rows = [("RTMW-l 原始输出", ab["raw"], C["grey"]),
            ("RTMW-l + 原清洗（无跟踪）", ab["full"], C["pink"])]
    sw = data / "smoothing-sweep" / "results.json"
    if sw.exists():
        from .smoothing_sweep import CANDIDATE
        best = next(r for r in json.loads(sw.read_text())["rows"] if r["params"] == CANDIDATE)
        rows.append(("RTMW-l + 候选清洗参数", best["scores"], C["sky"]))
    db = data / "detector-bench" / "results.json"
    if db.exists():                                        # production since 2026-10-23 (ITERATION_LOG I18)
        rows.append(("RTMW-l + 人物框跟踪（现用）", json.loads(db.read_text())["native+track"]["recapture"]["full"], C["blue"]))
    rows.append(("ViTPose++-B 原始输出", vt["vitpose++"]["raw"], C["grey"]))
    ot = data / "offline-trial" / "results.json"
    if ot.exists():
        o = json.loads(ot.read_text())
        if "gemx-2d" in o:
            rows.append(("GEM-X 二维（ViTPose-H）", o["gemx-2d"]["raw"], C["orange"]))
        if "gemx-3d-all" in o:
            rows.append(("GEM-X 三维运动（投影）", o["gemx-3d-all"]["raw"], C["green"]))
        if "sam3db" in o:
            rows.append(("SAM 3D Body", o["sam3db"]["raw"], C["green"]))
    calm = ("s1", "s2", "s3", "s4", "s7", "s8")
    panels = (("s5", "第五式"), ("s6", "第六式"), (calm, "其余六式平均"))
    fig, axes = plt.subplots(1, 3, figsize=(13, 0.5 * len(rows) + 1.5), sharey=True)
    for ax, (s, title) in zip(axes, panels):
        vals = [float(np.mean([r[1][x]["joint_p90"] for x in s])) if isinstance(s, tuple) else r[1][s]["joint_p90"] for r in rows]
        y = np.arange(len(rows))[::-1]
        ax.barh(y, vals, 0.62, color=[r[2] for r in rows], zorder=2)
        for yy, v in zip(y, vals):
            ax.text(v + 0.002, yy, f"{v:.3f}", va="center", fontsize=9, color=INK2)
        ax.set_yticks(y, [r[0] for r in rows])
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, 0.17)
        ax.set_title(f"{title}：误差 P90（躯干长）")
    fig.tight_layout()
    p = out / "memo_hard_routes.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/remote")
    ap.add_argument("--out", default="docs/figures")
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in (ablation, recapture_sections, hard_routes):
        print(f(data, out, plt))


if __name__ == "__main__":
    main()
