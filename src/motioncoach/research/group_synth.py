"""Synthetic multi-person video with a known answer, and its evaluation.

Composite (1920x1080, 30 fps), built with ffmpeg from the reference and the sample practitioner videos:

    P1  reference                         look-alike clothing (P1-P3 are the same demonstrator)
    P2  reference, 1.5 s late             -> sync offset P2 - P1 = +1.5 s
    P3  reference at 1.1x speed           -> P3 - P1 = -0.091 x reference time (increasingly ahead)
    P4  practitioner video, further back  (different person, own tempo)
    S1, S2  still frames, small, at the back: onlookers that must be screened out
    a grey bar sweeps across everybody between 40 and 48 s (each person occluded ~2 s)

    python -m motioncoach.research.group_synth build OUT.mp4
    python -m motioncoach.research.group_synth eval GROUP_DIR      # after `motioncoach group --video OUT.mp4`
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from .. import settings
from ..apps.assets import get_reference

W, H, FPS, DUR = 1920, 1080, 30, 150
# (name, x, y, height) of each placed video; feet at y + height
LAYOUT = {"P1": (60, 330, 720), "P2": (520, 330, 720), "P3": (980, 330, 720), "P4": (1480, 400, 560),
          "S1": (40, 40, 260), "S2": (1650, 60, 240)}
PRACTICE = "Weixin Videos2026-10-18_222517_138.mp4"


def _width(video: Path, height: int) -> int:   # displayed size (rotation applied)
    from ..media.ingest import probe
    info = probe(Path(video))
    return int(round(info["width"] * height / info["height"] / 2)) * 2


def ground_truth() -> dict:
    ref = get_reference("baduanjin_demo_a", settings.REFERENCES_DIR, settings.ROOT)
    out = {}
    for name, (x, y, h) in LAYOUT.items():
        src = ref.video if name in ("P1", "P2", "P3", "S2") else settings.ASSETS_DIR / PRACTICE
        out[name] = {"foot_x": x + _width(src, h) / 2, "foot_y": y + h, "performer": name.startswith("P")}
    return out


def build(out: Path) -> Path:
    ref = get_reference("baduanjin_demo_a", settings.REFERENCES_DIR, settings.ROOT).video
    prac = settings.ASSETS_DIR / PRACTICE
    L = LAYOUT
    graph = (
        f"color=c=0x6e6e6e:s={W}x{H}:r={FPS}:d={DUR}[bg];"
        f"[0:v]scale=-2:{L['P1'][2]},setpts=PTS-STARTPTS[p1];"
        f"[1:v]scale=-2:{L['P2'][2]},tpad=start_duration=1.5:start_mode=clone,setpts=PTS-STARTPTS[p2];"
        f"[2:v]setpts=(PTS-STARTPTS)/1.1,scale=-2:{L['P3'][2]}[p3];"
        f"[3:v]scale=-2:{L['P4'][2]},setpts=PTS-STARTPTS[p4];"
        f"[4:v]scale=-2:{L['S1'][2]}[s1];[5:v]scale=-2:{L['S2'][2]}[s2];"
        f"[bg][p1]overlay={L['P1'][0]}:{L['P1'][1]}:eof_action=pass[a];"
        f"[a][p2]overlay={L['P2'][0]}:{L['P2'][1]}:eof_action=pass[b];"
        f"[b][p3]overlay={L['P3'][0]}:{L['P3'][1]}:eof_action=pass[c];"
        f"[c][p4]overlay={L['P4'][0]}:{L['P4'][1]}:eof_action=pass[d];"
        f"[d][s1]overlay={L['S1'][0]}:{L['S1'][1]}[e];[e][s2]overlay={L['S2'][0]}:{L['S2'][1]}[f];"
        f"color=c=0x404040:s=240x780:r={FPS}:d=8,setpts=PTS+40/TB[bar];"
        f"[f][bar]overlay=x='-260+(t-40)*260':y=300:eof_action=pass[v]"
    )
    stills = []
    for src, at, tag in ((prac, 5, "s1"), (ref, 2, "s2")):    # onlookers: one frame, held
        png = Path(out).with_name(f"{Path(out).stem}_{tag}.png")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(at), "-i", str(src), "-frames:v", "1", str(png)], check=True)
        stills += ["-loop", "1", "-t", str(DUR), "-i", str(png)]
    cmd = ["ffmpeg", "-v", "error", "-y",
           "-i", str(ref), "-i", str(ref), "-i", str(ref), "-ss", "20", "-i", str(prac), *stills,
           "-filter_complex", graph, "-map", "[v]", "-t", str(DUR), "-r", str(FPS),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p", str(out)]
    subprocess.run(cmd, check=True)
    (Path(out).with_suffix(".truth.json")).write_text(json.dumps(ground_truth(), indent=1))
    return Path(out)


def evaluate(group_dir: Path, truth: dict) -> dict:
    g = json.loads((Path(group_dir) / "group.json").read_text(encoding="utf-8"))
    frames = np.load(Path(group_dir) / "identity_frames.npy")
    names = list(truth)
    gx = np.array([truth[n]["foot_x"] for n in names])
    gy = np.array([truth[n]["foot_y"] for n in names])
    match = {}
    for i in g["identities"]:
        d = np.hypot(gx - i["foot_px"][0], (gy - i["foot_px"][1]) * 0.5)
        match[i["id"]] = names[int(np.argmin(d))]
    perf_found = sorted(match[i["id"]] for i in g["identities"] if i["performer"])
    per_truth = {n: [i["id"] for i in g["identities"] if match[i["id"]] == n] for n in names}
    T = g["video"]["samples"]
    coverage = {n: round(float(sum(int((frames[:, 0] == k).sum()) for k in per_truth[n]) / T), 3) for n in names}
    sync = g["group"].get("sync", {})
    off = {match[int(k)]: v["offset_median_s"] for k, v in sync.items()}
    by_seg = {match[int(k)]: v["offset_by_segment_s"] for k, v in sync.items()}
    rel = {}
    if "P1" in off:
        for n in ("P2", "P3", "P4"):
            if n in off and off[n] is not None and off["P1"] is not None:
                rel[n] = round(off[n] - off["P1"], 2)
    p3 = {}
    if "P1" in by_seg and "P3" in by_seg:                 # expected -0.091 x reference time of the section
        ref = get_reference("baduanjin_demo_a", settings.REFERENCES_DIR, settings.ROOT)
        mids = {s["id"]: (s["start"] + s["end"]) / 2 for s in ref.raw["segments"]}
        p3 = {s: {"measured": round(by_seg["P3"][s] - by_seg["P1"][s], 2), "expected": round(-mids[s] * (1 - 1 / 1.1), 2)}
              for s in by_seg["P3"] if s in by_seg["P1"] and s in mids}
    return {"performers_expected": sorted(n for n in names if truth[n]["performer"]), "performers_found": perf_found,
            "identities_per_person": {n: len(v) for n, v in per_truth.items()}, "coverage": coverage,
            "sync_relative_to_P1_s": rel, "sync_expected": {"P2": 1.5, "P3": "-0.091 x 参考时间（见分段）"},
            "P3_by_segment": p3, "persons": g["persons"], "timings_s": g["timings_s"]}


if __name__ == "__main__":
    cmd, arg = sys.argv[1], Path(sys.argv[2])
    if cmd == "build":
        print(build(arg))
    else:
        truth = json.loads(Path(sys.argv[3]).read_text()) if len(sys.argv) > 3 else ground_truth()
        print(json.dumps(evaluate(arg, truth), ensure_ascii=False, indent=1))
