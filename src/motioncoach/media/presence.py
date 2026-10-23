"""Quick "is anyone in this video?" check, run before any heavy work.

ffmpeg fast-seeks to ``n`` evenly spaced moments of the *original* file (rotation applied, scaled to
640 px), and the lightest pose model looks for a person whose torso (both shoulders and hips) is
visible - the minimum every later step needs. A 4K/60 fps phone clip costs a few seconds instead of a
full transcode plus a full-length pose pass that would only fail at the end.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

from ..engine.errors import NoPersonError
from .ingest import _require, probe

TORSO = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
MIN_FRAMES = 3          # fewer frames with a visible torso than this -> not analysable
WARN_SHARE = 0.6        # below this share the analysis runs, with a warning


def _frame_at(src: Path, t: float, height: int = 640) -> np.ndarray | None:
    out = subprocess.run(
        [_require("ffmpeg"), "-v", "error", "-ss", f"{t:.3f}", "-i", str(src), "-frames:v", "1",
         "-vf", f"scale=-2:'min({height},ih)'", "-f", "image2pipe", "-vcodec", "png", "-"],
        capture_output=True, timeout=60)
    if out.returncode != 0 or not out.stdout:
        return None
    img = cv2.imdecode(np.frombuffer(out.stdout, np.uint8), cv2.IMREAD_COLOR)
    return img


def person_presence(src: Path, backend, n: int = 16) -> dict:
    """{"checked", "with_person", "share", "times"} for ``n`` frames spread over the video."""
    dur = float(probe(Path(src)).get("duration") or 0.0)
    times = [dur * (k + 0.5) / n for k in range(n)] if dur > 0 else [0.0]
    sk_idx = None
    hits, checked = [], 0
    backend.open()
    try:
        for k, t in enumerate(times):
            img = _frame_at(Path(src), t)
            if img is None:
                continue
            checked += 1
            xy, cf, _ = backend.estimate(img, int(1000 * t) + k)
            if xy is None or cf is None:
                continue
            if sk_idx is None:
                from ..engine.skeleton import get_skeleton
                sk = get_skeleton(backend.skeleton_name)
                sk_idx = [sk.index[j] for j in TORSO]
            if (cf[sk_idx] >= backend.conf_threshold).all() and np.isfinite(xy[sk_idx]).all():
                hits.append(round(t, 1))
    finally:
        backend.close()
    return {"checked": checked, "with_person": len(hits), "share": len(hits) / max(1, checked), "times": hits,
            "duration": dur}


def require_person(src: Path, backend, n: int = 16) -> dict:
    """Raise ``NoPersonError`` (a user-facing outcome, not a crash) when almost no frame shows a person."""
    p = person_presence(src, backend, n)
    if p["checked"] == 0:
        raise NoPersonError("无法读取视频画面，请确认文件完整、格式为常见手机视频（MP4/MOV）。", presence=p)
    if p["with_person"] < MIN_FRAMES:
        raise NoPersonError(
            f"视频中未检测到人物：均匀抽查 {p['checked']} 个画面，只有 {p['with_person']} 个能看到完整躯干（双肩与双髋）。"
            "请确认练习者在画面中、全身入画、光线充足，然后重新上传。", presence=p)
    return p
