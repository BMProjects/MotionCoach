"""Video ingest.

Every input is normalised once into a proxy (H.264/AAC MP4, rotation baked in, faststart,
original timestamps kept). Pose estimation, playback and overlays all use the same proxy,
so orientation and timebase cannot diverge between analysis and display. HEVC phone videos
and rotation metadata are the common cases this protects against.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

NORMALIZE_VERSION = "n1"
MAX_SHORT_SIDE = 1080


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        raise RuntimeError(f"'{tool}' not found on PATH (install ffmpeg)")
    return path


def probe(path: Path) -> dict:
    out = subprocess.run(
        [_require("ffprobe"), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    info = json.loads(out)
    v = next((s for s in info["streams"] if s.get("codec_type") == "video"), None)
    if v is None:
        raise ValueError(f"No video stream in {path}")
    rotation = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rotation = int(sd["rotation"])
    rotation = int(v.get("tags", {}).get("rotate", rotation))
    w, h = int(v["width"]), int(v["height"])
    if rotation % 180:
        w, h = h, w
    num, den = (v.get("avg_frame_rate") or "0/1").split("/")
    return {
        "codec": v.get("codec_name"),
        "width": w, "height": h,           # displayed size (rotation applied)
        "rotation": rotation,
        "fps": float(num) / float(den) if float(den) else None,
        "duration": float(info["format"].get("duration", 0.0)),
        "has_audio": any(s.get("codec_type") == "audio" for s in info["streams"]),
        "size_bytes": int(info["format"].get("size", 0)),
    }


def normalize(src: Path, dst: Path) -> Path:
    """Transcode to the analysis/playback proxy. Idempotent if ``dst`` exists."""
    dst = Path(dst)
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part.mp4")
    scale = f"scale='if(lt(iw,ih),min(iw,{MAX_SHORT_SIDE}),-2)':'if(lt(iw,ih),-2,min(ih,{MAX_SHORT_SIDE}))'"
    cmd = [
        _require("ffmpeg"), "-v", "error", "-y", "-i", str(src),
        "-vf", scale + ",format=yuv420p",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-fps_mode", "passthrough",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart", "-map_metadata", "-1",
        str(tmp),
    ]
    subprocess.run(cmd, check=True)
    tmp.replace(dst)
    return dst


def scene_cuts(proxy: Path, threshold: float = 0.35) -> list[float]:
    """Timestamps (s) of hard cuts, via ffmpeg scene detection; cached next to the proxy."""
    cache = Path(proxy).with_suffix(f".cuts{int(threshold * 100)}.json")
    if cache.exists():
        return json.loads(cache.read_text())
    out = subprocess.run(
        [_require("ffmpeg"), "-v", "error", "-i", str(proxy), "-an", "-vf",
         f"scale=160:-2,select='gt(scene,{threshold})',metadata=print:file=-", "-f", "null", "-"],
        capture_output=True, text=True, check=True,
    ).stdout
    cuts = [float(line.split("pts_time:")[1].split()[0]) for line in out.splitlines() if "pts_time:" in line]
    cache.write_text(json.dumps(cuts))
    return cuts
