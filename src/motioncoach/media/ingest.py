"""Video ingest (parameters: docs/VIDEO_SPEC.md).

Every input is decoded once (GPU when available) into two files with the same orientation, frame
rate and timebase, so analysis and display cannot diverge:

- analysis master  ``<id>_n3.mp4``: short side <= 1080, constant frame rate (30 fps when the source
  is faster), 1 s keyframes, limited-range yuv420p, near-transparent quality - pose estimation input;
- display proxy    ``<id>_n3_view.mp4``: short side <= 720, <= 2.5 Mb/s, 0.5 s keyframes, faststart,
  mono audio - what the browser streams and seeks in (overlays are scaled by frame size).

All of these are upper limits (n3): size, frame rate and bit rate never exceed the source's - a 1 Mb/s
source stays about 1 Mb/s instead of being re-encoded at a quality it never had.
HEVC / HDR / variable-frame-rate phone videos and rotation metadata are the cases this protects against.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

NORMALIZE_VERSION = "n3"            # n3: limits capped at the source; n2: CFR + display proxy; n1: single VFR proxy
MAX_SHORT_SIDE = 1080                # analysis master
VIEW_SHORT_SIDE = 720                # display proxy
MAX_FPS = 30
VIEW_MAXRATE = 2_500_000             # display proxy ceiling, b/s
# H.264 needs roughly this much more bit rate than the source codec for the same quality, so the
# re-encode is capped at source_rate * factor (HEVC / VP9 / AV1 sources) rather than at the source rate.
CODEC_FACTOR = {"hevc": 1.5, "vp9": 1.5, "av1": 1.7}


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
    a = next((s for s in info["streams"] if s.get("codec_type") == "audio"), None)
    a_rate = int(a["bit_rate"]) if a and str(a.get("bit_rate", "")).isdigit() else None
    v_rate = int(v["bit_rate"]) if str(v.get("bit_rate", "")).isdigit() else None
    if v_rate is None and str(info["format"].get("bit_rate", "")).isdigit():   # e.g. MKV/WebM: no per-stream rate
        v_rate = max(1, int(info["format"]["bit_rate"]) - (a_rate or 0))
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
        "has_audio": a is not None,
        "video_bitrate": v_rate, "audio_bitrate": a_rate,       # b/s, None when unknown
        "size_bytes": int(info["format"].get("size", 0)),
    }


_NVENC: bool | None = None


def nvenc_available() -> bool:
    """True when ffmpeg can actually encode with NVENC here (checked once with a tiny test encode)."""
    global _NVENC
    if _NVENC is None:
        try:
            r = subprocess.run([_require("ffmpeg"), "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.2",
                                "-c:v", "h264_nvenc", "-f", "null", "-"], capture_output=True, timeout=30)
            _NVENC = r.returncode == 0
        except (OSError, subprocess.SubprocessError):
            _NVENC = False
    return _NVENC


def view_path(master: Path) -> Path:
    """Display proxy belonging to an analysis master."""
    return Path(master).with_name(Path(master).stem + "_view.mp4")


def _short_side(n: int) -> str:
    return f"scale='if(lt(iw,ih),min(iw,{n}),-2)':'if(lt(iw,ih),-2,min(ih,{n}))'"


def normalize(src: Path, dst: Path) -> Path:
    """Analysis master ``dst`` + display proxy ``view_path(dst)`` from one decode. Idempotent."""
    dst = Path(dst)
    view = view_path(dst)
    if dst.exists() and view.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    info = probe(Path(src))
    fps = info["fps"] or MAX_FPS
    rate = MAX_FPS if fps > MAX_FPS + 1 else max(1, round(fps))           # 60 -> 30; 29.97 VFR -> 30 CFR
    # ceilings relative to the source: never spend more bits than the source had (with codec allowance)
    src_rate = info.get("video_bitrate")
    cap = int(src_rate * CODEC_FACTOR.get(info["codec"], 1.0)) if src_rate else None
    cap_v = min(VIEW_MAXRATE, cap) if cap else VIEW_MAXRATE
    # a source already within the display limits would give two near-identical files: encode one with
    # display keyframes and hard-link it as the display proxy
    single = min(info["width"], info["height"]) <= VIEW_SHORT_SIDE and cap is not None and cap <= VIEW_MAXRATE
    gop_m, gop_v = str(max(1, rate // 2) if single else rate), str(max(1, rate // 2))   # 1 s / 0.5 s keyframes
    graph = (f"[0:v]fps={rate},{_short_side(MAX_SHORT_SIDE)}:out_range=tv,format=yuv420p" +
             ("[m]" if single else f",split=2[m][v];[v]{_short_side(VIEW_SHORT_SIDE)}[vv]"))
    rate_m = ["-maxrate", str(cap), "-bufsize", str(2 * cap)] if cap else []
    rate_v = ["-maxrate", str(cap_v), "-bufsize", str(2 * cap_v)]
    a_src = info.get("audio_bitrate")
    a_m, a_v = (str(min(128_000, a_src)), str(min(80_000, a_src))) if a_src else ("128k", "80k")
    gpu = nvenc_available()
    if gpu:
        enc_m = ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "21", "-b:v", "0", *rate_m, "-g", gop_m]
        enc_v = ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "27", "-b:v", "0", *rate_v, "-g", gop_v]
    else:
        enc_m = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", *rate_m,
                 "-g", gop_m, "-keyint_min", gop_m, "-sc_threshold", "0"]
        enc_v = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "27", *rate_v,
                 "-g", gop_v, "-keyint_min", gop_v, "-sc_threshold", "0"]
    tag = f".part{os.getpid()}.mp4"                  # per process: CLI runs and services share the cache
    tm, tv = dst.with_suffix(tag), view.with_suffix(tag)
    common = ["-color_range", "tv", "-map_metadata", "-1", "-movflags", "+faststart"]

    def cmd(hw: list[str]) -> list[str]:
        out = [_require("ffmpeg"), "-v", "error", "-y", *hw, "-i", str(src), "-filter_complex", graph,
               "-map", "[m]", "-map", "0:a?", *enc_m, "-c:a", "aac", "-b:a", a_m, *common, str(tm)]
        if not single:
            out += ["-map", "[vv]", "-map", "0:a?", *enc_v, "-c:a", "aac", "-b:a", a_v, "-ac", "1", *common, str(tv)]
        return out

    try:   # GPU decode (NVDEC) first: 4K HEVC decoding is the slow part on CPU
        subprocess.run(cmd(["-hwaccel", "cuda"] if gpu else []), check=True, capture_output=True)
    except subprocess.CalledProcessError:
        if not gpu:
            raise
        subprocess.run(cmd([]), check=True, capture_output=True)
    tm.replace(dst)
    if single:
        view.unlink(missing_ok=True)
        try:
            os.link(dst, view)
        except OSError:          # filesystem without hard links
            shutil.copyfile(dst, view)
    else:
        tv.replace(view)
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
