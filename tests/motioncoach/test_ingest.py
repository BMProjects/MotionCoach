"""Ingest limits are ceilings: a low-quality source is never re-encoded into a bigger file."""

import shutil
import subprocess

import pytest

from motioncoach.media.ingest import normalize, probe, view_path

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def test_low_bitrate_source_is_not_inflated(tmp_path):
    src = tmp_path / "src.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=320x568:r=30:d=4",
                    "-c:v", "libx264", "-b:v", "300k", "-pix_fmt", "yuv420p", str(src)], check=True)
    info = probe(src)
    out = normalize(src, tmp_path / "x_n3.mp4")
    got = probe(out)
    assert (got["width"], got["height"]) == (320, 568) and round(got["fps"]) == 30   # no upscaling
    # capped at the source: maxrate is a VBV peak, so the average may sit a little above it on a 4 s clip
    # (measured 1.151x with x264; the inflation this guards against was 2-3x)
    assert got["video_bitrate"] <= 1.25 * info["video_bitrate"]
    view = view_path(out)
    assert view.exists() and view.stat().st_ino == out.stat().st_ino                # small source: one file, linked
