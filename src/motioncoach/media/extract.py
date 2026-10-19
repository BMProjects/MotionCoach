"""Pose extraction with a content-addressed cache.

Cache key = source content hash + normalisation version + model id + sample rate + time window.
The cache stores the *raw* sequence (before mirroring or cleaning). A cached full-length run
also serves any sub-window, so switching analysis intervals never re-runs the model.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np

from ..engine.sequence import PoseSequence
from ..perception.base import PoseBackend, run_on_video
from ..perception.mediapipe_tasks import file_digest
from .ingest import NORMALIZE_VERSION, normalize


def prepared_video(src: Path, data_dir: Path) -> tuple[str, Path]:
    """Return (content id, normalised proxy path)."""
    vid = file_digest(Path(src))
    proxy = Path(data_dir) / "videos" / f"{vid}_{NORMALIZE_VERSION}.mp4"
    normalize(Path(src), proxy)
    return vid, proxy


def _key(video_id: str, backend: PoseBackend, fps: float, win: str) -> str:
    tag = backend.model_id.replace("/", "_").replace("@", "_").replace(":", "_")
    return f"{video_id}_{NORMALIZE_VERSION}_g2_{tag}_{fps:g}fps_{win}"   # g2 = absolute k/fps sampling grid


def slice_window(seq: PoseSequence, t0: float, t1: float | None) -> PoseSequence:
    m = (seq.t >= t0) & (seq.t < (np.inf if t1 is None else t1))
    world = seq.world[m] if seq.world is not None else None
    meta = {**seq.meta, "window": [t0, t1]}
    return replace(seq, t=seq.t[m], xy=seq.xy[m], conf=seq.conf[m], world=world, meta=meta)


def extract_poses(proxy: Path, video_id: str, backend: PoseBackend, sample_fps: float, data_dir: Path,
                  t0: float = 0.0, t1: float | None = None,
                  progress: Callable[[float], None] | None = None) -> PoseSequence:
    poses_dir = Path(data_dir) / "poses"
    full = poses_dir / f"{_key(video_id, backend, sample_fps, 'full')}.npz"
    if full.exists():
        seq = PoseSequence.load(full)
        return seq if (t0 <= 0 and t1 is None) else slice_window(seq, t0, t1)
    win = "full" if (t0 <= 0 and t1 is None) else f"{t0:.1f}-{t1:.1f}"
    path = poses_dir / f"{_key(video_id, backend, sample_fps, win)}.npz"
    if path.exists():
        return PoseSequence.load(path)
    seq = run_on_video(backend, proxy, sample_fps, t0=max(0.0, t0), t1=t1, progress=progress)
    seq.meta.update({"video_id": video_id, "cache_key": path.stem})
    seq.save(path)
    return seq
