"""Pose extraction with a content-addressed cache.

Cache key = source content hash + normalisation version + model id + sample rate + time window.
The cache stores the *raw* sequence (before mirroring or cleaning). A cached full-length run
also serves any sub-window, so switching analysis intervals never re-runs the model.
"""

from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np

from ..engine.sequence import PoseSequence
from ..perception.base import PoseBackend, run_on_video
from ..perception.mediapipe_tasks import file_digest
from .ingest import NORMALIZE_VERSION, normalize


_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    """One lock per cache file: concurrent jobs sharing a video (the reference) build it once."""
    with _locks_guard:
        return _locks.setdefault(str(path), threading.Lock())


def prepared_video(src: Path, data_dir: Path) -> tuple[str, Path]:
    """Return (content id, normalised proxy path)."""
    vid = file_digest(Path(src))
    proxy = Path(data_dir) / "videos" / f"{vid}_{NORMALIZE_VERSION}.mp4"
    with _lock_for(proxy):
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
    win = "full" if (t0 <= 0 and t1 is None) else f"{t0:.1f}-{t1:.1f}"
    path = poses_dir / f"{_key(video_id, backend, sample_fps, win)}.npz"
    with _lock_for(path):
        if full.exists():
            seq = PoseSequence.load(full)
            return seq if (t0 <= 0 and t1 is None) else slice_window(seq, t0, t1)
        if path.exists():
            return PoseSequence.load(path)
        seq = run_on_video(backend, proxy, sample_fps, t0=max(0.0, t0), t1=t1, progress=progress)
        seq.meta.update({"video_id": video_id, "cache_key": path.stem})
        seq.save(path)
    return seq


def subsample(seq: PoseSequence, step: int) -> PoseSequence:
    """Every ``step``-th sample (a coarse view of a cached fine run, e.g. for locating the analysis interval)."""
    if step <= 1:
        return seq
    world = seq.world[::step] if seq.world is not None else None
    return replace(seq, t=seq.t[::step], xy=seq.xy[::step], conf=seq.conf[::step], world=world,
                   meta={**seq.meta, "subsample": step})


def extract_multi(proxy: Path, video_id: str, backend, sample_fps: float, data_dir: Path,
                  progress: Callable[[float], None] | None = None):
    """Every person in every sampled frame of the full video (``perception.multi``), cached like single poses."""
    from ..perception.multi import MultiPoses, run_multi_on_video

    path = Path(data_dir) / "poses" / f"{_key(video_id, backend, sample_fps, 'full')}_multi.npz"
    with _lock_for(path):
        if path.exists():
            return MultiPoses.load(path)
        mp = run_multi_on_video(backend, proxy, sample_fps, progress=progress)
        mp.meta.update({"video_id": video_id, "cache_key": path.stem})
        mp.save(path)
    return mp
