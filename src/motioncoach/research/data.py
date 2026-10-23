"""Shared inputs for the comparison: full-length poses, both filters, features, pose maps, annotation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .. import settings
from ..apps.assets import get_reference, load_profile
from ..engine.features import compute_features, lr_swap_features
from ..engine.normalize import bone_lengths, normalized
from ..engine.sequence import PoseSequence, clean
from ..media.extract import extract_poses, prepared_video
from ..media.ingest import probe
from ..perception.registry import DEFAULT_BACKEND, get_backend
from . import posemap
from .filters import jitter, one_euro_clean


@dataclass
class Side:
    t: np.ndarray
    raw: PoseSequence
    seq: dict[str, PoseSequence]            # filter name -> cleaned sequence (user: anatomical left/right)
    F: dict[str, np.ndarray]                # filter -> (T, K) features, user already left/right corrected
    N: dict[str, np.ndarray]                # filter -> (T, 13, 2) same-body normalised skeleton
    M: dict[str, np.ndarray]                # filter -> (T, P) pose maps


@dataclass
class Pair:
    name: str
    ref: Side
    user: Side
    names: list[str]
    scales: np.ndarray
    swap: bool
    ref_sections: list[tuple[str, float, float]]   # (id, start, end) reference seconds
    gt: dict | None = None                          # annotation (sections with reps) or None
    gt_offset: float | None = None                  # dance: user time = ref time + offset
    jitter: dict = field(default_factory=dict)
    video: dict = field(default_factory=dict)


FILTERS = ("sg", "oneeuro")


def _side(raw: PoseSequence, unmirror: bool, names: list[str], target_bones=None, unit=None, maps: bool = True) -> Side:
    base = raw.unmirror() if unmirror else raw
    thr = base.meta.get("conf_thr", 0.5)
    seqs = {"sg": clean(base, conf_thr=thr)[0], "oneeuro": one_euro_clean(base, conf_thr=thr),
            "raw": clean(base, conf_thr=thr, max_gap_s=0.0, smooth_window_s=0.0)[0]}   # masked only (noise studies)
    F, N, M = {}, {}, {}
    for k, s in seqs.items():
        F[k] = compute_features(s, names)[0]
        own = bone_lengths(s)
        tb = target_bones or own
        N[k] = normalized(s, tb, own, unit or tb[("hip_mid", "shoulder_mid")])
        if maps and k != "raw":
            M[k] = posemap.render(N[k])
    return Side(raw.t, raw, seqs, F, N, M)


def load_pair(name: str, fps: float = 10.0, backend: str = DEFAULT_BACKEND, maps: bool = True) -> Pair:
    """``backend``: pose model whose (cached) full-length poses are used; ``maps``: also render pose maps."""
    if name == "baduanjin":
        ann_path = settings.ROOT / "references" / "annotations" / "baduanjin_demo_a__weixin_222517.yaml"
        ann = yaml.safe_load(ann_path.read_text(encoding="utf-8"))
        ref = get_reference(ann["reference"], settings.REFERENCES_DIR, settings.ROOT)
        user_video = settings.ASSETS_DIR / ann["user_video"]
        swap = bool(ann.get("user_mirrored", False))
        sections = [(s["id"], float(s["ref"][0]), float(s["ref"][1])) for s in ann["sections"]]
        gt, offset = ann, None
    elif name == "dance":
        ref = get_reference("dance_demo_chuu", settings.REFERENCES_DIR, settings.ROOT)
        user_video = settings.ASSETS_DIR / "cyves.mov"
        swap = False
        prof = load_profile(ref.raw["profile"])
        sections = [(s.id, s.start, s.end) for s in ref.segments(prof) if s.start >= 5.8]   # cyves lacks the first 4.55 s
        gt, offset = None, -4.55
    else:
        raise ValueError(name)
    profile = load_profile(ref.raw["profile"])
    names = profile.features
    scales = np.array([profile.tolerance(n)["good"] for n in names], dtype=np.float64)
    be = get_backend(backend, settings.MODELS_DIR)
    rid, rpx = prepared_video(ref.video, settings.DATA_DIR)
    uid, upx = prepared_video(user_video, settings.DATA_DIR)
    r_raw = extract_poses(rpx, rid, be, fps, settings.DATA_DIR)          # full length, cached
    u_raw = extract_poses(upx, uid, be, fps, settings.DATA_DIR)
    R = _side(r_raw, bool(ref.raw.get("mirrored_video")), names, maps=maps)
    rb = bone_lengths(R.seq["sg"])
    U = _side(u_raw, swap, names, target_bones=rb, unit=rb[("hip_mid", "shoulder_mid")], maps=maps)
    if offset is not None:                     # keep sections whose counterpart lies inside the practitioner video
        sections = [(sid, a, b) for sid, a, b in sections if a + offset >= 0 and b + offset <= u_raw.t[-1] + 0.1]
    pair = Pair(name, R, U, names, scales, swap, sections, gt, offset)
    pair.jitter = {side: {"raw": jitter(S.raw.unmirror() if (side == "user" and swap) else S.raw),
                          **{k: jitter(v) for k, v in S.seq.items()}} for side, S in (("ref", R), ("user", U))}
    pair.video = {"ref": {"path": str(rpx), **probe(ref.video)}, "user": {"path": str(upx), **probe(user_video)}}
    return pair


def section_index(t: np.ndarray, a: float, b: float) -> tuple[int, int]:
    idx = np.flatnonzero((t >= a) & (t < b))
    return (int(idx[0]), int(idx[-1]) + 1) if len(idx) else (0, 0)


def swapped(F: np.ndarray, names: list[str]) -> np.ndarray:
    """Features of the other-side interpretation (the practitioner did the mirror-side repetition)."""
    return lr_swap_features(F, names)
