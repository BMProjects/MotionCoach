"""PoseSequence: the data contract between perception and comparison.

Conventions
-----------
- ``t``      real timestamps in seconds (not frame indices), so 30/60 fps or VFR videos compare by time.
- ``xy``     image coordinates in **pixels** of the displayed (rotation-applied) frame. Pixels keep
             x and y on the same scale; per-axis [0,1] normalisation distorts angles on non-square frames.
- ``conf``   per-joint confidence in [0,1]; 0 where the person was not detected.
- ``world``  optional model-estimated 3D coordinates (metres, hip-centred). Estimates, not measurements.
- ``mirrored`` True when the coordinates still come from a mirrored (selfie) recording. Use
             ``unmirror()`` to restore anatomical left/right before comparing.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import json

import numpy as np

from .skeleton import Skeleton, get_skeleton


@dataclass
class PoseSequence:
    t: np.ndarray                     # (T,) float64 seconds
    xy: np.ndarray                    # (T, J, 2) float32 pixels, NaN where missing
    conf: np.ndarray                  # (T, J) float32
    frame_size: tuple[int, int]       # (width, height) in pixels
    skeleton_name: str = "mediapipe33"
    world: np.ndarray | None = None   # (T, J, 3) float32
    mirrored: bool = False
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        T = len(self.t)
        J = len(self.skeleton.joints)
        if self.xy.shape != (T, J, 2) or self.conf.shape != (T, J):
            raise ValueError(f"Shape mismatch: t={self.t.shape} xy={self.xy.shape} conf={self.conf.shape}, J={J}")
        if T > 1 and np.any(np.diff(self.t) <= 0):
            raise ValueError("Timestamps must be strictly increasing")

    @property
    def skeleton(self) -> Skeleton:
        return get_skeleton(self.skeleton_name)

    def __len__(self) -> int:
        return len(self.t)

    @property
    def detected(self) -> np.ndarray:
        """(T,) bool: a person was detected in the frame."""
        return np.isfinite(self.xy).all(axis=2).any(axis=1)

    def joint(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (xy (T,2), conf (T,)) for a named joint."""
        i = self.skeleton.index[name]
        return self.xy[:, i], self.conf[:, i]

    def unmirror(self) -> "PoseSequence":
        """Undo a horizontal mirror: flip x and swap left/right joint identities."""
        perm = self.skeleton.lr_permutation()
        xy = self.xy[:, perm].copy()
        xy[..., 0] = self.frame_size[0] - xy[..., 0]
        world = None
        if self.world is not None:
            world = self.world[:, perm].copy()
            world[..., 0] = -world[..., 0]
        return replace(self, xy=xy, conf=self.conf[:, perm].copy(), world=world, mirrored=not self.mirrored)

    # ---- persistence -------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"t": self.t, "xy": self.xy, "conf": self.conf}
        if self.world is not None:
            arrays["world"] = self.world
        header = {
            "frame_size": list(self.frame_size),
            "skeleton_name": self.skeleton_name,
            "mirrored": self.mirrored,
            "meta": self.meta,
        }
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, header=np.array(json.dumps(header)), **arrays)
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "PoseSequence":
        with np.load(Path(path), allow_pickle=False) as z:
            header = json.loads(str(z["header"]))
            return cls(
                t=z["t"], xy=z["xy"], conf=z["conf"],
                world=z["world"] if "world" in z.files else None,
                frame_size=tuple(header["frame_size"]),
                skeleton_name=header["skeleton_name"],
                mirrored=header["mirrored"],
                meta=header["meta"],
            )


def clean(seq: PoseSequence, conf_thr: float = 0.5, max_gap_s: float = 0.4,
          smooth_window_s: float = 0.5) -> tuple[PoseSequence, np.ndarray]:
    """Mask low-confidence joints, fill short gaps, and smooth.

    Returns the cleaned sequence and an (T, J) bool array marking interpolated samples.
    Gaps longer than ``max_gap_s`` stay NaN so that they are reported as unmeasurable
    instead of being invented.
    """
    from scipy.signal import savgol_filter

    xy = seq.xy.astype(np.float64).copy()
    valid = (seq.conf >= conf_thr) & np.isfinite(xy).all(axis=2)
    xy[~valid] = np.nan
    interp = np.zeros_like(valid)
    t = seq.t
    T, J = valid.shape
    dt = float(np.median(np.diff(t))) if T > 1 else 1.0
    win = max(3, int(round(smooth_window_s / dt)) | 1)  # odd

    for j in range(J):
        v = valid[:, j]
        if v.sum() < 2:
            continue
        idx = np.flatnonzero(v)
        # fill interior gaps no longer than max_gap_s
        for a, b in zip(idx[:-1], idx[1:]):
            if b - a > 1 and t[b] - t[a] <= max_gap_s + 1e-9:
                w = (t[a + 1:b] - t[a]) / (t[b] - t[a])
                xy[a + 1:b, j] = xy[a, j] * (1 - w)[:, None] + xy[b, j] * w[:, None]
                interp[a + 1:b, j] = True
        ok = np.isfinite(xy[:, j, 0])
        # smooth each contiguous finite run independently
        edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
        for s, e in zip(edges[::2], edges[1::2]):
            if e - s >= win:
                xy[s:e, j] = savgol_filter(xy[s:e, j], win, 2, axis=0)

    meta = dict(seq.meta)
    meta["preprocess"] = {"conf_thr": conf_thr, "max_gap_s": max_gap_s, "smooth_window_s": smooth_window_s,
                          "method": "mask+linear_gap_fill+savgol2"}
    out = replace(seq, xy=xy.astype(np.float32), meta=meta)
    return out, interp
