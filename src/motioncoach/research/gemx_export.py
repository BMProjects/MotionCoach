"""Export a GEM-X Body Motion prediction to plain arrays (runs in the GEM-X environment, not the app's).

    python gemx_export.py OUT.npz MOTION.pt [MOTION2.pt ...] --soma-assets assets/soma [--device cuda]

GEM-X runs at most 60 s per call, so long videos are run as overlapping chunks (``--start``); the chunks are
stitched here, each overlap cut at its middle so every frame comes from the chunk where it is farther from an
edge (the temporal model sees context on both sides).

Writes, per 30 Hz GEM-X frame:
    t          (T,)        seconds in the source video
    kp2d       (T, 77, 3)  GEM-X's own 2D observations (ViTPose-DINOv3-H, SOMA77 joints), x, y, confidence
    proj2d     (T, 77, 2)  the temporal 3D motion's SOMA77 joints projected with the per-frame intrinsics
    joints3d   (T, 77, 3)  the same joints in camera space, metres
    observed   (T,)        GEM-X tracked the primary person in that frame (otherwise its output is interpolated)
Only numpy goes into the file, so the app environment can read it without GEM-X installed
(docs/EXPERIMENT_PLAN.md E4).
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from gem.motion.geometry import MotionGeometry


def export_one(path: str, geo: MotionGeometry) -> dict:
    d = torch.load(path, map_location="cpu", weights_only=False)
    if d.get("format") != "gem_body_video_predictions_v1":
        raise ValueError(f"{path}: unexpected GEM-X output format {d.get('format')!r}")
    J = geo(d["pred_soma_params_incam"])["joints"][0].double()          # (T, 77, 3) camera metres
    K = d["conditions"]["K_fullimg"][0].double()                          # (T, 3, 3) source pixels
    uvw = torch.einsum("tij,tkj->tki", K, J)
    proj = (uvw[..., :2] / uvw[..., 2:3].clamp(min=1e-6)).numpy()
    obs = np.zeros(J.shape[0], bool)
    tr = d.get("tracking") or {}
    if "frame_ids" in tr and "observed_primary" in tr:
        ids = np.asarray(tr["frame_ids"])
        ok = np.asarray(tr["observed_primary"], bool)
        obs[ids[ids < len(obs)]] = ok[ids < len(obs)]
    start = float(d["video"]["source"].get("requested_start") or 0.0)
    t = d["frame_times"][0].numpy().astype(np.float64)
    if start > 0 and t[0] < start - 0.5:                                  # clip-relative timestamps
        t = t + start
    return {"t": t, "kp2d": d["conditions"]["kp2d_native77"][0].numpy(), "proj2d": proj.astype(np.float32),
            "joints3d": J.numpy().astype(np.float32), "observed": obs,
            "width": d["video"]["source"]["width"], "height": d["video"]["source"]["height"]}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("motion", nargs="+")
    ap.add_argument("--soma-assets", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    geo = MotionGeometry(a.soma_assets, device=a.device)
    parts = sorted((export_one(p, geo) for p in a.motion), key=lambda c: c["t"][0])
    keep = []
    for i, c in enumerate(parts):
        lo = 0.5 * (c["t"][0] + parts[i - 1]["t"][-1]) if i else -np.inf
        hi = 0.5 * (parts[i + 1]["t"][0] + c["t"][-1]) if i + 1 < len(parts) else np.inf
        keep.append((c["t"] >= lo) & (c["t"] < hi))
    arrays = {k: np.concatenate([c[k][m] for c, m in zip(parts, keep)]) for k in ("t", "kp2d", "proj2d", "joints3d", "observed")}
    np.savez_compressed(a.out, **arrays, width=parts[0]["width"], height=parts[0]["height"])
    print(a.out, arrays["joints3d"].shape, f"{arrays['t'][0]:.2f}-{arrays['t'][-1]:.2f} s", f"observed {arrays['observed'].mean():.3f}")


if __name__ == "__main__":
    main()
