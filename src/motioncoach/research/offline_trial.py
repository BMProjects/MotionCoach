"""E4: score externally run pose models (GEM-X, SAM 3D Body) on the re-capture video, like vitpose_trial.

The models run in their own environments (`gemx_export.py`, `sam3db_export.py`) and leave plain .npz files.
Here each is resampled onto the RTMW-l 10 fps grid of the same video (nearest frame), mapped to COCO-17, and
scored with the same stages and per-section scores as the other candidates:

    python -m motioncoach.research.offline_trial OUT_DIR '<gt json>' RECAPTURE_VIDEO \\
        --run gemx-2d=REF.npz,USER.npz --run gemx-3d=REF.npz,USER.npz --run sam3db=REF.npz,USER.npz

Box-conditioned models get the same person boxes as the other candidates: ``--dump-boxes`` writes them first.

Variants: ``gemx-2d`` = GEM-X's own 2D observations (ViTPose-DINOv3-H), ``gemx-3d`` = its temporal 3D motion
projected into the image on frames where GEM-X tracked the person (``gemx-3d-all``: also the frames
it bridged without a detection), ``sam3db`` = SAM 3D Body's per-frame 2D keypoints (projected MHR joints).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .. import settings
from ..apps.assets import get_reference, load_profile
from ..engine.normalize import bone_lengths
from ..engine.sequence import PoseSequence
from ..engine.skeleton import get_skeleton
from ..media.extract import extract_poses, prepared_video
from ..perception.registry import get_backend
from .cleaning_ablation import section_scores, stages

COCO = get_skeleton("coco17").joints
# source joint index (or a tuple to average) for each COCO-17 joint; None = not available
SOMA77 = {"nose": (9, 10), "left_eye": 9, "right_eye": 10, "left_ear": None, "right_ear": None,
          "left_shoulder": 12, "right_shoulder": 40, "left_elbow": 13, "right_elbow": 41, "left_wrist": 14,
          "right_wrist": 42, "left_hip": 67, "right_hip": 72, "left_knee": 68, "right_knee": 73,
          "left_ankle": 69, "right_ankle": 74}
# MHR70 (sam_3d_body/metadata/mhr70.py): COCO order without the wrists for 0-14, wrists at 41 (right) / 62 (left)
MHR70 = {**{n: i for i, n in enumerate(c for c in COCO if not c.endswith("wrist"))}, "right_wrist": 41, "left_wrist": 62}
SOURCES = {"gemx-2d": ("kp2d", SOMA77), "gemx-3d": ("proj2d", SOMA77), "gemx-3d-all": ("proj2d", SOMA77),
           "sam3db": ("kp2d", MHR70)}


def dump_boxes(video: Path, out: Path, sample_fps: float = 10.0) -> Path:
    """Person box per sampled frame from rtmlib's YOLOX-m (RTMPose-m's detector, as in the ViTPose++ trial), so
    a box-conditioned model sees the same crops: the largest box, kept on the previous one when they overlap.
    NaN where nobody is detected."""
    from ..perception.base import sampled_frames
    rtm = get_backend("rtmpose-m", settings.MODELS_DIR)
    rtm.open()
    det = rtm._model.det_model
    vid, px = prepared_video(Path(video), settings.DATA_DIR)
    ts, boxes, prev = [], [], None
    for ms, frame in sampled_frames(px, sample_fps, 0.0, None, None):
        b = np.asarray(det(frame), dtype=np.float64).reshape(-1, 4)
        pick = np.full(4, np.nan)
        if len(b):
            area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
            pick = b[int(np.argmax(area))]
            if prev is not None:
                x0, y0 = np.maximum(b[:, 0], prev[0]), np.maximum(b[:, 1], prev[1])
                x1, y1 = np.minimum(b[:, 2], prev[2]), np.minimum(b[:, 3], prev[3])
                inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
                iou = inter / (area + (prev[2] - prev[0]) * (prev[3] - prev[1]) - inter)
                if iou.max() > 0.3:
                    pick = b[int(np.argmax(iou))]
            prev = pick
        ts.append(ms / 1000.0)
        boxes.append(pick)
    np.savez_compressed(out, t=np.asarray(ts), boxes=np.asarray(boxes), video=str(px))
    return out


def to_sequence(npz: Path, variant: str, grid: PoseSequence) -> PoseSequence:
    """Resample an exported model output onto ``grid``'s timestamps as a COCO-17 PoseSequence."""
    key, jmap = SOURCES[variant]
    z = np.load(npz)
    t_src, arr = z["t"], z[key]
    if len(t_src) < 2:
        raise ValueError(f"{npz}: no frames")
    xy_src = arr[..., :2].astype(np.float64)
    if arr.shape[-1] == 3:
        cf_src = arr[..., 2]
    else:                                             # 3D projections carry no confidence: tracked frames = 1
        obs = z["observed"] if "observed" in z and not variant.endswith("-all") else np.ones(len(t_src), bool)
        cf_src = np.repeat(np.where(obs, 1.0, 0.0)[:, None], arr.shape[1], axis=1)
    if "valid" in z:                                   # per-frame "a person was found" (SAM 3D Body)
        cf_src = cf_src * z["valid"][:, None]
    k = np.clip(np.searchsorted(t_src, grid.t), 1, len(t_src) - 1)
    k = np.where(np.abs(t_src[k - 1] - grid.t) <= np.abs(t_src[k] - grid.t), k - 1, k)
    far = np.abs(t_src[k] - grid.t) > 0.05            # outside the exported range
    T = len(grid.t)
    xy = np.full((T, len(COCO), 2), np.nan)
    cf = np.zeros((T, len(COCO)), np.float32)
    for j, name in enumerate(COCO):
        src = jmap.get(name)
        if src is None:
            continue
        idx = list(src) if isinstance(src, tuple) else [src]
        xy[:, j] = xy_src[k][:, idx].mean(axis=1)
        cf[:, j] = cf_src[k][:, idx].min(axis=1)
    xy[far] = np.nan
    cf[far] = 0.0
    xy[cf <= 0] = np.nan
    return PoseSequence(t=grid.t.copy(), xy=xy.astype(np.float32), conf=cf, frame_size=grid.frame_size,
                        skeleton_name="coco17", meta={"model": variant, "backend": variant, "conf_thr": 0.3,
                                                      "source": str(npz)})


def _angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle in degrees between row vectors (2D or 3D)."""
    c = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def hip_angles(gt: dict, segs, grids: list[PoseSequence], ref_npz: Path, user_npz: Path) -> dict:
    """Trunk-thigh angle (180 = upright) per side, reference vs re-capture at the true correspondence: from
    GEM-X's 3D joints (camera-independent) and from RTMW-l's 2D keypoints (what a frontal view gives)."""
    from .recapture import piecewise_map

    def soma(npz):
        z = np.load(npz)
        J = z["joints3d"].astype(np.float64)
        trunk = J[:, 3] - J[:, 0]                                      # Chest - Hips
        return z["t"], {side: _angle(trunk, J[:, knee] - J[:, hip]) for side, hip, knee in (("left", 67, 68), ("right", 72, 73))}

    def flat(seq):
        P = {n: seq.joint(n)[0].astype(np.float64) for n in ("left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_knee", "right_knee")}
        trunk = 0.5 * (P["left_shoulder"] + P["right_shoulder"]) - 0.5 * (P["left_hip"] + P["right_hip"])
        return seq.t, {side: _angle(trunk, P[f"{side}_knee"] - P[f"{side}_hip"]) for side in ("left", "right")}

    def at(t_src, v, t):
        k = np.clip(np.searchsorted(t_src, t), 1, len(t_src) - 1)
        k = np.where(np.abs(t_src[k - 1] - t) <= np.abs(t_src[k] - t), k - 1, k)
        return v[k]

    out = {}
    from ..engine.filters import one_euro_clean
    clean = [one_euro_clean(g, conf_thr=float(g.meta.get("conf_thr", 0.3))) for g in grids]   # production cleaning
    for name, (r, u) in (("gemx-3d", (soma(ref_npz), soma(user_npz))), ("rtmw-l-2d", (flat(clean[0]), flat(clean[1])))):
        row = {}
        for g in segs:
            tr = grids[0].t[(grids[0].t >= g.start) & (grids[0].t < g.end)]
            tu = piecewise_map(gt, tr)
            d, rng = [], []
            for side in ("left", "right"):
                a_r, a_u = at(r[0], r[1][side], tr), at(u[0], u[1][side], tu)
                d.append(np.abs(a_r - a_u))
                rng.append(a_r)
            d, rng = np.concatenate(d), np.concatenate(rng)
            ok = np.isfinite(d)
            row[g.id] = {"diff_med": round(float(np.median(d[ok])), 2) if ok.any() else None,
                         "diff_p90": round(float(np.percentile(d[ok], 90)), 2) if ok.any() else None,
                         "ref_min": round(float(np.nanmin(rng)), 1) if np.isfinite(rng).any() else None}
        out[name] = row
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("gt")
    ap.add_argument("recapture")
    ap.add_argument("--run", action="append", default=[], help="VARIANT=REF.npz,USER.npz")
    ap.add_argument("--angles", default="", help="REF.npz,USER.npz (GEM-X exports): 3D vs 2D hip angle consistency")
    ap.add_argument("--dump-boxes", action="store_true", help="only write OUT/boxes_{ref,user}.npz for box-conditioned models")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gt = json.loads(a.gt)
    ref = get_reference("baduanjin_demo_a", settings.REFERENCES_DIR, settings.ROOT)
    if a.dump_boxes:
        for tag, video in (("ref", ref.video), ("user", Path(a.recapture))):
            print(dump_boxes(Path(video), out / f"boxes_{tag}.npz"))
        return
    prof = load_profile(ref.raw["profile"])
    names, segs = prof.features, ref.segments(prof)
    tol = np.array([prof.tolerance(n)["good"] for n in names])
    be = get_backend("rtmw-l", settings.MODELS_DIR)
    grids = []
    for video in (ref.video, Path(a.recapture)):
        vid, px = prepared_video(Path(video), settings.DATA_DIR)
        grids.append(extract_poses(px, vid, be, 10.0, settings.DATA_DIR))    # cached; supplies the time grid
    res = {}
    for spec in a.run:
        variant, files = spec.split("=", 1)
        rf, uf = files.split(",")
        R, U = to_sequence(Path(rf), variant, grids[0]), to_sequence(Path(uf), variant, grids[1])
        SR, SU = stages(R), stages(U)
        rb, ub = bone_lengths(SR["full"]), bone_lengths(SU["full"])
        res[variant] = {st: section_scores(SR[st], SU[st], rb, ub, gt, segs, names, tol) for st in ("raw", "conf", "full")}
        res[variant]["coverage"] = [round(float(np.isfinite(S.xy).all(axis=2).any(axis=1).mean()), 4) for S in (R, U)]
        print(variant, res[variant]["coverage"],
              {s: (res[variant]["raw"][s]["joint_med"], res[variant]["raw"][s]["joint_p90"], res[variant]["raw"][s]["feat_med"])
               for s in ("s1", "s4", "s5", "s6", "s7")})
    if a.angles:
        rf, uf = a.angles.split(",")
        res["hip_angle"] = hip_angles(gt, segs, grids, Path(rf), Path(uf))
        for m, row in res["hip_angle"].items():
            print("hip angle", m, {s: (row[s]["diff_med"], row[s]["diff_p90"], row[s]["ref_min"]) for s in ("s1", "s5", "s6", "s7")})
    prev = out / "results.json"
    if prev.exists():                                  # add to earlier variants rather than replace them
        res = {**json.loads(prev.read_text()), **res}
    prev.write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
