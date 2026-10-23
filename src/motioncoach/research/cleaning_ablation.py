"""Which cleaning stage helps or hurts, section by section (review 2026-10-22, docs/EXPERIMENT_PLAN.md E1).

Stages, all from the same cached model output:
    raw         every keypoint as the model returned it (no confidence mask)
    conf        low-confidence keypoints masked
    bone_old    + the former bone check (distal joint masked when a bone exceeds 1.35 x its *median* projection)
    bone_new    + the current bone check (1.35 x the bone's 95th-percentile projection, left/right pooled)
    full        production cleaning: bone_new + short-gap fill + zero-phase One Euro
    full_observed  the same, scored on observed samples only (separates smoothing from gap filling)

Two inputs:
- the re-capture video (known answer: the practitioner *is* the reference, replayed at known speeds;
  docs/RECAPTURE_STUDY.md): per section, the same-body joint distance and the indicator error between the two
  renditions at true correspondence, and the share of joints each stage leaves usable;
- the sample practitioner (no answer): how many joints each bone check removes per section.

    python -m motioncoach.research.cleaning_ablation OUT_DIR '<gt json>' RECAPTURE_VIDEO [--backend rtmw-l]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .. import settings
from ..apps.assets import get_reference, load_profile
from ..engine.features import compute_features
from ..engine.filters import LIMBS, one_euro_clean, reject_long_bones
from ..engine.normalize import CANON, bone_lengths, normalized
from ..engine.sequence import PoseSequence
from ..media.extract import extract_poses, prepared_video
from ..perception.registry import get_backend
from .recapture import piecewise_map

PRACTICE = "Weixin Videos2026-10-18_222517_138.mp4"


def old_reject(seq: PoseSequence, max_ratio: float = 1.35) -> tuple[PoseSequence, int]:
    """The bone check before 2026-10-22 (reference = median projected length), kept for the ablation."""
    sk = seq.skeleton
    xy = seq.xy.astype(np.float64).copy()
    n = 0
    for a, b in LIMBS:
        L = np.linalg.norm(xy[:, sk.index[b]] - xy[:, sk.index[a]], axis=1)
        med = np.nanmedian(L) if np.isfinite(L).any() else np.nan
        bad = np.isfinite(L) & (L > max_ratio * med)
        xy[bad, sk.index[b]] = np.nan
        n += int(bad.sum())
    conf = seq.conf.copy()
    conf[~np.isfinite(xy).all(axis=2)] = 0.0
    return replace(seq, xy=xy.astype(np.float32), conf=conf), n


def masked(seq: PoseSequence, thr: float) -> PoseSequence:
    xy = seq.xy.astype(np.float32).copy()
    xy[seq.conf < thr] = np.nan
    return replace(seq, xy=xy)


def stages(seq: PoseSequence) -> dict[str, PoseSequence]:
    thr = float(seq.meta.get("conf_thr", 0.3))
    conf = masked(seq, thr)
    full = one_euro_clean(seq, conf_thr=thr)
    obs = full.xy.copy()
    obs[full.meta["filled"]] = np.nan                     # the same output, observed samples only
    return {"raw": seq, "conf": conf, "bone_old": masked(old_reject(seq)[0], thr),
            "bone_new": masked(reject_long_bones(seq)[0], thr), "full": full, "full_observed": replace(full, xy=obs)}


def section_scores(r: PoseSequence, u: PoseSequence, rb: dict, ub: dict, gt: dict, segs, names, tol) -> dict:
    """Per section: same-body joint distance (torso lengths) and indicator error (tolerance units) between the
    reference ``r`` and the re-capture ``u`` at the true correspondence ``gt``. ``rb``/``ub``: bone lengths
    (one body model per video, shared by every variant scored)."""
    Nr = normalized(r, rb, rb, rb[("hip_mid", "shoulder_mid")])
    Nu = normalized(u, rb, ub, rb[("hip_mid", "shoulder_mid")])
    Fr, Fu = compute_features(r, names)[0], compute_features(u, names)[0]
    ui = np.clip(np.round((piecewise_map(gt, r.t) - u.t[0]) * 10).astype(int), 0, len(u.t) - 1)
    D = np.linalg.norm(Nu[ui] - Nr, axis=2)                        # (T, 13) torso lengths
    E = np.abs(Fu[ui] - Fr) / tol
    row = {}
    for g in segs:
        m = (r.t >= g.start) & (r.t < g.end)
        d = D[m]
        fin = np.isfinite(d).any()
        row[g.id] = {"joint_med": round(float(np.nanmedian(d)), 4) if fin else None,
                     "joint_p90": round(float(np.nanpercentile(d, 90)), 4) if fin else None,
                     "feat_med": round(float(np.nanmedian(E[m])), 3) if np.isfinite(E[m]).any() else None,
                     "usable": round(float(np.isfinite(d).mean()), 3),
                     "left_ankle": round(float(np.nanmedian(d[:, CANON.index("left_ankle")])), 4)
                     if np.isfinite(d[:, CANON.index("left_ankle")]).any() else None}
    return row


def removed_by_section(seq: PoseSequence, segs, t_map=None) -> dict:
    """Share of limb-end joint samples each bone check removes (among confident ones), per section."""
    thr = float(seq.meta.get("conf_thr", 0.3))
    ends = [seq.skeleton.index[b] for _, b in LIMBS]
    ok0 = (seq.conf[:, ends] >= thr) & np.isfinite(seq.xy[:, ends]).all(axis=2)
    out = {}
    for name, fn in (("bone_old", old_reject), ("bone_new", reject_long_bones)):
        s2, _ = fn(seq)
        gone = ok0 & ~np.isfinite(s2.xy[:, ends]).all(axis=2)
        row = {}
        for g in segs:
            a, b = (g.start, g.end) if t_map is None else (t_map(g.start), t_map(g.end))
            m = (seq.t >= a) & (seq.t < b)
            row[g.id] = round(float(gone[m].sum() / max(1, ok0[m].sum())), 4)
        out[name] = row
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("gt")
    ap.add_argument("recapture")
    ap.add_argument("--backend", default="rtmw-l")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gt = json.loads(a.gt)
    ref = get_reference("baduanjin_demo_a", settings.REFERENCES_DIR, settings.ROOT)
    prof = load_profile(ref.raw["profile"])
    names, segs = prof.features, ref.segments(prof)
    tol = np.array([prof.tolerance(n)["good"] for n in names])
    be = get_backend(a.backend, settings.MODELS_DIR)
    rid, rpx = prepared_video(ref.video, settings.DATA_DIR)
    uid, upx = prepared_video(Path(a.recapture), settings.DATA_DIR)
    R = extract_poses(rpx, rid, be, 10.0, settings.DATA_DIR)
    U = extract_poses(upx, uid, be, 10.0, settings.DATA_DIR)
    SR, SU = stages(R), stages(U)
    result = {"backend": a.backend, "recapture": {}, "removed": {}}
    rb, ub = bone_lengths(SR["full"]), bone_lengths(SU["full"])     # one body model for all stages
    for st in SR:
        result["recapture"][st] = section_scores(SR[st], SU[st], rb, ub, gt, segs, names, tol)
    result["removed"]["recapture_reference"] = removed_by_section(R, segs)
    result["removed"]["recapture_practitioner"] = removed_by_section(U, segs, lambda t: float(piecewise_map(gt, np.array([t]))[0]))
    pid, ppx = prepared_video(settings.ASSETS_DIR / PRACTICE, settings.DATA_DIR)
    P = extract_poses(ppx, pid, be, 10.0, settings.DATA_DIR)
    # the sample practitioner's sections from the annotation (practitioner seconds)
    import yaml
    ann = yaml.safe_load((settings.ROOT / "references" / "annotations" / "baduanjin_demo_a__weixin_222517.yaml").read_text())
    from ..engine.compare import Segment
    psegs = [Segment(s["id"], s["id"], float(s["user"][0]), float(s["user"][1])) for s in ann["sections"]]
    result["removed"]["sample_practitioner"] = removed_by_section(P, psegs)
    (out / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(json.dumps(result, ensure_ascii=False)[:3000])


if __name__ == "__main__":
    main()
