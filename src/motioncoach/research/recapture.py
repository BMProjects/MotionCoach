"""Re-capture check with a known answer: the 'practitioner' video is the reference video itself, replayed
(here: mpv at 1.0 / 1.1^3 / 1.1^4 speed, projected on a wall, filmed with a phone). Every reported deviation
is a system error. The true time map is piecewise-constant speed (docs/RECAPTURE_STUDY.md).

    python -m motioncoach.research.recapture OUT_DIR '{"c0":..,"t1":..,"t2":..,"k1":..,"k2":..}' USER_VIDEO         mediapipe-full rtmw-l+mediapipe-full ...        # backend[+bounds backend]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from .. import settings
from ..apps.analysis import AnalysisOptions, run_analysis
from ..apps.assets import get_reference, load_profile
from ..engine.features import compute_features
from ..engine.filters import one_euro_clean
from ..engine.normalize import CANON, bone_lengths, normalized
from ..media.extract import extract_poses, prepared_video
from ..perception.registry import get_backend


def piecewise_map(gt: dict, t: np.ndarray) -> np.ndarray:
    """Reference time -> practitioner time for playback at 1x, then 1/k1, then 1/k2 speed."""
    c0, t1, t2, k1, k2 = gt["c0"], gt["t1"], gt["t2"], gt["k1"], gt["k2"]
    u1 = t1 + c0
    u2 = u1 + (t2 - t1) / k1
    return np.where(t < t1, t + c0, np.where(t < t2, u1 + (t - t1) / k1, u2 + (t - t2) / k2))


def main(argv: list[str]) -> None:
    out, gt, user_video = Path(argv[0]), json.loads(argv[1]), Path(argv[2])
    configs = argv[3:]
    out.mkdir(parents=True, exist_ok=True)

    def gt_map(t):
        return piecewise_map(gt, t)

    ref = get_reference("baduanjin_demo_a", settings.REFERENCES_DIR, settings.ROOT)
    prof = load_profile(ref.raw["profile"])
    names = prof.features
    tol = np.array([prof.tolerance(n)["good"] for n in names])
    segs = ref.segments(prof)
    templates = ref.rep_templates()
    results = {"gt": gt, "configs": {}}

    for cfg in configs:
        be_name, _, bounds = cfg.partition("+")
        t0 = time.time()
        opts = AnalysisOptions(backend=be_name, bounds_backend=bounds or be_name)
        rep = run_analysis(ref, user_video, out / cfg.replace("+", "__"), settings.DATA_DIR, settings.MODELS_DIR, opts)
        elapsed = time.time() - t0
        # ---- measurement error under the true time map (independent of the system's alignment)
        be = get_backend(be_name, settings.MODELS_DIR)
        rid, rpx = prepared_video(ref.video, settings.DATA_DIR)
        uid, upx = prepared_video(user_video, settings.DATA_DIR)
        rs = one_euro_clean(r := extract_poses(rpx, rid, be, 10.0, settings.DATA_DIR), conf_thr=r.meta.get("conf_thr", 0.5))
        us = one_euro_clean(u := extract_poses(upx, uid, be, 10.0, settings.DATA_DIR), conf_thr=u.meta.get("conf_thr", 0.5))
        Fr, Fu = compute_features(rs, names)[0], compute_features(us, names)[0]
        rb = bone_lengths(rs)
        Nr = normalized(rs, rb, rb, rb[("hip_mid", "shoulder_mid")])
        Nu = normalized(us, rb, bone_lengths(us), rb[("hip_mid", "shoulder_mid")])
        ui = np.clip(np.round((gt_map(rs.t) - us.t[0]) * 10).astype(int), 0, len(us.t) - 1)
        E = np.abs(Fu[ui] - Fr) / tol                                   # tolerance units, per ref sample
        D = np.linalg.norm(Nu[ui] - Nr, axis=2)                         # per joint, torso lengths
        per_seg = {}
        for g in segs:
            m = (rs.t >= g.start) & (rs.t < g.end)
            fe = {n: (float(np.nanmedian(E[m, k])) if np.isfinite(E[m, k]).any() else None,
                      float(np.isfinite(Fu[ui[m], k]).mean())) for k, n in enumerate(names)}
            # the statistic the rules judge: |deviation| at the annotated holds (whole section if not repeated)
            mh = np.zeros_like(m)
            for _, _, h0, h1 in templates.get(g.id) or []:
                mh |= (rs.t >= h0) & (rs.t < h1)
            mh = mh if mh.any() else m
            he = {n: float(np.nanmedian(E[mh, k]) * tol[k]) if np.isfinite(E[mh, k]).any() else None
                  for k, n in enumerate(names)}
            per_seg[g.id] = {"feat_err": fe, "hold_err": he, "skel_med": float(np.nanmedian(D[m])),
                             "skel_p90": float(np.nanpercentile(D[m], 90)),
                             "joint_med": {j: float(np.nanmedian(D[m, i])) if np.isfinite(D[m, i]).any() else None
                                           for i, j in enumerate(CANON)},
                             "user_valid": float(np.isfinite(Nu[ui[m]]).all(axis=2).mean())}
        # ---- outcome of the real analysis
        tr = np.array(rep["series"]["t_ref"], float)
        uu = np.array([np.nan if x is None else x for x in rep["series"]["user_t"]], float)
        ok = np.isfinite(uu)
        terr = np.abs(uu[ok] - gt_map(tr[ok]))
        flagged = [{"seg": s["id"], "feat": n, "status": c["status"], "value": c.get("value"), "signed": c.get("signed"),
                    "good": c["tolerance"]["good"], "focus": c.get("focus")}
                   for s in rep["segments"] for n, c in s["features"].items() if c.get("status") in ("warn", "bad")]
        reps = {s["id"]: [len((s.get("reps") or {}).get("user") or []), len(templates.get(s["id"]) or []) or 1]
                for s in rep["segments"]}
        bound_err = {s["id"]: float(s["user_start"] - gt_map(np.array([s["ref_start"]]))[0]) for s in rep["segments"]}
        stature = float(np.nanmedian(np.nanmax(us.xy[:, :, 1], axis=1) - np.nanmin(us.xy[:, :, 1], axis=1)))
        results["configs"][cfg] = {
            "counts": rep["summary"]["counts"], "flagged": flagged, "reps": reps, "bound_err": bound_err,
            "time_err": {"p50": float(np.percentile(terr, 50)), "p90": float(np.percentile(terr, 90)),
                         "p99": float(np.percentile(terr, 99))},
            "align_cost": rep["timing"]["align_cost"], "elapsed_s": elapsed, "timings_s": rep.get("timings_s"),
            "per_seg": per_seg, "person_height_frac": stature / us.frame_size[1], "frame_size": list(us.frame_size)}
        print(cfg, "done", round(elapsed, 1), "s", rep["summary"]["counts"], flush=True)
        (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1:])
