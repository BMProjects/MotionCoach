"""E1b: smoothing and gap-fill settings vs error in fast, large movements (docs/EXPERIMENT_PLAN.md E1b).

The E1 ablation found that production cleaning *raises* the tail error of form 5 (P90 0.104 -> 0.147 torso
lengths) while helping the calmer forms. Suspects: a low One Euro floor cutoff (0.6 Hz at 10 fps), the
confidence weighting that smooths low-confidence joints even harder (exactly the fast, blurred ones), and the
0.4 s gap fill. Each variant re-cleans the same cached model output; scores are the re-capture video's
per-section joint distance at the true correspondence, plus the sample practitioner's jitter (the reason
smoothing exists).

    python -m motioncoach.research.smoothing_sweep OUT_DIR '<gt json>' RECAPTURE_VIDEO [--backend rtmw-l]

The re-capture video is both the tuning and the test material here: a chosen setting must still be checked on
the E3 hold-out set.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from .. import settings
from ..apps.assets import get_reference, load_profile
from ..engine.filters import jitter, one_euro_clean
from ..engine.normalize import bone_lengths
from ..media.extract import extract_poses, prepared_video
from ..perception.registry import get_backend
from .cleaning_ablation import PRACTICE, masked, section_scores

GRID = {"min_cutoff": (0.6, 1.0, 1.5, 2.5), "beta": (0.004, 0.01, 0.03), "conf_weight": (0.7, 0.0), "max_gap_s": (0.4, 0.2)}
HARD, CALM = ("s5", "s6"), ("s1", "s2", "s3", "s4", "s7", "s8")
# first-round candidate (2026-10-22): lowest hard-section tail error among settings whose practitioner jitter
# stays <= 0.006 (raw: 0.018); production keeps the defaults until the E3 hold-out set confirms it
CANDIDATE = {"min_cutoff": 0.6, "beta": 0.03, "conf_weight": 0.0, "max_gap_s": 0.2}


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
    seqs = {}
    for key, video in (("ref", ref.video), ("user", Path(a.recapture)), ("practice", settings.ASSETS_DIR / PRACTICE)):
        vid, px = prepared_video(Path(video), settings.DATA_DIR)
        seqs[key] = extract_poses(px, vid, be, 10.0, settings.DATA_DIR)
    R, U, P = seqs["ref"], seqs["user"], seqs["practice"]
    thr = float(R.meta.get("conf_thr", 0.3))
    base_r, base_u = one_euro_clean(R, conf_thr=thr), one_euro_clean(U, conf_thr=thr)
    rb, ub = bone_lengths(base_r), bone_lengths(base_u)          # one body model for every variant
    rows = [{"name": "observed (no smoothing)", "params": None,
             "scores": section_scores(masked(R, thr), masked(U, thr), rb, ub, gt, segs, names, tol),
             "jitter": jitter(masked(P, thr))}]
    keys = list(GRID)
    for vals in itertools.product(*GRID.values()):
        kw = dict(zip(keys, vals))
        r, u = one_euro_clean(R, conf_thr=thr, **kw), one_euro_clean(U, conf_thr=thr, **kw)
        rows.append({"name": " ".join(f"{k}={v}" for k, v in kw.items()), "params": kw,
                     "scores": section_scores(r, u, rb, ub, gt, segs, names, tol),
                     "jitter": jitter(one_euro_clean(P, conf_thr=thr, **kw))})
    for row in rows:
        sc = row["scores"]
        row["hard_p90"] = round(float(np.mean([sc[s]["joint_p90"] for s in HARD])), 4)
        row["calm_p90"] = round(float(np.mean([sc[s]["joint_p90"] for s in CALM])), 4)
        row["calm_feat"] = round(float(np.mean([sc[s]["feat_med"] for s in CALM])), 3)
        row["hard_feat"] = round(float(np.mean([sc[s]["feat_med"] for s in HARD])), 3)
    (out / "results.json").write_text(json.dumps({"backend": a.backend, "grid": GRID, "rows": rows}, ensure_ascii=False, indent=1))
    rows_sorted = sorted(rows[1:], key=lambda r: r["hard_p90"] + r["calm_p90"])
    for row in [rows[0], next(r for r in rows if r["params"] == {"min_cutoff": 0.6, "beta": 0.004, "conf_weight": 0.7, "max_gap_s": 0.4})] + rows_sorted[:8]:
        print(f"{row['name']:<60} hard P90 {row['hard_p90']:.4f}  calm P90 {row['calm_p90']:.4f}  "
              f"feat hard {row['hard_feat']:.3f} calm {row['calm_feat']:.3f}  jitter {row['jitter']:.4f}")


if __name__ == "__main__":
    main()
