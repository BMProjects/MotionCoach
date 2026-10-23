"""Person detector x box tracking for the top-down RTMW-l (docs/ITERATION_LOG.md I18, EXPERIMENT_PLAN.md E2).

For each variant (detector tiny / m / x, box tracking on / off) on the reference, the re-capture and the
sample practitioner:
- completeness: share of sampled frames with >= 10 of the 12 body joints confident, overall and per section
  (practitioner sections from its annotation), and frames with no skeleton at all;
- the re-capture's per-section joint error at the true correspondence (raw and production-cleaned);
- speed (ms per frame, detector + pose) and how the boxes came about (detected / bridged / enlarged / lost).

    python -m motioncoach.research.detector_bench OUT_DIR '<gt json>' RECAPTURE_VIDEO
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from .. import settings
from ..apps.assets import get_reference, load_profile
from ..engine.compare import Segment
from ..engine.normalize import bone_lengths
from ..media.extract import extract_poses, prepared_video
from ..perception.rtm import BODY, RtmlibBackend
from .cleaning_ablation import PRACTICE, section_scores, stages

VARIANTS = (("native", False), ("native", True), ("m", True), ("x", False), ("x", True))
ANNOTATION = "baduanjin_demo_a__weixin_222517.yaml"


def completeness(seq, segs=None) -> dict:
    ok = seq.conf[:, BODY] >= float(seq.meta.get("conf_thr", 0.3))
    full, none = ok.sum(axis=1) >= 10, ok.sum(axis=1) == 0
    out = {"complete": round(float(full.mean()), 4), "empty": round(float(none.mean()), 4)}
    for g in segs or []:
        m = (seq.t >= g.start) & (seq.t < g.end)
        out[g.id] = round(float(full[m].mean()), 4) if m.any() else None
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("gt")
    ap.add_argument("recapture")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gt = json.loads(a.gt)
    ref = get_reference("baduanjin_demo_a", settings.REFERENCES_DIR, settings.ROOT)
    prof = load_profile(ref.raw["profile"])
    names, segs = prof.features, ref.segments(prof)
    tol = np.array([prof.tolerance(n)["good"] for n in names])
    ann = yaml.safe_load((settings.ROOT / "references" / "annotations" / ANNOTATION).read_text())
    psegs = [Segment(s["id"], s["id"], float(s["user"][0]), float(s["user"][1])) for s in ann["sections"]]
    videos = {"reference": Path(ref.video), "recapture": Path(a.recapture), "practitioner": settings.ASSETS_DIR / PRACTICE}
    prepared = {k: prepared_video(v, settings.DATA_DIR) for k, v in videos.items()}
    res = {}
    for det, track in VARIANTS:
        be = RtmlibBackend("rtmw-l", detector=det, track=track)
        tag = f"{det}{'+track' if track else ''}"
        seqs = {k: extract_poses(px, vid, be, 10.0, settings.DATA_DIR) for k, (vid, px) in prepared.items()}
        SR, SU = stages(seqs["reference"]), stages(seqs["recapture"])
        rb, ub = bone_lengths(SR["full"]), bone_lengths(SU["full"])
        res[tag] = {"model_id": be.model_id,
                    "ms_per_frame": {k: round(float(s.meta.get("ms_per_frame") or 0), 2) for k, s in seqs.items()},
                    "boxes": {k: s.meta.get("boxes") for k, s in seqs.items()},
                    "completeness": {"reference": completeness(seqs["reference"], segs),
                                     "recapture": completeness(seqs["recapture"]),
                                     "practitioner": completeness(seqs["practitioner"], psegs)},
                    "recapture": {st: section_scores(SR[st], SU[st], rb, ub, gt, segs, names, tol) for st in ("raw", "full")}}
        r = res[tag]
        print(f"{tag:<12} ms {r['ms_per_frame']}  practitioner complete {r['completeness']['practitioner']['complete']} "
              f"(s5 {r['completeness']['practitioner'].get('s5')}, s6 {r['completeness']['practitioner'].get('s6')}) "
              f"empty {r['completeness']['practitioner']['empty']}  recapture full P90 "
              + " ".join(f"{s}:{r['recapture']['full'][s]['joint_p90']}" for s in ("s1", "s4", "s5", "s6", "s7")), flush=True)
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
