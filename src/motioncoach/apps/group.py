"""Multi-person analysis: one video with several practitioners (and possibly onlookers).

    ingest -> every person in every sampled frame (perception.multi, cached)
           -> identities, floor layout and performer screening (engine.group)
           -> each performer through the single-person comparison (apps.analysis, same model throughout)
           -> group metrics: synchronisation and consistency (engine.group.group_metrics)

Output ``group.json`` (``motioncoach.group/0``) plus one ordinary ``motioncoach.report`` per performer, so
each performer opens in the existing workbench.
"""

from __future__ import annotations

import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .. import __version__
from ..contracts import GROUP_SCHEMA
from ..engine.features import FEATURES
from ..engine.group import (GroupParams, build_identities, consensus_sequence, group_metrics, group_timeline,
                            identity_sequence, signal_sync)
from ..media.camera import apply, camera_motion, drift
from ..media.extract import extract_multi, prepared_video
from ..media.ingest import probe, view_path
from ..perception.registry import get_backend
from .analysis import AnalysisOptions, Progress, run_analysis, to_json
from .assets import Reference, load_profile
from .scope import ScopeRequest


def run_group(reference: Reference, video: Path, out_dir: Path, data_dir: Path, models_dir: Path,
              options: AnalysisOptions, params: GroupParams = GroupParams(),
              progress: Progress | None = None) -> dict:
    progress = progress or (lambda stage, f: None)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    timings = {}
    profile = load_profile(reference.raw["profile"])

    s = time.time()
    progress("ingest", 0.0)
    vid, proxy = prepared_video(Path(video), data_dir)
    timings["ingest"] = time.time() - s

    s = time.time()
    backend = get_backend(options.backend, models_dir)
    mp = extract_multi(proxy, vid, backend, options.sample_fps, data_dir, progress=lambda f: progress("poses", f))
    timings["poses"] = time.time() - s

    # hand-held / panning camera: identities and layout are built in first-frame coordinates; the overlay
    # boxes and each person's measurements keep the original pixels (same detection indices)
    s = time.time()
    progress("identities", 0.0)
    cam = None
    if params.stabilize:
        boxes = [mp.box[mp.frame == k] for k in range(len(mp.t))]
        A = camera_motion(proxy, options.sample_fps, boxes, mp.frame_size,
                          cache=Path(data_dir) / "poses" / f"{mp.meta.get('cache_key', vid)}_cam.npz")
        cam = drift(A)
        corners = apply(A, mp.frame, mp.box.reshape(-1, 2, 2))
        box_s = np.concatenate([corners.min(axis=1), corners.max(axis=1)], axis=1).astype(np.float32)
        mp_geo = replace(mp, xy=apply(A, mp.frame, mp.xy).astype(np.float32), box=box_s)
    else:
        mp_geo = mp
    timings["camera"] = time.time() - s
    s = time.time()
    ids, diag = build_identities(mp_geo, params)
    timings["identities"] = time.time() - s
    performers = [i for i in ids if i.performer]

    reports, persons = {}, []
    opts = replace(options, bounds_backend=None)        # one model throughout: identities come from it

    # ---- one time axis for the whole group: the analysis interval and the section bounds are located once,
    # on the group's median pose, then every performer is compared inside the same sections (shifted by
    # their own lag), so the per-person results and the group metrics refer to the same moments
    s = time.time()
    progress("persons", 0.0)
    common = _common_axis(reference, Path(video), out_dir, data_dir, models_dir, opts, mp, performers, params)
    timings["common_axis"] = time.time() - s
    for n, idn in enumerate(performers):
        progress("persons", (n + 1) / (len(performers) + 1))
        for old in out_dir.glob(f"person_{idn.id}/report.json"):   # a re-run after identity edits
            old.unlink()
        d = out_dir / f"person_{idn.id}"
        o = opts
        if common.get("section_bounds"):
            lag = float(idn.screening.get("lag_s") or 0.0)
            ur = common["user_range"]
            o = replace(opts, scope=ScopeRequest("manual", common["segments"][0], common["segments"][-1],
                                                 max(0.0, ur[0] + lag), ur[1] + lag),
                        section_bounds={k: (max(0.0, a + lag), b + lag) for k, (a, b) in common["section_bounds"].items()})
        try:
            rep = run_analysis(reference, Path(video), d, data_dir, models_dir, o, user_poses=identity_sequence(mp, idn))
        except Exception as e:                           # one person failing must not hide the others
            if isinstance(e, ValueError) and "too short" in str(e):
                persons.append({"id": idn.id, "status": "skipped", "reason": "在统一分析区间内可见的时间太短，未比较"})
            else:
                persons.append({"id": idn.id, "status": "error", "error": f"{type(e).__name__}: {e}"})
            continue
        reports[idn.id] = rep
        sm = rep["summary"]
        segs = []
        for sg in rep["segments"]:
            st = [c["status"] for c in sg["features"].values() if c.get("focus")]
            segs.append({"id": sg["id"], "name": sg["name"], "ref_start": sg["ref_start"], "ref_end": sg["ref_end"],
                         "counts": {k: st.count(k) for k in set(st)}, "pass_rate": sg.get("pass_rate"),
                         "reps": len((sg.get("reps") or {}).get("user") or [])})
        persons.append({"id": idn.id, "status": "done", "segments": segs, "lag_s": idn.screening.get("lag_s"),
                        "top_issues": [{k: it[k] for k in ("segment", "segment_name", "label", "status", "text", "time")}
                                       for it in sm.get("top_issues", [])[:3]],
                        "pass_text": sm.get("pass_text"), "counts": sm.get("counts"),
                        "tempo_ratio": rep["timing"]["tempo_ratio"], "align_cost": rep["timing"]["align_cost"],
                        "scope": rep["scope"]["segments"]})
    timings["persons"] = time.time() - t_start - sum(timings.values())

    s = time.time()
    metrics = group_metrics(reports, profile.features)
    tol = {n: profile.tolerance(n)["good"] for n in profile.features}
    metrics["timeline"] = group_timeline(mp, [i for i in performers if i.id in reports], profile.features, tol,
                                         common.get("section_bounds"))
    # synchronisation from the motion signals (robust to repeated identical moves), over the common sections
    done = [i for i in performers if i.id in reports]
    ss = signal_sync(mp, done, common.get("section_bounds"))
    if ss:
        metrics.update(sync=ss["sync"], sync_series=ss["sync_series"])
        metrics.setdefault("definition", {})["offset"] = ss["sync_definition"]
    timings["group"] = time.time() - s
    timings["total"] = time.time() - t_start

    out = {
        "schema": GROUP_SCHEMA, "engine_version": __version__, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "reference": {"id": reference.id, "title": reference.raw.get("title")},
        "profile": {"id": profile.id, "title": profile.raw.get("title"), "status": profile.raw.get("status")},
        "features": [{"name": n, "label": FEATURES[n].label} for n in profile.features],
        "video": {"filename": Path(video).name, "frame_size": list(mp.frame_size), "samples": int(len(mp.t)),
                  "sample_fps": options.sample_fps, "duration": float(probe(Path(video))["duration"]),
                  "video_url": f"/media/{view_path(proxy).name}"},
        "model": {"backend": backend.name, "id": backend.model_id, "device": backend.device},
        "params": asdict(params),
        "identities": [{
            "id": i.id, "performer": i.performer, "tracklets": len(i.tracklets), "samples": int(len(i.frames)),
            "first_s": float(mp.t[i.frames[0]]) if len(i.frames) else None,
            "last_s": float(mp.t[i.frames[-1]]) if len(i.frames) else None,
            "foot_px": [round(v, 1) for v in i.foot], "stature_px": round(i.stature_px, 1),
            "layout": i.layout, "screening": i.screening, "report": i.id in reports,
            "runs": _runs(mp, i), "joins": _joins(mp, i)} for i in ids],
        "tracks": {str(i.id): _track(mp, i) for i in ids},
        "diagnostics": diag, "camera": cam, "common_axis": common, "persons": persons, "group": metrics, "timings_s": timings,
    }
    (out_dir / "group.json").write_text(to_json(out, indent=1), encoding="utf-8")
    np.save(out_dir / "identity_frames.npy", np.array([[i.id, int(f)] for i in ids for f in i.frames], dtype=np.int32))
    return out


def _runs(mp, idn, gap_s: float = 1.0) -> list[list[float]]:
    """Seen intervals [start, end] seconds (gaps shorter than ``gap_s`` bridged)."""
    if not len(idn.frames):
        return []
    t = mp.t[idn.frames]
    cut = np.flatnonzero(np.diff(t) > gap_s)
    starts, ends = np.r_[0, cut + 1], np.r_[cut, len(t) - 1]
    return [[round(float(t[a]), 2), round(float(t[b]), 2)] for a, b in zip(starts, ends)]


def _joins(mp, idn) -> list[float]:
    """Times where consecutive detections come from different tracklets: the stitch points to review."""
    if len(idn.frames) < 2:
        return []
    k = np.flatnonzero(np.diff(idn.trk) != 0)
    return [round(float(mp.t[idn.frames[i + 1]]), 2) for i in k]


def _track(mp, idn, step: int = 2) -> dict:
    """Boxes for the overview overlay, every ``step`` samples: t (s) and integer x0 y0 x1 y1 (pixels)."""
    fr, de = idn.frames[::step], idn.dets[::step]
    return {"t": [round(float(v), 2) for v in mp.t[fr]], "b": np.round(mp.box[de]).astype(int).tolist()}


def _common_axis(reference, video: Path, out_dir: Path, data_dir: Path, models_dir: Path, opts: AnalysisOptions,
                 mp, performers, params: GroupParams) -> dict:
    """The group's analysis interval and section bounds, used for every performer.

    1. ``params.sections`` (set by a person in the overview, "修正动作段"): used as given.
    2. Vote: the best-measured performers (seen longest, large enough, moving most like the group) each locate
       the sections on their own; a section is kept when at least 40 % of them (and two) found it, its bounds are the median of
       theirs (each shifted by the voter's own lag). Averaging poses across people first, as the consensus
       skeleton does, cancels movements done in slightly different phase or direction (2026-10-22: form 5).
    3. Fallback when fewer than two voters succeed: the consensus skeleton."""
    if params.sections:
        b = {k: (float(v[0]), float(v[1])) for k, v in params.sections.items() if float(v[1]) > float(v[0])}
        order = [g.id for g in reference.segments(load_profile(reference.raw["profile"]))]
        segs = [k for k in order if k in b]
        if segs:
            return {"method": "manual", "segments": segs, "section_bounds": {k: b[k] for k in segs},
                    "user_range": [min(b[k][0] for k in segs), max(b[k][1] for k in segs)],
                    "reason": "人工指定的动作段", "members": [i.id for i in performers]}
    if len(performers) < 2:
        return {}
    ranked = sorted(performers, key=lambda i: -(i.screening.get("presence", 0) * max(0.0, i.screening.get("similarity") or 0.5)))
    voters = [i for i in ranked if i.screening.get("presence", 0) >= 0.6][: params.voters]
    votes, errors = [], []
    for v in voters:
        try:
            rep = run_analysis(reference, video, out_dir / "axis" / f"voter_{v.id}", data_dir, models_dir, opts,
                               user_poses=identity_sequence(mp, v))
        except Exception as e:
            errors.append(f"#{v.id}: {type(e).__name__}: {e}")
            continue
        lag = float(v.screening.get("lag_s") or 0.0)
        votes.append({"id": v.id, "bounds": {g["id"]: (g["user_start"] - lag, g["user_end"] - lag) for g in rep["segments"]
                                             if g.get("user_end") is not None and g["user_end"] > g["user_start"]},
                      "needs_review": rep["scope"].get("needs_review", False)})
    if len(votes) >= 2:
        order = [g.id for g in reference.segments(load_profile(reference.raw["profile"]))]
        need = max(2, int(np.ceil(0.4 * len(votes))))   # 40 %: a recording that starts mid-section splits the voters
        count = {k: sum(k in vt["bounds"] for vt in votes) for k in order}
        segs = [k for k in order if count[k] >= need]
        if segs:
            b = {k: [float(np.median([vt["bounds"][k][0] for vt in votes if k in vt["bounds"]])),
                     float(np.median([vt["bounds"][k][1] for vt in votes if k in vt["bounds"]]))] for k in segs}
            for k0, k1 in zip(segs[:-1], segs[1:]):          # consecutive sections must not overlap
                if b[k0][1] > b[k1][0]:
                    mid = 0.5 * (b[k0][1] + b[k1][0])
                    b[k0][1], b[k1][0] = mid, mid
            return {"method": "vote", "segments": segs, "section_bounds": {k: tuple(b[k]) for k in segs},
                    "user_range": [max(0.0, b[segs[0]][0]), b[segs[-1]][1]],
                    "reason": f"{len(votes)} 位画面最好的练习者各自分段，多数认定的动作段",
                    "votes": count, "voters": [vt["id"] for vt in votes], "errors": errors,
                    "members": [i.id for i in performers]}
    cons = consensus_sequence(mp, performers)
    if cons is None:
        return {"errors": errors}
    try:
        rep = run_analysis(reference, video, out_dir / "consensus", data_dir, models_dir, opts, user_poses=cons)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "errors": errors}
    sc = rep["scope"]
    return {"method": "consensus", "segments": sc["segments"], "ref_range": sc["ref_range"], "user_range": sc["user_range"],
            "reason": sc["reason"], "candidates": [{k: c[k] for k in ("segment", "eligible", "reasons", "user_range")}
                                                   for c in sc.get("candidates", [])],
            "section_bounds": {g["id"]: (g["user_start"], g["user_end"]) for g in rep["segments"]},
            "members": [i.id for i in performers], "tempo_ratio": rep["timing"]["tempo_ratio"], "errors": errors}
