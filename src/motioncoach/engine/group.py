"""Several people in one video: stable identities, approximate floor layout, performer screening, group metrics.

1. **Tracklets** come from ``engine.tracking`` (strict, so a tracklet is one person).
2. **Baseline** (the first ``baseline_s`` seconds): every long tracklet there becomes an identity "slot" with
   its foot position, stature, appearance and limb proportions.
3. **Layout** without assuming equal statures (real heights differ by 10-20 % in a mixed group, children far
   more). Depth comes from up to three cues, combined by their precision:
   - *head scale*: the distance between the eyes (adult mean 63.4 mm, SD 3.8 mm, 6 %; men and women differ by
     4 %) and between the ears (converted to eye units by the group's own median ratio, so no absolute head
     breadth is needed). The head changes far less with age and sex than stature, but it is small in the
     image, so the robust 80th percentile over all frames is used (frontal frames give the largest value);
   - *floor contact*: on a common floor and a level camera ``y_foot - y_horizon = f * h_cam / Z`` - no body
     size at all. The horizon and ``f * h_cam`` are fitted across people against the head-scale depth;
   - *stature* with the adult average, only as a weak fallback.
   Then ``X = (x_foot - cx) * Z / f`` and each person's real stature ``stature_px * Z / f`` (reported).
   Metric values need the focal length, from an assumed horizontal field of view; ratios do not. The floor
   residual (foot point vs. the fitted floor at the head-scale depth) marks people off that floor.
4. **Stitching**: tracklets, longest first, join the slot with the lowest cost - distance in body heights
   (scale-free, so near and far people are treated alike), stature ratio, appearance and limb proportions -
   provided they do not overlap in time with the slot's other tracklets. Position dominates because
   similar clothes make appearance weak while a practising group keeps its places.
5. **Performer screening**: an identity is a performer when it is large enough to be measured, present long
   enough, and moves; one far behind the group is an onlooker when it is also off the group's floor, or moves
   unlike the group (lagged correlation of a mirror-invariant signal) and only a little. Every decision carries its
   reasons. **Edits** (merge, split, set performer / onlooker) are replayed in order on every run.
6. **Group metrics** from the per-person comparisons with the reference: synchronisation (each person's time
   offset against the group median for the same reference moment) and consistency (spread of the
   indicators across performers at the same reference moment).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .sequence import PoseSequence
from .skeleton import get_skeleton
from .tracking import TrackParams, track

LIMBS = (("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"),
         ("right_elbow", "right_wrist"), ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
         ("right_hip", "right_knee"), ("right_knee", "right_ankle"))
HEAD = ("nose", "left_eye", "right_eye", "left_ear", "right_ear")
FEET = ("left_ankle", "right_ankle", "left_heel", "right_heel")


@dataclass
class GroupParams:
    baseline_s: float = 10.0          # identities and their references are set up in this opening window
    max_persons: int = 12
    height_m: float = 1.64            # adult average stature: fallback depth cue only (engine/scale.py)
    ipd_m: float = 0.0634             # interpupillary distance, adult mean (Dodgson 2004, ANSUR data, SD 3.8 mm)
    head_q: float = 80.0              # percentile of the per-frame head scale (frontal frames are the largest)
    hfov_deg: float = 65.0            # assumed horizontal field of view for metric depth (phone main camera)
    # stitching cost weights and limit
    w_pos: float = 1.0                # per body height of foot distance
    w_height: float = 2.0             # per unit |log stature ratio|
    w_app: float = 1.0                # per unit (1 - Bhattacharyya coefficient)
    w_bone: float = 2.0               # per unit mean |limb / torso difference|
    max_cost: float = 1.2
    max_overlap: float = 0.05         # consecutive tracklets of one person may overlap by this share (>= 3 samples)
    min_new_identity_s: float = 3.0   # a later tracklet starts a new identity only if at least this long
    # performer screening
    max_depth_ratio: float = 1.8      # farther than this x the group's median depth counts as "behind the group"
    min_height_frac: float = 0.10     # stature below this fraction of the frame height cannot be measured
    min_presence: float = 0.3         # share of the analysed time the identity is seen
    min_motion_rel: float = 0.3       # motion energy >= this x the group median
    min_active_rel: float = 0.6       # a far person unlike the group must move at least this much to stay
    min_similarity: float = 0.3       # lagged correlation with the others' consensus
    crowd_size: int = 6               # from this many measurable people on, similarity to the group decides
    crowd_min_similarity: float = 0.55   # (2128.mp4: practitioners 0.66-0.92, placard bearers 0.0-0.45)
    max_floor_residual: float = 0.25  # |foot y - floor line| / stature; above it the person is off the floor
    include: tuple[int, ...] = ()     # manual overrides by identity id
    exclude: tuple[int, ...] = ()
    # ordered identity edits, replayed on every run (ids as they are at that point of the sequence):
    # {"op": "merge", "from": a, "into": b} | {"op": "split", "id": a, "at": seconds} | {"op": "role", "id": a, "performer": bool}
    edits: tuple[dict, ...] = ()
    sections: dict | None = None      # section id -> (start, end) video seconds set by a person; replaces the vote
    voters: int = 5                   # best-measured performers who locate the sections on their own
    stabilize: bool = True            # compensate camera motion before tracking / stitching / layout (media/camera.py)
    track: TrackParams = field(default_factory=TrackParams)


def _bhatt(a: np.ndarray, b: np.ndarray) -> float:
    """1 - Bhattacharyya coefficient over the two appearance halves present in both (0 = same colours)."""
    h = len(a) // 2
    vals = []
    for s in (slice(0, h), slice(h, None)):
        if a[s].sum() > 0 and b[s].sum() > 0:
            vals.append(1.0 - float(np.sqrt(a[s] * b[s]).sum()))
    return float(np.mean(vals)) if vals else 0.5


class _Geo:
    """Per-detection geometry from keypoints: foot point, stature, limb proportions."""

    def __init__(self, mp, conf_thr: float):
        sk = get_skeleton(mp.skeleton_name)
        J = {j: i for i, j in enumerate(sk.joints)}
        ok = mp.conf >= conf_thr
        xy = np.where(ok[..., None], mp.xy, np.nan)
        pick = lambda names: [J[n] for n in names if n in J]   # noqa: E731
        with np.errstate(all="ignore"):
            head = np.nanmin(xy[:, pick(HEAD), 1], axis=1)
            feet = xy[:, pick(FEET)]
            foot_y = np.nanmax(feet[..., 1], axis=1)
            foot_x = np.nanmean(feet[..., 0], axis=1)
            hip = 0.5 * (xy[:, J["left_hip"]] + xy[:, J["right_hip"]])
            sh = 0.5 * (xy[:, J["left_shoulder"]] + xy[:, J["right_shoulder"]])
            torso = np.linalg.norm(sh - hip, axis=1)
            # head-to-foot, completed for the part above the eyes (~6 % of stature, engine/scale.py);
            # without head or feet: from the torso (shoulder - hip = 0.288 H), feet at hip + 0.53 H
            stature = (foot_y - head) / 0.94
            stature = np.where(np.isfinite(stature) & (stature > 0), stature, torso / 0.288)
            foot_y = np.where(np.isfinite(foot_y), foot_y, hip[:, 1] + 0.53 * stature)
            self.foot = np.stack([np.where(np.isfinite(foot_x), foot_x, hip[:, 0]), foot_y], axis=1)
            self.stature = stature
            self.bones = np.stack([np.linalg.norm(xy[:, J[a]] - xy[:, J[b]], axis=1) / torso
                                   for a, b in LIMBS if a in J and b in J], axis=1)
            self.torso = torso
            self.hip = hip
            eye = [J[n] for n in ("left_eye", "right_eye") if n in J]
            ear = [J[n] for n in ("left_ear", "right_ear") if n in J]
            self.eye_d = np.linalg.norm(xy[:, eye[0]] - xy[:, eye[1]], axis=1) if len(eye) == 2 else np.full(len(xy), np.nan)
            self.ear_d = np.linalg.norm(xy[:, ear[0]] - xy[:, ear[1]], axis=1) if len(ear) == 2 else np.full(len(xy), np.nan)


@dataclass
class Identity:
    id: int
    tracklets: list[int]
    frames: np.ndarray                 # sample indices where the identity has a detection
    dets: np.ndarray                   # detection index per frame in ``frames``
    trk: np.ndarray                    # tracklet index per frame in ``frames`` (stitch points for review)
    foot: tuple[float, float] = (0.0, 0.0)
    stature_px: float = 0.0
    layout: dict = field(default_factory=dict)
    screening: dict = field(default_factory=dict)
    performer: bool = False


def _tracklet_summary(tr, geo: _Geo, mp) -> dict:
    d = np.array([x[1] for x in tr])
    f = np.array([x[0] for x in tr])
    with np.errstate(all="ignore"):
        return {"frames": f, "dets": d, "t0": int(f[0]), "t1": int(f[-1]),
                "foot0": np.nanmedian(geo.foot[d[:5]], axis=0), "foot1": np.nanmedian(geo.foot[d[-5:]], axis=0),
                "foot": np.nanmedian(geo.foot[d], axis=0), "stature": float(np.nanmedian(geo.stature[d])),
                "app": np.nanmean(mp.app[d], axis=0), "bones": np.nanmedian(geo.bones[d], axis=0)}


def build_identities(mp, p: GroupParams = GroupParams()) -> tuple[list[Identity], dict]:
    """MultiPoses -> identities (stable ids over the whole video) and diagnostics."""
    fps = float(mp.meta.get("sample_fps", 10.0))
    conf_thr = float(mp.meta.get("conf_thr", 0.3))
    T = len(mp.t)
    tracklets = track(mp.frame, mp.box, mp.score, mp.xy, mp.conf, T, fps, conf_thr, p.track)
    geo = _Geo(mp, conf_thr)
    S = [_tracklet_summary(tr, geo, mp) for tr in tracklets]
    base_n = int(round(p.baseline_s * fps))

    # ---- baseline slots: long tracklets in the opening window, largest first
    cand = [k for k, s in enumerate(S) if s["t0"] < base_n and len(s["frames"]) >= 0.5 * min(base_n, T)]
    cand.sort(key=lambda k: -S[k]["stature"])
    slots: list[dict] = []
    for k in cand[: p.max_persons]:
        slots.append({"members": [k], "ref": S[k], "frames": set(S[k]["frames"].tolist())})

    def cost(s: dict, slot: dict) -> float:
        ref = slot["ref"]
        h = max(1.0, ref["stature"])
        # foot distance to the slot's baseline place, or to where its previous tracklet ended (people may move)
        prev = max((S[m] for m in slot["members"] if S[m]["t1"] < s["t0"]), key=lambda x: x["t1"], default=None)
        d = np.linalg.norm(s["foot0"] - ref["foot"]) / h
        if prev is not None:
            d = min(d, np.linalg.norm(s["foot0"] - prev["foot1"]) / h)
        dh = abs(np.log(max(1.0, s["stature"]) / h))
        db = float(np.nanmean(np.abs(s["bones"] - ref["bones"]))) if np.isfinite(s["bones"]).any() else 0.5
        c = p.w_pos * d + p.w_height * dh + p.w_app * _bhatt(s["app"], ref["app"]) + p.w_bone * (db if np.isfinite(db) else 0.5)
        return float(c) if np.isfinite(c) else np.inf

    def free(slot, s) -> bool:
        """No (or only a brief hand-over) overlap in time: the tracker may start a new track just before
        dropping the old one, but two different people are seen together for long stretches."""
        both = len(slot["frames"].intersection(s["frames"].tolist()))
        return both <= max(3, p.max_overlap * len(s["frames"]))

    assigned = {k for sl in slots for k in sl["members"]}
    rejected = []
    for k in sorted((k for k in range(len(S)) if k not in assigned), key=lambda k: -len(S[k]["frames"])):
        s = S[k]
        opts = [(cost(s, sl), i) for i, sl in enumerate(slots) if free(sl, s)]
        best = min(opts, default=(np.inf, -1))
        if best[0] <= p.max_cost:
            sl = slots[best[1]]
            sl["members"].append(k)
            sl["frames"].update(s["frames"].tolist())
        elif len(s["frames"]) >= p.min_new_identity_s * fps and len(slots) < 3 * p.max_persons:
            slots.append({"members": [k], "ref": s, "frames": set(s["frames"].tolist())})   # someone who arrives later
        else:
            rejected.append({"tracklet": k, "frames": len(s["frames"]), "best_cost": None if not np.isfinite(best[0]) else round(float(best[0]), 2)})

    ids = []
    for i, sl in enumerate(slots):
        fr = np.concatenate([S[m]["frames"] for m in sl["members"]])
        de = np.concatenate([S[m]["dets"] for m in sl["members"]])
        tk = np.concatenate([np.full(len(S[m]["frames"]), m) for m in sl["members"]])
        ids.append(_identity(i, fr, de, tk, geo))
    applied = []
    for e in p.edits:                                          # structural edits, in the order they were made
        ids, ok = _apply_edit(ids, e, mp, geo)
        if ok:
            applied.append(e)
    _layout(ids, mp, geo, p)
    _screen(ids, mp, geo, p, T)
    for e in p.edits:                                          # role edits act on the final identities
        if e.get("op") == "role":
            for i in ids:
                if i.id == int(e["id"]):
                    i.performer = bool(e["performer"])
                    i.screening["reasons"] = i.screening.get("reasons", []) + ["人工指定为练习者" if i.performer else "人工排除"]
                    applied.append(e)
    switches = [{"identity": idn.id, "tracklets": len(idn.tracklets)} for idn in ids if len(idn.tracklets) > 1]
    diag = {"tracklets": len(S), "identities": len(ids), "rejected_tracklets": rejected,
            "stitched": switches, "edits_applied": applied, "params": asdict(p)}
    return ids, diag


def _identity(i: int, fr: np.ndarray, de: np.ndarray, tk: np.ndarray, geo: _Geo) -> Identity:
    o = np.argsort(fr, kind="stable")
    fr, de, tk = fr[o], de[o], tk[o]
    keep = np.r_[True, np.diff(fr) > 0] if len(fr) else np.zeros(0, bool)   # one detection per frame
    fr, de, tk = fr[keep], de[keep], tk[keep]
    with np.errstate(all="ignore"):
        foot = tuple(float(v) for v in np.nanmedian(geo.foot[de], axis=0)) if len(de) else (np.nan, np.nan)
        st = float(np.nanmedian(geo.stature[de])) if len(de) else np.nan
    return Identity(i, sorted(set(int(t) for t in tk)), fr, de, tk, foot, st)


def _apply_edit(ids: list[Identity], e: dict, mp, geo: _Geo) -> tuple[list[Identity], bool]:
    by = {i.id: i for i in ids}
    op = e.get("op")
    if op == "merge" and int(e["from"]) in by and int(e["into"]) in by and int(e["from"]) != int(e["into"]):
        a, b = by[int(e["from"])], by[int(e["into"])]
        own = np.isin(a.frames, b.frames, invert=True)          # where both were seen, the target keeps its own
        m = _identity(b.id, np.r_[b.frames, a.frames[own]], np.r_[b.dets, a.dets[own]], np.r_[b.trk, a.trk[own]], geo)
        return [m if i.id == b.id else i for i in ids if i.id != a.id], True
    if op == "split" and int(e["id"]) in by:
        a = by[int(e["id"])]
        after = mp.t[a.frames] >= float(e["at"])
        if after.any() and (~after).any():
            new_id = max(i.id for i in ids) + 1
            keep = _identity(a.id, a.frames[~after], a.dets[~after], a.trk[~after], geo)
            moved = _identity(new_id, a.frames[after], a.dets[after], a.trk[after], geo)
            return [keep if i.id == a.id else i for i in ids] + [moved], True
    return ids, False


def _layout(ids: list[Identity], mp, geo: _Geo, p: GroupParams) -> None:
    """Floor positions (metres) from head scale and floor contact; stature only as a fallback (module docstring)."""
    W, _ = mp.frame_size
    f_px = (W / 2) / np.tan(np.radians(p.hfov_deg) / 2)
    fps = float(mp.meta.get("sample_fps", 10.0))

    def head_len(d: np.ndarray) -> float:
        """High percentile of a 1 s rolling median: frontal frames, without keypoint jitter inflating it."""
        d = d[np.isfinite(d)]
        if len(d) < 10:
            return np.nan
        w = max(3, int(fps) | 1)
        sm = np.array([np.median(d[max(0, k - w // 2):k + w // 2 + 1]) for k in range(len(d))])
        return float(np.percentile(sm, p.head_q))

    eye = np.array([head_len(geo.eye_d[i.dets]) for i in ids])
    ear = np.array([head_len(geo.ear_d[i.dets]) for i in ids])
    both = np.isfinite(eye) & np.isfinite(ear)
    ratio = float(np.median(ear[both] / eye[both])) if both.any() else None      # ear distance in eye units
    head = np.where(np.isfinite(ear) & (ratio is not None), ear / (ratio or 1.0), eye)   # eye-equivalent pixels
    # relative depth errors (1 sigma): population spread + ~1.5 px keypoint error on the measured length
    z_head = f_px * p.ipd_m / head
    e_head = np.sqrt(0.06 ** 2 + (1.5 / np.where(np.isfinite(ear) & (ratio is not None), ear, eye)) ** 2)
    st = np.array([i.stature_px for i in ids], float)
    z_st = f_px * p.height_m / st
    e_st = np.sqrt(0.09 ** 2 + (3.0 / st) ** 2)
    fy = np.array([i.foot[1] for i in ids], float)
    # floor: y_foot = y_h + c / Z, fitted against the best size-based depth of each person (robust)
    z_size = np.where(np.isfinite(z_head), z_head, z_st)
    okf = np.isfinite(z_size) & np.isfinite(fy)
    y_h = c = None
    if okf.sum() >= 3 and np.ptp(1 / z_size[okf]) > 0.1 * np.median(1 / z_size[okf]):
        from scipy.stats import theilslopes
        c, y_h, _, _ = theilslopes(fy[okf], 1 / z_size[okf])
        if not (c > 0):
            y_h = c = None
    for k, i in enumerate(ids):
        cues = {}
        if np.isfinite(z_head[k]):
            cues["head"] = (float(z_head[k]), float(e_head[k]))
        if np.isfinite(z_st[k]):
            cues["stature"] = (float(z_st[k]), float(e_st[k]))
        res = None
        if y_h is not None and np.isfinite(fy[k]) and fy[k] - y_h > 5:
            cues["floor"] = (float(c / (fy[k] - y_h)), float(np.sqrt(0.05 ** 2 + (4.0 / (fy[k] - y_h)) ** 2)))
            if np.isfinite(z_size[k]) and st[k] > 0:
                res = float((fy[k] - (y_h + c / z_size[k])) / st[k])
        if not cues:
            i.layout = {"x_m": None, "z_m": None, "height_m": None, "floor_residual": None, "cues": {}}
            continue
        # off-floor people (raised, seated) must not use the floor cue
        use = {n: v for n, v in cues.items() if not (n == "floor" and res is not None and abs(res) > p.max_floor_residual)}
        w = {n: 1 / e ** 2 for n, (_, e) in use.items()}
        z = float(np.exp(sum(w[n] * np.log(z_) for n, (z_, _) in use.items()) / sum(w.values())))
        i.layout = {"x_m": round(float((i.foot[0] - W / 2) * z / f_px), 2), "z_m": round(z, 2),
                    "height_m": round(float(st[k] * z / f_px), 2) if st[k] > 0 else None,
                    "floor_residual": None if res is None else round(res, 3),
                    "cues": {n: round(v[0], 2) for n, v in cues.items()},
                    "note": "按头部尺度与脚点着地位置估算，假定视场角；仅表示大致前后左右关系"}


def _motion_signal(mp, geo: _Geo, idn: Identity, T: int) -> np.ndarray:
    """Per-sample body-normalised joint spread (mirror invariant), NaN where the identity is absent."""
    out = np.full(T, np.nan)
    d = idn.dets
    with np.errstate(all="ignore"):
        rel = (mp.xy[d] - geo.hip[d][:, None]) / geo.torso[d][:, None, None]
        vis = mp.conf[d] >= float(mp.meta.get("conf_thr", 0.3))
        rel = np.where(vis[..., None], rel, np.nan)
        out[idn.frames] = np.nanmean(np.abs(rel[..., 0]), axis=1) + np.nanmean(rel[..., 1], axis=1)
    return out


def _lagged_corr(a: np.ndarray, b: np.ndarray, max_lag: int, with_lag: bool = False):
    """Best correlation of ``a`` against ``b`` within +-max_lag samples; with the lag (> 0: ``a`` later)."""
    best, best_lag = -1.0, 0
    for lag in range(-max_lag, max_lag + 1, 2):
        x, y = (a[lag:], b[:len(b) - lag]) if lag >= 0 else (a[:lag], b[-lag:])
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() > 30 and np.std(x[m]) > 1e-6 and np.std(y[m]) > 1e-6:
            c = float(np.corrcoef(x[m], y[m])[0, 1])
            if c > best:
                best, best_lag = c, lag
    return (best, best_lag) if with_lag else best


def _screen(ids: list[Identity], mp, geo: _Geo, p: GroupParams, T: int) -> None:
    fps = float(mp.meta.get("sample_fps", 10.0))
    if not ids:
        return
    sig = {i.id: _motion_signal(mp, geo, i, T) for i in ids}
    def amplitude(v: np.ndarray) -> float:
        """Range of the 1 s-smoothed signal (P90 - P10): movement, not keypoint jitter."""
        ok = np.isfinite(v)
        if ok.sum() <= 2 * fps:
            return 0.0
        x = np.convolve(np.interp(np.arange(len(v)), np.flatnonzero(ok), v[ok]), np.ones(int(fps)) / int(fps), "same")
        x = x[ok]
        return float(np.percentile(x, 90) - np.percentile(x, 10))

    energy = {k: amplitude(v) for k, v in sig.items()}
    # the group: identities seen long enough and big enough to measure; its median depth is the reference
    big = [i.id for i in ids if len(i.frames) >= p.min_presence * T and i.stature_px >= p.min_height_frac * mp.frame_size[1]] \
        or [i.id for i in ids]
    e_med = float(np.median([energy[k] for k in big])) if big else 0.0
    zs = [i.layout.get("z_m") for i in ids if i.id in big and i.layout.get("z_m") and energy[i.id] >= p.min_motion_rel * e_med]
    z_ref = float(np.median(zs)) if zs else None
    sims = {}
    for i in ids:                                         # similarity to the others' median motion, and the lag
        others = [sig[k] for k in big if k != i.id]
        if others:
            with np.errstate(all="ignore"):
                cons = np.nanmedian(np.stack(others), axis=0)
            sim, lag = _lagged_corr(sig[i.id], cons, int(5 * fps), with_lag=True)
            sims[i.id] = sim
            i.screening["lag_s"] = round(lag / fps, 1)
    coherent = sum(v is not None and v >= p.crowd_min_similarity for v in sims.values()) >= p.crowd_size
    for i in ids:
        reasons, ok = [], True
        presence = len(i.frames) / max(1, T)
        small_abs = i.stature_px < p.min_height_frac * mp.frame_size[1]
        z = i.layout.get("z_m")
        far = z is not None and z_ref is not None and z > p.max_depth_ratio * z_ref
        if small_abs:
            ok = False
            reasons.append(f"人物过小（身高 {i.stature_px:.0f} px，低于画高的 {p.min_height_frac * 100:.0f}%），无法测量")
        elif far:
            reasons.append(f"位置远在人群之后（约 {z:.1f} m，人群约 {z_ref:.1f} m），可能是背景")
        if presence < p.min_presence:
            ok = False
            reasons.append(f"出现时间短（{presence * 100:.0f}%）")
        sim = sims.get(i.id)
        moving = energy[i.id] >= p.min_motion_rel * e_med if e_med > 0 else True
        # many people moving together: moving like the group is what makes a practitioner. Only when such a
        # group exists - a group standing still has no common motion, and similarity is then just noise
        crowd = len(big) >= p.crowd_size and coherent
        similar = sim is None or sim >= (p.crowd_min_similarity if crowd else p.min_similarity)
        res = i.layout.get("floor_residual")
        off_floor = res is not None and abs(res) > p.max_floor_residual
        if not moving:
            ok = False
            reasons.append(f"几乎不动（运动量为群体中位数的 {energy[i.id] / e_med * 100:.0f}%）")
        if not similar:
            reasons.append(f"动作与集体不一致（相关 {sim:.2f}），可能是领操、标牌手或观众" if crowd
                           else f"动作与其他人不同步或不相似（相关 {sim:.2f}）")
        if crowd and not similar:
            ok = False
        if off_floor:
            reasons.append(f"脚点不在主体人群所在地面上（偏离 {res:+.2f} 身高）")
        # behind the group AND off its floor -> onlooker (stands, raised seats); behind the group, unlike it AND moving
        # little -> onlooker; a back row on the same floor that clearly moves stays a practitioner (reasons listed)
        weak = energy[i.id] < p.min_active_rel * e_med if e_med > 0 else False
        if far and (off_floor or (not similar and weak)):
            ok = False
        if i.id in p.include:
            ok, reasons = True, reasons + ["人工指定为练习者"]
        if i.id in p.exclude:
            ok, reasons = False, reasons + ["人工排除"]
        i.performer = ok
        i.screening = {**i.screening, "presence": round(presence, 3), "depth_rel": round(z / z_ref, 2) if z and z_ref else None,
                       "motion_rel": round(energy[i.id] / e_med, 2) if e_med else None,
                       "similarity": None if sim is None else round(sim, 2), "reasons": reasons}


def identity_sequence(mp, idn: Identity) -> PoseSequence:
    """One identity as a single-person sequence on the full sampling grid (NaN where not seen)."""
    T, J = len(mp.t), mp.xy.shape[1]
    xy = np.full((T, J, 2), np.nan, np.float32)
    conf = np.zeros((T, J), np.float32)
    xy[idn.frames] = mp.xy[idn.dets]
    conf[idn.frames] = mp.conf[idn.dets]
    return PoseSequence(t=mp.t.copy(), xy=xy, conf=conf, world=None, frame_size=tuple(mp.frame_size),
                        skeleton_name=mp.skeleton_name,
                        meta={**mp.meta, "identity": idn.id, "window": [float(mp.t[0]), None]})


def consensus_sequence(mp, ids: list[Identity], min_share: float = 0.3) -> PoseSequence | None:
    """The group's median pose per frame: each performer's skeleton normalised (hip-midpoint origin,
    own median torso length as unit), the median across performers, drawn at a fixed size. A group
    practising together moves together, so this sequence locates the routine's sections once for all."""
    if not ids:
        return None
    sk = get_skeleton(mp.skeleton_name)
    J = {j: k for k, j in enumerate(sk.joints)}
    thr = float(mp.meta.get("conf_thr", 0.3))
    T = len(mp.t)
    stack = []
    for idn in ids:
        seq = identity_sequence(mp, idn)
        xy = np.where((seq.conf >= thr)[..., None], seq.xy.astype(float), np.nan)
        hip = 0.5 * (xy[:, J["left_hip"]] + xy[:, J["right_hip"]])
        sh = 0.5 * (xy[:, J["left_shoulder"]] + xy[:, J["right_shoulder"]])
        with np.errstate(all="ignore"):
            torso = float(np.nanmedian(np.linalg.norm(sh - hip, axis=1)))
        if not np.isfinite(torso) or torso <= 0:
            continue
        stack.append((xy - hip[:, None]) / torso)
    if not stack:
        return None
    A = np.stack(stack)                                          # (P, T, J, 2)
    n = np.isfinite(A).all(axis=3).sum(axis=0)                   # (T, J) performers seeing each joint
    with np.errstate(all="ignore"):
        med = np.nanmedian(A, axis=0)
    ok = n >= max(2, int(np.ceil(min_share * len(stack))))
    unit, origin = 150.0, np.array([400.0, 520.0])               # a 800 x 1200 px canvas
    xy = np.where(ok[..., None], med * unit + origin, np.nan).astype(np.float32)
    conf = np.where(ok, 0.9, 0.0).astype(np.float32)
    return PoseSequence(t=mp.t.copy(), xy=xy, conf=conf, world=None, frame_size=(800, 1200),
                        skeleton_name=mp.skeleton_name,
                        meta={**mp.meta, "identity": "consensus", "members": [i.id for i in ids], "conf_thr": thr})


def _fill(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(v)
    if ok.sum() < 2:
        return np.zeros_like(v), ok
    return np.interp(np.arange(len(v)), np.flatnonzero(ok), v[ok]), ok


def _track_pair(x, xok, y, yok, centers, w, ml) -> tuple[np.ndarray, np.ndarray]:
    """Lag of x behind y (samples) in each window, searched around the previous window's lag (slow drift is
    followed, a repeated move cannot make it jump by a whole repetition); correlation per window."""
    T = len(x)
    lags, rs, prev = np.zeros(len(centers), int), np.full(len(centers), -1.0), 0
    for k, c in enumerate(centers):
        a = c - w // 2
        if yok[a:a + w].mean() < 0.6:
            continue
        yy = y[a:a + w] - y[a:a + w].mean()
        cand = [lag for lag in range(prev - ml, prev + ml + 1) if 0 <= a + lag and a + lag + w <= T and xok[a + lag:a + lag + w].mean() >= 0.6]
        if not cand or not np.any(yy):
            continue
        X = np.stack([x[a + lag:a + lag + w] for lag in cand])
        X = X - X.mean(axis=1, keepdims=True)
        den = np.linalg.norm(X, axis=1) * np.linalg.norm(yy)
        r = np.where(den > 0, X @ yy / np.where(den > 0, den, 1), -1)
        j = int(np.argmax(r))
        lags[k], rs[k] = cand[j], r[j]
        if r[j] >= 0.5:
            prev = cand[j]
    return lags, rs


def signal_sync(mp, ids: list[Identity], bounds: dict | None = None, win_s: float = 8.0, step_s: float = 2.0,
                max_lag_s: float = 3.0, min_r: float = 0.5) -> dict:
    """Synchronisation from the motion signals themselves, not from the repetition pairing (which can slip by a
    whole repetition when the same move repeats every few seconds).

    For every pair of performers, the lag in sliding windows (``_track_pair``); pairs that do not move
    alike in a window (correlation < ``min_r``) are left out. Per window, one offset per performer is fitted
    to all pairwise lags (weighted least squares, o_i - o_j = lag_ij), then expressed relative to the median
    performer (> 0: later than the group). ``bounds``: section id -> (start, end) video seconds."""
    if len(ids) < 2:
        return {}
    fps = float(mp.meta.get("sample_fps", 10.0))
    geo = _Geo(mp, float(mp.meta.get("conf_thr", 0.3)))
    T = len(mp.t)
    sig = [_fill(_motion_signal(mp, geo, i, T)) for i in ids]
    w, st, ml = int(win_s * fps), int(step_s * fps), int(max_lag_s * fps)
    centers = list(range(w // 2, T - w // 2 - 1, st))
    n, K = len(ids), len(centers)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            lag, r = _track_pair(sig[i][0], sig[i][1], sig[j][0], sig[j][1], centers, w, ml)
            pairs.append((i, j, lag, r))
    off = np.full((n, K), np.nan)
    for k in range(K):
        rows, rhs, wts = [], [], []
        for i, j, lag, r in pairs:
            if r[k] >= min_r:
                row = np.zeros(n)
                row[i], row[j] = 1.0, -1.0
                rows.append(row)
                rhs.append(lag[k] / fps)
                wts.append(r[k])
        if not rows:
            continue
        A, b, W = np.array(rows), np.array(rhs), np.sqrt(np.array(wts))
        seen = np.flatnonzero(np.abs(A).sum(axis=0) > 0)
        if len(seen) < 2:
            continue
        A2 = np.vstack([A[:, seen] * W[:, None], np.ones(len(seen))])   # sum of offsets = 0 fixes the origin
        o, *_ = np.linalg.lstsq(A2, np.r_[b * W, 0.0], rcond=None)
        off[seen, k] = o - np.median(o)
    series, per = {}, {}
    for r_i, i in enumerate(ids):
        v = off[r_i]
        series[str(i.id)] = [None if not np.isfinite(x) else round(float(x), 1) for x in v]
        ok = np.isfinite(v)
        seg = {}
        for sid, (sa, sb) in (bounds or {}).items():
            m = ok & np.array([sa <= mp.t[c] < sb for c in centers])
            if m.sum() >= 2:
                seg[sid] = round(float(np.median(v[m])), 1)
        per[i.id] = {"offset_median_s": round(float(np.median(v[ok])), 1) if ok.any() else None,
                     "offset_abs_p90_s": round(float(np.percentile(np.abs(v[ok]), 90)), 1) if ok.any() else None,
                     "offset_by_segment_s": seg, "windows": int(ok.sum())}
    return {"sync": per, "sync_series": {"t": [round(float(mp.t[c]), 1) for c in centers], "offset": series, "axis": "video"},
            "sync_definition": f"视频中同一时刻，该练习者的动作相对集体中位的时间差（{win_s:g} s 滑动窗口，两两比较动作信号后联合求解；"
                               f"正 = 落后；动作不相似（相关 < {min_r}）的组合不计）"}


def group_timeline(mp, ids: list[Identity], names: list[str], tol: dict, bounds: dict | None = None,
                   step: int = 5, cap: float = 3.0) -> dict:
    """Group indicators along the video's own time (what the overview timeline scrubs through), every ``step``
    samples. At each moment, from every performer's own indicators (same moment, so timing differences count):
    - ``spread[f]``: standard deviation across performers / tolerance (how uneven the group is);
    - ``consistency``: median of the spreads (one number per moment; lower = more uniform);
    - ``deviation[id]``: mean |indicator - group median| / tolerance of one performer (capped);
    - ``section``: the common-axis section the moment belongs to."""
    from .features import compute_features
    from .filters import one_euro_clean

    T = len(mp.t)
    keep = np.arange(0, T, step)
    tl = np.array([tol[n] for n in names], float)
    F = []
    for i in ids:
        seq = identity_sequence(mp, i)
        seq = one_euro_clean(seq, conf_thr=float(mp.meta.get("conf_thr", 0.3)))
        F.append(compute_features(seq, names)[0][keep] / tl)
    if len(F) < 2:
        return {}
    X = np.stack(F)                                               # (P, K, F) tolerance units
    import warnings
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)           # all-NaN moments are expected (nobody visible)
        n_ok = np.isfinite(X).sum(axis=0)                         # (K, F)
        sd = np.where(n_ok >= 2, np.nanstd(X, axis=0), np.nan)
        med = np.nanmedian(X, axis=0)
        cons = np.nanmedian(sd, axis=1)
        dev = np.minimum(np.nanmean(np.abs(X - med[None]), axis=2), cap)   # (P, K)
    r = lambda v: None if not np.isfinite(v) else round(float(v), 2)   # noqa: E731
    tt = mp.t[keep]
    sec = [next((sid for sid, (a, b) in (bounds or {}).items() if a <= t < b), None) for t in tt]
    return {"t": [round(float(t), 2) for t in tt], "section": sec,
            "consistency": [r(v) for v in cons],
            "spread": {n: [r(v) for v in sd[:, k]] for k, n in enumerate(names)},
            "deviation": {str(i.id): [r(v) for v in dev[k]] for k, i in enumerate(ids)},
            "seen": [int(v) for v in np.isfinite(X).any(axis=2).sum(axis=0)],
            "definition": "同一时刻各练习者指标的标准差 / 允许偏差；偏离 = 个人与集体中位之差 / 允许偏差（平均，上限 3）"}


# ----------------------------------------------------------------------------- group metrics

def group_metrics(reports: dict[int, dict], names: list[str]) -> dict:
    """Synchronisation and consistency from per-person reports. Each report may cover a different interval,
    so everything is keyed by the reference time (10 fps grid)."""
    if not reports:
        return {}
    key = lambda t: int(round(float(t) * 10))   # noqa: E731
    ids = sorted(reports)
    grid = sorted({key(t) for r in reports.values() for t in r["series"]["t_ref"]})
    col = {g: i for i, g in enumerate(grid)}
    t_ref = np.array(grid, float) / 10
    M = np.full((len(ids), len(grid)), np.nan)              # practitioner time of each reference moment
    V = {f: np.full((len(ids), len(grid)), np.nan) for f in names}
    for r_i, k in enumerate(ids):
        ser = reports[k]["series"]
        cols = [col[key(t)] for t in ser["t_ref"]]
        M[r_i, cols] = [np.nan if v is None else v for v in ser["user_t"]]
        for f in names:
            u = ser.get("user", {}).get(f)
            if u is not None:
                V[f][r_i, cols] = [np.nan if v is None else v for v in u]
    with np.errstate(all="ignore"):
        med = np.nanmedian(M, axis=0)
    segs = max((r["segments"] for r in reports.values()), key=len)
    tol = {x["name"]: x["tolerance"]["good"] for x in next(iter(reports.values()))["features"]}
    per_person = {}
    for row, k in zip(M, ids):
        off = row - med                                    # > 0: later than the group (lagging)
        n_others = np.isfinite(M).sum(axis=0) - np.isfinite(row)
        ok = np.isfinite(off) & (n_others >= 1)
        seg_off = {}
        for s in segs:
            m = (t_ref >= s["ref_start"]) & (t_ref < s["ref_end"]) & ok
            if m.sum() >= 5:
                seg_off[s["id"]] = round(float(np.median(off[m])), 2)
        per_person[k] = {"offset_median_s": round(float(np.median(off[ok])), 2) if ok.any() else None,
                         "offset_abs_p90_s": round(float(np.percentile(np.abs(off[ok]), 90)), 2) if ok.any() else None,
                         "offset_by_segment_s": seg_off}
    consistency = {}
    for s in segs:
        m = (t_ref >= s["ref_start"]) & (t_ref < s["ref_end"])
        row = {}
        for f in names:
            X = V[f][:, m]
            enough = np.isfinite(X).sum(axis=0) >= 2
            if enough.any() and tol.get(f):
                with np.errstate(all="ignore"):
                    row[f] = round(float(np.nanmedian(np.nanstd(X[:, enough], axis=0)) / tol[f]), 2)
        consistency[s["id"]] = row
    step = 10                                               # 1 s on the 10 fps reference grid
    keep = [k for k in range(len(grid)) if grid[k] % step == 0]
    with np.errstate(all="ignore"):
        off_all = M - med
    n_seen = np.isfinite(M).sum(axis=0)
    series = {"t": [round(float(t_ref[k]), 1) for k in keep],
              "offset": {str(k_id): [None if not (np.isfinite(off_all[r, k]) and n_seen[k] >= 2) else round(float(off_all[r, k]), 2)
                                     for k in keep] for r, k_id in enumerate(ids)}}
    return {"sync": per_person, "sync_series": series, "consistency": consistency,
            "definition": {"offset": "同一标准时刻，该练习者的时间减去群体中位数（秒，正 = 落后）",
                           "consistency": "同一标准时刻各练习者指标的标准差 / 允许偏差（越小越整齐）"}}
