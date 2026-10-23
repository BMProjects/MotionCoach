"""Scoring of a method's correspondence and of the playback it implies.

Two time maps are derived from every result:
  u2r  practitioner-master playback: for each practitioner sample, which reference sample to show;
  r2u  reference-master playback: for each reference sample, which practitioner sample to show.
"loop" policy (proposed): inside paired repetitions follow the in-repetition correspondence; between
them play at ~1x and *jump* (loop the reference template back / skip extra practitioner repetitions)
instead of fast-forwarding. "ui" policy reproduces the current workbench (M1 only).
"""

from __future__ import annotations

import numpy as np

from ..engine.normalize import CANON
from .data import Pair, section_index
from .methods import DT, SectionOut

_LR = [CANON.index(j.replace("left_", "right_")) if j.startswith("left_") else
       CANON.index(j.replace("right_", "left_")) if j.startswith("right_") else i for i, j in enumerate(CANON)]


# ----------------------------------------------------------------------------- maps

def paired_map(secs: list[SectionOut], n_user: int) -> tuple[np.ndarray, np.ndarray]:
    """Reference index (float) for each practitioner sample inside a paired repetition, and the rep's mirror flag."""
    r = np.full(n_user, np.nan)
    mir = np.zeros(n_user, dtype=bool)
    for s in secs:
        for u in s.user_reps:
            if u.paired and u.path is not None:
                ui, ri = u.path
                r[ui] = ri
                mir[ui] = u.mirror
    return r, mir


def _fill(anchors: list[tuple[int, np.ndarray]], X0: int, X1: int, Y0: float, Y1: float) -> np.ndarray:
    """anchors: (x_start, y values for x_start..). Gaps: linear if the implied speed is within [0.25, 4],
    otherwise ~1x out of the previous anchor, a jump, and ~1x into the next one."""
    y = np.full(X1 - X0, np.nan)
    pts = [(X0, np.array([Y0]))] + sorted(anchors, key=lambda a: a[0]) + [(X1 - 1, np.array([Y1]))]
    for xs, ys in pts:
        lo, hi = max(xs, X0), min(xs + len(ys), X1)
        if hi > lo:
            y[lo - X0:hi - X0] = ys[lo - xs:hi - xs]
    for (xa, ya), (xb, yb) in zip(pts[:-1], pts[1:]):
        g0, g1 = xa + len(ya), xb                  # gap [g0, g1)
        if g1 <= g0:
            continue
        y_from, y_to = float(ya[-1]), float(yb[0])
        g = g1 - g0 + 1
        speed = (y_to - y_from) / g
        idx = np.arange(g0, g1)
        if 0.25 <= speed <= 4.0:
            y[idx - X0] = y_from + (idx - g0 + 1) * speed
        else:
            half = (g0 + g1) // 2
            y[np.arange(g0, half) - X0] = y_from + (np.arange(g0, half) - g0 + 1)
            y[np.arange(half, g1) - X0] = y_to - (g1 - np.arange(half, g1))
    return y


def playback_loop(pair: Pair, secs: list[SectionOut]) -> tuple[tuple[int, np.ndarray], tuple[int, np.ndarray]]:
    """(U0, u2r over user[U0:U1]), (R0, r2u over ref[R0:R1]) in sample indices (floats)."""
    U0, U1 = secs[0].u0, secs[-1].u1
    R0, R1 = secs[0].r0, secs[-1].r1
    ua, ra = [], []
    for s in secs:
        for u in s.user_reps:
            if u.paired and u.path is not None:
                ui, ri = u.path
                ua.append((int(ui[0]), ri.astype(np.float64)))
                if u.default:   # inverse path: practitioner sample for each reference sample of the rep
                    rr = np.arange(int(ri.min()), int(ri.max()) + 1)
                    order = np.argsort(ri, kind="stable")
                    uu = np.interp(rr, ri[order].astype(np.float64), ui[order].astype(np.float64))
                    ra.append((int(rr[0]), uu))
    ua = _drop_overlaps(ua)
    ra = _drop_overlaps(ra)
    u2r = _fill(ua, U0, U1, R0, R1 - 1)
    r2u = _fill(ra, R0, R1, U0, U1 - 1)
    return (U0, u2r), (R0, r2u)


def _drop_overlaps(anchors):
    out, end = [], -1
    for xs, ys in sorted(anchors, key=lambda a: a[0]):
        if xs >= end:
            out.append((xs, ys))
            end = xs + len(ys)
    return out


def playback_ui_m1(pair: Pair, secs: list[SectionOut]):
    """The current workbench: ref-master = continuous interpolated map; practitioner-master = per-repetition
    linear map, waiting at the next repetition's start inside a section, else the inverse map."""
    ex = secs[0].extra
    tr = pair.ref.t
    ut = np.asarray(ex["user_t_for_ref"], dtype=np.float64)
    U0, U1 = secs[0].u0, secs[-1].u1
    R0, R1 = secs[0].r0, secs[-1].r1
    r2u = ut[R0:R1] / DT
    ok = np.isfinite(ut)
    rt, utt = tr[ok], ut[ok]
    M = sorted(ex["user_map"], key=lambda m: m[0])
    seg_of = lambda t: next((k for k, s in enumerate(secs) if s.r0 * DT <= t < s.r1 * DT), -1)   # noqa: E731
    u2r = np.empty(U1 - U0)
    for i, u in enumerate(np.arange(U0, U1) * DT):
        val = None
        for k, (us, ue, rs, re) in enumerate(M):
            if us <= u <= ue:
                val = rs + (u - us) / max(1e-9, ue - us) * (re - rs)
                break
            if k + 1 < len(M) and ue < u < M[k + 1][0] and seg_of(rs) == seg_of(M[k + 1][2]):
                val = M[k + 1][2]
                break
        if val is None:
            val = float(np.interp(u, utt, rt)) if len(utt) else u
        u2r[i] = val / DT
    return (U0, u2r), (R0, r2u)


# ----------------------------------------------------------------------------- speed / balance

def speed_stats(y: np.ndarray, valid: np.ndarray | None = None, win: int = 10) -> dict:
    """y: target sample index per source sample. Local speed over 1 s windows; jumps (> 1.5 s) excluded."""
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n < win + 2:
        return {"in_range": None, "rush": None, "longest_stall_s": None, "jumps": 0}
    dy = np.diff(y)
    jump = np.abs(dy) > 1.5 / DT          # a real discontinuity (loop back / skip), not a fast-forward
    h = win // 2
    i = np.arange(h, n - h)
    sp = (y[i + h] - y[i - h]) / win
    cj = np.r_[0, np.cumsum(jump)]
    clean = (cj[i + h] - cj[i - h]) == 0
    m = clean & np.isfinite(sp)
    if valid is not None:
        m &= valid[i]
    s = sp[m]
    stall = (sp < 0.25) & m
    best = cur = 0
    for v in stall:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return {"in_range": float(((s >= 0.5) & (s <= 2.0)).mean()) if len(s) else None,
            "rush": float((s > 4).mean()) if len(s) else None,
            "longest_stall_s": best * DT, "jumps": int(jump.sum())}


def balance(secs: list[SectionOut]) -> dict:
    """Inside paired repetitions: local speed of the correspondence, longest stall, and the share of
    practitioner samples that land on a reference sample receiving >= 4 practitioner samples (one-to-many)."""
    speeds, longest, many = [], 0.0, []
    for s in secs:
        for u in s.user_reps:
            if not (u.paired and u.path is not None):
                continue
            ui, ri = u.path
            st = speed_stats(ri.astype(np.float64))
            if st["in_range"] is not None:
                speeds.append((st["in_range"], len(ui)))
                longest = max(longest, st["longest_stall_s"])
            lo = int(ri.min())
            cnt = np.bincount(ri - lo)
            many.append(((cnt[ri - lo] >= 4).mean(), len(ri)))   # practitioner time piled onto one reference sample
    w = lambda pairs: float(np.average([p[0] for p in pairs], weights=[p[1] for p in pairs])) if pairs else None   # noqa: E731
    return {"speed_in_range": w(speeds), "longest_stall_s": longest, "one_to_many_share": w(many)}


def residual(pair: Pair, secs: list[SectionOut]) -> float | None:
    """Mean same-body joint distance (torso units) between corresponding frames, over paired user time."""
    Nr, Nu = pair.ref.N["sg"], pair.user.N["sg"]
    Nu_m = Nu[:, _LR].copy()
    Nu_m[..., 0] *= -1
    r, mir = paired_map(secs, len(Nu))
    idx = np.flatnonzero(np.isfinite(r))
    if not len(idx):
        return None
    ri = np.clip(np.round(r[idx]).astype(int), 0, len(Nr) - 1)
    U = np.where(mir[idx, None, None], Nu_m[idx], Nu[idx])
    d = np.linalg.norm(U - Nr[ri], axis=2)
    return float(np.nanmean(d))


# ----------------------------------------------------------------------------- against the annotation

def _iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def _match(pred: list[tuple[float, float]], gt: list[tuple[float, float]], thr: float = 0.5):
    pairs, used = [], set()
    cand = sorted(((_iou(p, g), i, k) for i, p in enumerate(pred) for k, g in enumerate(gt)), reverse=True)
    gi = set()
    for iou, i, k in cand:
        if iou < thr or i in used or k in gi:
            continue
        used.add(i)
        gi.add(k)
        pairs.append((i, k, iou))
    return pairs


def _interp_map(start: int, y: np.ndarray, x_s: float) -> float:
    xi = x_s / DT - start
    if xi < 0 or xi > len(y) - 1 or not np.isfinite(y).any():
        return np.nan
    ok = np.isfinite(y)
    return float(np.interp(xi, np.flatnonzero(ok), y[ok])) * DT


def vs_annotation(pair: Pair, secs: list[SectionOut], u2r, r2u) -> dict:
    gt = {s["id"]: s for s in pair.gt["sections"]}
    det = {"user": [], "ref": []}
    count_err = {"user": [], "ref": []}
    bound_err = []
    hold_u = []      # practitioner-master: GT practitioner hold centre -> shown reference time
    hold_r = []      # reference-master: GT reference hold centre -> shown practitioner time
    alt = []
    for s in secs:
        g = gt.get(s.id)
        bound_err.append(abs(s.u0 * DT - g["user"][0]))
        if not g.get("user_reps"):
            continue
        for side, pred_reps, gt_reps in (("user", s.user_reps, g["user_reps"]), ("ref", s.ref_reps, g["ref_reps"])):
            if not s.has_reps:
                continue
            P = [(u.start * DT, u.end * DT) for u in pred_reps]
            G = [(q[0], q[1]) for q in gt_reps]
            m = _match(P, G)
            det[side].append({"tp": len(m), "n_pred": len(P), "n_gt": len(G),
                              "start_err": [abs(P[i][0] - G[k][0]) for i, k, _ in m],
                              "end_err": [abs(P[i][1] - G[k][1]) for i, k, _ in m]})
            count_err[side].append(abs(len(P) - len(G)))
            if side == "user" and len(g["ref_reps"]) == 2 and len(m) >= 2:
                mk = sorted(m, key=lambda x: x[1])
                refs = [pred_reps[i].ref if pred_reps[i].paired else None for i, _, _ in mk]
                alt += [int(a is not None and b is not None and a != b) for a, b in zip(refs[:-1], refs[1:])]
        holds_r = [(q[2], q[3]) for q in g["ref_reps"]]
        holds_u = [(q[2], q[3]) for q in g["user_reps"]]
        for q in g["user_reps"]:
            c = 0.5 * (q[2] + q[3])
            r = _interp_map(u2r[0], u2r[1], c)
            hit = np.isfinite(r) and any(a - 0.5 <= r <= b + 0.5 for a, b in holds_r)
            err = min(abs(r - 0.5 * (a + b)) for a, b in holds_r) if np.isfinite(r) else np.nan
            hold_u.append((hit, err))
        for q in g["ref_reps"]:
            c = 0.5 * (q[2] + q[3])
            u = _interp_map(r2u[0], r2u[1], c)
            hit = np.isfinite(u) and any(a - 0.5 <= u <= b + 0.5 for a, b in holds_u)
            hold_r.append(hit)

    def agg(L):
        if not L:
            return None
        tp, npred, ngt = sum(d["tp"] for d in L), sum(d["n_pred"] for d in L), sum(d["n_gt"] for d in L)
        se = [e for d in L for e in d["start_err"]] + [e for d in L for e in d["end_err"]]
        p, r = tp / max(1, npred), tp / max(1, ngt)
        return {"precision": p, "recall": r, "f1": 2 * p * r / max(1e-9, p + r), "n_pred": npred, "n_gt": ngt,
                "boundary_mae_s": float(np.mean(se)) if se else None}
    errs = [e for _, e in hold_u if np.isfinite(e)]
    return {
        "user_reps": agg(det["user"]), "ref_reps": agg(det["ref"]),
        "count_abs_err": {k: (float(np.mean(v)) if v else None) for k, v in count_err.items()},
        "hold_hit_user_master": float(np.mean([h for h, _ in hold_u])) if hold_u else None,
        "hold_err_user_master_s": float(np.median(errs)) if errs else None,
        "hold_hit_ref_master": float(np.mean(hold_r)) if hold_r else None,
        "alternation_ok": float(np.mean(alt)) if alt else None,
        "section_start_mae_s": float(np.mean(bound_err)),
    }


def vs_offset(pair: Pair, u2r, r2u) -> dict:
    """Dance: the true map is user = ref + offset (audio-aligned)."""
    off = pair.gt_offset
    U0, y = u2r
    u = (U0 + np.arange(len(y))) * DT
    e_u = np.abs(y * DT - (u - off))
    R0, z = r2u
    r = (R0 + np.arange(len(z))) * DT
    e_r = np.abs(z * DT - (r + off))
    f = lambda e: {"median_s": float(np.nanmedian(e)), "p90_s": float(np.nanpercentile(e, 90))}   # noqa: E731
    return {"user_master": f(e_u), "ref_master": f(e_r)}


def score(pair: Pair, secs: list[SectionOut], policy: str = "loop") -> dict:
    u2r, r2u = playback_ui_m1(pair, secs) if policy == "ui" else playback_loop(pair, secs)
    out = {
        "balance": balance(secs),
        "residual": residual(pair, secs),
        "playback_user_master": speed_stats(u2r[1]),
        "playback_ref_master": speed_stats(r2u[1]),
        "paired_user_share": float(np.isfinite(paired_map(secs, len(pair.user.t))[0][secs[0].u0:secs[-1].u1]).mean()),
    }
    if pair.gt is not None:
        out["annotation"] = vs_annotation(pair, secs, u2r, r2u)
    if pair.gt_offset is not None:
        out["offset"] = vs_offset(pair, u2r, r2u)
    return out


__all__ = ["score", "playback_loop", "playback_ui_m1", "paired_map", "section_index"]
